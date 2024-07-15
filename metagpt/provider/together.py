# In metagpt/provider/together.py

from typing import Optional, Union
from together import Together

from metagpt.provider.base_llm import BaseLLM
from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.provider.llm_provider_registry import register_provider
from metagpt.const import USE_CONFIG_TIMEOUT
from metagpt.logs import logger
from metagpt.utils.cost_manager import CostManager
from metagpt.utils.token_counter import count_message_tokens, get_max_completion_tokens

@register_provider(LLMType.TOGETHER)
class TogetherLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        self.config = config
        self._init_together()
        self.auto_max_tokens = False
        self.cost_manager: Optional[CostManager] = None

    def _init_together(self):
        self.model = self.config.model
        self.client = Together(api_key=self.config.api_key)
        if self.config.base_url:
            self.client.base_url = self.config.base_url

    def _cons_kwargs(self, messages: list[dict], stream: bool = False, **kwargs) -> dict:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self._get_max_tokens(messages),
            "stream": stream,
        }
        return kwargs

    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        kwargs = self._cons_kwargs(messages, timeout=self.get_timeout(timeout))
        response = self.client.chat.completions.create(**kwargs)
        self._update_costs(response.usage)
        return response

    async def _achat_completion_stream(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        kwargs = self._cons_kwargs(messages, stream=True, timeout=self.get_timeout(timeout))
        response = self.client.chat.completions.create(**kwargs)
        collected_messages = []
        async for chunk in response:
            if chunk.choices:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                yield chunk_message

        full_content = "".join(collected_messages)
        usage = self._calc_usage(messages, full_content)
        self._update_costs(usage)

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    async def acompletion_text(self, messages: list[dict], stream=False, timeout=USE_CONFIG_TIMEOUT) -> str:
        if stream:
            collected_messages = []
            async for chunk in self._achat_completion_stream(messages, timeout=timeout):
                collected_messages.append(chunk)
            return "".join(collected_messages)

        response = await self._achat_completion(messages, timeout=self.get_timeout(timeout))
        return self.get_choice_text(response)

    def _calc_usage(self, messages: list[dict], response_text: str):
        prompt_tokens = count_message_tokens(messages, self.model)
        completion_tokens = count_message_tokens([{"role": "assistant", "content": response_text}], self.model)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }

    def get_choice_text(self, rsp) -> str:
        return rsp.choices[0].message.content if rsp.choices else ""

    def _get_max_tokens(self, messages: list[dict]) -> int:
        if not self.auto_max_tokens:
            return self.config.max_token
        return get_max_completion_tokens(messages, self.model, self.config.max_token)

    def _update_costs(self, usage):
        if self.cost_manager:
            self.cost_manager.update_cost(usage, self.model)
