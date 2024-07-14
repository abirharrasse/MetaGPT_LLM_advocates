import asyncio
import platform
import fire
from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team

# Global variables for debate content
QUESTION = "Should AI be regulated?"
ANSWER1 = "Yes, it should"
ANSWER2 = "No, no need for regulation"

scores = []

class DefendAnswer(Action):
    PROMPT_TEMPLATE: str = """
    You're an eloquent advocate defending this answer: {answer} to the question: {question}. Your opponent
    defends: {opponent_answer}. Convince the audience your argument is better. Use the latest feedback and your opponent's last argument to improve your case.
    Latest feedback: {feedback}
    Opponent's last argument: {opponent_argument}
    Respond in under 80 words.
    Your defense:
    """
    name: str = "DefendAnswer"

    async def run(self, question: str, answer: str, opponent_answer: str, opponent_argument: str = "", feedback: str = ""):
        prompt = self.PROMPT_TEMPLATE.format(question=question, answer=answer, opponent_answer=opponent_answer, 
                                             opponent_argument=opponent_argument, feedback=feedback)
        return await self._aask(prompt)


class JudgeAnswer(Action):
    PROMPT_TEMPLATE: str = """
    You're a fair, impartial judge. Your role is to see into the defenses of two advocates who are defending two different answers: answer 1: 
    {answer1} and answer 2: {answer2} to a given question: {question}. 
    You have to give feedback that will help them improve their defenses and will help you make more difference between the 2 answers
    to give them scores at the end that are widely distant from each other. You will be provided with the defenses of the 2 advocates.
    Defense for 1st answer: {defense1}
    Defense for 2nd answer: {defense2} 
    Give your feedback directly in under 50 words.
    Your feedback: 
    """
    name: str = "JudgeAnswer"

    async def run(self, question: str, answer1: str, answer2: str, defense1: str, defense2: str):
        prompt = self.PROMPT_TEMPLATE.format(question=question, answer1=answer1, answer2=answer2, defense1=defense1, defense2=defense2)
        rsp = await self._aask(prompt)
        return rsp


class ScoreAnswer(Action):
    PROMPT_TEMPLATE: str = """
    You're a fair, impartial judge. Your role is to see into the defenses of two advocates who are defending two different answers: answer 1: 
    {answer1} and answer 2: {answer2} to a given question: {question}. 
    A debate has been going between the 2 advocates and now they are at their last round. They will provide their latest defenses that you should
    see into and then score. Scores should be between 1 and 20. The best answer/defense should get a score close to 20, the other one, a score close
    to 1.
    Last defense for 1st answer: {defense1}
    Last defense for 2nd answer: {defense2} 
    Now, return the score in the format (score1, score2), where score1 is the score of the 1st answer, and score2 of the 2nd answer.
    Example: if scores are 2 and 18, return only the tuple (2, 18).
    Your scores: 
    """
    name: str = "ScoreAnswer"

    async def run(self, question: str, answer1: str, answer2: str, defense1: str, defense2: str):
        prompt = self.PROMPT_TEMPLATE.format(question=question, answer1=answer1, answer2=answer2, defense1=defense1, defense2=defense2)
        rsp = await self._aask(prompt)
        return rsp





class Advocate(Role):
    def __init__(self, name: str, answer: str, opponent_answer: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.answer = answer
        self.opponent_answer = opponent_answer
        self.defend_action = DefendAnswer()
        self.set_actions([self.defend_action])
        self._watch([DefendAnswer])

    async def _act(self) -> Message:
        logger.info(f"{self.name}: Preparing argument")
        memories = self.rc.memory.get_by_role(role=self.name)
        opponent_memories = self.rc.memory.get_by_role(role=f"Opponent of {self.name}")
        
        opponent_argument = opponent_memories[-1].content if opponent_memories else ""
        feedback = self.rc.memory.get_by_role(role="Judge")[-1].content if self.rc.memory.get_by_role(role="Judge") else ""

        new_defense = await self.defend_action.run(question=QUESTION, answer=self.answer, opponent_answer=self.opponent_answer,
                                     opponent_argument=opponent_argument, feedback=feedback)

        msg = Message(content=new_defense, role=self.name, cause_by=DefendAnswer)
        self.rc.memory.add(msg)
        return msg

class Judge(Role):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Judge"
        self.judge_action = JudgeAnswer()
        self.set_actions([self.judge_action])
        self._watch([DefendAnswer])

    async def _act(self) -> Message:
        logger.info("Judge: Evaluating arguments")
        memories = self.rc.memory.get(k=2)
        if len(memories) < 2:
            return Message(content="Waiting for more arguments.", role=self.name)

        advocate1_arg = memories[-2].content
        advocate2_arg = memories[-1].content

        evaluation = await self.judge_action.run(question=QUESTION, answer1=ANSWER1, answer2=ANSWER2, defense1=advocate1_arg, defense2=advocate2_arg)

        msg = Message(content=evaluation, role=self.name)
        self.rc.memory.add(msg)
        return msg

class Scorer(Role):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Scorer"
        self.score_action = ScoreAnswer()
        self.set_actions([self.score_action])
        self._watch([DefendAnswer])

    async def _act(self) -> Message:
        logger.info("Judge: Evaluating arguments")
        memories = self.rc.memory.get(k=2)
        if len(memories) < 2:
            return Message(content="Waiting for more arguments.", role=self.name)

        advocate1_arg = memories[-2].content
        advocate2_arg = memories[-1].content

        evaluation = await self.score_action.run(question=QUESTION, answer1=ANSWER1, answer2=ANSWER2, defense1=advocate1_arg, defense2=advocate2_arg)

        msg = Message(content=evaluation, role=self.name)
        self.rc.memory.add(msg)
        return msg

async def debate(investment: float = 3.0, n_round: int = 5):
    advocate1 = Advocate(name="Advocate1", answer=ANSWER1, opponent_answer=ANSWER2)
    advocate2 = Advocate(name="Advocate2", answer=ANSWER2, opponent_answer=ANSWER1)
    judge = Judge()
    scorer = Scorer()
    
    print(f"Debate Question: {QUESTION}")
    print(f"Advocate1 defends: {ANSWER1}")
    print(f"Advocate2 defends: {ANSWER2}\n")

    # Manually start the debate
    initial_msg = Message(content=QUESTION, role="Human", cause_by=DefendAnswer)
    advocate1.rc.memory.add(initial_msg)

    for i in range(n_round):
        print(f"Round {i+1}:")
        
        # Advocate1's turn
        msg1 = await advocate1._act()
        advocate2.rc.memory.add(Message(content=msg1.content, role="Opponent of Advocate2", cause_by=DefendAnswer))
        judge.rc.memory.add(msg1)  # Add to Judge's memory
        scorer.rc.memory.add(msg1)
        print(f"Advocate1: {msg1.content}")

        # Advocate2's turn
        msg2 = await advocate2._act()
        advocate1.rc.memory.add(Message(content=msg2.content, role="Opponent of Advocate1", cause_by=DefendAnswer))
        judge.rc.memory.add(msg2)  # Add to Judge's memory
        scorer.rc.memory.add(msg2)
        print(f"Advocate2: {msg2.content}")

        # Judge's turn
        judge_msg = await judge._act()
        advocate1.rc.memory.add(judge_msg)
        advocate2.rc.memory.add(judge_msg)
        score_msg = await scorer._act()
        print(f"Judge: {judge_msg.content}")
        print(f"Scorer: {score_msg.content}")
        scores.append(score_msg.content)

        print()  # Add a blank line between rounds

def main(investment: float = 0.1, n_round: int = 3):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(debate(investment, n_round))

if __name__ == "__main__":
    fire.Fire(main)
