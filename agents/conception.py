"""Conception agents: Guide, Generator, IdeaCritic, Ranker, Structurer."""

import logging
from llm import llm

logger = logging.getLogger(__name__)


class GuideAgent:
    SYSTEM = "You are a research guide. Help refine research questions into actionable directions."

    async def run(self, query, papers=""):
        result = await llm(f"Query: {query}\nPapers:\n{papers[:2000]}\nGuide.",
                           system=self.SYSTEM, agent="researcher")
        return result or "No guidance generated — LLM returned empty."


class GeneratorAgent:
    SYSTEM = "You are a research idea generator. Produce 5-10 novel ideas with title, hypothesis, methodology, feasibility (1-10)."

    async def run(self, topic, papers=""):
        result = await llm(f"Topic: {topic}\nLiterature:\n{papers[:3000]}\nGenerate ideas.",
                           system=self.SYSTEM, agent="researcher")
        return result or "No ideas generated — LLM returned empty."


class IdeaCriticAgent:
    SYSTEM = "You are a rigorous research critic. Evaluate ideas on novelty, soundness, feasibility, impact, clarity (1-10 each)."

    async def run(self, ideas, papers=""):
        result = await llm(f"Ideas:\n{ideas}\nLiterature:\n{papers[:2000]}\nCritique.",
                           system=self.SYSTEM, agent="critic")
        return result or "No critique generated — LLM returned empty."


class RankerAgent:
    SYSTEM = "You are a research idea ranker. Rank by expected value: novelty x impact x feasibility. Return top-3."

    async def run(self, ideas_with_critiques):
        result = await llm(f"Ideas+critiques:\n{ideas_with_critiques}\nRank top-3.",
                           system=self.SYSTEM, agent="critic")
        return result or "No ranking generated — LLM returned empty."


class StructurerAgent:
    SYSTEM = "You are a research structurer. Transform idea into formal plan: problem, questions, methodology, design, timeline."

    async def run(self, idea, papers=""):
        result = await llm(f"Idea:\n{idea}\nLiterature:\n{papers[:2000]}\nStructure.",
                           system=self.SYSTEM, agent="researcher")
        return result or "No structure generated — LLM returned empty."


class ConceptionService:
    """Full ideation pipeline: Guide -> Generate -> Critique -> Rank -> Structure."""

    def __init__(self):
        self.guide = GuideAgent()
        self.generator = GeneratorAgent()
        self.critic = IdeaCriticAgent()
        self.ranker = RankerAgent()
        self.structurer = StructurerAgent()

    async def run_full(self, query, papers="") -> dict:
        guidance = await self.guide.run(query, papers)
        ideas = await self.generator.run(query + "\n" + guidance, papers)
        critiques = await self.critic.run(ideas, papers)
        ranked = await self.ranker.run(ideas + "\n" + critiques)
        plan = await self.structurer.run(ranked, papers)
        return {"guidance": guidance, "ideas": ideas, "critiques": critiques,
                "ranked": ranked, "plan": plan}
