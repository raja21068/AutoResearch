"""Decision engine: GO/NO_GO gates for research pipeline."""

import json, logging, re, time
from enum import Enum
from llm import llm

logger = logging.getLogger(__name__)

class DecisionType(str, Enum):
    GO = "go"; NO_GO = "no_go"; PIVOT = "pivot"; REFINE = "refine"

class DecisionEngine:
    SYSTEM = "You are a research decision engine. Decide: GO, NO_GO, PIVOT, or REFINE. Return JSON: {type, reasoning, confidence (0-1)}."

    async def decide(self, context, stage=""):
        raw = await llm(f"Stage: {stage}\nContext:\n{context[:4000]}\nDecide.", system=self.SYSTEM, agent="critic")
        if not raw.strip():
            logger.warning("DecisionEngine received empty LLM response — defaulting to REFINE")
            return {"type": DecisionType.REFINE.value, "reasoning": "LLM returned empty", "confidence": 0.0}
        m = re.search(r'"type"\s*:\s*"(\w+)"', raw)
        dtype = DecisionType(m.group(1).lower()) if m else DecisionType.REFINE
        c = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
        conf = float(c.group(1)) if c else 0.5
        return {"type": dtype.value, "reasoning": raw[:2000], "confidence": conf}

class DecisionService:
    def __init__(self): self.engine = DecisionEngine(); self.history = []
    async def gate(self, context, name, min_conf=0.6):
        d = await self.engine.decide(context, name)
        passed = d["type"] == "go" and d["confidence"] >= min_conf
        self.history.append({"gate": name, "decision": d, "passed": passed, "ts": time.time()})
        return passed
