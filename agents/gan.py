"""
GAN-style multi-agent harness.

Inspired by Anthropic's harness design (March 2026).
Three agents iterate adversarially:
  1. Planner: expands prompt → full product spec + eval rubric
  2. Generator: builds the implementation
  3. Evaluator: ruthlessly tests against rubric, provides feedback
  
Loop continues until Evaluator score >= threshold or max iterations reached.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from llm import llm

logger = logging.getLogger(__name__)


@dataclass
class GANIteration:
    iteration: int
    code: str = ""
    feedback: str = ""
    score: float = 0.0
    passed: bool = False


@dataclass
class GANResult:
    spec: str = ""
    rubric: str = ""
    iterations: list = field(default_factory=list)
    final_code: str = ""
    final_score: float = 0.0
    passed: bool = False
    total_iterations: int = 0
    elapsed_sec: float = 0.0


GAN_PLANNER_SYSTEM = (
    "You are the Planner in a GAN-style multi-agent harness. "
    "Take a brief user prompt and expand it into a comprehensive product specification. "
    "Be deliberately ambitious — push for 12-16 features, rich design, polished UX. "
    "Output: 1) Product Specification with features list, 2) Evaluation Rubric with "
    "10 criteria scored 1-10. Return as structured text."
)

GAN_GENERATOR_SYSTEM = (
    "You are the Generator in a GAN-style harness. "
    "Read the spec and any previous feedback. Build a complete, working implementation. "
    "Address EVERY piece of feedback from the evaluator. Return ONLY code."
)

GAN_EVALUATOR_SYSTEM = (
    "You are the Evaluator in a GAN-style harness. Be RUTHLESSLY strict. "
    "Test the implementation against the eval rubric. For each criterion: "
    "score 1-10 with specific evidence. Calculate overall score (average). "
    "List every bug, missing feature, and UX issue. "
    "Return JSON: {\"scores\": {\"criterion\": score, ...}, \"overall\": float, "
    "\"bugs\": [...], \"missing\": [...], \"feedback\": \"...\"}"
)


class GANHarness:
    """Adversarial multi-agent loop: Plan → Generate → Evaluate → repeat."""

    def __init__(self, max_iterations: int = 10, pass_threshold: float = 7.5):
        self.max_iterations = max_iterations
        self.pass_threshold = pass_threshold

    async def run(self, prompt: str, context: str = "") -> GANResult:
        """Run the full GAN harness loop."""
        t0 = time.time()
        result = GANResult()

        # Phase 1: Plan
        logger.info("GAN Planner: expanding prompt")
        plan_prompt = f"User brief: {prompt}\n\nContext:\n{context[:2000]}\n\nWrite the spec and eval rubric."
        plan_output = await llm(plan_prompt, system=GAN_PLANNER_SYSTEM, agent="planner")
        result.spec = plan_output
        result.rubric = plan_output  # Both in same output

        # Phase 2-3: Generate → Evaluate loop
        feedback = ""
        for i in range(1, self.max_iterations + 1):
            logger.info("GAN iteration %d/%d", i, self.max_iterations)
            iteration = GANIteration(iteration=i)

            # Generate
            gen_prompt = (
                f"Spec:\n{result.spec[:3000]}\n\n"
                f"Iteration {i}.\n"
            )
            if feedback:
                gen_prompt += f"PREVIOUS FEEDBACK (address ALL issues):\n{feedback[:2000]}\n\n"
            gen_prompt += "Build the complete implementation."

            code = await llm(gen_prompt, system=GAN_GENERATOR_SYSTEM, agent="coder")
            iteration.code = code

            # Evaluate
            eval_prompt = (
                f"Eval Rubric:\n{result.rubric[:2000]}\n\n"
                f"Implementation (iteration {i}):\n{code[:4000]}\n\n"
                "Evaluate ruthlessly. Return JSON with scores, overall, bugs, missing, feedback."
            )
            eval_output = await llm(eval_prompt, system=GAN_EVALUATOR_SYSTEM, agent="critic")
            iteration.feedback = eval_output

            # Parse score
            try:
                import re
                m = re.search(r'"overall"\s*:\s*([0-9]+\.?[0-9]*)', eval_output)
                if m:
                    iteration.score = float(m.group(1))
            except Exception:
                iteration.score = 0.0

            iteration.passed = iteration.score >= self.pass_threshold
            feedback = eval_output
            result.iterations.append(iteration)

            if iteration.passed:
                logger.info("GAN passed at iteration %d (score=%.1f)", i, iteration.score)
                break

        # Final result
        result.final_code = result.iterations[-1].code if result.iterations else ""
        result.final_score = result.iterations[-1].score if result.iterations else 0.0
        result.passed = result.final_score >= self.pass_threshold
        result.total_iterations = len(result.iterations)
        result.elapsed_sec = time.time() - t0
        return result
