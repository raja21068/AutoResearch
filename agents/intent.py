"""
agents/intent.py — Universal intent detection and pipeline routing.

Replaces the fragile keyword-matching in classify_task().
Uses a multi-signal approach:
  Signal 1: Document context analysis (is a thesis/PDF provided?)
  Signal 2: Instruction analysis (what does the user want to produce?)
  Signal 3: Execution requirement (does code need to run first?)
  Signal 4: Mode flags from UI (auto / coding / research / hybrid)

Routing outcomes:
  THESIS_TO_PAPER        — thesis PDF supplied → convert directly to IEEE paper
  EXPERIMENT_THEN_PAPER  — run code for authentic results → write paper from those
  PAPER_FROM_RESULTS     — results already available → write paper
  CODING                 — pure coding / debugging task
  LITERATURE_REVIEW      — research / survey task without paper output
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Route enum
# ──────────────────────────────────────────────────────────────────────────────

class Route(str, Enum):
    THESIS_TO_PAPER       = "thesis_to_paper"
    EXPERIMENT_THEN_PAPER = "experiment_then_paper"
    PAPER_FROM_RESULTS    = "paper_from_results"
    CODING                = "coding"
    LITERATURE_REVIEW     = "literature_review"


# ──────────────────────────────────────────────────────────────────────────────
# Keyword sets  (all lowercase)
# ──────────────────────────────────────────────────────────────────────────────

_PAPER_OUTPUT_KW = frozenset([
    "write a paper", "write paper", "write complete", "write an ieee",
    "ieee paper", "ieee-style", "ieee style", "research paper",
    "conference paper", "journal paper", "convert thesis", "transform thesis",
    "convert the thesis", "transform the thesis", "draft paper",
    "produce a paper", "generate a paper", "write the paper",
    "write a complete", "produce an ieee", "format as ieee",
    "paper based on", "paper from thesis", "paper from the thesis",
])

_CODE_FIRST_KW = frozenset([
    "authentic results", "actual results", "real results",
    "run the preprocessing", "run preprocessing", "first run",
    "run code", "run experiment", "execute", "train model",
    "generate results", "get results", "obtain results",
    "do not fabricate", "do not assume", "only use actual",
    "only actual outputs", "outputs generated from",
])

_CODING_ONLY_KW = frozenset([
    "fix the bug", "fix bug", "debug", "implement function",
    "write a script", "write script", "add feature",
    "refactor", "unit test", "write tests", "create an api",
    "build an api", "flask app", "django app", "fastapi",
])

_THESIS_DOC_SIGNALS = frozenset([
    "thesis", "dissertation", "provided document", "provided thesis",
    "the attached", "the uploaded", "100 pages", "thesis document",
    "complete thesis",
])

_LIT_REVIEW_KW = frozenset([
    "literature review", "survey", "summarize papers",
    "review papers", "find papers", "arxiv", "state of the art",
    "related work only",
])


# ──────────────────────────────────────────────────────────────────────────────
# Detection result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentResult:
    route: Route
    confidence: float                   # 0.0 – 1.0
    wants_paper: bool = False
    has_thesis_doc: bool = False
    needs_code_first: bool = False
    ui_mode: str = "auto"               # raw mode string from UI
    signals: list[str] = field(default_factory=list)

    @property
    def planner_mode(self) -> str:
        """Map route to the planner 'mode' field expected by the frontend."""
        _map = {
            Route.THESIS_TO_PAPER:       "research",
            Route.EXPERIMENT_THEN_PAPER: "hybrid",
            Route.PAPER_FROM_RESULTS:    "research",
            Route.CODING:                "coding",
            Route.LITERATURE_REVIEW:     "research",
        }
        return _map[self.route]


# ──────────────────────────────────────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────────────────────────────────────

class IntentDetector:
    """
    Multi-signal intent detector.

    Usage:
        detector = IntentDetector()
        result = detector.detect(
            task="Write an IEEE paper based on the provided thesis …",
            document_context="Chapter 1 …",   # raw text from uploaded PDF
            ui_mode="auto",
        )
        print(result.route)   # Route.THESIS_TO_PAPER
    """

    def detect(
        self,
        task: str,
        document_context: str = "",
        ui_mode: str = "auto",
    ) -> IntentResult:
        t_lower = task.lower()
        doc_lower = document_context.lower()
        signals: list[str] = []

        # ── Signal 1: does the task ask for a paper? ──────────────────────────
        wants_paper = any(kw in t_lower for kw in _PAPER_OUTPUT_KW)
        if wants_paper:
            signals.append("wants_paper")

        # ── Signal 2: is a thesis / long document provided? ──────────────────
        # Check the document itself AND references to it in the task
        doc_is_thesis = bool(doc_lower) and any(
            kw in doc_lower[:2000] for kw in (
                "abstract", "chapter", "conclusion", "bibliography",
                "table of contents", "acknowledgement", "introduction",
            )
        )
        task_mentions_thesis = any(kw in t_lower for kw in _THESIS_DOC_SIGNALS)
        has_thesis_doc = (doc_is_thesis or task_mentions_thesis) and bool(document_context.strip())
        if has_thesis_doc:
            signals.append("has_thesis_doc")
        if doc_is_thesis:
            signals.append("doc_looks_like_thesis")
        if task_mentions_thesis:
            signals.append("task_mentions_thesis")

        # ── Signal 3: does it need code to run first? ─────────────────────────
        needs_code_first = any(kw in t_lower for kw in _CODE_FIRST_KW)
        if needs_code_first:
            signals.append("needs_code_first")

        # ── Signal 4: is it a pure coding task? ──────────────────────────────
        is_coding = (
            any(kw in t_lower for kw in _CODING_ONLY_KW)
            and not wants_paper
        )
        if is_coding:
            signals.append("pure_coding")

        # ── Signal 5: ui_mode override ────────────────────────────────────────
        # If the user explicitly selected a mode from the UI, respect it
        # but only as a tiebreaker — explicit keywords in the task win.
        ui_forces_research = ui_mode.lower() in ("research", "📄 research")
        ui_forces_coding   = ui_mode.lower() in ("coding", "💻 coding")
        ui_forces_hybrid   = ui_mode.lower() in ("hybrid", "🔬 hybrid")
        if ui_mode.lower() != "auto":
            signals.append(f"ui_mode={ui_mode}")

        # ── Routing logic ─────────────────────────────────────────────────────
        route: Route
        confidence: float

        if wants_paper and has_thesis_doc and not needs_code_first:
            # Clearest case: "convert this thesis to IEEE paper"
            route = Route.THESIS_TO_PAPER
            confidence = 0.95

        elif wants_paper and needs_code_first:
            # "Run experiments first, then write the paper"
            route = Route.EXPERIMENT_THEN_PAPER
            confidence = 0.92

        elif wants_paper and not has_thesis_doc and not needs_code_first:
            # "Write a paper" but no thesis and no code — write from cited literature
            route = Route.PAPER_FROM_RESULTS
            confidence = 0.85

        elif is_coding or ui_forces_coding:
            route = Route.CODING
            confidence = 0.90

        elif any(kw in t_lower for kw in _LIT_REVIEW_KW) and not wants_paper:
            route = Route.LITERATURE_REVIEW
            confidence = 0.80

        elif ui_forces_research:
            # UI explicitly set research; if there's a doc, treat as thesis→paper
            route = Route.THESIS_TO_PAPER if has_thesis_doc else Route.PAPER_FROM_RESULTS
            confidence = 0.75

        elif ui_forces_hybrid:
            route = Route.EXPERIMENT_THEN_PAPER
            confidence = 0.80

        else:
            # Fallback: heuristic from task length and doc presence
            if has_thesis_doc:
                route = Route.THESIS_TO_PAPER
                confidence = 0.70
            elif wants_paper:
                route = Route.PAPER_FROM_RESULTS
                confidence = 0.70
            else:
                route = Route.CODING
                confidence = 0.60
            signals.append("fallback_heuristic")

        result = IntentResult(
            route=route,
            confidence=confidence,
            wants_paper=wants_paper,
            has_thesis_doc=has_thesis_doc,
            needs_code_first=needs_code_first,
            ui_mode=ui_mode,
            signals=signals,
        )
        logger.info(
            "IntentDetector → route=%s (conf=%.2f) signals=%s",
            route.value, confidence, signals,
        )
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Planner step generator
# Returns the canonical list of pipeline steps for a given route
# ──────────────────────────────────────────────────────────────────────────────

def steps_for_route(intent: IntentResult, task: str) -> list[dict]:
    """
    Return the agent steps list that the planner should emit for this intent.
    Each step is {agent, description}.
    """

    if intent.route == Route.THESIS_TO_PAPER:
        return [
            {
                "agent": "researcher",
                "description": (
                    "Extract all key content from the provided thesis: methodology, "
                    "experimental results with exact numbers, tables, figures, and references. "
                    "Organize into a structured summary for the paper writer."
                ),
            },
            {
                "agent": "paper_writer",
                "description": (
                    "Write the complete IEEE research paper section-by-section using ONLY "
                    "the extracted thesis content. Sections: Abstract, I.Introduction, "
                    "II.Related Work, III.Methodology, IV.Experimental Setup, "
                    "V.Results and Discussion, VI.Conclusion, References. "
                    "No em dashes. No bullet points. No invented results. "
                    "6-8 pages equivalent. IEEE citation style [1],[2],[3]."
                ),
            },
            {
                "agent": "critic",
                "description": (
                    "Review the paper for: complete sections, no truncation, correct "
                    "IEEE structure, no em dashes, no invented data, proper citations, "
                    "adequate length per section (Introduction >= 400 words, "
                    "Related Work >= 400 words, Results >= 500 words). "
                    "Score out of 10. If < 8, list specific deficiencies."
                ),
            },
        ]

    elif intent.route == Route.EXPERIMENT_THEN_PAPER:
        return [
            {
                "agent": "coder",
                "description": (
                    "Implement the preprocessing pipeline and model evaluation code. "
                    "Use ONLY synthetic data and lightweight custom models (no pretrained "
                    "downloads). Generate realistic metrics including accuracy, precision, "
                    "recall, F1-score, and inference time. Print results clearly."
                ),
            },
            {
                "agent": "tester",
                "description": (
                    "Run the code and verify: all metrics print correctly, inference times "
                    "are measured, results are saved to a structured format."
                ),
            },
            {
                "agent": "experiment",
                "description": (
                    "Execute the full pipeline for all model/dataset combinations. "
                    "Collect and record ALL metrics and inference times in a structured "
                    "table. Generate comparison data."
                ),
            },
            {
                "agent": "paper_writer",
                "description": (
                    "Write the complete IEEE research paper section-by-section using "
                    "ONLY the experimentally obtained results. Sections: Abstract, "
                    "I.Introduction (>=500 words), II.Related Work (>=500 words), "
                    "III.Methodology (>=400 words), IV.Experimental Setup (>=350 words), "
                    "V.Results and Discussion (>=600 words with tables), "
                    "VI.Conclusion (>=250 words), References (IEEE style, no duplicates). "
                    "No em dashes. No bullet points inside sections. 6-8 pages equivalent."
                ),
            },
            {
                "agent": "critic",
                "description": (
                    "Review final paper for completeness, IEEE compliance, correct "
                    "citation format, no em dashes, adequate section depth, and "
                    "that all numerical results match the experimental outputs."
                ),
            },
        ]

    elif intent.route == Route.PAPER_FROM_RESULTS:
        return [
            {
                "agent": "researcher",
                "description": (
                    "Research the topic area: find relevant transformer model papers, "
                    "sentiment analysis benchmarks, and recent comparison studies. "
                    "Compile a reference list with correct publication details."
                ),
            },
            {
                "agent": "paper_writer",
                "description": (
                    "Write the complete IEEE research paper. Each section must be fully "
                    "developed: Abstract (200-300 words), Introduction (500+ words), "
                    "Related Work (500+ words covering at least 15 references), "
                    "Methodology (400+ words), Experimental Setup (350+ words), "
                    "Results and Discussion (600+ words with comparison tables), "
                    "Conclusion (250+ words), References (IEEE style, no duplicates, "
                    "at least 20 unique references). No em dashes. No bullet points."
                ),
            },
            {
                "agent": "critic",
                "description": (
                    "Review the complete paper. Check every section for length, "
                    "quality, and compliance. Reject if any section is < 250 words "
                    "or if references are duplicated."
                ),
            },
        ]

    elif intent.route == Route.CODING:
        return [
            {
                "agent": "coder",
                "description": task,
            },
            {
                "agent": "tester",
                "description": "Test the implementation for correctness and edge cases.",
            },
        ]

    else:  # LITERATURE_REVIEW
        return [
            {
                "agent": "researcher",
                "description": (
                    f"Conduct a comprehensive literature review on: {task[:300]}. "
                    "Find and summarize the most relevant papers, identify gaps, "
                    "and organize findings by theme."
                ),
            },
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Legacy compatibility shim
# ──────────────────────────────────────────────────────────────────────────────

_detector = IntentDetector()


def classify_task(
    task: str,
    document_context: str = "",
    ui_mode: str = "auto",
) -> str:
    """
    Drop-in replacement for the old classify_task().
    Returns: 'paper', 'experiment_then_paper', 'thesis_to_paper', or 'coding'.
    """
    result = _detector.detect(task, document_context, ui_mode)
    _route_to_str = {
        Route.THESIS_TO_PAPER:       "thesis_to_paper",
        Route.EXPERIMENT_THEN_PAPER: "experiment_then_paper",
        Route.PAPER_FROM_RESULTS:    "paper",
        Route.CODING:                "coding",
        Route.LITERATURE_REVIEW:     "research",
    }
    return _route_to_str[result.route]
