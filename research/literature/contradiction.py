"""Contradiction detection engine — find conflicting claims across sources.

Compares claims extracted from multiple papers/sources and identifies
direct contradictions, partial conflicts, and tension pairs.

Pipeline:
  1. ``ClaimExtractor``   — pull atomic factual claims from each source
  2. ``ContradictionDetector`` — compare claims pairwise via LLM + heuristics
  3. ``ContradictionReport``   — ranked list of conflicts with evidence

Severity levels:
  - ``DIRECT``    : Source A explicitly states the opposite of Source B
  - ``PARTIAL``   : Sources agree on direction but disagree on magnitude/scope
  - ``TENSION``   : Sources emphasize incompatible aspects without direct clash
  - ``OUTDATED``  : Older source contradicts newer (likely superseded, not wrong)

Zero extra dependencies — stdlib + the existing LLMClient protocol.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM protocol (structural typing — mirrors agents/base.py convention)
# ---------------------------------------------------------------------------


class _LLMClientLike(Protocol):  # pragma: no cover
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Public enums & dataclasses
# ---------------------------------------------------------------------------


class ContradictionSeverity(str, Enum):
    DIRECT = "direct"
    PARTIAL = "partial"
    TENSION = "tension"
    OUTDATED = "outdated"


@dataclass
class Claim:
    """A single atomic factual claim extracted from a source."""

    text: str
    source_id: str          # paper_id, URL, or doc identifier
    source_title: str
    source_year: int = 0
    confidence: float = 1.0  # extractor confidence 0.0–1.0
    quote: str = ""          # verbatim excerpt supporting the claim

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_id": self.source_id,
            "source_title": self.source_title,
            "source_year": self.source_year,
            "confidence": round(self.confidence, 3),
            "quote": self.quote,
        }


@dataclass
class ContradictionPair:
    """A detected contradiction between two claims from different sources."""

    claim_a: Claim
    claim_b: Claim
    severity: ContradictionSeverity
    explanation: str
    confidence: float           # detector confidence 0.0–1.0
    topic: str = ""             # shared topic these claims address

    @property
    def is_significant(self) -> bool:
        return (
            self.severity in (ContradictionSeverity.DIRECT, ContradictionSeverity.PARTIAL)
            and self.confidence >= 0.6
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_a": self.claim_a.to_dict(),
            "claim_b": self.claim_b.to_dict(),
            "severity": self.severity.value,
            "explanation": self.explanation,
            "confidence": round(self.confidence, 3),
            "topic": self.topic,
            "is_significant": self.is_significant,
        }


@dataclass
class ContradictionReport:
    """Full contradiction analysis for a set of sources."""

    total_claims: int = 0
    total_pairs_checked: int = 0
    contradictions: list[ContradictionPair] = field(default_factory=list)
    topics_with_conflict: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Derived metrics                                                       #
    # ------------------------------------------------------------------ #

    @property
    def direct_count(self) -> int:
        return sum(
            1 for c in self.contradictions
            if c.severity == ContradictionSeverity.DIRECT
        )

    @property
    def significant_count(self) -> int:
        return sum(1 for c in self.contradictions if c.is_significant)

    @property
    def conflict_rate(self) -> float:
        """Fraction of checked pairs that produced a contradiction."""
        if self.total_pairs_checked == 0:
            return 0.0
        return round(len(self.contradictions) / self.total_pairs_checked, 3)

    @property
    def trust_penalty(self) -> float:
        """0.0–1.0 penalty to apply to overall confidence score.

        Direct contradictions hurt most; tensions hurt least.
        """
        score = 0.0
        for c in self.contradictions:
            if c.severity == ContradictionSeverity.DIRECT:
                score += 0.15 * c.confidence
            elif c.severity == ContradictionSeverity.PARTIAL:
                score += 0.07 * c.confidence
            elif c.severity == ContradictionSeverity.TENSION:
                score += 0.03 * c.confidence
            # OUTDATED has no penalty — it's informational
        return min(round(score, 3), 1.0)

    def top_contradictions(self, n: int = 5) -> list[ContradictionPair]:
        """Return top-N contradictions sorted by severity then confidence."""
        _order = {
            ContradictionSeverity.DIRECT: 0,
            ContradictionSeverity.PARTIAL: 1,
            ContradictionSeverity.TENSION: 2,
            ContradictionSeverity.OUTDATED: 3,
        }
        return sorted(
            self.contradictions,
            key=lambda c: (_order[c.severity], -c.confidence),
        )[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "total_pairs_checked": self.total_pairs_checked,
            "contradiction_count": len(self.contradictions),
            "direct_count": self.direct_count,
            "significant_count": self.significant_count,
            "conflict_rate": self.conflict_rate,
            "trust_penalty": self.trust_penalty,
            "topics_with_conflict": self.topics_with_conflict,
            "contradictions": [c.to_dict() for c in self.contradictions],
        }

    def format_summary(self) -> str:
        """Human-readable summary for inclusion in research reports."""
        if not self.contradictions:
            return "No contradictions detected across sources."

        lines = [
            f"⚠️  Found {len(self.contradictions)} contradiction(s) across "
            f"{self.total_claims} claims ({self.direct_count} direct).",
            "",
        ]
        for i, pair in enumerate(self.top_contradictions(5), 1):
            icon = {"direct": "🔴", "partial": "🟡",
                    "tension": "🟠", "outdated": "⬜"}[pair.severity.value]
            lines.append(
                f"{i}. {icon} [{pair.severity.value.upper()}] {pair.topic or 'General'}"
            )
            lines.append(f"   • {pair.claim_a.source_title}: \"{pair.claim_a.text}\"")
            lines.append(f"   • {pair.claim_b.source_title}: \"{pair.claim_b.text}\"")
            lines.append(f"   → {pair.explanation}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claim extractor
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """\
You are a scientific fact extractor. Given a document excerpt, extract
atomic factual claims — one idea per claim, stated as a complete sentence.

Rules:
- Each claim must be independently verifiable
- Avoid vague claims ("this is important")
- Include quantitative claims when present ("accuracy improved by 12%")
- Maximum 10 claims per document
- Respond ONLY with valid JSON — no preamble, no markdown fences

Format:
{
  "claims": [
    {"text": "...", "confidence": 0.9, "quote": "verbatim excerpt"},
    ...
  ]
}
"""


class ClaimExtractor:
    """Extract atomic factual claims from document text using an LLM."""

    def __init__(self, llm: _LLMClientLike) -> None:
        self._llm = llm

    def extract(
        self,
        text: str,
        source_id: str,
        source_title: str,
        source_year: int = 0,
        max_claims: int = 10,
    ) -> list[Claim]:
        """Extract claims from *text* and tag them with source metadata."""
        truncated = text[:6000]  # keep within token budget
        prompt = (
            f"Extract up to {max_claims} factual claims from this document:\n\n"
            f"{truncated}"
        )

        try:
            response = self._llm.chat(
                [{"role": "user", "content": prompt}],
                system=_EXTRACT_SYSTEM,
                max_tokens=1500,
                temperature=0.1,
                json_mode=True,
            )
            raw = response.content if hasattr(response, "content") else str(response)
            data = _safe_json(raw)
            raw_claims = data.get("claims", []) if isinstance(data, dict) else []
        except Exception as exc:
            logger.warning("ClaimExtractor LLM call failed for %s: %s", source_id, exc)
            return []

        claims: list[Claim] = []
        for item in raw_claims[:max_claims]:
            if not isinstance(item, dict) or not item.get("text"):
                continue
            claims.append(
                Claim(
                    text=str(item["text"]).strip(),
                    source_id=source_id,
                    source_title=source_title,
                    source_year=source_year,
                    confidence=float(item.get("confidence", 0.8)),
                    quote=str(item.get("quote", ""))[:300],
                )
            )

        logger.debug(
            "Extracted %d claims from '%s'", len(claims), source_title[:60]
        )
        return claims


# ---------------------------------------------------------------------------
# Contradiction detector
# ---------------------------------------------------------------------------

_COMPARE_SYSTEM = """\
You are a scientific contradiction detector. Given two factual claims from
different sources, determine if they contradict each other.

Severity levels:
- "direct"   : The claims make opposite assertions about the same fact
- "partial"  : They agree directionally but disagree on degree, scope, or conditions
- "tension"  : They emphasize incompatible aspects without direct logical conflict
- "outdated" : The older claim was likely superseded by the newer one
- "none"     : No meaningful contradiction

Respond ONLY with valid JSON:
{
  "severity": "direct" | "partial" | "tension" | "outdated" | "none",
  "confidence": 0.0-1.0,
  "explanation": "one concise sentence",
  "topic": "the shared subject these claims address"
}
"""


class ContradictionDetector:
    """Detect contradictions between pairs of claims using an LLM."""

    def __init__(self, llm: _LLMClientLike, confidence_threshold: float = 0.5) -> None:
        self._llm = llm
        self._threshold = confidence_threshold

    def compare(self, claim_a: Claim, claim_b: Claim) -> ContradictionPair | None:
        """Compare two claims. Returns a ``ContradictionPair`` or ``None``."""
        # Fast heuristic skip: same source → not a contradiction
        if claim_a.source_id == claim_b.source_id:
            return None

        prompt = (
            f"Claim A (from '{claim_a.source_title}', {claim_a.source_year}):\n"
            f"\"{claim_a.text}\"\n\n"
            f"Claim B (from '{claim_b.source_title}', {claim_b.source_year}):\n"
            f"\"{claim_b.text}\""
        )

        try:
            response = self._llm.chat(
                [{"role": "user", "content": prompt}],
                system=_COMPARE_SYSTEM,
                max_tokens=300,
                temperature=0.0,
                json_mode=True,
            )
            raw = response.content if hasattr(response, "content") else str(response)
            data = _safe_json(raw)
        except Exception as exc:
            logger.debug("Contradiction compare failed: %s", exc)
            return None

        if not isinstance(data, dict):
            return None

        severity_str = str(data.get("severity", "none")).lower()
        if severity_str == "none":
            return None

        try:
            severity = ContradictionSeverity(severity_str)
        except ValueError:
            return None

        confidence = float(data.get("confidence", 0.0))
        if confidence < self._threshold:
            return None

        return ContradictionPair(
            claim_a=claim_a,
            claim_b=claim_b,
            severity=severity,
            explanation=str(data.get("explanation", "")).strip(),
            confidence=confidence,
            topic=str(data.get("topic", "")).strip(),
        )


# ---------------------------------------------------------------------------
# Top-level analysis function
# ---------------------------------------------------------------------------


def analyze_contradictions(
    sources: list[dict[str, Any]],
    llm: _LLMClientLike,
    *,
    max_claims_per_source: int = 8,
    confidence_threshold: float = 0.5,
    max_pairs: int = 200,
) -> ContradictionReport:
    """Full contradiction analysis across a list of sources.

    Args:
        sources: List of dicts with keys:
            - ``id``    (str)  source identifier
            - ``title`` (str)  human-readable name
            - ``text``  (str)  document body
            - ``year``  (int, optional) publication year
        llm: Any object matching ``_LLMClientLike``.
        max_claims_per_source: Cap per-source claim extraction.
        confidence_threshold: Minimum confidence to record a contradiction.
        max_pairs: Limit pairwise comparisons for large source sets.

    Returns:
        ``ContradictionReport`` with all findings.
    """
    extractor = ClaimExtractor(llm)
    detector = ContradictionDetector(llm, confidence_threshold=confidence_threshold)
    report = ContradictionReport()

    # ---- Step 1: extract claims from every source ----------------------- #
    all_claims: list[Claim] = []
    for src in sources:
        claims = extractor.extract(
            text=src.get("text", ""),
            source_id=str(src.get("id", "")),
            source_title=str(src.get("title", "Unknown")),
            source_year=int(src.get("year", 0)),
            max_claims=max_claims_per_source,
        )
        all_claims.extend(claims)

    report.total_claims = len(all_claims)
    logger.info("Extracted %d total claims from %d sources", len(all_claims), len(sources))

    # ---- Step 2: pairwise comparison ------------------------------------ #
    pairs_checked = 0
    seen_topics: set[str] = set()

    for i, claim_a in enumerate(all_claims):
        for claim_b in all_claims[i + 1:]:
            if pairs_checked >= max_pairs:
                break
            pairs_checked += 1
            result = detector.compare(claim_a, claim_b)
            if result is not None:
                report.contradictions.append(result)
                if result.topic:
                    seen_topics.add(result.topic)
        if pairs_checked >= max_pairs:
            break

    report.total_pairs_checked = pairs_checked
    report.topics_with_conflict = sorted(seen_topics)

    logger.info(
        "Contradiction analysis complete: %d pairs checked, %d contradictions "
        "(%d direct) — trust_penalty=%.2f",
        pairs_checked,
        len(report.contradictions),
        report.direct_count,
        report.trust_penalty,
    )
    return report


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _safe_json(raw: str) -> Any:
    """Parse JSON, stripping markdown fences if present."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
