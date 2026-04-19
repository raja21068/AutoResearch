"""Claim-level citation registry — track every claim back to its source.

Every factual claim the LLM produces during paper writing is registered
here with a pointer to the source document, quote, and confidence score.
At any point you can audit which claims are grounded vs. potentially
hallucinated.

Components:
  ``ClaimRegistry``     — central store, persisted as JSONL
  ``CitedClaim``        — a single traceable claim
  ``RegistryReport``    — aggregate audit with hallucination risk score
  ``claim_guard``       — context-manager that auto-registers claims
                          produced by an LLM writing call

Integration points:
  - Call ``registry.register()`` immediately after every LLM generation step
  - Call ``registry.audit()`` before QUALITY_GATE (Stage 20)
  - The ``trust_penalty`` from ``ContradictionReport`` combines with
    ``hallucination_risk`` here to produce the final confidence score
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterator, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM protocol (structural typing)
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
# Core data classes
# ---------------------------------------------------------------------------


class GroundingStatus(str):
    """Grounding status constants."""
    GROUNDED = "grounded"        # claim verified against source quote
    INFERRED = "inferred"        # reasonable inference, no direct quote
    UNVERIFIED = "unverified"    # no source provided yet
    HALLUCINATED = "hallucinated"  # LLM marked it as likely fabricated


@dataclass
class SourceRef:
    """Pointer to the source document backing a claim."""

    source_id: str          # paper_id, URL, or doc key
    source_title: str
    source_year: int = 0
    url: str = ""
    quote: str = ""         # verbatim excerpt (≤ 300 chars)
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source_id": self.source_id,
            "source_title": self.source_title,
            "source_year": self.source_year,
            "url": self.url,
            "quote": self.quote,
        }
        if self.page is not None:
            d["page"] = self.page
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceRef:
        return cls(
            source_id=str(data.get("source_id", "")),
            source_title=str(data.get("source_title", "")),
            source_year=int(data.get("source_year", 0)),
            url=str(data.get("url", "")),
            quote=str(data.get("quote", "")),
            page=data.get("page"),
        )


@dataclass
class CitedClaim:
    """A single traceable factual claim produced during paper writing."""

    claim_id: str
    text: str                       # the claim as written in the paper
    stage: str                      # pipeline stage that produced it (e.g. "PAPER_DRAFT")
    grounding: str                  # GroundingStatus value
    confidence: float               # 0.0–1.0
    sources: list[SourceRef] = field(default_factory=list)
    created_at: str = ""
    section: str = ""               # paper section (e.g. "Introduction")
    llm_model: str = ""             # model that generated this claim

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.claim_id:
            self.claim_id = str(uuid.uuid4())[:8]

    @property
    def is_risky(self) -> bool:
        """True if this claim needs human review before publishing."""
        return (
            self.grounding in (GroundingStatus.UNVERIFIED, GroundingStatus.HALLUCINATED)
            or (self.grounding == GroundingStatus.INFERRED and self.confidence < 0.6)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "stage": self.stage,
            "grounding": self.grounding,
            "confidence": round(self.confidence, 3),
            "sources": [s.to_dict() for s in self.sources],
            "created_at": self.created_at,
            "section": self.section,
            "llm_model": self.llm_model,
            "is_risky": self.is_risky,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CitedClaim:
        sources = [
            SourceRef.from_dict(s)
            for s in data.get("sources", [])
            if isinstance(s, dict)
        ]
        return cls(
            claim_id=str(data.get("claim_id", "")),
            text=str(data.get("text", "")),
            stage=str(data.get("stage", "")),
            grounding=str(data.get("grounding", GroundingStatus.UNVERIFIED)),
            confidence=float(data.get("confidence", 0.5)),
            sources=sources,
            created_at=str(data.get("created_at", "")),
            section=str(data.get("section", "")),
            llm_model=str(data.get("llm_model", "")),
        )


# ---------------------------------------------------------------------------
# Registry report
# ---------------------------------------------------------------------------


@dataclass
class RegistryReport:
    """Audit summary of all claims in the registry."""

    total: int = 0
    grounded: int = 0
    inferred: int = 0
    unverified: int = 0
    hallucinated: int = 0
    risky_claims: list[CitedClaim] = field(default_factory=list)

    @property
    def hallucination_risk(self) -> float:
        """0.0–1.0 risk score — higher means more ungrounded claims."""
        if self.total == 0:
            return 0.0
        risky = self.unverified + self.hallucinated + (self.inferred * 0.3)
        return round(min(risky / self.total, 1.0), 3)

    @property
    def grounding_rate(self) -> float:
        """Fraction of claims that are fully grounded."""
        if self.total == 0:
            return 1.0
        return round(self.grounded / self.total, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "grounded": self.grounded,
            "inferred": self.inferred,
            "unverified": self.unverified,
            "hallucinated": self.hallucinated,
            "hallucination_risk": self.hallucination_risk,
            "grounding_rate": self.grounding_rate,
            "risky_claims": [c.to_dict() for c in self.risky_claims],
        }

    def format_summary(self) -> str:
        """Human-readable summary for research reports."""
        icon = "✅" if self.hallucination_risk < 0.2 else ("⚠️" if self.hallucination_risk < 0.5 else "🔴")
        lines = [
            f"{icon} Citation audit: {self.total} claims — "
            f"{self.grounded} grounded, {self.inferred} inferred, "
            f"{self.unverified} unverified, {self.hallucinated} hallucinated",
            f"   Grounding rate: {self.grounding_rate:.0%}  |  "
            f"Hallucination risk: {self.hallucination_risk:.0%}",
        ]
        if self.risky_claims:
            lines.append(f"\n   ⚠️  {len(self.risky_claims)} claims need review:")
            for c in self.risky_claims[:5]:
                lines.append(f"   • [{c.grounding}] \"{c.text[:80]}...\"")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry (persistence layer)
# ---------------------------------------------------------------------------

_REGISTRY_FILENAME = "claim_registry.jsonl"


class ClaimRegistry:
    """Persistent store of all cited claims produced during a pipeline run.

    Backed by a JSONL file (one claim per line) following the same pattern
    as ``MemoryStore``.  Append-only during a run; loaded on resume.

    Usage::

        registry = ClaimRegistry(run_dir / "stage-17")
        registry.load()

        claim = registry.register(
            text="NVIDIA holds 80% of the AI accelerator market.",
            stage="PAPER_DRAFT",
            sources=[SourceRef(source_id="s2:abc", source_title="TechReport 2024",
                               source_year=2024, quote="NVIDIA... 80%...")],
            grounding=GroundingStatus.GROUNDED,
            confidence=0.92,
            section="Introduction",
        )

        report = registry.audit()
        print(report.format_summary())
    """

    def __init__(self, store_dir: str | Path) -> None:
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / _REGISTRY_FILENAME
        self._claims: list[CitedClaim] = []

    # ------------------------------------------------------------------ #
    # Persistence                                                           #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load claims from disk (call on pipeline resume)."""
        if not self._path.exists():
            return
        loaded = 0
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._claims.append(CitedClaim.from_dict(json.loads(line)))
                    loaded += 1
                except Exception as exc:
                    logger.debug("Skipping malformed claim line: %s", exc)
        logger.info("Loaded %d claims from %s", loaded, self._path)

    def _append_to_disk(self, claim: CitedClaim) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(claim.to_dict(), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    # Registration                                                          #
    # ------------------------------------------------------------------ #

    def register(
        self,
        text: str,
        stage: str,
        *,
        sources: list[SourceRef] | None = None,
        grounding: str = GroundingStatus.UNVERIFIED,
        confidence: float = 0.5,
        section: str = "",
        llm_model: str = "",
    ) -> CitedClaim:
        """Register a new claim and persist it immediately."""
        claim = CitedClaim(
            claim_id=str(uuid.uuid4())[:8],
            text=text.strip(),
            stage=stage,
            grounding=grounding,
            confidence=confidence,
            sources=sources or [],
            section=section,
            llm_model=llm_model,
        )
        self._claims.append(claim)
        self._append_to_disk(claim)
        return claim

    def register_bulk(self, claims: list[CitedClaim]) -> None:
        """Register multiple pre-built claims at once."""
        for c in claims:
            self._claims.append(c)
            self._append_to_disk(c)

    def update_grounding(
        self,
        claim_id: str,
        grounding: str,
        confidence: float | None = None,
        sources: list[SourceRef] | None = None,
    ) -> bool:
        """Update grounding status of an existing claim (re-writes JSONL)."""
        for c in self._claims:
            if c.claim_id == claim_id:
                c.grounding = grounding
                if confidence is not None:
                    c.confidence = confidence
                if sources is not None:
                    c.sources = sources
                self._rewrite()
                return True
        return False

    # ------------------------------------------------------------------ #
    # Auditing                                                              #
    # ------------------------------------------------------------------ #

    def audit(self) -> RegistryReport:
        """Produce a full audit report of all registered claims."""
        report = RegistryReport(total=len(self._claims))
        for c in self._claims:
            if c.grounding == GroundingStatus.GROUNDED:
                report.grounded += 1
            elif c.grounding == GroundingStatus.INFERRED:
                report.inferred += 1
            elif c.grounding == GroundingStatus.HALLUCINATED:
                report.hallucinated += 1
            else:
                report.unverified += 1
            if c.is_risky:
                report.risky_claims.append(c)
        return report

    def claims_for_section(self, section: str) -> list[CitedClaim]:
        return [c for c in self._claims if c.section.lower() == section.lower()]

    def risky_claims(self) -> list[CitedClaim]:
        return [c for c in self._claims if c.is_risky]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _rewrite(self) -> None:
        """Rewrite entire JSONL file (used after update_grounding)."""
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for c in self._claims:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        tmp.replace(self._path)

    def __len__(self) -> int:
        return len(self._claims)


# ---------------------------------------------------------------------------
# LLM-assisted claim extraction from generated text
# ---------------------------------------------------------------------------

_EXTRACT_CLAIMS_SYSTEM = """\
You are a scientific claim extractor. Given a passage of text from a research
paper draft, extract every distinct factual claim. For each claim, also identify
the best supporting source from the provided source list (by source_id).

Rules:
- One claim per sentence/assertion
- Skip hedged meta-statements ("We believe", "This paper proposes")
- Include quantitative values, comparisons, and causal statements
- Assign grounding: "grounded" if a source quote supports it directly,
  "inferred" if it follows logically, "unverified" if no source applies
- Respond ONLY with valid JSON — no prose, no markdown fences

Format:
{
  "claims": [
    {
      "text": "...",
      "grounding": "grounded|inferred|unverified",
      "confidence": 0.0-1.0,
      "source_id": "s2:abc123 or empty string",
      "quote": "verbatim supporting excerpt from source"
    }
  ]
}
"""


def extract_claims_from_text(
    text: str,
    stage: str,
    sources: list[dict[str, Any]],
    llm: _LLMClientLike,
    *,
    section: str = "",
    llm_model: str = "",
) -> list[CitedClaim]:
    """Extract and annotate claims from LLM-generated paper text.

    Args:
        text:    Generated paper section text.
        stage:   Pipeline stage name (e.g. ``"PAPER_DRAFT"``).
        sources: List of source dicts with ``id``, ``title``, ``year``, ``text``.
        llm:     LLM client for extraction.
        section: Paper section name.
        llm_model: Model identifier for provenance.

    Returns:
        List of ``CitedClaim`` objects ready to register.
    """
    # Build a compact source index for the LLM
    source_index = "\n".join(
        f"[{s.get('id','')}] {s.get('title','')} ({s.get('year','')})"
        for s in sources[:20]
    )
    prompt = (
        f"Available sources:\n{source_index}\n\n"
        f"Paper text:\n{text[:5000]}"
    )

    try:
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            system=_EXTRACT_CLAIMS_SYSTEM,
            max_tokens=2000,
            temperature=0.0,
            json_mode=True,
        )
        raw = response.content if hasattr(response, "content") else str(response)
        data = _safe_json(raw)
        raw_claims = data.get("claims", []) if isinstance(data, dict) else []
    except Exception as exc:
        logger.warning("extract_claims_from_text failed: %s", exc)
        return []

    # Build source lookup for SourceRef construction
    src_map = {str(s.get("id", "")): s for s in sources}

    claims: list[CitedClaim] = []
    for item in raw_claims:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        src_id = str(item.get("source_id", "")).strip()
        source_refs: list[SourceRef] = []
        if src_id and src_id in src_map:
            s = src_map[src_id]
            source_refs.append(
                SourceRef(
                    source_id=src_id,
                    source_title=str(s.get("title", "")),
                    source_year=int(s.get("year", 0)),
                    url=str(s.get("url", "")),
                    quote=str(item.get("quote", ""))[:300],
                )
            )
        claims.append(
            CitedClaim(
                claim_id=str(uuid.uuid4())[:8],
                text=str(item["text"]).strip(),
                stage=stage,
                grounding=str(item.get("grounding", GroundingStatus.UNVERIFIED)),
                confidence=float(item.get("confidence", 0.5)),
                sources=source_refs,
                section=section,
                llm_model=llm_model,
            )
        )

    logger.debug("Extracted %d claims from generated text (section=%s)", len(claims), section)
    return claims


# ---------------------------------------------------------------------------
# Context manager helper
# ---------------------------------------------------------------------------


@contextmanager
def claim_guard(
    registry: ClaimRegistry,
    stage: str,
    section: str = "",
) -> Generator[list[CitedClaim], None, None]:
    """Context manager that collects and registers claims on exit.

    Usage::

        with claim_guard(registry, stage="PAPER_DRAFT", section="Results") as claims:
            # ... LLM generates text ...
            claims.extend(extract_claims_from_text(text, stage, sources, llm))
        # claims are now persisted in registry
    """
    bucket: list[CitedClaim] = []
    yield bucket
    for c in bucket:
        if not c.stage:
            c.stage = stage
        if not c.section:
            c.section = section
    registry.register_bulk(bucket)
    logger.debug("claim_guard: registered %d claims for stage=%s", len(bucket), stage)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _safe_json(raw: str) -> Any:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Combined confidence score helper
# ---------------------------------------------------------------------------


def compute_final_confidence(
    base_score: float,
    *,
    hallucination_risk: float,
    trust_penalty: float,
    verified_citations_rate: float = 1.0,
) -> dict[str, Any]:
    """Compute the unified confidence score surfaced in the final report.

    Combines:
      - base_score              (from PaperScorer.overall, 0–1)
      - hallucination_risk      (from ClaimRegistry.audit, 0–1)
      - trust_penalty           (from ContradictionReport, 0–1)
      - verified_citations_rate (from VerificationReport.integrity_score, 0–1)

    Returns a dict with the final score (0–100) and a plain-English reason.
    """
    adjusted = base_score
    adjusted -= hallucination_risk * 0.30
    adjusted -= trust_penalty * 0.20
    adjusted *= 0.5 + 0.5 * verified_citations_rate  # partial scaling by citation quality

    final = max(0.0, min(1.0, adjusted))
    pct = round(final * 100)

    reasons: list[str] = []
    if hallucination_risk > 0.3:
        reasons.append(f"high ungrounded claim rate ({hallucination_risk:.0%})")
    if trust_penalty > 0.2:
        reasons.append("direct contradictions detected in sources")
    if verified_citations_rate < 0.7:
        reasons.append(f"low citation verification rate ({verified_citations_rate:.0%})")
    if not reasons:
        reasons.append("all quality checks passed")

    return {
        "confidence": pct,
        "confidence_float": round(final, 3),
        "reasoning": "; ".join(reasons),
        "components": {
            "base_score": round(base_score, 3),
            "hallucination_risk": round(hallucination_risk, 3),
            "trust_penalty": round(trust_penalty, 3),
            "verified_citations_rate": round(verified_citations_rate, 3),
        },
    }
