"""
Verification Engine: checks references, evidence alignment, citation validity.

Three verification stages:
  1. Citation Verification — do cited papers exist? DOI/arXiv resolvable?
  2. Evidence Alignment   — do claims match experiment data?
  3. Consistency Check    — no contradictions between sections?
"""

import json, logging, re
from llm import llm

logger = logging.getLogger(__name__)


class CitationVerifier:
    """Verifies that cited papers are real and properly formatted."""

    SYSTEM = (
        "You are a citation verification expert. Given a paper's reference list, "
        "check each citation for: 1) Is the title/author plausible? 2) Is the venue real? "
        "3) Is the year reasonable? 4) Are there formatting errors? "
        "Return JSON: {ref_key: {valid: bool, issue: str|null}} for each reference."
    )

    async def verify(self, bibtex_or_refs: str) -> dict:
        raw = await llm(
            f"Verify these references:\n{bibtex_or_refs[:5000]}\n\nReturn JSON.",
            system=self.SYSTEM, agent="critic"
        )
        if not raw.strip():
            logger.warning("CitationVerifier received empty LLM response")
            return {"error": "LLM returned empty response"}
        try:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            return json.loads(m.group()) if m else {"raw": raw[:500]}
        except Exception as e:
            logger.warning("CitationVerifier JSON parse failed: %s", e)
            return {"raw": raw[:500]}

    def check_bibtex_format(self, bibtex: str) -> list[str]:
        """Static checks on BibTeX formatting."""
        issues = []
        entries = re.findall(r'@\w+\{([^,]+),', bibtex)
        # Check for duplicate keys
        seen = set()
        for key in entries:
            if key in seen:
                issues.append(f"Duplicate BibTeX key: {key}")
            seen.add(key)
        # Check for missing fields
        for entry in re.split(r'@\w+\{', bibtex)[1:]:
            key = entry.split(',')[0].strip()
            if 'title' not in entry.lower():
                issues.append(f"{key}: missing 'title' field")
            if 'year' not in entry.lower():
                issues.append(f"{key}: missing 'year' field")
        return issues


class EvidenceAligner:
    """Checks that paper claims match experimental data."""

    SYSTEM = (
        "You are a scientific integrity checker. Compare claims in a paper draft "
        "against the raw experimental log. Flag: 1) Numbers that don't match, "
        "2) Claims not supported by data, 3) Overstated results, 4) Missing caveats. "
        "Return JSON: {aligned: bool, issues: [{claim: str, evidence: str, verdict: str}]}"
    )

    async def check(self, paper_tex: str, experiment_log: str) -> dict:
        raw = await llm(
            f"Paper claims:\n{paper_tex[:4000]}\n\nExperimental ground truth:\n{experiment_log[:4000]}\n\n"
            "Check alignment. Return JSON.",
            system=self.SYSTEM, agent="critic"
        )
        if not raw.strip():
            logger.warning("EvidenceAligner received empty LLM response")
            return {"aligned": None, "error": "LLM returned empty response"}
        try:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            return json.loads(m.group()) if m else {"aligned": None, "raw": raw[:500]}
        except Exception as e:
            logger.warning("EvidenceAligner JSON parse failed: %s", e)
            return {"aligned": None, "raw": raw[:500]}


class ConsistencyChecker:
    """Checks for contradictions between paper sections."""

    SYSTEM = (
        "You are a manuscript consistency checker. Read the paper and flag: "
        "1) Contradictions between Abstract and Results, "
        "2) Method details that don't match Experiments, "
        "3) Figures/tables referenced but missing, "
        "4) Notation inconsistencies. "
        "Return JSON: {consistent: bool, issues: [str]}"
    )

    async def check(self, paper_tex: str) -> dict:
        raw = await llm(
            f"Check this paper for internal consistency:\n{paper_tex[:6000]}\n\nReturn JSON.",
            system=self.SYSTEM, agent="critic"
        )
        if not raw.strip():
            logger.warning("ConsistencyChecker received empty LLM response")
            return {"consistent": None, "error": "LLM returned empty response"}
        try:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            return json.loads(m.group()) if m else {"consistent": None, "raw": raw[:500]}
        except Exception as e:
            logger.warning("ConsistencyChecker JSON parse failed: %s", e)
            return {"consistent": None, "raw": raw[:500]}


class VerificationEngine:
    """Runs all verification checks on a paper."""

    def __init__(self):
        self.citation = CitationVerifier()
        self.evidence = EvidenceAligner()
        self.consistency = ConsistencyChecker()

    async def verify_all(self, paper_tex: str, bibtex: str = "",
                          experiment_log: str = "") -> dict:
        """Run all three verification stages."""
        results = {"passed": True, "stages": {}}

        # 1. Citation verification
        if bibtex:
            fmt_issues = self.citation.check_bibtex_format(bibtex)
            cite_check = await self.citation.verify(bibtex)
            results["stages"]["citations"] = {
                "format_issues": fmt_issues,
                "verification": cite_check,
                "passed": len(fmt_issues) == 0
            }
            if fmt_issues:
                results["passed"] = False

        # 2. Evidence alignment
        if experiment_log:
            alignment = await self.evidence.check(paper_tex, experiment_log)
            results["stages"]["evidence"] = alignment
            if alignment.get("aligned") is False:
                results["passed"] = False

        # 3. Internal consistency
        consistency = await self.consistency.check(paper_tex)
        results["stages"]["consistency"] = consistency
        if consistency.get("consistent") is False:
            results["passed"] = False

        return results

    async def quick_check(self, paper_tex: str) -> dict:
        """Fast consistency-only check (no LLM calls for citations)."""
        # Static checks
        issues = []
        # Check for orphan references
        cites = set(re.findall(r'\\cite\{([^}]+)\}', paper_tex))
        bibitems = set(re.findall(r'\\bibitem\{([^}]+)\}', paper_tex))
        for c in cites:
            for key in c.split(','):
                key = key.strip()
                if bibitems and key not in bibitems:
                    issues.append(f"\\cite{{{key}}} has no matching \\bibitem")

        # Check figure references
        figrefs = set(re.findall(r'\\ref\{(fig:[^}]+)\}', paper_tex))
        figlabels = set(re.findall(r'\\label\{(fig:[^}]+)\}', paper_tex))
        for ref in figrefs:
            if ref not in figlabels:
                issues.append(f"\\ref{{{ref}}} references undefined figure")

        # Check for common LaTeX errors
        for env in ['figure', 'figure*', 'table', 'table*', 'equation', 'equation*', 'align', 'align*']:
            escaped = env.replace('*', r'\*')
            opens = len(re.findall(r'\\begin\{' + escaped + r'\}', paper_tex))
            closes = len(re.findall(r'\\end\{' + escaped + r'\}', paper_tex))
            if opens != closes:
                issues.append(f"Mismatched \\begin/{env} ({opens}) vs \\end/{env} ({closes})")

        return {"passed": len(issues) == 0, "issues": issues}
