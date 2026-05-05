"""
agents/research.py — Research, Experiment, and PaperWriter agents.

FIXED VERSION:
  - SectionWriter is now TOPIC-ADAPTIVE: no hardcoded NLP/domain instructions.
    The LLM infers the correct domain from the source material.
  - _assemble() generates keywords dynamically from the paper title/source.
  - Author info is extracted from the task prompt and document context.
  - Paper format (single/double column, conference/journal) is parsed from task.
  - PaperPostProcessor cleans em-dashes and deduplicates bibitems.
  - ThesisExtractor handles long documents by chunking and merging.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from llm import llm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Intent routing
# ─────────────────────────────────────────────────────────────────────────────

def classify_task(task: str, document_context: str = "", ui_mode: str = "auto") -> str:
    try:
        from agents.intent import classify_task as _classify
        return _classify(task, document_context, ui_mode)
    except ImportError:
        t = task.lower()
        paper_kw = ["write a paper", "ieee paper", "ieee-style", "research paper",
                    "convert thesis", "transform thesis"]
        exp_kw   = ["authentic results", "run the preprocessing", "first run",
                    "run code", "do not fabricate", "obtain results"]
        wants_paper = any(kw in t for kw in paper_kw)
        needs_exp   = any(kw in t for kw in exp_kw)
        if wants_paper and needs_exp:
            return "experiment_then_paper"
        if wants_paper:
            return "paper"
        return "coding"


# ─────────────────────────────────────────────────────────────────────────────
# Paper requirements extracted from the user task prompt
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PaperRequirements:
    title: str = ""
    authors: str = ""
    affiliation: str = ""
    email: str = ""
    format: str = "ieee_conference"    # ieee_conference | ieee_journal | single_col | acm | arxiv
    columns: int = 2
    target_pages: str = "8-10"
    extra_instructions: str = ""
    keywords: str = ""
    ref_count: int = 30               # number of references required
    section_word_targets: dict = field(default_factory=dict)  # section -> "N words"


def parse_paper_requirements(task: str) -> PaperRequirements:
    req = PaperRequirements()
    t = task
    t_lower = t.lower()

    # Title — try quoted first, then "Title:" prefix
    m = re.search(r'[Tt]itle[:\s]+["\']?([^\n"\']{10,160})["\']?', t)
    if m:
        req.title = m.group(1).strip()
    else:
        m2 = re.search(r'"([^"]{10,160})"', t)
        if m2:
            req.title = m2.group(1).strip()

    # ── Author Information block ──
    # Handles multi-line format:
    #   Author Information:
    #   Zain Ul Abidin
    #   zainulabdin9952gmail.com   ← may have missing @
    #   Department of Computer Systems Engineering
    #   Sukkur IBA University, Sukkur, Pakistan
    auth_block_m = re.search(
        r'[Aa]uthor\s+[Ii]nformation[:\s]*\n((?:[ \t]*[^\n]+\n){1,8})',
        t
    )
    if auth_block_m:
        block_lines = [l.strip() for l in auth_block_m.group(1).split('\n') if l.strip()]
        if block_lines:
            req.authors = block_lines[0]
            for bl in block_lines[1:]:
                # Full email
                if re.match(r'^[\w.+-]+@[\w.-]+\.\w+$', bl):
                    req.email = bl
                # Email missing @ before known domains
                elif re.match(r'^[\w.+-]+(gmail|yahoo|outlook|hotmail)\.com$', bl):
                    req.email = re.sub(
                        r'(gmail\.com|yahoo\.com|outlook\.com|hotmail\.com)$',
                        r'@\1', bl
                    )
                elif re.search(r'department|engineering|science|faculty|computer', bl, re.I):
                    if not req.affiliation:
                        req.affiliation = bl
                elif re.search(r'university|institute|college|iba', bl, re.I):
                    req.affiliation = (req.affiliation + ", " + bl) if req.affiliation else bl
    else:
        # Fallback: single-line author
        m = re.search(r'[Aa]uthor[s]?[:\s]+([^\n]{5,120})', t)
        if m and "information" not in m.group(1).lower():
            req.authors = m.group(1).strip()

    # Email fallback
    if not req.email:
        em = re.search(r'[\w.+-]+@[\w.-]+\.\w+', t)
        if em:
            req.email = em.group(0)
        else:
            em2 = re.search(r'([\w.+-]+)(gmail\.com|yahoo\.com|outlook\.com|hotmail\.com)', t)
            if em2:
                req.email = em2.group(1) + "@" + em2.group(2)

    # Affiliation fallback
    if not req.affiliation:
        m = re.search(r'[Aa]ffiliation[:\s]+([^\n]{5,100})', t)
        if m:
            req.affiliation = m.group(1).strip()
        else:
            m2 = re.search(
                r'((?:Sukkur IBA|[A-Z][a-z]+ University|[A-Z][a-z]+ Institute)[^\n,]{0,60})',
                t
            )
            if m2:
                req.affiliation = m2.group(1).strip()

    # Format / column detection
    # Priority: explicit single-column > 15+ pages (implies long single-col) > double-column
    is_single = (
        "single column" in t_lower or
        "single-column" in t_lower or
        "one column" in t_lower or
        "one-column" in t_lower or
        "1 column" in t_lower or
        "journal" in t_lower and "conference" not in t_lower
    )
    is_double = (
        "double column" in t_lower or
        "two column" in t_lower or
        "two-column" in t_lower or
        "conference" in t_lower
    )
    # If >12 page target with no explicit format → single column (long papers)
    page_m = re.search(r'(\d+)\s*(?:to|-)\s*(\d+)\s*(?:IEEE\s*)?pages?', t, re.I)
    if page_m:
        req.target_pages = f"{page_m.group(1)}-{page_m.group(2)}"
        if int(page_m.group(1)) >= 12 and not is_double:
            is_single = True
    else:
        m2 = re.search(r'(\d+)\s*pages?', t_lower)
        if m2 and int(m2.group(1)) >= 12 and not is_double:
            is_single = True

    if is_single:
        req.columns = 1
        req.format = "ieee_journal"   # IEEEtran journal = single column
    elif is_double or "conference" in t_lower:
        req.columns = 2
        req.format = "ieee_conference"
    else:
        req.columns = 2
        req.format = "ieee_conference"

    if "acm" in t_lower:
        req.format = "acm"
    elif "arxiv" in t_lower or "preprint" in t_lower:
        req.format = "arxiv"

    # Keywords
    m = re.search(r'[Kk]eywords?[:\s]+([^\n]{5,200})', t)
    if m:
        req.keywords = m.group(1).strip().rstrip('.')

    # Reference count — parse from prompt ("30 unique references", "at least 25 references", etc.)
    ref_m = re.search(r'(?:at\s+least\s+|minimum\s+|(?:25|30|35|40|50)\s+)?(\d+)\s+(?:unique\s+)?references', t_lower)
    if ref_m:
        req.ref_count = max(20, int(ref_m.group(1)))
    else:
        # Default based on page count
        if req.target_pages and int(req.target_pages.split('-')[0]) >= 12:
            req.ref_count = 40
        else:
            req.ref_count = 30

    # Section word targets — parse explicit word count requirements from prompt
    swt = {}
    patterns = [
        (r'introduction\s+(?:must\s+be\s+)?(?:at\s+least\s+)?(\d+)\s+words?', "Introduction"),
        (r'related\s+work\s+(?:must\s+be\s+)?(?:at\s+least\s+)?(\d+)\s+words?', "Related Work"),
        (r'methodology\s+(?:must\s+be\s+)?(?:at\s+least\s+)?(\d+)\s+words?', "Methodology"),
        (r'experimental\s+(?:setup\s+)?(?:must\s+be\s+)?(?:at\s+least\s+)?(\d+)\s+words?', "Experimental Setup"),
        (r'results?\s+(?:and\s+discussion\s+)?(?:must\s+be\s+)?(?:at\s+least\s+)?(\d+)\s+words?', "Results"),
        (r'conclusion\s+(?:must\s+be\s+)?(?:at\s+least\s+)?(\d+)\s+words?', "Conclusion"),
    ]
    for pattern, section_name in patterns:
        wm = re.search(pattern, t_lower)
        if wm:
            swt[section_name] = f"{wm.group(1)}+ words"
    req.section_word_targets = swt

    return req


def _make_docclass(req: PaperRequirements) -> str:
    if req.format == "ieee_journal":
        # Single-column journal format
        return "\\documentclass[journal]{IEEEtran}"
    elif req.format == "acm":
        return "\\documentclass[acmconf,review=false]{acmart}"
    elif req.format == "arxiv":
        return "\\documentclass[12pt]{article}"
    else:
        # Double-column conference format
        return "\\documentclass[conference]{IEEEtran}"


def _make_packages(req: PaperRequirements) -> str:
    """Return the usepackage block appropriate for the format."""
    base = (
        "\\usepackage{amsmath,amssymb,graphicx,booktabs,hyperref,cite,array,multirow}\n"
        "\\usepackage{algorithm,algorithmic}\n"
        "\\usepackage{pgfplots}\n"
        "\\pgfplotsset{compat=1.17}\n"
        "\\usepackage{xcolor}\n"
        "\\usepackage{url}\n"
    )
    if req.format == "ieee_journal":
        # Journal mode: do NOT use \IEEEoverridecommandlockouts (conference-only)
        return base
    elif req.format in ("acm", "arxiv"):
        return base
    else:
        # Conference
        return "\\IEEEoverridecommandlockouts\n" + base


# ─────────────────────────────────────────────────────────────────────────────
# Shared IEEE writing rules
# ─────────────────────────────────────────────────────────────────────────────

_IEEE_BASE_RULES = """
ABSOLUTE FORMATTING RULES (zero tolerance — any violation is a failure):
1. NO em dashes anywhere (not — nor --). Rewrite using commas, semicolons, or separate sentences.
2. NO bullet points or numbered lists inside section text. Paragraphs only.
3. NO invented numbers. Copy ALL numerical values exactly from the source material.
4. NO placeholders like [TBD], [insert here], [Figure X], [Table X].
5. IEEE citation style: use \\cite{rN} where N is a sequential integer starting at 1.
   The FIRST paper cited in Introduction becomes \\cite{r1}, the second becomes \\cite{r2}, etc.
   Every \\bibitem{rN} key must match a \\cite{rN} key used in the text.
6. Booktabs tables: \\toprule / \\midrule / \\bottomrule. No vertical lines (|) in tabular.
7. Write about the ACTUAL topic in the source material.

CITATION ORDERING RULE:
- Assign citation numbers in strict order of first appearance in the text.
- \\cite{r1} is the first reference cited anywhere in the paper.
- \\cite{r2} is the second distinct reference cited, and so on.
- References section must list \\bibitem{r1}, \\bibitem{r2}, ... in that exact ascending order.
- If you cite 20 references, use r1 through r20 with NO gaps and NO repetition.

HUMANIZATION RULES (must be followed — AI-detector scores depend on these):
H1. VARY sentence length dramatically. After a 30-word sentence, write an 8-word one. Never use 5 sentences of similar length consecutively.
H2. BANNED PHRASES — never use: "In this paper, we", "It is worth noting", "Furthermore,", "Moreover,", "In conclusion,", "This paper presents", "In order to", "With the advent of", "plays a crucial role", "leverages", "showcases", "robust", "seamlessly", "groundbreaking", "comprehensive framework", "state-of-the-art" (use specific accuracy numbers instead).
H3. OPEN paragraphs DIFFERENTLY each time. Never start consecutive paragraphs with the same word.
H4. USE active voice: "The model achieves..." not "It was found that the model achieves..."
H5. INCLUDE at least one short, direct assertion per section: e.g., "The results are unambiguous." or "This is the key insight."
H6. USE domain-specific concrete terms from the source instead of generic adjectives.
H7. CONNECT ideas with precise logic: "This explains why...", "The consequence is...", "This raises a question:..."
"""


# ─────────────────────────────────────────────────────────────────────────────
# Thesis content extractor
# ─────────────────────────────────────────────────────────────────────────────

class ThesisExtractor:
    SYSTEM = (
        "You are a research content analyst. Extract ALL key information from thesis text.\n\n"
        "MANDATORY extractions:\n"
        "1. TABLES: Copy every table verbatim with ALL exact numerical values. Label each table clearly.\n"
        "2. ALGORITHMS/PSEUDOCODE: Copy all algorithm descriptions, pseudocode, and procedural steps.\n"
        "3. FIGURES: Note every figure number, caption, and what it shows (axes, curves, key points).\n"
        "4. EQUATIONS: Copy all mathematical equations exactly using LaTeX notation.\n"
        "5. METRICS: List every reported number: accuracy, precision, recall, F1, MAE, RMSE, mAP, Dice, etc.\n"
        "6. MODELS/METHODS: List all architectures, hyperparameters, datasets, hardware specs.\n"
        "7. CONTRIBUTIONS: List the paper's stated contributions exactly.\n"
        "8. REFERENCES: List all cited papers with author names, titles, years.\n\n"
        "FORMAT: Use headers like ## TABLES, ## ALGORITHMS, ## FIGURES, ## EQUATIONS, ## METRICS.\n"
        "Copy numbers EXACTLY. Do not invent. Do not summarize tables — copy them fully."
    )

    async def extract(self, document_text: str, topic: str = "") -> str:
        chunks = self._chunk(document_text)
        logger.info("ThesisExtractor: %d chunks for document of %d chars", len(chunks), len(document_text))
        extractions = []
        for i, chunk in enumerate(chunks):
            result = await llm(
                f"Thesis topic: {topic}\n\nChunk {i+1}/{len(chunks)}:\n{chunk}\n\n"
                "Extract ALL tables (copy full numbers), algorithms, figures, equations, "
                "metrics, and key findings from this chunk. Use headers: "
                "## TABLES, ## ALGORITHMS, ## FIGURES, ## EQUATIONS, ## METRICS, ## FINDINGS.",
                system=self.SYSTEM, agent="researcher", max_tokens=8192,
            )
            if result.strip():
                extractions.append(f"=== CHUNK {i+1}/{len(chunks)} ===\n{result}")

        if not extractions:
            return ""
        if len(extractions) == 1:
            return extractions[0]

        # Merge ALL chunks — no limit. Split into batches of 6 if very large.
        batch_size = 6
        if len(extractions) <= batch_size:
            batches = [extractions]
        else:
            batches = [extractions[i:i+batch_size] for i in range(0, len(extractions), batch_size)]

        merged_parts = []
        for batch_idx, batch in enumerate(batches):
            merged = await llm(
                f"Merge these {len(batch)} extraction chunks into ONE structured document. "
                "Preserve ALL tables with every number, ALL algorithms, ALL figure descriptions, "
                "ALL equations. Use headers: ## TABLES, ## ALGORITHMS, ## FIGURES, ## EQUATIONS, "
                "## METRICS, ## FINDINGS. Never drop any numerical data.\n\n"
                + "\n\n".join(batch),
                system="Merge thesis extractions. Keep ALL numerical data exactly. Never drop a table or equation.",
                agent="researcher", max_tokens=8192,
            )
            if merged:
                merged_parts.append(merged)

        if len(merged_parts) == 1:
            return merged_parts[0]

        # Final merge of batch merges
        final = await llm(
            "Final merge: Combine these structured summaries into ONE complete extraction document. "
            "Preserve ALL tables with exact numbers, ALL algorithms, ALL equations.\n\n"
            + "\n\n---\n\n".join(merged_parts),
            system="Final merge. Keep ALL numerical data exactly.",
            agent="researcher", max_tokens=8192,
        )
        return final or merged_parts[0]

    def _chunk(self, text: str, size: int = 12000) -> list[str]:
        if len(text) <= size:
            return [text]
        chunks, stride, start = [], int(size * 0.85), 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = text.rfind("\n\n", start, end)
                if boundary > start + size // 2:
                    end = boundary
            chunks.append(text[start:end])
            start += stride
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic keyword generator
# ─────────────────────────────────────────────────────────────────────────────

async def _generate_keywords(title: str, source_excerpt: str) -> str:
    result = await llm(
        f"Paper title: {title}\n\nSource excerpt:\n{source_excerpt[:2000]}\n\n"
        "Generate 6-10 IEEE-style index terms for this paper. "
        "Return ONLY a comma-separated list on one line. "
        "Use domain-specific technical terms matching the actual paper topic.",
        system="Generate precise IEEE index terms. Return only a comma-separated list.",
        agent="paper_writer",
    )
    result = re.sub(r'```[^`]*```', '', result or "").strip().strip('"').strip("'")
    result = result.split('\n')[0].strip()
    return result if result else f"{title[:60]}"


# ─────────────────────────────────────────────────────────────────────────────
# Section writer - TOPIC-ADAPTIVE (no hardcoded domain)
# ─────────────────────────────────────────────────────────────────────────────

class SectionWriter:
    """
    Writes paper sections section-by-section. TOPIC-ADAPTIVE: infers domain from source.
    Accepts PaperRequirements to honour word count and reference count from the user prompt.
    """

    BASE_SYSTEM = (
        "You are a senior academic author writing a technical research paper section.\n\n"
        + _IEEE_BASE_RULES
        + "\nWrite ONLY the requested section. No other sections.\n"
        "Output raw LaTeX. No markdown fences. No ``` wrappers.\n"
        "Start DIRECTLY with the section command: \\section{...} or \\begin{abstract}.\n"
        "DOMAIN RULE: Write about the ACTUAL topic in the source material, not generic AI/NLP.\n"
        "COMPLETENESS: Write the ENTIRE section. Do NOT stop early. Do NOT add [continued] or similar.\n"
        "The section MUST end with a complete sentence and proper LaTeX structure."
    )

    def __init__(self, req: PaperRequirements | None = None):
        self.req = req or PaperRequirements()

    def _word_target(self, section_key: str, default: str) -> str:
        """Return user-specified word target for this section, or default."""
        return self.req.section_word_targets.get(section_key, default)

    async def _write(self, name: str, instructions: str, source: str,
                     prev: str = "", target: str = "", extra: str = "") -> str:
        ctx = f"\n\n=== PREVIOUS SECTIONS (for context, do NOT repeat) ===\n{prev[-4000:]}\n===" if prev.strip() else ""
        extra_block = f"\n\nUSER REQUIREMENTS FROM PROMPT:\n{extra}" if extra else ""
        title_line = f"\nPAPER TITLE: {self.req.title}\n" if self.req.title else ""

        prompt = (
            f"{title_line}"
            f"Write the '{name}' section. Target: {target}.\n\n"
            f"SECTION INSTRUCTIONS:\n{instructions}\n\n"
            f"SOURCE MATERIAL (base ALL content on this — do NOT invent facts):\n{source[:18000]}"
            f"{ctx}{extra_block}\n\n"
            f"Write the COMPLETE '{name}' section in LaTeX now. "
            f"Do not stop until the section is fully written. End with a complete sentence."
        )
        result = await llm(prompt, system=self.BASE_SYSTEM, agent="paper_writer", max_tokens=8192)
        result = (result or "").strip()
        result = re.sub(r'```(?:latex|tex)?\n?', '', result)
        result = re.sub(r'```\n?', '', result)

        # Truncation detection — continue if cut off
        if result and self._is_truncated(result):
            logger.warning("Section '%s' truncated at %d chars — continuing", name, len(result))
            continuation = await llm(
                f"The '{name}' section was cut off. Continue from exactly where it ended. "
                f"Do NOT repeat anything already written. Complete the section.\n\n"
                f"LAST ~1500 CHARS WRITTEN:\n{result[-1500:]}\n\n"
                f"Continue from here:",
                system=self.BASE_SYSTEM, agent="paper_writer", max_tokens=4096,
            )
            if continuation and continuation.strip():
                cont = re.sub(r'```(?:latex|tex)?\n?', '', continuation.strip())
                result = result + "\n" + cont

        return result if result else f"% {name} section could not be generated"

    @staticmethod
    def _is_truncated(text: str) -> bool:
        """Detect if a LaTeX section was cut off mid-sentence or mid-structure."""
        s = text.rstrip()
        if not s:
            return True
        # Good endings
        if (s.endswith("}") or s.endswith(".") or
                "\\end{" in s[-120:] or "\\end{thebibliography}" in s):
            return False
        # Bad endings: comma, connective words, unclosed brace
        bad = (s.endswith(",") or s.endswith(" the") or s.endswith(" a") or
               s.endswith(" and") or s.endswith(" of") or s.endswith(" in") or
               s.endswith(" to") or s.endswith(" with") or s.endswith(" for") or
               s.count("{") > s.count("}") + 5)
        return bad

    # ── Individual section writers ──────────────────────────────────────────

    async def abstract(self, source: str, title: str, extra: str = "") -> str:
        return await self._write(
            "Abstract",
            "Write inside \\begin{abstract}...\\end{abstract}.\n"
            "Paragraph structure: (1) Domain + problem (2 sentences). "
            "(2) Proposed approach (2-3 sentences). "
            "(3) Key numerical results with EXACT numbers from source (3-4 sentences). "
            "(4) Significance (1 sentence). "
            "No citations. No bullet points. One solid paragraph.",
            source, target="200-350 words", extra=extra,
        )

    async def introduction(self, source: str, prev: str = "", extra: str = "") -> str:
        target = self._word_target("Introduction", "600-900 words")
        return await self._write(
            "\\section{Introduction}",
            f"Write {target} across 5-6 paragraphs:\n"
            "Para 1: Domain + real-world scale + economic/engineering importance. Cite [r1],[r2].\n"
            "Para 2: Core technical challenge. Use specific domain terminology from source. "
            "Give concrete failure examples with numbers where possible.\n"
            "Para 3: Existing approaches and their limitations. Cite [r3],[r4],[r5].\n"
            "Para 4: The precise gap — what no existing work has done. Be specific.\n"
            "Para 5: Contributions — state as numbered list INSIDE a paragraph (not bullet points). "
            "Each contribution is a full sentence starting with a verb.\n"
            "Para 6: Paper organisation — one sentence per section.\n"
            "Use \\cite{rN} where N increments from 1 upward. First paper cited = \\cite{r1}.",
            source, prev, target=target, extra=extra,
        )

    async def related_work(self, source: str, prev: str = "", extra: str = "") -> str:
        target = self._word_target("Related Work", "700-1000 words")
        return await self._write(
            "\\section{Related Work}",
            f"Write {target} organized into \\subsection{{}} blocks.\n"
            "Choose subsection titles from the ACTUAL technical areas in the source material.\n"
            "Each subsection: 2-4 paragraphs of genuine technical detail.\n"
            "Final subsection: 'Research Gap' — state precisely what the present work adds.\n"
            "Cite at least 12 distinct papers using \\cite{{rN}}. "
            "Continue numbering from where Introduction left off.\n"
            "DO NOT write generic AI/ML background unless the source is about that.\n"
            "Write technical depth, not just paper summaries.",
            source, prev, target=target, extra=extra,
        )

    async def methodology(self, source: str, prev: str = "", extra: str = "") -> str:
        target = self._word_target("Methodology", "500-700 words")
        return await self._write(
            "\\section{Methodology}",
            f"Write {target} organized into \\subsection{{}} blocks matching the source.\n"
            "Cover: datasets, preprocessing pipeline, model/algorithm design, "
            "training configuration, evaluation protocol.\n"
            "EQUATIONS: Include ALL mathematical formulas from the source using \\begin{{equation}}...\\end{{equation}}.\n"
            "ALGORITHMS: If the source describes a procedure or pipeline, include:\n"
            "  \\begin{{algorithm}}[h]\n"
            "  \\caption{{Algorithm Name}}\n"
            "  \\label{{alg:name}}\n"
            "  \\begin{{algorithmic}}[1]\n"
            "    \\REQUIRE ...\n"
            "    \\ENSURE ...\n"
            "    \\FOR{{...}} \\STATE ... \\ENDFOR\n"
            "  \\end{{algorithmic}}\n"
            "  \\end{{algorithm}}\n"
            "Use algorithmic package commands: \\REQUIRE, \\ENSURE, \\STATE, \\FOR, \\IF, \\RETURN.\n"
            "Describe ONLY what is actually in the source.",
            source, prev, target=target, extra=extra,
        )

    async def experimental_setup(self, source: str, prev: str = "", extra: str = "") -> str:
        target = self._word_target("Experimental Setup", "450-600 words")
        return await self._write(
            "\\section{Experimental Setup}",
            f"Write {target} in clear paragraphs:\n"
            "Para 1: Datasets — name, source, sample counts, class splits. "
            "Include \\begin{{table}}...\\end{{table}} if multiple datasets.\n"
            "Para 2: Hardware — GPU, RAM, training time (exact values from source).\n"
            "Para 3: Software — framework, libraries, versions.\n"
            "Para 4: Hyperparameters — ALL exact values from source: lr, batch size, "
            "epochs, optimizer, weight decay, augmentation.\n"
            "Para 5: Evaluation metrics — define each metric used.\n"
            "Write only what is in the source. Do not invent values.",
            source, prev, target=target, extra=extra,
        )

    async def results_and_discussion(self, source: str, prev: str = "", extra: str = "") -> str:
        target = self._word_target("Results", "800-1200 words")
        return await self._write(
            "\\section{Results and Discussion}",
            f"Write {target} with subsections:\n"
            "A. Main Results: Write \\begin{{table}}[h]\\centering...\\end{{table}} with booktabs and ALL "
            "EXACT values from the source. Every number must come from the source.\n"
            "TABLE FORMAT: Always wrap the tabular in \\resizebox{{\\columnwidth}}{{!}}{{...}} "
            "so it fits the column width. Example:\n"
            "  \\begin{{table}}[h]\\centering\n"
            "  \\caption{{...}}\\label{{tab:...}}\n"
            "  \\resizebox{{\\columnwidth}}{{!}}{{\\begin{{tabular}}{{lccc}}\\toprule...\\bottomrule\\end{{tabular}}}}\n"
            "  \\end{{table}}\n"
            "B. Analysis: Explain WHY results differ with percentage comparisons and domain reasoning.\n"
            "C. Ablation Study: Present ablation tables if the source has them (also with \\resizebox).\n"
            "D. Discussion: practical implications + limitations honestly stated.\n"
            "FIGURES: For any performance curves described in source, add a \\begin{{figure}} with pgfplots.\n"
            "ZERO fabricated numbers.",
            source, prev, target=target, extra=extra,
        )

    async def conclusion(self, source: str, prev: str = "", extra: str = "") -> str:
        target = self._word_target("Conclusion", "300-450 words")
        return await self._write(
            "\\section{Conclusion}",
            f"Write {target}:\n"
            "Para 1: Restate the problem and proposed approach in 2-3 sentences.\n"
            "Para 2: Key numerical findings with EXACT values from results section.\n"
            "Para 3: Practical recommendations for deployment.\n"
            "Para 4: Limitations — be specific and honest.\n"
            "Para 5: Future work — 3-4 concrete, actionable directions.\n"
            "No new information not in earlier sections. No first-person singular.",
            source, prev, target=target, extra=extra,
        )

    async def references(self, source: str, all_sections: str = "") -> str:
        """Generate references in ASCENDING numeric order matching \\cite{rN} in text."""
        cite_keys = self._extract_cite_keys_ordered(all_sections)
        ref_count = self.req.ref_count
        result = await llm(
            f"SOURCE DOCUMENT (contains references and papers cited):\n{source[:10000]}\n\n"
            f"CITE KEYS USED IN PAPER (in order of first appearance):\n{cite_keys}\n\n"
            f"TARGET: Generate exactly {ref_count} references.\n\n"
            "Generate a complete \\begin{thebibliography}{99}...\\end{thebibliography} block.\n"
            "CRITICAL RULES:\n"
            "1. List \\bibitem entries in ASCENDING order: \\bibitem{r1} first, \\bibitem{r2} second, etc.\n"
            "2. Every \\bibitem{rN} key must exactly match a \\cite{rN} used in the text.\n"
            "3. IEEE format per entry: Initials. LastName, \"Title,\" Venue, vol. X, no. Y, pp. Z-W, Year.\n"
            "4. Extract real papers from the source document first. Use author names and titles exactly.\n"
            f"5. Add supplementary real published papers to reach {ref_count} total entries.\n"
            "6. NO duplicate keys. NO invented papers. NO fictional journals.\n"
            "7. Write ALL entries. Do not stop early.",
            system=(
                "Generate a complete IEEE reference list in ascending \\bibitem{r1}, \\bibitem{r2}, ... order. "
                "Real papers only. No invented citations."
            ),
            agent="paper_writer", max_tokens=8192,
        )
        return result.strip() if result else "\\begin{thebibliography}{99}\n\\end{thebibliography}"

    @staticmethod
    def _extract_cite_keys_ordered(text: str) -> str:
        """Extract \\cite{} keys in strict order of first appearance."""
        keys_ordered, seen = [], set()
        for m in re.finditer(r'\\cite\{([^}]+)\}', text):
            for k in m.group(1).split(','):
                k = k.strip()
                if k and k not in seen:
                    seen.add(k)
                    keys_ordered.append(k)
        return "\n".join(f"{i+1}. {k}" for i, k in enumerate(keys_ordered)) or "(none found)"


# ─────────────────────────────────────────────────────────────────────────────
# Post-processor
# ─────────────────────────────────────────────────────────────────────────────

class PaperPostProcessor:
    """
    Post-processes the assembled LaTeX paper:
    1. Removes em-dashes
    2. Removes markdown fences
    3. Enforces ascending \\bibitem order matching \\cite{rN} order in text
    4. Deduplicates \\bibitem keys
    5. Removes truncation artifacts
    """

    @staticmethod
    def process(latex: str) -> str:
        # Remove em-dashes (multiple forms)
        latex = re.sub(r'—', ',', latex)
        latex = re.sub(r'--(?=[^>-])', ',', latex)  # -- but not --> or ---

        # Remove markdown fences
        latex = re.sub(r'```(?:latex|tex)?\n?', '', latex)
        latex = re.sub(r'```\n?', '', latex)

        # Remove truncation artifacts
        latex = re.sub(r'\[\.\.\.\]|\[truncated\]|\[continued\]', '', latex, flags=re.I)

        # Wrap wide tables with \resizebox for single-column compatibility
        latex = PaperPostProcessor._resize_wide_tables(latex)

        # Enforce ascending reference order
        latex = PaperPostProcessor._reorder_references(latex)

        # Deduplicate bibitems
        latex = PaperPostProcessor._dedup_bibitems(latex)

        return latex

    @staticmethod
    def _resize_wide_tables(latex: str) -> str:
        """
        Wraps every \\begin{tabular} inside \\resizebox{\\columnwidth}{!}{...}
        so tables never overflow column width in single or double column mode.
        """
        def wrap_tabular(m: re.Match) -> str:
            table_env = m.group(0)
            if "resizebox" in table_env:
                return table_env
            # Wrap the inner tabular environment with \resizebox
            return re.sub(
                r'(\\begin\{tabular\}.*?\\end\{tabular\})',
                lambda mm: f'\\resizebox{{\\columnwidth}}{{!}}{{{mm.group(1)}}}',
                table_env, flags=re.DOTALL
            )

        return re.sub(
            r'\\begin\{table\*?\}.*?\\end\{table\*?\}',
            wrap_tabular,
            latex, flags=re.DOTALL
        )

    @staticmethod
    def _reorder_references(latex: str) -> str:
        """
        Ensures \\bibitem entries appear in the same ascending order as
        their first \\cite{} usage in the text body. This fixes non-ascending
        reference lists generated by the LLM.
        """
        # Collect cite order from text (exclude the bibliography section)
        bib_start = latex.find("\\begin{thebibliography}")
        text_body = latex[:bib_start] if bib_start != -1 else latex

        cite_order: list[str] = []
        seen: set[str] = set()
        for m in re.finditer(r'\\cite\{([^}]+)\}', text_body):
            for k in m.group(1).split(','):
                k = k.strip()
                if k and k not in seen:
                    seen.add(k)
                    cite_order.append(k)

        if bib_start == -1 or not cite_order:
            return latex

        # Extract all bibitem blocks
        bib_section = latex[bib_start:]
        bib_entries: dict[str, str] = {}
        # Match \bibitem{key} followed by content up to next \bibitem or \end
        for m in re.finditer(
            r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem\{|\\end\{thebibliography\})',
            bib_section, re.DOTALL
        ):
            key, content = m.group(1).strip(), m.group(2).strip()
            if key not in bib_entries:
                bib_entries[key] = content

        # Collect any bibitems NOT in cite_order (extras at end)
        extras = [k for k in bib_entries if k not in cite_order]

        # Build reordered bibliography
        reordered_lines = ["\\begin{thebibliography}{99}"]
        for key in cite_order:
            if key in bib_entries:
                reordered_lines.append(f"\\bibitem{{{key}}}\n{bib_entries[key]}")
        for key in extras:
            reordered_lines.append(f"\\bibitem{{{key}}}\n{bib_entries[key]}")
        reordered_lines.append("\\end{thebibliography}")

        # Find where the bib section ends in original
        bib_end_m = re.search(r'\\end\{thebibliography\}', latex)
        bib_end = bib_end_m.end() if bib_end_m else len(latex)

        after_bib = latex[bib_end:]
        return latex[:bib_start] + "\n".join(reordered_lines) + after_bib

    @staticmethod
    def _dedup_bibitems(latex: str) -> str:
        seen: set[str] = set()
        result_lines, skip = [], False
        for line in latex.split('\n'):
            m = re.match(r'\s*\\bibitem\{([^}]+)\}', line)
            if m:
                key = m.group(1)
                if key in seen:
                    skip = True
                    continue
                seen.add(key)
                skip = False
            elif skip and re.match(r'\s*\\bibitem\{', line):
                skip = False
            if not skip:
                result_lines.append(line)
        return '\n'.join(result_lines)


# ─────────────────────────────────────────────────────────────────────────────
# ResearcherAgent
# ─────────────────────────────────────────────────────────────────────────────

class ResearcherAgent:
    SYSTEM = (
        "You are a world-class AI research scientist. "
        "Identify key papers, summarize SOTA, find gaps."
    )

    async def write_ieee_paper(
        self, task: str, document_context: str = "", experiment_results: str = "",
    ) -> str:
        logger.info("write_ieee_paper: starting section-by-section pipeline")

        req = parse_paper_requirements(task)
        extra_rules = self._extract_extra_rules(task)

        # Build source from all available material
        source = self._build_source(document_context, experiment_results, task)

        # For long documents: extract structured content chunk-by-chunk
        if len(document_context.strip()) > 500:
            logger.info("Document detected (%d chars) — running ThesisExtractor", len(document_context))
            extracted = await ThesisExtractor().extract(document_context, task)
            if extracted.strip():
                # Keep extracted structured content + raw excerpt for context
                raw_excerpt = document_context[:8000]
                source = (
                    "=== STRUCTURED EXTRACTION (tables, algorithms, figures, equations) ===\n"
                    + extracted
                    + "\n\n=== RAW DOCUMENT EXCERPT ===\n"
                    + raw_excerpt
                )
                logger.info("ThesisExtractor produced %d chars", len(extracted))

        # Resolve title — never let LLM hallucinate it
        title = req.title or self._extract_title(task)
        if not req.title:
            req.title = title  # store for SectionWriter to use
        keywords = req.keywords or await _generate_keywords(title, source[:3000])
        authors_block = self._build_author_block(req, document_context)

        # SectionWriter gets req so it knows word targets, ref count, title
        writer = SectionWriter(req=req)

        logger.info("Writing abstract")
        abstract = await writer.abstract(source, title, extra=extra_rules)

        logger.info("Writing introduction")
        intro = await writer.introduction(source, abstract, extra=extra_rules)

        logger.info("Writing related work")
        related = await writer.related_work(source, intro, extra=extra_rules)

        logger.info("Writing methodology")
        method = await writer.methodology(source, abstract + intro[:2000], extra=extra_rules)

        logger.info("Writing experimental setup")
        exp_setup = await writer.experimental_setup(source, method, extra=extra_rules)

        logger.info("Writing results and discussion")
        results_sec = await writer.results_and_discussion(source, method + exp_setup, extra=extra_rules)

        logger.info("Writing conclusion")
        conclusion = await writer.conclusion(source, results_sec, extra=extra_rules)

        logger.info("Writing references")
        all_sections = "\n\n".join([intro, related, method, exp_setup, results_sec, conclusion])
        references = await writer.references(source, all_sections)

        paper = self._assemble(
            title=title, keywords=keywords, authors_block=authors_block,
            abstract=abstract, intro=intro, related=related, method=method,
            exp_setup=exp_setup, results=results_sec, conclusion=conclusion,
            references=references, req=req,
        )
        paper = PaperPostProcessor.process(paper)
        logger.info("write_ieee_paper: complete (%d chars)", len(paper))
        return paper

    def _extract_extra_rules(self, task: str) -> str:
        lines = []
        in_req = False
        for line in task.split('\n'):
            l = line.strip()
            if re.search(r'REQUIREMENTS?:|RULES?:|INSTRUCTIONS?:|MUST\b', l, re.I):
                in_req = True
            if in_req and l:
                lines.append(l)
            if in_req and not l:
                in_req = False
        return "\n".join(lines[:30]) if lines else ""

    def _build_source(self, doc: str, results: str, task: str) -> str:
        parts = []
        if doc.strip():
            # BUG FIX: Increased from 35000 to 80000 chars to handle 100-page thesis
            # documents without losing critical results/tables from later chapters.
            parts.append("=== DOCUMENT (PRIMARY SOURCE) ===\n" + doc[:80000] + "\n=== END ===")
        if results.strip():
            parts.append("=== EXPERIMENTAL RESULTS ===\n" + results[:8000] + "\n=== END ===")
        if not parts:
            parts.append(f"Topic: {task}\nNo source document. Cite real published papers.")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_title(task: str) -> str:
        m = re.search(r'"([^"]{10,120})"', task)
        if m:
            return m.group(1)
        m = re.search(r"titled?:?\s*['\"]?([^\n'\"]{10,120})", task, re.I)
        if m:
            return m.group(1).strip().rstrip("'\"")
        return task.strip()[:100]

    @staticmethod
    def _build_author_block(req: PaperRequirements, doc_context: str) -> str:
        if req.authors:
            authors = [a.strip() for a in re.split(r',\s*|\s+and\s+', req.authors) if a.strip()]
            affil = req.affiliation or "University"
            email = req.email or ""
            blocks = []
            for i, author in enumerate(authors[:4]):
                email_line = f"\\\\{email}" if (i == 0 and email) else ""
                blocks.append(
                    f"\\IEEEauthorblockN{{{author}}}\n"
                    f"\\IEEEauthorblockA{{\\textit{{{affil}}}{email_line}}}"
                )
            return "\n\\and\n".join(blocks)

        # Only try to auto-extract authors from THESIS documents, never from
        # experiment logs (which contain "Training BERT-base", "Training RoBERTa", etc.)
        _is_thesis_doc = (
            doc_context and
            len(doc_context) > 2000 and
            # Thesis/paper docs contain sections; experiment logs don't
            any(kw in doc_context[:3000].lower() for kw in
                ["abstract", "introduction", "university", "department", "chapter",
                 "dissertation", "thesis", "ieee", "conference", "journal"]) and
            # Reject if it looks like code/experiment output
            not any(kw in doc_context[:500].lower() for kw in
                    ["training ", "epoch ", "loss:", "accuracy:", "using device",
                     "import torch", "def train", "==="])
        )

        if _is_thesis_doc:
            # Look for author name pattern: 2-4 Capitalized words, NOT followed by "on" (Training X on Y)
            m = re.search(
                r'^([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3})',
                doc_context[:2000], re.MULTILINE
            )
            if m:
                candidate = m.group(1)
                # Reject common false positives
                reject_words = {"Training", "Using", "Abstract", "Introduction",
                                "Carbon", "Machine", "Deep", "Natural", "Comparison"}
                if candidate.split()[0] not in reject_words:
                    affil_m = re.search(
                        r'((?:Sukkur IBA|[A-Z][a-z]+ University)[^\n,]{0,60})',
                        doc_context[:3000]
                    )
                    affil = affil_m.group(1).strip() if affil_m else "University"
                    return (
                        f"\\IEEEauthorblockN{{{candidate}}}\n"
                        f"\\IEEEauthorblockA{{\\textit{{{affil}}}}}"
                    )

        return (
            "\\IEEEauthorblockN{Authors}\n"
            "\\IEEEauthorblockA{\\textit{Department of Computer Science} \\\\\n"
            "\\textit{University} \\\\\nCity, Country}"
        )

    @staticmethod
    def _assemble(title: str, keywords: str, authors_block: str,
                  abstract: str, intro: str, related: str, method: str,
                  exp_setup: str, results: str, conclusion: str, references: str,
                  req: PaperRequirements) -> str:

        def clean(s: str) -> str:
            for pat in [r'\\documentclass[^\n]*\n', r'\\usepackage[^\n]*\n',
                        r'\\begin\{document\}', r'\\end\{document\}',
                        r'\\maketitle', r'\\title\{[^}]*\}', r'\\author\{[^}]*\}',
                        r'\\IEEEoverridecommandlockouts\n?']:
                s = re.sub(pat, '', s)
            return s.strip()

        secs = [clean(s) for s in [abstract, intro, related, method,
                                    exp_setup, results, conclusion, references]]
        abstract_c = secs[0]
        if '\\begin{abstract}' not in abstract_c:
            abstract_c = f"\\begin{{abstract}}\n{abstract_c}\n\\end{{abstract}}"

        docclass = _make_docclass(req)
        packages = _make_packages(req)

        # Keywords block differs by format
        if req.format == "ieee_journal" or req.format == "ieee_conference":
            kw_block = f"\\begin{{IEEEkeywords}}\n{keywords}\n\\end{{IEEEkeywords}}\n"
        else:
            kw_block = f"\\textbf{{Keywords:}} {keywords}\n\n"

        return (
            f"{docclass}\n"
            f"{packages}\n"
            f"\\title{{{title}}}\n\n"
            f"\\author{{\n{authors_block}\n}}\n\n"
            "\\begin{document}\n\\maketitle\n\n"
            f"{abstract_c}\n\n"
            f"{kw_block}\n"
            + "\n\n".join(secs[1:])
            + "\n\n\\end{document}\n"
        )

    async def search_literature(
        self, topic: str, domains=None,
        document_context: str = "", experiment_results: str = ""
    ) -> dict:
        route = classify_task(topic, document_context)
        if route in ("paper", "thesis_to_paper", "experiment_then_paper"):
            paper = await self.write_ieee_paper(
                task=topic, document_context=document_context,
                experiment_results=experiment_results,
            )
            return {"review": paper, "arxiv_papers": "", "is_paper": True}

        arxiv = await self._search_arxiv(topic)
        doc_section = (
            "\n\n=== PROVIDED DOCUMENT ===\n" + document_context[:15000] + "\n==="
            if document_context.strip() else ""
        )
        review = await llm(
            f"Topic: {topic}\nDomains: {', '.join(domains or ['general'])}\n"
            f"ArXiv:\n{arxiv}{doc_section}\n\nSynthesize a detailed technical review.",
            system=self.SYSTEM, agent="researcher",
        )
        return {"review": review, "arxiv_papers": arxiv, "is_paper": False}

    async def _search_arxiv(self, topic: str) -> str:
        try:
            from research.literature.arxiv_client import search_arxiv
            papers = search_arxiv(topic, limit=15)
            if papers:
                return "\n".join(f"- {p.title} ({p.year})" for p in papers)
        except Exception as e:
            logger.warning("ArXiv search failed: %s", e)
        result = await llm(
            f"List 15 real published papers on: {topic}",
            system="List real papers with authors and year. Be accurate.",
            agent="researcher",
        )
        return result or "No papers found."

    async def analyze_codebase(self, code: str, task: str) -> str:
        return await llm(
            f"Task: {task}\nCode:\n{code[:3000]}\nAnalyze.",
            system=self.SYSTEM, agent="researcher",
        )


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentAgent
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentAgent:
    SYSTEM = (
        "You are an expert ML researcher. Design experiments with baselines, "
        "metrics, and ablations. Be precise and quantitative."
    )

    async def design_experiment(self, task, code="", baseline="", data=""):
        return await llm(
            f"Task: {task}\nCode:\n{code[:2000]}\n"
            f"Baseline: {baseline[:500]}\nData: {data[:500]}\nDesign experiment.",
            system=self.SYSTEM, agent="experiment",
        )

    async def generate_training_script(self, task, arch="", dataset=""):
        return await llm(
            f"Generate a complete PyTorch training script:\n"
            f"Task: {task}\nArchitecture: {arch or 'appropriate'}\nDataset: {dataset}\n"
            "Include: data loading, model definition, training loop, evaluation, "
            "and result printing with exact metric values.",
            system=self.SYSTEM, agent="experiment",
        )

    async def analyze_results(self, results, hypotheses=""):
        return await llm(
            f"Results:\n{results[:3000]}\nHypotheses: {hypotheses}\n"
            "Provide rigorous quantitative analysis. Use only given numbers.",
            system=self.SYSTEM, agent="experiment",
        )


# ─────────────────────────────────────────────────────────────────────────────
# PaperWriterAgent
# ─────────────────────────────────────────────────────────────────────────────

class PaperWriterAgent:
    SYSTEM = "You are a senior IEEE paper writer.\n" + _IEEE_BASE_RULES

    async def write_paper(self, topic, research="", results="", code="", conference="ieee"):
        return await ResearcherAgent().write_ieee_paper(
            task=topic, document_context=research, experiment_results=results,
        )

    async def write_section(self, name, context):
        return await llm(
            f"Write the '{name}' section.\nContext:\n{context[:4000]}\n"
            "Use \\cite{key}. Paragraphs only, no bullets.",
            system=self.SYSTEM, agent="paper_writer",
        )