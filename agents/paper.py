"""
Paper agents: powered by PaperOrchestra (Song et al., 2026, arXiv:2604.05018).

5-step pipeline:
  Step 1: Outline Agent     — idea + experiments → JSON outline
  Step 2: Plotting Agent    — outline → figures (parallel with Step 3)
  Step 3: Literature Agent  — outline → citations + intro/related work
  Step 4: Section Writer    — one-shot draft of remaining sections
  Step 5: Refinement Agent  — iterative peer-review + revision

Plus utilities:
  - Paper Autoraters  — Citation F1, lit review quality, SxS comparison
  - Research Aggregator — agent logs → structured inputs
  - Writing Bench — benchmark construction from existing papers
"""

import asyncio, logging, os, json, re
from pathlib import Path
from llm import llm

logger = logging.getLogger(__name__)


def _load_skill_prompt(skill_name: str) -> str:
    """Load SKILL.md for a paper-pipeline skill."""
    try:
        from skills.paper_pipeline_loader import get_paper_pipeline
        return get_paper_pipeline().get_prompt(skill_name)
    except Exception:
        return ""


def _load_reference(skill_name: str, ref_name: str) -> str:
    """Load a reference doc from a paper-pipeline skill."""
    try:
        from skills.paper_pipeline_loader import get_paper_pipeline
        return get_paper_pipeline().get_reference(skill_name, ref_name)
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════
# Step 1: Outline Agent
# ═══════════════════════════════════════════════════════

class OutlineAgent:
    """Converts (idea, experiments, template, guidelines) → JSON outline."""

    async def run(self, idea: str, experiments: str = "", guidelines: str = "") -> str:
        skill = _load_skill_prompt("outline-agent")
        prompt_ref = _load_reference("outline-agent", "prompt.md")
        schema_ref = _load_reference("outline-agent", "outline-schema.md")

        system = skill[:3000] if skill else (
            "You are an expert paper outliner. Convert research idea and experiments "
            "into a JSON outline with: plotting_plan, intro_related_work_plan, section_plan."
        )
        prompt = (
            f"Idea:\n{idea[:3000]}\n\n"
            f"Experimental log:\n{experiments[:3000]}\n\n"
            f"Guidelines:\n{guidelines[:1000]}\n\n"
        )
        if prompt_ref:
            prompt += f"\n--- Prompt Reference ---\n{prompt_ref[:2000]}\n"
        if schema_ref:
            prompt += f"\n--- Schema ---\n{schema_ref[:1500]}\n"
        prompt += "\nProduce the JSON outline."

        result = await llm(prompt, system=system, agent="paper_writer")
        if not result.strip():
            logger.warning("OutlineAgent received empty LLM response")
            return '{"plotting_plan": [], "intro_related_work_plan": {}, "section_plan": []}'
        return result


# ═══════════════════════════════════════════════════════
# Step 2: Plotting Agent
# ═══════════════════════════════════════════════════════

class PlottingAgent:
    """Generates matplotlib figures + captions from experiment data."""

    async def run(self, outline: str, experiments: str = "", idea: str = "") -> str:
        skill = _load_skill_prompt("plotting-agent")
        system = skill[:3000] if skill else (
            "You are a scientific figure expert. Generate matplotlib code for "
            "publication-quality figures with proper captions."
        )
        prompt = (
            f"Plotting plan from outline:\n{outline[:2000]}\n\n"
            f"Experimental data:\n{experiments[:3000]}\n\n"
            f"Research idea:\n{idea[:1000]}\n\n"
            "Generate matplotlib code and LaTeX figure blocks."
        )
        result = await llm(prompt, system=system, agent="paper_writer")
        if not result.strip():
            logger.warning("PlottingAgent received empty LLM response")
            return "% No figures generated — LLM returned empty response"
        return result


# ═══════════════════════════════════════════════════════
# Step 3: Literature Review Agent
# ═══════════════════════════════════════════════════════

class LiteratureReviewAgent:
    """Discovers citations, verifies via S2, drafts intro + related work."""

    async def run(self, outline: str, idea: str = "", experiments: str = "") -> str:
        skill = _load_skill_prompt("literature-review-agent")
        discovery = _load_reference("literature-review-agent", "discovery-pipeline.md")
        verification = _load_reference("literature-review-agent", "verification-rules.md")
        density = _load_reference("literature-review-agent", "citation-density-rule.md")

        system = skill[:3000] if skill else (
            "You are a literature review expert. Find relevant papers, verify them, "
            "and draft Introduction + Related Work with proper citations."
        )
        prompt = (
            f"Outline search plan:\n{outline[:2000]}\n\n"
            f"Idea:\n{idea[:1500]}\n\n"
            f"Experiments:\n{experiments[:1000]}\n\n"
        )
        if discovery:
            prompt += f"\n--- Discovery Pipeline ---\n{discovery[:1500]}\n"
        if verification:
            prompt += f"\n--- Verification Rules ---\n{verification[:1000]}\n"
        if density:
            prompt += f"\n--- Citation Density Rule ---\n{density[:800]}\n"
        prompt += "\nDraft Introduction + Related Work with BibTeX."

        result = await llm(prompt, system=system, agent="researcher")
        if not result.strip():
            logger.warning("LiteratureReviewAgent received empty LLM response")
            return "% Introduction and Related Work could not be generated"
        return result


# ═══════════════════════════════════════════════════════
# Step 4: Section Writing Agent
# ═══════════════════════════════════════════════════════

class SectionWriterAgent:
    """ONE comprehensive call to draft remaining sections (Abstract, Method, Experiments, Conclusion)."""

    async def run(self, outline: str, idea: str = "", experiments: str = "",
                  intro_relwork: str = "", figures: str = "") -> str:
        skill = _load_skill_prompt("section-writing-agent")
        prompt_ref = _load_reference("section-writing-agent", "prompt.md")
        table_ref = _load_reference("section-writing-agent", "latex-table-patterns.md")
        fig_ref = _load_reference("section-writing-agent", "figure-integration.md")

        system = skill[:3000] if skill else (
            "You are an expert academic writer. Write Abstract, Methodology, "
            "Experiments, and Conclusion in ONE comprehensive LaTeX draft."
        )
        prompt = (
            f"Outline:\n{outline[:2000]}\n\n"
            f"Idea:\n{idea[:2000]}\n\n"
            f"Experimental log:\n{experiments[:3000]}\n\n"
            f"Intro + Related Work (already written):\n{intro_relwork[:2000]}\n\n"
            f"Figures:\n{figures[:1000]}\n\n"
        )
        if prompt_ref:
            prompt += f"\n--- Writing Prompt ---\n{prompt_ref[:2000]}\n"
        if table_ref:
            prompt += f"\n--- Table Patterns ---\n{table_ref[:1000]}\n"
        if fig_ref:
            prompt += f"\n--- Figure Integration ---\n{fig_ref[:800]}\n"
        prompt += "\nWrite the complete remaining sections in LaTeX."

        result = await llm(prompt, system=system, agent="paper_writer")
        if not result.strip():
            logger.warning("SectionWriterAgent received empty LLM response")
            return "% Remaining sections could not be generated"
        return result


# ═══════════════════════════════════════════════════════
# Step 5: Content Refinement Agent
# ═══════════════════════════════════════════════════════

class ContentRefinementAgent:
    """Iterative peer review + revision with accept/revert halt rules."""

    async def run(self, paper_tex: str, experiments: str = "",
                  max_iterations: int = 3) -> dict:
        skill = _load_skill_prompt("content-refinement-agent")
        rubric = _load_reference("content-refinement-agent", "reviewer-rubric.md")
        halt_rules = _load_reference("content-refinement-agent", "halt-rules.md")
        safe_rules = _load_reference("content-refinement-agent", "safe-revision-rules.md")

        review_system = (
            "You are a harsh NeurIPS/ICML reviewer. Score 1-10, list weaknesses, "
            "questions, and actionable suggestions. Be specific."
        )
        if rubric:
            review_system += f"\n\n--- Rubric ---\n{rubric[:2000]}"

        revise_system = skill[:2000] if skill else (
            "You are a paper revision expert. Address EVERY reviewer concern. "
            "Preserve what works. Return complete revised LaTeX."
        )
        if safe_rules:
            revise_system += f"\n\n--- Safe Revision Rules ---\n{safe_rules[:1500]}"

        current = paper_tex
        history = []

        for i in range(1, max_iterations + 1):
            # Review
            review = await llm(
                f"Paper (iteration {i}):\n{current[:6000]}\n\nReview rigorously.",
                system=review_system, agent="critic"
            )
            if not review.strip():
                logger.warning("Refinement reviewer returned empty at iteration %d", i)
                break

            # Parse score
            score = 0.0
            m = re.search(r'(?:score|overall)[:\s]*(\d+(?:\.\d+)?)', review, re.I)
            if m:
                score = float(m.group(1))

            history.append({"iteration": i, "review": review[:1000], "score": score})

            # Halt check
            if score >= 7.5:
                logger.info("Refinement accepted at iteration %d (score=%.1f)", i, score)
                break

            # Revise
            revised = await llm(
                f"Paper:\n{current[:5000]}\n\nReview:\n{review[:2000]}\n\n"
                f"Experiments (ground truth):\n{experiments[:1500]}\n\n"
                "Revise addressing every concern. Return complete LaTeX.",
                system=revise_system, agent="paper_writer"
            )
            if not revised.strip():
                logger.warning("Revision returned empty at iteration %d — keeping previous", i)
                break
            current = revised

        return {"final_tex": current, "iterations": len(history),
                "history": history, "final_score": history[-1]["score"] if history else 0}


# ═══════════════════════════════════════════════════════
# Paper Autoraters
# ═══════════════════════════════════════════════════════

class PaperAutorater:
    """Quality scoring from PaperOrchestra (arXiv:2604.05018, App. F.3)."""

    async def citation_f1(self, paper_tex: str, references: str) -> str:
        prompt_ref = _load_reference("paper-autoraters", "citation-f1-prompt.md")
        system = prompt_ref[:2000] if prompt_ref else (
            "Partition the reference list into P0 (must-cite) and P1 (good-to-cite). "
            "Return JSON {ref_num: P0|P1}."
        )
        result = await llm(f"Paper:\n{paper_tex[:4000]}\n\nReferences:\n{references[:2000]}",
                           system=system, agent="critic")
        return result or '{"error": "empty response"}'

    async def lit_review_quality(self, paper_tex: str) -> str:
        prompt_ref = _load_reference("paper-autoraters", "litreview-quality-prompt.md")
        system = prompt_ref[:2000] if prompt_ref else (
            "Score Intro+Related Work on 6 axes (0-100): coverage, positioning, "
            "recency, density, coherence, critical analysis."
        )
        result = await llm(f"Paper:\n{paper_tex[:5000]}", system=system, agent="critic")
        return result or '{"error": "empty response"}'

    async def sxs_comparison(self, paper_a: str, paper_b: str) -> str:
        prompt_ref = _load_reference("paper-autoraters", "sxs-paper-quality-prompt.md")
        system = prompt_ref[:2000] if prompt_ref else (
            "Compare two papers side-by-side. Return {winner: paper_1|paper_2|tie, reasoning: ...}"
        )
        result = await llm(f"Paper A:\n{paper_a[:3000]}\n\nPaper B:\n{paper_b[:3000]}",
                           system=system, agent="critic")
        return result or '{"error": "empty response"}'


# ═══════════════════════════════════════════════════════
# Full PaperOrchestra Orchestrator
# ═══════════════════════════════════════════════════════

SECTIONS = ["Abstract", "Introduction", "Related Work", "Methodology",
            "Experiments", "Results", "Discussion", "Conclusion"]


class PaperOrchestrator:
    """
    Full PaperOrchestra pipeline (Song et al., 2026):
      Step 1: Outline
      Step 2: Plotting  }  parallel
      Step 3: Literature }
      Step 4: Section writing (one-shot)
      Step 5: Content refinement (iterative)
    """

    def __init__(self):
        self.outline = OutlineAgent()
        self.plotter = PlottingAgent()
        self.literature = LiteratureReviewAgent()
        self.writer = SectionWriterAgent()
        self.refiner = ContentRefinementAgent()
        self.autorater = PaperAutorater()

    async def run_full(self, topic: str, idea: str = "", experiments: str = "",
                       guidelines: str = "", do_refinement: bool = True,
                       do_scoring: bool = False) -> dict:
        """Run the complete pipeline.

        BUG FIX: The original 5-step pipeline produces tiny stubs (530-848 chars)
        because each agent gets minimal context and the assemble step produces
        a bare \documentclass wrapper with empty sections.

        Solution: Delegate to ResearcherAgent.write_ieee_paper() which uses a
        proper section-by-section pipeline with high max_tokens, thesis extraction,
        and post-processing. This is the path that produces 45K+ char complete papers.
        """
        logger.info("PaperOrchestra: %s", topic[:80])

        # Build a rich source document from all available context
        source_parts = []
        if idea and len(idea) > 100:
            source_parts.append(f"=== IDEA / SOURCE ===\n{idea}\n=== END ===")
        if experiments and len(experiments) > 50:
            source_parts.append(f"=== EXPERIMENTAL RESULTS ===\n{experiments}\n=== END ===")
        document_context = "\n\n".join(source_parts)

        try:
            from agents.research import ResearcherAgent as _RA
            researcher = _RA()
            paper_tex = await researcher.write_ieee_paper(
                task=topic,
                document_context=document_context,
                experiment_results=experiments,
            )
        except Exception as e:
            logger.error("ResearcherAgent.write_ieee_paper failed: %s — falling back to stub", e)
            paper_tex = self._assemble(topic, idea[:2000] if idea else topic, experiments[:2000], "")

        return {
            "topic": topic,
            "outline": "",
            "figures": "",
            "intro_related_work": "",
            "draft": "",
            "paper_tex": paper_tex,
            "refinement": {"iterations": 0, "final_score": 0, "history": []},
            "scores": {},
        }

    def _assemble(self, topic, intro_relwork, sections, figures):
        """
        Minimal fallback — only reached if write_ieee_paper() throws.
        Extracts just the title from topic so the task body never goes into \title{}.
        """
        import re as _re
        # Extract the paper title (quoted string or first non-instruction line)
        _title = ""
        m = _re.search(r'"([^"]{10,120})"', topic)
        if m:
            _title = m.group(1).strip()
        else:
            for line in topic.split('\n'):
                line = line.strip()
                if (line and not line.startswith('#') and
                        not any(kw in line.lower() for kw in
                                ['write a', 'step ', 'pipeline', 'format', 'requirement',
                                 'section', 'include', 'must', 'target'])):
                    _title = line[:120]
                    break
        _title = _title or "Research Paper"

        # Extract experiment table if present in sections (first 3000 chars)
        _exp_block = ""
        if "Accuracy" in sections or "Dataset" in sections:
            _exp_block = sections[:3000]

        return (
            "\\documentclass[conference]{IEEEtran}\n"
            "\\usepackage{amsmath,amssymb,graphicx,booktabs,hyperref,cite,algorithm,algorithmic}\n"
            f"\\title{{{_title}}}\n"
            "\\author{\\IEEEauthorblockN{Authors}\\IEEEauthorblockA{University}}\n"
            "\\begin{document}\n\\maketitle\n\n"
            "\\begin{abstract}\n"
            "Paper generation encountered an error. Experimental results are available below.\n"
            "\\end{abstract}\n\n"
            "\\section{Experimental Results}\n"
            f"The following results were obtained from the experiment pipeline.\n\n"
            f"{_exp_block}\n\n"
            "\\end{document}\n"
        )

    def save(self, result: dict, output_dir: str = "./output/paper") -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        saved = {}
        for name, content in [
            ("paper.tex", result.get("paper_tex", "")),
            ("outline.json", result.get("outline", "")),
            ("figures.tex", result.get("figures", "")),
            ("intro_relwork.tex", result.get("intro_related_work", "")),
        ]:
            if content:
                path = os.path.join(output_dir, name)
                Path(path).write_text(str(content))
                saved[name] = path
        return saved
