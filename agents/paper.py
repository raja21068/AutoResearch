"""Paper agents: Outline, SectionWriter, Citation, Figure, Reviewer, Revision + Orchestrator."""

import asyncio, os, logging
from pathlib import Path
from llm import llm

logger = logging.getLogger(__name__)

class OutlineAgent:
    SYSTEM = "You are an expert academic writer. Produce a complete paper outline with sections and key points."
    async def run(self, topic, research="", results=""): return await llm(f"Topic: {topic}\nResearch:\n{research[:3000]}\nResults:\n{results[:2000]}\nOutline.", system=self.SYSTEM, agent="paper_writer")

class SectionWriterAgent:
    SYSTEM = "You are an expert academic writer. Write a rigorous section in LaTeX. No placeholders."
    async def write(self, section, research="", results="", outline=""): return await llm(f"Section: {section}\nOutline:\n{outline[:1000]}\nResearch:\n{research[:2000]}\nResults:\n{results[:2000]}\nWrite LaTeX.", system=self.SYSTEM, agent="paper_writer")

class CitationAgent:
    SYSTEM = "You are a citation expert. Produce BibTeX entries for all cited works."
    async def run(self, text, research=""): return await llm(f"Paper text:\n{text[:3000]}\nLiterature:\n{research[:3000]}\nGenerate BibTeX.", system=self.SYSTEM, agent="researcher")

class FigureAgent:
    SYSTEM = "You are a scientific figure expert. Produce matplotlib code + LaTeX figure blocks."
    async def run(self, results): return await llm(f"Results:\n{results[:2500]}\nGenerate figure code.", system=self.SYSTEM, agent="paper_writer")

class ReviewerAgent:
    SYSTEM = "You are a NeurIPS area chair. Produce rigorous review: summary, strengths, weaknesses, score 1-10, recommendation."
    async def run(self, paper): return await llm(f"Paper:\n{paper[:6000]}\nReview.", system=self.SYSTEM, agent="critic")

class RevisionAgent:
    SYSTEM = "You are a revision expert. Revise paper addressing every reviewer concern."
    async def run(self, paper, review): return await llm(f"Paper:\n{paper[:5000]}\nReview:\n{review[:3000]}\nRevise.", system=self.SYSTEM, agent="paper_writer")

SECTIONS = ["Abstract","Introduction","Related Work","Methodology","Experiments","Results","Discussion","Conclusion"]

class PaperOrchestrator:
    """End-to-end: outline → sections → citations → figures → review → revise."""
    def __init__(self):
        self.outline = OutlineAgent(); self.writer = SectionWriterAgent()
        self.citation = CitationAgent(); self.figure = FigureAgent()
        self.reviewer = ReviewerAgent(); self.revision = RevisionAgent()

    async def run_full(self, topic, research="", results="", do_revision=True) -> dict:
        outline = await self.outline.run(topic, research, results)
        secs = await asyncio.gather(*[self.writer.write(s, research, results, outline) for s in SECTIONS])
        section_map = dict(zip(SECTIONS, secs))
        bib = await self.citation.run("\n".join(secs), research)
        figs = await self.figure.run(results)
        body = ""
        for s, c in section_map.items():
            body += f"\\begin{{abstract}}\n{c}\n\\end{{abstract}}\n\n" if s == "Abstract" else f"\\section{{{s}}}\n{c}\n\n"
        tex = f"\\documentclass{{article}}\n\\usepackage{{amsmath,graphicx,booktabs}}\n\\title{{{topic}}}\n\\begin{{document}}\n\\maketitle\n{body}{figs}\n\\begin{{thebibliography}}{{99}}\n{bib}\n\\end{{thebibliography}}\n\\end{{document}}\n"
        review = await self.reviewer.run(tex)
        revised = tex
        if do_revision:
            revised = await self.revision.run(tex, review)
        return {"outline": outline, "sections": section_map, "bibtex": bib, "figures": figs,
                "paper_tex": tex, "review": review, "revised_tex": revised}
