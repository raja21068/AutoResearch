"""Research agents: Researcher, Experiment, PaperWriter."""

import logging
from llm import llm

logger = logging.getLogger(__name__)


class ResearcherAgent:
    SYSTEM = "You are a world-class AI research scientist. Identify key papers, summarize SOTA, find gaps."

    async def search_literature(self, topic, domains=None) -> dict:
        arxiv = await self._search_arxiv(topic)
        review = await llm(
            f"Topic: {topic}\nDomains: {', '.join(domains or ['general'])}\nArXiv:\n{arxiv}\nSynthesize review.",
            system=self.SYSTEM, agent="researcher")
        return {"review": review, "arxiv_papers": arxiv}

    async def _search_arxiv(self, topic):
        try:
            from research.literature.arxiv_client import search_arxiv
            papers = search_arxiv(topic, limit=15)
            if papers:
                return "\n".join(f"- {p.title} ({p.year})" for p in papers)
        except Exception as e:
            logger.warning("ArXiv: %s", e)
        return await llm(f"List 15 recent papers on: {topic}", system="List real papers.", agent="researcher")

    async def analyze_codebase(self, code, task):
        return await llm(f"Task: {task}\nCode:\n{code[:3000]}\nAnalyze.", system=self.SYSTEM, agent="researcher")


class ExperimentAgent:
    SYSTEM = "You are an expert ML researcher. Design experiments with baselines, metrics, ablations."

    async def design_experiment(self, task, code="", baseline="", data=""):
        return await llm(
            f"Task: {task}\nCode:\n{code[:2000]}\nBaseline: {baseline[:500]}\nData: {data[:500]}\nDesign experiment.",
            system=self.SYSTEM, agent="experiment")

    async def generate_training_script(self, task, arch="", dataset=""):
        return await llm(
            f"Generate PyTorch training script:\nTask: {task}\nArch: {arch or 'appropriate'}\nData: {dataset}\n"
            "Include: data loading, model, train loop, eval, checkpointing.",
            system=self.SYSTEM, agent="experiment")

    async def analyze_results(self, results, hypotheses=""):
        return await llm(f"Results:\n{results[:3000]}\nHypotheses: {hypotheses}\nAnalyze.",
                         system=self.SYSTEM, agent="experiment")


class PaperWriterAgent:
    SYSTEM = "You are an expert academic writer. Write publication-ready LaTeX."

    async def write_paper(self, topic, research="", results="", code="", conference="neurips"):
        sections = {}
        for sec in ["abstract","introduction","related work","methodology","experiments","conclusion"]:
            sections[sec] = await llm(
                f"Write {sec} for paper on: {topic}\nResearch:\n{research[:1500]}\n"
                f"Results:\n{results[:1500]}\nCode:\n{code[:800]}",
                system=self.SYSTEM, agent="paper_writer")
        return self._assemble(topic, sections)

    async def write_section(self, name, context):
        return await llm(f"Write {name} section.\nContext:\n{context[:4000]}", system=self.SYSTEM, agent="paper_writer")

    @staticmethod
    def _assemble(topic, sections):
        body = ""
        for key, (title, is_abs) in {
            "abstract": ("abstract", True), "introduction": ("Introduction", False),
            "related work": ("Related Work", False), "methodology": ("Methodology", False),
            "experiments": ("Experiments", False), "conclusion": ("Conclusion", False),
        }.items():
            c = sections.get(key, "")
            body += f"\\begin{{abstract}}\n{c}\n\\end{{abstract}}\n\n" if is_abs else f"\\section{{{title}}}\n{c}\n\n"
        return f"\\documentclass{{article}}\n\\usepackage{{amsmath,graphicx,booktabs,hyperref}}\n\\title{{{topic}}}\n\\begin{{document}}\n\\maketitle\n{body}\\end{{document}}\n"
