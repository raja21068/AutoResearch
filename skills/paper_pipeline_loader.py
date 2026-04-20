"""
skills/paper_pipeline_loader.py — Loads PaperOrchestra skills.

9 specialized skills for paper writing (from arXiv:2604.05018):
  1. paper-orchestra        — top-level orchestrator
  2. outline-agent          — Step 1: idea → JSON outline
  3. plotting-agent         — Step 2: outline → figures
  4. literature-review-agent— Step 3: outline → citations + intro/related work
  5. section-writing-agent  — Step 4: one-shot draft of remaining sections
  6. content-refinement-agent— Step 5: iterative peer-review refinement
  7. paper-autoraters       — quality scoring (Citation F1, SxS, etc.)
  8. agent-research-aggregator — pre-pipeline: agent logs → structured inputs
  9. paper-writing-bench    — benchmark construction from existing papers
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).parent / "paper-pipeline"


@dataclass
class PaperSkill:
    name: str
    description: str
    skill_md: str          # full SKILL.md content
    scripts: dict = field(default_factory=dict)    # {name: content}
    references: dict = field(default_factory=dict)  # {name: content}
    step: int = 0          # pipeline step number (0 = utility)


def _load_skill(skill_dir: Path) -> PaperSkill:
    """Load a single paper skill from its directory."""
    name = skill_dir.name
    skill_md = ""
    skill_path = skill_dir / "SKILL.md"
    if skill_path.exists():
        skill_md = skill_path.read_text(encoding="utf-8", errors="ignore")

    # Extract description from frontmatter
    desc = ""
    if "description:" in skill_md:
        for line in skill_md.split("\n"):
            if line.strip().startswith("description:"):
                desc = line.split("description:", 1)[1].strip()
                break

    # Load scripts
    scripts = {}
    scripts_dir = skill_dir / "scripts"
    if scripts_dir.exists():
        for f in scripts_dir.iterdir():
            if f.is_file():
                scripts[f.name] = f.read_text(encoding="utf-8", errors="ignore")

    # Load references
    refs = {}
    refs_dir = skill_dir / "references"
    if refs_dir.exists():
        for f in refs_dir.iterdir():
            if f.is_file():
                refs[f.name] = f.read_text(encoding="utf-8", errors="ignore")

    # Determine step number
    step_map = {
        "outline-agent": 1, "plotting-agent": 2,
        "literature-review-agent": 3, "section-writing-agent": 4,
        "content-refinement-agent": 5,
    }
    step = step_map.get(name, 0)

    return PaperSkill(name=name, description=desc, skill_md=skill_md,
                       scripts=scripts, references=refs, step=step)


class PaperPipelineRegistry:
    """Registry of all PaperOrchestra skills."""

    def __init__(self):
        self.skills: dict[str, PaperSkill] = {}
        self._load()

    def _load(self):
        if not PIPELINE_DIR.exists():
            logger.warning("Paper pipeline dir not found: %s", PIPELINE_DIR)
            return
        for d in sorted(PIPELINE_DIR.iterdir()):
            if d.is_dir() and (d / "SKILL.md").exists():
                skill = _load_skill(d)
                self.skills[skill.name] = skill
        logger.info("Loaded %d paper-pipeline skills", len(self.skills))

    def get(self, name: str) -> PaperSkill:
        return self.skills.get(name)

    def get_step(self, step: int) -> PaperSkill:
        for s in self.skills.values():
            if s.step == step:
                return s
        return None

    def get_prompt(self, name: str) -> str:
        """Get the main prompt/SKILL.md for a skill."""
        s = self.get(name)
        return s.skill_md if s else ""

    def get_reference(self, skill_name: str, ref_name: str) -> str:
        """Get a specific reference document."""
        s = self.get(skill_name)
        if s:
            return s.references.get(ref_name, "")
        return ""

    def get_script(self, skill_name: str, script_name: str) -> str:
        """Get a specific script."""
        s = self.get(skill_name)
        if s:
            return s.scripts.get(script_name, "")
        return ""

    def pipeline_order(self) -> list[PaperSkill]:
        """Return skills in pipeline order (steps 1-5)."""
        return [s for s in sorted(self.skills.values(), key=lambda x: x.step) if s.step > 0]

    def list_all(self) -> list[str]:
        return sorted(self.skills.keys())

    def __len__(self):
        return len(self.skills)


_registry = None

def get_paper_pipeline() -> PaperPipelineRegistry:
    global _registry
    if _registry is None:
        _registry = PaperPipelineRegistry()
    return _registry
