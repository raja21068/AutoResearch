"""
Memory agent: cross-run knowledge persistence + meta-learning.

Stores:
  - Task results (success/failure)
  - Failure-derived lessons (what went wrong, how it was fixed)
  - Reusable skills extracted from successful runs
  - Pattern memory (recurring error → fix mapping)
"""

import json, logging, time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Lesson:
    """A failure-derived lesson."""
    task: str
    error_type: str
    error_msg: str
    fix_applied: str
    success: bool
    timestamp: float = 0.0

    def to_dict(self):
        return self.__dict__


@dataclass
class Skill:
    """A reusable skill extracted from a successful run."""
    name: str
    pattern: str        # what triggers this skill
    solution: str       # the reusable solution template
    times_used: int = 0
    success_rate: float = 1.0

    def to_dict(self):
        return self.__dict__


class MemoryAgent:
    """Cross-run knowledge persistence with meta-learning."""

    def __init__(self, persist_dir: str = ""):
        self._results: list[dict] = []
        self._lessons: list[Lesson] = []
        self._skills: list[Skill] = []
        self._error_fix_map: dict[str, list[str]] = {}  # error_pattern → [fixes]
        self._persist_dir = Path(persist_dir) if persist_dir else None
        if self._persist_dir:
            self._load()

    # ── Basic store/retrieve ──

    def store(self, task: str, result: str, success: bool = True) -> None:
        self._results.append({"task": task, "result": result, "success": success, "ts": time.time()})
        self._results = self._results[-100:]

    def retrieve(self, query: str, top_k: int = 5) -> str:
        q = query.lower()
        hits = [e for e in self._results
                if any(w in e["task"].lower() for w in q.split())]
        return "\n".join(e["result"][:300] for e in hits[:top_k])

    # ── Meta-learning: failure-derived lessons ──

    def record_failure(self, task: str, error_type: str, error_msg: str,
                       fix_applied: str, fix_worked: bool) -> None:
        """Record what went wrong and how it was fixed (or not)."""
        lesson = Lesson(
            task=task, error_type=error_type, error_msg=error_msg[:500],
            fix_applied=fix_applied[:500], success=fix_worked, timestamp=time.time()
        )
        self._lessons.append(lesson)
        self._lessons = self._lessons[-200:]

        # Build error → fix mapping for pattern matching
        key = self._error_key(error_type, error_msg)
        if key not in self._error_fix_map:
            self._error_fix_map[key] = []
        if fix_worked and fix_applied not in self._error_fix_map[key]:
            self._error_fix_map[key].append(fix_applied)
            self._error_fix_map[key] = self._error_fix_map[key][-5:]

        logger.info("Lesson recorded: %s → %s (%s)", error_type,
                     "fixed" if fix_worked else "failed", key[:40])

    def get_fix_suggestions(self, error_type: str, error_msg: str) -> list[str]:
        """Look up past fixes for similar errors."""
        key = self._error_key(error_type, error_msg)
        if key in self._error_fix_map:
            return self._error_fix_map[key]
        # Fuzzy match: try just error_type
        for k, fixes in self._error_fix_map.items():
            if error_type.lower() in k:
                return fixes
        return []

    def get_lessons(self, error_type: str = "", top_k: int = 5) -> list[dict]:
        """Retrieve past lessons, optionally filtered by error type."""
        lessons = self._lessons
        if error_type:
            lessons = [l for l in lessons if error_type.lower() in l.error_type.lower()]
        return [l.to_dict() for l in lessons[-top_k:]]

    # ── Reusable skills ──

    def extract_skill(self, name: str, pattern: str, solution: str) -> None:
        """Store a reusable skill from a successful run."""
        existing = next((s for s in self._skills if s.name == name), None)
        if existing:
            existing.times_used += 1
            existing.solution = solution
        else:
            self._skills.append(Skill(name=name, pattern=pattern, solution=solution))
            self._skills = self._skills[-50:]

    def find_skill(self, query: str) -> Skill | None:
        """Find a matching skill for a task."""
        q = query.lower()
        for skill in reversed(self._skills):
            if any(w in skill.pattern.lower() for w in q.split()):
                skill.times_used += 1
                return skill
        return None

    def list_skills(self) -> list[dict]:
        return [s.to_dict() for s in self._skills]

    # ── Context for agents ──

    def get_context(self, task: str) -> str:
        """Build a context string for agent prompts with relevant memory."""
        parts = []

        # Past results
        relevant = self.retrieve(task, top_k=3)
        if relevant:
            parts.append(f"=== PAST RESULTS ===\n{relevant}")

        # Relevant skills
        skill = self.find_skill(task)
        if skill:
            parts.append(f"=== REUSABLE SKILL: {skill.name} ===\n{skill.solution[:500]}")

        # Recent lessons
        lessons = self.get_lessons(top_k=3)
        if lessons:
            lesson_text = "\n".join(
                f"- {l['error_type']}: {l['fix_applied'][:100]} ({'worked' if l['success'] else 'failed'})"
                for l in lessons
            )
            parts.append(f"=== LEARNED LESSONS ===\n{lesson_text}")

        return "\n\n".join(parts)

    # ── Stats ──

    @property
    def stats(self) -> dict:
        total = len(self._lessons)
        fixed = sum(1 for l in self._lessons if l.success)
        return {
            "results_stored": len(self._results),
            "lessons_recorded": total,
            "lessons_success_rate": round(fixed / total * 100, 1) if total else 0,
            "skills_extracted": len(self._skills),
            "error_patterns": len(self._error_fix_map),
        }

    # ── Persistence ──

    def _error_key(self, error_type: str, error_msg: str) -> str:
        """Create a normalized key for error pattern matching."""
        msg = error_msg.lower().strip()
        # Remove line numbers and file paths for generalization
        import re
        msg = re.sub(r'line \d+', 'line N', msg)
        msg = re.sub(r'"/[^"]*"', '"FILE"', msg)
        msg = re.sub(r"'[^']*'", "'X'", msg)
        return f"{error_type.lower()}:{msg[:100]}"

    def _load(self):
        if not self._persist_dir or not self._persist_dir.exists():
            return
        try:
            p = self._persist_dir / "memory.json"
            if p.exists():
                data = json.loads(p.read_text())
                self._results = data.get("results", [])
                self._lessons = [Lesson(**l) for l in data.get("lessons", [])]
                self._skills = [Skill(**s) for s in data.get("skills", [])]
                self._error_fix_map = data.get("error_fix_map", {})
        except Exception as e:
            logger.warning("Memory load failed: %s", e)

    def save(self):
        if not self._persist_dir:
            return
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "results": self._results[-100:],
            "lessons": [l.to_dict() for l in self._lessons[-200:]],
            "skills": [s.to_dict() for s in self._skills[-50:]],
            "error_fix_map": self._error_fix_map,
        }
        (self._persist_dir / "memory.json").write_text(json.dumps(data, indent=2, default=str))
