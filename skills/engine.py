"""
skills/rule_engine.py — Language-specific coding rules engine.

Loads rules from skills/rules/<language>/ (coding-style, testing, hooks,
patterns, security) and provides them as context to coding agents.

Supported languages: python, java, kotlin, dart, golang, rust, cpp, csharp,
                     swift, php, perl, typescript, web
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

RULE_CATEGORIES = ["coding-style", "testing", "hooks", "patterns", "security"]
LANG_ALIASES = {
    "py": "python", "js": "typescript", "ts": "typescript",
    "go": "golang", "c++": "cpp", "c#": "csharp", "cs": "csharp",
    "rb": "ruby", "rs": "rust", "kt": "kotlin",
}
EXT_TO_LANG = {
    ".py": "python", ".js": "typescript", ".ts": "typescript",
    ".tsx": "typescript", ".jsx": "typescript",
    ".java": "java", ".kt": "kotlin", ".go": "golang",
    ".rs": "rust", ".cpp": "cpp", ".c": "cpp", ".h": "cpp",
    ".cs": "csharp", ".swift": "swift", ".php": "php",
    ".pl": "perl", ".dart": "dart", ".html": "web", ".css": "web",
}


class RuleEngine:
    """Provides language-specific coding rules from skills/rules/."""

    def __init__(self, rules_dir: str = "skills/rules"):
        self.rules_dir = Path(rules_dir)
        self._cache: dict[str, dict[str, str]] = {}
        self._common: dict[str, str] = {}
        self._load_common()

    def _load_common(self):
        """Load common rules that apply to all languages."""
        common = self.rules_dir / "common"
        if common.exists():
            for f in common.glob("*.md"):
                self._common[f.stem] = f.read_text(encoding="utf-8", errors="ignore")

    def get_rules(self, language: str, categories: list[str] | None = None) -> str:
        """Get concatenated rules for a language."""
        lang = self._normalize(language)
        cats = categories or RULE_CATEGORIES

        if lang not in self._cache:
            self._cache[lang] = self._load_lang(lang)

        parts = []
        # Common rules first
        for cat in cats:
            if cat in self._common:
                parts.append(f"=== Common: {cat} ===\n{self._common[cat][:1500]}")
        # Language-specific rules
        for cat in cats:
            if cat in self._cache.get(lang, {}):
                parts.append(f"=== {lang}: {cat} ===\n{self._cache[lang][cat][:2000]}")

        return "\n\n".join(parts) if parts else ""

    def get_rules_for_file(self, filepath: str, categories: list[str] | None = None) -> str:
        """Auto-detect language from file extension and return rules."""
        ext = Path(filepath).suffix.lower()
        lang = EXT_TO_LANG.get(ext, "")
        if not lang:
            return ""
        return self.get_rules(lang, categories)

    def get_security_rules(self, language: str) -> str:
        """Get security-specific rules."""
        return self.get_rules(language, ["security"])

    def get_testing_rules(self, language: str) -> str:
        """Get testing-specific rules."""
        return self.get_rules(language, ["testing"])

    def available_languages(self) -> list[str]:
        """List all languages with rules."""
        if not self.rules_dir.exists():
            return []
        return sorted([
            d.name for d in self.rules_dir.iterdir()
            if d.is_dir() and d.name not in ("common", "zh", "__pycache__")
        ])

    def _normalize(self, lang: str) -> str:
        return LANG_ALIASES.get(lang.lower(), lang.lower())

    def _load_lang(self, lang: str) -> dict[str, str]:
        d = self.rules_dir / lang
        if not d.exists():
            return {}
        rules = {}
        for f in d.glob("*.md"):
            rules[f.stem] = f.read_text(encoding="utf-8", errors="ignore")
        return rules


# Singleton
_engine: RuleEngine | None = None

def get_rule_engine() -> RuleEngine:
    global _engine
    if _engine is None:
        _engine = RuleEngine()
    return _engine
