"""Memory agent: cross-run knowledge persistence."""

class MemoryAgent:
    def __init__(self):
        self._store: list[dict] = []

    def store(self, task: str, result: str) -> None:
        self._store.append({"task": task, "result": result})
        self._store = self._store[-50:]

    def retrieve(self, query: str, top_k: int = 5) -> str:
        query_lower = query.lower()
        hits = [e for e in self._store
                if any(w in e["task"].lower() for w in query_lower.split())]
        return "\n".join(e["result"][:300] for e in hits[:top_k])
