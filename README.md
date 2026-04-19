# 🧠 AutoResearch

Unified multi-agent research & coding framework.

## Structure

```
NexusAI/
├── main.py              ← app + routes + entry
├── orchestrator.py      ← brain: auto code→test→debug→fix loop
├── llm.py               ← LLM provider (100+ models)
│
├── agents/              ← ALL agents (one file per domain)
│   ├── engineering.py   ← Planner, Coder, Tester, Debugger, Critic
│   ├── research.py      ← Researcher, Experiment, PaperWriter
│   ├── conception.py    ← Guide, Generator, IdeaCritic, Ranker, Structurer
│   ├── paper.py         ← Outline, Section, Citation, Figure, Reviewer, Revision
│   ├── experiment.py    ← Planner, CodeGen, Runner, Tracker, Evaluator
│   ├── decision.py      ← GO/NO_GO research gates
│   ├── memory.py        ← cross-run knowledge
│   ├── gan.py           ← adversarial generate-evaluate loop
│   ├── hooks.py         ← session lifecycle events
│   ├── context_modes.py ← dev/research/review switching
│   └── registry.py      ← unified lookup (68 agents)
│
├── tools/               ← sandbox, file reader, executor, output manager
│
├── skills/              ← knowledge files (no Python logic)
│   ├── agents/          ← 58 agent definitions (.md)
│   ├── rules/           ← 13 languages × 5 categories
│   ├── commands/        ← 34 workflow templates
│   └── contexts/        ← dev/research/review modes
│
├── research/            ← research library (curated)
│   ├── literature/      ← ArXiv, Semantic Scholar search
│   ├── pipeline/        ← 23-stage state machine
│   ├── templates/       ← LaTeX styles (NeurIPS/ICML/ICLR)
│   ├── domains/         ← domain detection
│   ├── knowledge/       ← knowledge base
│   ├── trends/          ← trend analysis
│   ├── web/             ← web crawler
│   ├── hitl/            ← human-in-the-loop
│   ├── assessor/        ← quality scoring
│   ├── memory/          ← advanced memory (decay, embeddings)
│   └── mcp/             ← MCP integration
│
├── static/              ← web GUI
├── tests/
└── eval/
```

## Quick start

```bash
bash deploy.sh
source .venv/bin/activate
python main.py          # → http://localhost:8000
```

## How it works

```
Input: Papers (PDF/DOCX) + GitHub repo + Dataset + Instructions
  ↓
Planner → creates execution plan
  ↓
Coder → generates code (streaming)
  ↓
Tester → generates + runs pytest
  ↓ PASS? → done ✅
  ↓ FAIL? → Debugger → fixes → re-test (up to 3x)
  ↓
Critic → final quality review
  ↓
Output: code/ + experiments/ + paper/ + knowledge/ + summary.json
```

## API

| Endpoint | What it does |
|----------|-------------|
| `POST /api/agent/run` | Run task synchronously |
| `POST /api/agent/stream` | Run with SSE streaming |
| `POST /api/upload` | Upload PDF/DOCX/CSV/JSON |
| `POST /api/conception/ideate` | Research ideation pipeline |
| `POST /api/experiment/run` | Experiment pipeline |
| `POST /api/paper/write` | Paper generation pipeline |
| `POST /api/gan/run` | Adversarial generate-evaluate |
| `GET /api/skills/agents` | List 58 skill agents |
| `GET /api/skills/rules/{lang}` | Get language rules |
| `GET /api/outputs` | Browse saved outputs |
