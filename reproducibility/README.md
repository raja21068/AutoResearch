# Reproducibility Artifact

Complete reproduction package for the paper experiments.

## Structure

```
reproducibility/
├── configs/
│   ├── experiment_config.json    — full experiment configuration
│   └── requirements_pinned.txt   — exact package versions
├── prompts/
│   ├── system_prompts.json       — system prompts for all 8 agents
│   └── prompt_templates.json     — user prompt templates with variables
├── data/
│   └── crossdomain10_tasks.json  — T3 benchmark task definitions
├── scripts/
│   ├── run_experiments.sh        — runs all experiments
│   └── analyze_results.py        — generates paper tables from results
└── results/                      — (populated after running experiments)
```

## How to Reproduce

1. **Install**: `pip install -r configs/requirements_pinned.txt`
2. **Set API key**: `export OPENAI_API_KEY=sk-...`
3. **Run**: `bash scripts/run_experiments.sh`
4. **Analyze**: `python scripts/analyze_results.py`

## Experiment Configuration

- **Random seed**: 42 (all experiments)
- **Temperature**: 0 (deterministic)
- **Models**: See `configs/experiment_config.json`
- **Self-healing**: K=3, test prepend injection
- **Sandbox**: subprocess, 60s timeout, temp directory isolation

## Total API Cost

~$284.50 across all experiments (T1 + T2 + T3 + ablation + routing).
