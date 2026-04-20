# Experiment Results

All results from the paper experiments.

## Directory Structure

```
results/
├── logs/                  — 120 run logs with timestamped events
│   ├── run_000.json ... run_119.json
│   └── summary.json
├── T1/                    — HumanEval benchmark (164 tasks)
│   ├── humaneval_results.json
│   ├── humaneval_results.csv
│   └── README.md
├── T2/                    — MBPP benchmark (427 tasks)
│   ├── mbpp_results.json
│   ├── mbpp_results.csv
│   └── README.md
├── T3/                    — CrossDomain-10 benchmark (10 tasks)
│   ├── CD-01/ ... CD-10/  — per-task outputs
│   │   ├── scores.json
│   │   ├── generated_code.py
│   │   └── experiment_output.md
│   ├── crossdomain_summary.json
│   └── README.md
├── ablation/              — Agent ablation + K sweep + error distribution
│   ├── ablation_results.json
│   └── error_distribution.json
└── routing/               — Per-agent model routing strategies
    └── routing_results.json
```

## Key Results (from paper)

| Benchmark | Pass@1 | After CL-SHCG (K=3) | Δ |
|-----------|--------|---------------------|---|
| HumanEval | 67.1% | 89.6% | +22.5% |
| MBPP | 72.4% | 87.3% | +14.9% |
| CrossDomain-10 | 40% (E2E) | 80% (E2E) | +40% |
