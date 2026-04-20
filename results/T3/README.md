# T3: CrossDomain-10 Benchmark Results

## Summary
10 hybrid tasks requiring both research and code generation.

| Config | Code (0-10) | Research (0-10) | Paper (0-10) | E2E Complete |
|--------|-------------|-----------------|--------------|-------------|
| Engineering only | 6.2 | — | — | 4/10 |
| Research only | — | 7.1 | 5.8 | 5/10 |
| Sequential | 6.5 | 7.0 | 6.1 | 6/10 |
| **Proposed (CDCIP)** | **7.8** | **7.9** | **7.4** | **8/10** |

## Per-Task Directories
Each CD-XX/ folder contains:
- `scores.json` — per-config human evaluation scores
- `generated_code.py` — code produced by the system
- `experiment_output.md` — experiment results and baseline comparison

## Evaluators
3 independent evaluators (CS PhD students), scores averaged.
