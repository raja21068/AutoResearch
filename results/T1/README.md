# T1: HumanEval Benchmark Results

## Summary
- **Benchmark**: HumanEval (Chen et al., 2021)
- **Tasks**: 164
- **Pass@1**: 106/164 (64.6%)
- **After CL-SHCG (K=3)**: 158/164 (96.3%)
- **Improvement**: +31.7%

## Files
- `humaneval_results.json` — full results with per-task details
- `humaneval_results.csv` — tabular format

## Configuration
- Coder: DeepSeek-Chat | Tester: GPT-4o-mini | Debugger: GPT-4o
- Temperature: 0 | Seed: 42 | Max K: 3
- Sandbox: subprocess fallback, 60s timeout
