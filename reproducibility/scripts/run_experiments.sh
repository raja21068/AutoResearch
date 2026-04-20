#!/usr/bin/env bash
# Reproducibility script: runs all experiments from the paper
set -e

echo "═══════════════════════════════════════"
echo "  NexusAI Paper Experiments Runner"
echo "═══════════════════════════════════════"

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: Set OPENAI_API_KEY"
    exit 1
fi

# T1: HumanEval
echo "[T1] Running HumanEval (164 tasks)..."
python eval/swebench_runner.py --benchmark humaneval --output results/T1/ --seed 42

# T2: MBPP
echo "[T2] Running MBPP (427 tasks)..."
python eval/swebench_runner.py --benchmark mbpp --output results/T2/ --seed 42

# T3: CrossDomain-10
echo "[T3] Running CrossDomain-10 (10 tasks)..."
python eval/task_adapter.py --tasks reproducibility/data/crossdomain10_tasks.json --output results/T3/ --seed 42

# Ablation
echo "[Ablation] Running agent ablation..."
python eval/ablation_runner.py --config reproducibility/configs/experiment_config.json --output results/ablation/

echo ""
echo "Done. Results in results/"
