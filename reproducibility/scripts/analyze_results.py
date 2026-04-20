"""Analyze experiment results and generate paper tables."""
import json, os, sys

def main():
    # Load T1
    t1 = json.load(open("results/T1/humaneval_results.json"))
    print(f"T1 HumanEval: pass@1={t1['pass_at_1_rate']}% → final={t1['final_rate']}%")
    
    # Load T2
    t2 = json.load(open("results/T2/mbpp_results.json"))
    print(f"T2 MBPP: pass@1={t2['pass_at_1_rate']}% → final={t2['final_rate']}%")
    
    # Load ablation
    abl = json.load(open("results/ablation/ablation_results.json"))
    print("\nAblation:")
    for a in abl["ablations"]:
        print(f"  -{a['removed']}: {a['pass_at_1']}% (Δ={a['delta']})")
    
    # Load routing
    rt = json.load(open("results/routing/routing_results.json"))
    print("\nRouting:")
    for s in rt["strategies"]:
        print(f"  {s['name']}: {s['pass']}% @ ${s['cost']}/task")

if __name__ == "__main__":
    main()
