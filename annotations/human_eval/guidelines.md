# Human Evaluation Guidelines

## Task
Evaluate the outputs of an AI multi-agent system on 10 cross-domain tasks.
Each task was run under 4 configurations (blinded, randomized order).

## Scoring Criteria (1-10 scale)

| Criterion | 1-3 (Poor) | 4-6 (Adequate) | 7-8 (Good) | 9-10 (Excellent) |
|-----------|-----------|----------------|------------|-------------------|
| Code Correctness | Crashes or wrong output | Partial correctness | Mostly correct, minor bugs | Fully correct, handles edge cases |
| Code Readability | Unstructured, no comments | Some structure | Clean, commented | Publication-quality, well-documented |
| Test Coverage | No tests | Basic happy-path | Edge cases covered | Comprehensive with mutation testing |
| Research Depth | No citations | Surface-level review | Solid coverage | Thorough, identifies gaps |
| Paper Clarity | Incoherent | Readable but vague | Clear methodology | Publication-ready prose |
| Overall Quality | Unusable | Functional prototype | Good quality | Production/publication ready |

## Protocol
1. Each evaluator independently scores all 40 outputs (10 tasks × 4 configs)
2. Configs are presented in randomized order without labels
3. Evaluators may add free-text comments
4. Disagreements > 3 points are flagged for discussion
5. Final scores are averaged across 3 evaluators

## Compensation
Evaluators received no compensation (voluntary participation as co-authors).
