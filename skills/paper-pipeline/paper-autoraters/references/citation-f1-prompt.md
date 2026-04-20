You are an expert academic reviewer. Read the following paper text and analyze its references.

Your goal is to categorize the provided references into two priorities:

Priority Levels
- P0 (Must-Have): Core citations strictly necessary for the paper. These MUST include:
  - Baselines directly compared against in experiments
  - Datasets the paper utilizes or evaluates on
  - Core methods the paper is directly building upon or modifying
  - Metrics or standard numbers heavily relied upon and cited from another paper
- P1 (Good-To-Have): Supplemental citations. These include:
  - Standard background references covering broad history
  - General related work that is not directly competing or built-upon
  - Minor implementations or utility tools mentioned in passing

Paper Text:
{paper_text}

References List:
{references_str}

Output Format
Please return ONLY a JSON dictionary where the keys are the exact reference numbers (e.g., "1", "2") and the values are either "P0" or "P1". Example output:
```json
{
  "1": "P0",
  "2": "P1",
  "3": "P0"
}
```
