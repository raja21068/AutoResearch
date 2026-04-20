# Annotations

Human evaluation and citation validation artifacts.

## Structure

```
annotations/
├── human_eval/
│   ├── annotation_sheet.csv      — 120 annotations (3 evaluators × 10 tasks × 4 configs)
│   ├── annotations.json          — same data in JSON
│   ├── inter_annotator_agreement.json — Krippendorff's α, Cohen's κ
│   ├── evaluator_profiles.json   — evaluator backgrounds (anonymized)
│   └── guidelines.md             — scoring rubric and protocol
└── citations/
    ├── citation_validation.json  — 20 citations with DOIs, arXiv IDs
    ├── citation_validation.csv   — tabular format
    └── README.md
```
