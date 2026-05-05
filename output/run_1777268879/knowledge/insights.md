Task: Write a complete IEEE conference paper (double-column) on:

"Comparison of BERT, RoBERTa, and DistilBERT on Sentiment Analysis Datasets: Accuracy vs Speed Trade-off"

PIPELINE:
Step 1 — Run experiments first using synthetic data only (no pretrained model downloads):
- Implement IMDB, SST-2, Twitter sentiment datasets (60 samples each, vocab=1000, max_len=20)
- Fine-tune and evaluate BERT-base, RoBERTa-base, DistilBERT-base on each dataset
- Measure accuracy, precision, recall, F1-score, and inference time in milliseconds
- Save results in a structured table

Step 2 — Write the paper using ONLY those results. Do NOT fabricate any numbers.

SECTION REQUIREMENTS:
- Introduction must be at least 600 words
- Related Work must be at least 700 words with minimum 20 citations
- Methodology must be at least 500 words including algorithm pseudocode
- Experimental Setup must be at least 450 words
- Results and Discussion must be at least 800 words with comparison tables and accuracy-vs-speed figures
- Conclusion must be at least 300 words

FORMATTING: No em dashes. No bullet points inside section text. Paragraph form only.
REFERENCES: IEEE style [1],[2],[3]. Include at least 30 unique references in ascending order.
TARGET: 8-10 IEEE pages, publication-ready quality.
Passed: False
Review: FAIL: The pipeline output does not contain a complete paper. The final step only shows a LaTeX preamble and the beginning of an abstract, but the paper is truncated and missing all required sections: Introduction (≥600 words), Related Work (≥700 words with ≥20 citations), Methodology (≥500 words with pseudocode), Experimental Setup (≥450 words), Results and Discussion (≥800 words with tables and figures), Conclusion (≥300 words), and References (≥30 IEEE-style references). The output ends mid-sentence in the abstract, indicating the paper was not fully generated. Additionally, the experimental results are from synthetic data with random labels, which would not constitute a valid publication-quality paper, but the primary failure is the incomplete paper generation.

Score: 2/10