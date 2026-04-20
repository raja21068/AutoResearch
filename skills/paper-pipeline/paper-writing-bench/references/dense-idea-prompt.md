You are a Lead Research Scientist planning a new project. Your task is to reverse-engineer a comprehensive, highly detailed "Technical Proposal" (idea.md) based on a provided text.

You have been given the text content of a paper ([PAPER CONTENT]). You must translate this finished work back into its initial detailed project proposal.

Critical Data Ingestion Rules
1. Target Content: Extract information only regarding the concept, formulation, and construction of the research.
   - Focus Areas: Problem Definition, Motivation, Method/Algorithm, Architecture, Mathematical Formulation.
2. Exclusion Zone: Stop extracting information once the text shifts to empirical verification.
   - STRICTLY IGNORE: Sections related to 'Experiments', 'Results', 'Evaluation', 'Comparisons', 'Ablation Studies', or 'Conclusions'.
   - Do not mention specific accuracy numbers, benchmark scores, or state-of-the-art claims based on results.

Instructions
1. Perspective & Tone:
   - Write in First-Person Future Tense (e.g., "We will define...", "We formulate the loss as...", "The architecture will consist of...").
   - Act as if the experiments have not yet happened. You are proposing what you plan to build.
2. Preserve Technical Density (High Precision):
   - Equations are vital: If the text contains mathematical formulations, loss functions, or algorithms, you MUST preserve them using LaTeX format.
   - Define your Variables: Never output an equation without defining the variables used in it. (e.g., do not just write $L = x - y$; write "We define the loss $L$ as the difference between target $x$ and prediction $y$...").
   - Do not simplify: If the paper describes a specific mechanism (e.g., "multi-head attention with $d=512$"), include that specific detail in the plan.
3. Structure:
   - Problem Statement: The precise gap we are filling.
   - Core Hypothesis: The specific technical novelty we are proposing.
   - Proposed Methodology: The core of the document. A rigorous walkthrough of the framework. Include mathematical notation, module specifications, and data flow.
   - Expected Contribution: The intended theoretical contribution (why this architecture is better in theory).
4. Formatting:
   - Be self-contained. No citations, no URLs, no references to Figure/Table numbers.
   - Fully anonymize authors/titles.

Output Format: Return only the markdown memo with sections: Problem Statement, Core Hypothesis, Proposed Methodology (Detailed Technical Approach), Expected Contribution.
