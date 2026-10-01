# CHI format notes — verified 2026-10-01

These notes record the **current CHI 2027 Papers requirements as a formatting benchmark**. They are not assumed to govern a later CHI cycle without re-verification.

## Current official requirements

Source pages:

- https://chi2027.acm.org/authors/papers/
- https://chi2027.acm.org/chi-publication-formats/

Verified on 2026-10-01:

- Review submissions use the ACM **single-column** manuscript format.
- LaTeX review submissions use `\documentclass[manuscript,review,anonymous]{acmart}` when anonymized.
- Submissions between **5,000 and 8,000 words are encouraged**.
- Typical long CHI papers are approximately **7,000–8,000 words**, excluding references, figure/table captions, and appendices.
- Submissions above **12,000 words** risk desk rejection unless exceptional length is strongly justified.
- Abstracts must be **150 words or fewer**.
- The paper must stand alone; essential evidence cannot be outsourced to supplementary material.
- Anonymization applies to the manuscript and review-stage supplementary/external materials.
- Accessibility is an explicit submission requirement.

The current manuscript abstract in `frontmatter_conclusion.md` is 133 words under simple whitespace tokenization and therefore fits the current 150-word cap.

## Calendar boundary

The CHI 2027 initial full-paper deadline was **2026-09-10 AoE** and has passed. Unless this manuscript corresponds to an already-submitted CHI 2027 paper, the 2027 rules above are used only as the current CHI formatting benchmark. The actual target-cycle call and templates must be re-verified before submission.

## Writing target for this manuscript

Use a working main-text target of approximately **7,000–8,000 words** rather than trying to maximize the 12,000-word ceiling. The contribution is methodological and should remain focused enough that reviewers can see the claim-centered workflow, the SRL propagation result, and the independent MCFW validation without wading through implementation inventory.

Prefer compression in this order:

1. remove repeated explanation of the same robustness distinction across Introduction, Results, and Discussion;
2. move implementation-level provenance details that are not needed for scientific evaluation to supplementary/repository materials while retaining enough main-text detail for review;
3. collapse repetitive detector/AOI descriptions once their frozen contracts are defined;
4. retain the full decision-universe definition, failure semantics, primary estimator, and the empirical quantities needed to assess the contribution;
5. do **not** solve length pressure by removing relevant prior work, methodological detail needed for correctness, or limitations.

## Submission checks to add later

Before a real CHI submission:

- regenerate a single-column anonymous ACM PDF;
- verify main-text word count under the target cycle's definition;
- verify abstract word count mechanically;
- verify all references resolve and BibTeX renders author names correctly;
- include accessible figure alt text/descriptions and inspect contrast/legibility;
- strip deanonymizing repository/user identifiers from review materials or provide an anonymized artifact path;
- review the target year's ACM AI-authorship/use policy and disclose tool use as required;
- re-check concurrent/prior-submission requirements against any related eyeprocess/eyeprocesspy software papers.
