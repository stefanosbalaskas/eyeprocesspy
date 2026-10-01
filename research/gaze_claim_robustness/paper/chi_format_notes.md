# CHI format notes — verified 2026-10-01

These notes record the **current CHI 2027 Papers requirements as a formatting and submission benchmark**. They are not assumed to govern a later CHI cycle without re-verification.

## Current official requirements

Primary sources:

- https://chi2027.acm.org/authors/papers/
- https://chi2027.acm.org/contributions-to-chi/
- https://chi2027.acm.org/chi-publication-formats/
- https://chi2027.acm.org/chi-anonymization-policy/
- https://sigchi.org/resources/guides-for-authors/accessibility/

Verified on 2026-10-01:

- Review submissions use the ACM **single-column** manuscript format.
- Anonymous LaTeX review submissions use `\documentclass[manuscript,review,anonymous]{acmart}`.
- Submissions between **5,000 and 8,000 words are encouraged**; the average is approximately 7,000–8,000 words excluding references, figure/table captions, and appendices.
- Submissions above **12,000 words** can be desk-rejected when excessive length is not strongly justified.
- The paper must stand alone; essential evidence cannot be outsourced to supplementary material.
- Anonymization applies to the manuscript, supplementary materials, and author-controlled external links. Normal citations to prior work should remain visible rather than being replaced by `Anonymous`.
- Any non-video supplementary material should be **one ZIP** containing a README.
- Closely related concurrent submissions based on the same study, artifact, or dataset must be disclosed in PCS with an anonymized copy where required.
- Human-participant work requires a short reviewer note giving the applicable ethics-review context.
- CHI 2027 requires reviewer-expertise descriptors and four review-responsibility slots assigned to qualified authors unless an applicable exemption is granted.
- Accessibility is an explicit review requirement. SIGCHI asks authors to use semantic structure, avoid color-only encodings, provide `\Description{...}` text for figures, and make the review PDF accessible.
- ACM policy requires disclosure of material generative-AI use; AI systems are not authors and human authors remain responsible for the work.

The current manuscript build contains a **140-word abstract** and approximately **7.6k main-text words** under the repository's mechanical metric, keeping the paper within CHI's encouraged range rather than approaching the 12,000-word desk-rejection threshold.

## Calendar boundary

The CHI 2027 initial full-paper deadline was **2026-09-10 AoE** and has passed. Unless this manuscript corresponds to an already-submitted CHI 2027 paper, the 2027 rules above are a current benchmark only and cannot create a new CHI 2027 Papers submission after the deadline.

For an already-submitted CHI 2027 paper, keep the PCS metadata constraints in view: author addition/removal is not permitted after the original deadline, submission affiliations are treated as final under the stated policy, and title changes are permitted only when specifically requested by the Papers Chairs.

## Writing target for this manuscript

Use a working main-text target of approximately **7,000–8,000 words**. The contribution is methodological and should remain focused enough that reviewers can see the claim-centered workflow, the SRL propagation result, and the independent MCFW validation without reading an implementation inventory.

Prefer compression in this order:

1. remove repeated explanations of the same robustness distinction across Introduction, Results, and Discussion;
2. move implementation-level provenance details that are not needed for scientific evaluation to the anonymized supplement while retaining enough main-text detail for review;
3. collapse repetitive detector/AOI descriptions once their frozen contracts are defined;
4. retain the full decision-universe definition, failure semantics, primary estimator, and empirical quantities needed to assess the contribution;
5. do **not** solve length pressure by removing relevant prior work, methodological detail needed for correctness, or limitations.

## Automated checks now implemented

The paper workflow now:

- assembles the manuscript deterministically;
- checks abstract and manuscript metrics;
- regenerates frozen publication figures under version-pinned rendering;
- requires figures to use non-color encodings in addition to color;
- inserts `\Description{...}` text for all manuscript figures;
- builds the official anonymous single-column ACM review source;
- compiles the review PDF;
- scans extracted review text for known deanonymizing repository/workflow identifiers;
- verifies the explicit ChatGPT-use disclosure is present;
- builds one anonymized non-video supplementary ZIP with a README and deterministic manifest;
- scans the supplement for known author/repository/workflow identifiers.

## Author/PCS checks that cannot be automated from the repository

Before an actual submission/resubmission, authors must still:

- confirm the paper is eligible for the current CHI 2027 submission stage;
- verify the exact frozen PCS author list, affiliations, conflicts, and contact details;
- disclose any closely related concurrent submission as required;
- enter the requested short ethics-context note for this secondary-data study;
- choose focused reviewer-expertise descriptors available in PCS;
- fill all four review-responsibility slots with qualified paper authors or use an approved exemption;
- run a final anonymity inspection on the exact uploaded PDF/ZIP and every author-controlled external link;
- run an accessibility Full Check on the exact review PDF and remediate document tags, alternative text, title/language metadata, reading order, table headers, and tab order where necessary;
- open the files after upload to PCS and inspect those uploaded copies, not only the local files.

See `chi2027_submission_compliance.md` for the complete gate-by-gate ledger.
