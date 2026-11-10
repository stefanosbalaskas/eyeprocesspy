# CHI 2027 paper submission compliance ledger

This is the author-facing control document for **From Gaze Signals to HCI Claims: Propagating Eye-Tracking Measurement Uncertainty Through Analysis Pipelines**. It distinguishes requirements that the repository/build can verify from requirements that must be confirmed or completed by the authors in PCS/ACM systems.

Official sources checked on 2026-10-01:

- CHI 2027, Contributions to CHI: https://chi2027.acm.org/contributions-to-chi/
- CHI 2027, Papers: https://chi2027.acm.org/authors/papers/
- CHI 2027, Publication Formats: https://chi2027.acm.org/chi-publication-formats/
- CHI 2027, Anonymization Policy: https://chi2027.acm.org/chi-anonymization-policy/
- CHI 2027, Identifying Reviewers for Your Paper: https://chi2027.acm.org/authors/papers/identifying-reviewers-for-your-paper/
- SIGCHI, Accessibility Guide for Authors: https://sigchi.org/resources/guides-for-authors/accessibility/
- ACM, Policy on Authorship / generative AI disclosure, as linked from the CHI 2027 Papers page.

When this ledger conflicts with the live CHI website or PCS, the live official source wins.

## Status vocabulary

- **VERIFIED** — checked in the current manuscript/build package.
- **CI-GATED** — enforced automatically by the paper build workflow.
- **AUTHOR/PCS ACTION** — cannot be truthfully completed from the repository; an author must confirm or enter it in PCS/ACM systems.
- **POST-ACCEPTANCE** — relevant only after conditional acceptance.

## 1. Submission timing

| Requirement | Status | Current evidence / action |
| --- | --- | --- |
| Initial CHI 2027 Papers deadline | **AUTHOR/PCS ACTION** | The official deadline was **2026-09-10 AoE**. Because this ledger is dated 2026-10-01, confirm that this paper already has a valid CHI 2027 Papers submission in PCS. A new initial Papers submission cannot be created after the deadline. |
| Review release | Informational | 2026-11-05. |
| Revise-and-resubmit window | Informational | 2026-11-05 through 2026-12-03 for papers invited to revise. |
| Final notification | Informational | 2026-12-17. |

## 2. Contribution fit: Methodology

CHI states that methodology contributions should improve how HCI researchers/designers conduct design or evaluation and asks whether the methodology is novel relative to prior methods, clearly positioned, sufficiently broad, described in enough detail to use, and demonstrably valuable.

| CHI methodology criterion | Status | Evidence in this paper |
| --- | --- | --- |
| Novelty contextualized against prior work | **VERIFIED** | Introduction/Related Work position the contribution against multiverse/specification-curve methods, eye-movement cleaning multiverses, detector comparison, AOI uncertainty, data quality, and reporting guidance. Novelty is framed narrowly as claim-centered propagation of a frozen multi-stage measurement decision space, not as inventing multiverse analysis. |
| Positioning relative to existing methodologies | **VERIFIED** | Related Work distinguishes general multiverse/specification-curve tooling from eye-tracking-specific measurement uncertainty and from detector benchmarking. |
| Applicability beyond one narrow user group | **VERIFIED** | The framework is written for HCI researchers using eye tracking; it is not tied to one participant demographic or one application domain. The two empirical cases use different hardware, sampling rates, and interaction contexts. |
| Sufficient detail for reuse | **VERIFIED** | Methods specify the claim contract, branch taxonomy, freeze rule, failure accounting, SRL 144-specification universe, statistical model, independent MCFW validation, and reporting dimensions. |
| Demonstrated methodological value | **VERIFIED** | SRL demonstrates end-to-end claim propagation; MCFW independently demonstrates structured detector-dependent measurement divergence. The paper does not select a preferred detector from these results. |

The paper should continue to foreground **methodological value to HCI**, not package size or feature count.

## 3. Review format and manuscript length

| Requirement | Status | Current evidence / action |
| --- | --- | --- |
| Official ACM single-column anonymous review format | **CI-GATED** | Review source must contain `\\documentclass[manuscript,review,anonymous]{acmart}`. Do not substitute a two-column or experimental class. |
| Paper length proportional to contribution | **VERIFIED** | Current assembled main-text estimate is approximately **7,578 words**, excluding headings/code/figure captions under the repository metric; CHI encourages 5,000–8,000 words and warns that submissions above 12,000 words may be desk-rejected absent strong justification. Recheck on the final PDF/source before submission. |
| Abstract concise | **CI-GATED** | Current abstract is 140 words; build fails if the configured 150-word limit is exceeded. |
| Paper stands alone | **VERIFIED, maintain** | Main paper contains the substantive method, datasets, model contracts, principal quantitative results, limitations, and contribution claims. Supplement is audit/reproducibility support only. |
| English and conference-paper form | **VERIFIED** | Current manuscript is an English full research paper. |

## 4. Anonymization

CHI can desk-reject anonymization violations in the paper, supplement, or external links.

| Requirement | Status | Current evidence / action |
| --- | --- | --- |
| No author/institution identity in title/header | **CI-GATED** | Review build uses anonymous ACM mode and `Anonymous Author(s)`. |
| No identifying repository/workflow links in review PDF | **CI-GATED** | Review-text scan rejects known repository/user strings, workflow IDs, and frozen commit identifiers. |
| No author identity in PDF metadata | **CI-GATED / manual final check** | Build is anonymous; inspect the final submitted PDF metadata once more before PCS upload. |
| Prior own work cited normally, in third person | **AUTHOR CHECK** | CHI says prior work should not be replaced with `Anonymous`; verify all self-citations use normal bibliographic entries and third-person phrasing. |
| Supplement anonymized | **CI-GATED** | Supplement builder scans all bundled text for known author/repository/workflow identifiers and emits no live repository link. |
| External links anonymized | **AUTHOR/PCS ACTION** | Do not enter a non-anonymous GitHub/OSF/project URL in PCS or manuscript during review. Public third-party dataset DOI links are acceptable because they identify the source datasets, not the submitting authors. |

## 5. Supplementary material

| Requirement | Status | Current evidence / action |
| --- | --- | --- |
| Non-video supplement is one ZIP | **CI-GATED** | `build_anonymized_supplement.py` emits one deterministic review ZIP. |
| ZIP contains README | **CI-GATED** | Builder includes `README.txt`. |
| Supplement supports but does not replace main paper | **VERIFIED, maintain** | Bundle contains decision contracts, selected analysis mechanics, compact frozen summaries, and figure snapshots; no essential claim is intended to exist only in the ZIP. |
| No private/participant data redistributed | **CI-GATED by curation** | Raw public datasets are not bundled; only compact derived summaries/contracts/code are included. |
| Supplement remains anonymous | **CI-GATED** | Known identifying strings are rejected before ZIP creation and after archive emission. |

## 6. Accessibility

SIGCHI asks authors to use semantic document structure, avoid relying only on colour, provide descriptions for all figures, and submit an accessible review PDF. For LaTeX, `\\Description{...}` must be present in source, but SIGCHI notes that these descriptions are not automatically exported into the review PDF and recommends adding/checking missing PDF accessibility metadata separately.

| Requirement | Status | Current evidence / action |
| --- | --- | --- |
| Semantic LaTeX structure | **VERIFIED** | Review builder emits sections/lists/figures using LaTeX structure rather than visual-only formatting. |
| Do not rely only on colour | **CI-GATED** | Figures use marker shape and/or line style in addition to colour. |
| Figure text descriptions in source | **CI-GATED** | All three manuscript figures receive explicit `\\Description{...}` text and includegraphics alt text. |
| Tables/equations not rasterized | **VERIFIED, maintain** | Current manuscript does not use image screenshots for analytical tables/equations. |
| PDF title/language metadata | **CI-GATED** | Review source emits document language/PDF metadata; inspect final PDF properties after compilation/remediation. |
| Fully accessible review PDF | **AUTHOR FINAL ACTION** | After generating the official `acmart` review PDF, run Acrobat Accessibility Full Check (or equivalent remediation workflow), add/fix document tags and figure alternative text if absent, set reading order/tab order as needed, and save the remediated PDF. Do this again after any regenerated final PDF. Do **not** change the mandated ACM review class solely to obtain automatic tagging. |

## 7. Research ethics and secondary data

| Requirement | Status | Current evidence / action |
| --- | --- | --- |
| Human-participant ethics context in manuscript | **VERIFIED** | Methods state that this is secondary analysis of previously released public research data, with no new recruitment/intervention/contact, and that source-study ethics/consent procedures apply. |
| Short ethics note to reviewers | **AUTHOR/PCS ACTION** | CHI explicitly asks for a short note giving the applicable ethics-review context. Enter a concise secondary-data statement in the reviewer/ethics field requested by PCS. |
| No invented ethics approval | **VERIFIED** | Manuscript does not claim a new IRB/ethics approval for this secondary analysis. |
| Source dataset ethics represented accurately | **AUTHOR FINAL CHECK** | Before submission/resubmission, verify wording against the source articles/data documentation and cite them appropriately. |

Suggested PCS ethics note (adapt to the exact field and source documentation):

> This paper reports secondary analyses of two previously released public eye-tracking research datasets. The present study involved no new participant recruitment, intervention, or participant contact and did not collect new identifiable human-subject data. Ethics approval/consent procedures for the original data collections are those reported by the respective source studies. The submitted supplement contains only derived summaries, analysis code, and prespecified decision records; it does not redistribute participant-level source archives.

## 8. Generative-AI / authorship policy

| Requirement | Status | Current evidence / action |
| --- | --- | --- |
| AI is not listed as an author | **VERIFIED** | No AI system is represented as an author. |
| Material generative-AI use disclosed | **VERIFIED in manuscript; AUTHOR FINAL CHECK** | Methods disclose OpenAI ChatGPT use in planning, implementation/test scaffolding, literature-search planning, and manuscript drafting, while assigning all scientific decisions/verification to the human authors. Keep this disclosure in the submitted version unless current ACM policy/chairs instruct a different location. |
| Authors remain accountable for generated content/citations/code | **VERIFIED by stated process; AUTHOR RESPONSIBILITY** | Methods state human verification of sources, code, and numerical claims. Authors must perform the final verification. |
| No AI-generated participant data | **VERIFIED** | The empirical datasets are public third-party source datasets; no participant data were generated or altered by AI. |

## 9. Originality, related/concurrent work, and metadata integrity

| Requirement | Status | Current evidence / action |
| --- | --- | --- |
| Original and not concurrently under review elsewhere | **AUTHOR/PCS ACTION** | Confirm for this exact manuscript. |
| Closely related concurrent submissions disclosed | **AUTHOR/PCS ACTION** | CHI requires an anonymized copy in the concurrent-submissions field for work based on the same study, artifact, or dataset or work built directly on material under review elsewhere. Audit all current submissions before PCS/resubmission. |
| All closely related prior work cited | **AUTHOR FINAL CHECK** | This is a desk-rejection issue in CHI review guidance; verify own and external prior work. |
| Final author list complete by original deadline | **AUTHOR/PCS ACTION** | CHI says authors cannot be added/removed after the paper deadline; confirm the submitted PCS author list is complete. |
| Affiliations in PCS are correct/final | **AUTHOR/PCS ACTION** | CHI states submission affiliations are final (secondary affiliation may later be added under ACM policy). |
| Title changes after deadline | **DO NOT CHANGE without chair request** | CHI states title changes are only permitted when specifically requested by the Papers Chairs. If a CHI 2027 submission already exists, keep its submitted title unless the chairs authorize a change. |

## 10. Reviewer matching and review responsibility (CHI 2027-specific)

| Requirement | Status | Current evidence / action |
| --- | --- | --- |
| Reviewer-expertise descriptors selected deliberately | **AUTHOR/PCS ACTION** | CHI 2027 uses descriptors rather than topical subcommittee selection. Use a focused set covering eye tracking/HCI evaluation, quantitative/robustness methodology, and methodology as the primary contribution; select only descriptors actually present in PCS. |
| Four review-responsibility slots filled | **AUTHOR/PCS ACTION** | CHI 2027 requires four slots assigned to qualified **authors of this paper** (one author may fill multiple slots) unless an applicable exemption was granted. Do not nominate a non-author. |
| Conflicts/profile information current | **AUTHOR/PCS ACTION** | Ensure each author's PCS/ACM profile and conflicts are current enough for reviewer matching. |

## 11. Reproducibility and transparency

| Requirement | Status | Current evidence / action |
| --- | --- | --- |
| Core analysis described sufficiently for review | **VERIFIED** | Methods contain detector parameters, eye handling, visual-angle assumptions, AOI rules, quality/cohort branches, transition semantics, GLMM formula, convergence criteria, and MCFW validation definitions. |
| Frozen alternatives established before focal result inspection | **VERIFIED** | Decision registries/plans and manuscript describe the pre-result freeze; later presentation-only operations are separated from scientific execution. |
| Failures/non-evaluable branches retained | **VERIFIED** | Manuscript reports non-evaluable rows and does not silently convert failures/missing values to zero. |
| No result-driven preferred detector | **VERIFIED** | MCFW is described as measurement generalization; detector agreement is not treated as a ground-truth ranking. |
| Compact reproducibility supplement available | **CI-GATED** | Anonymized ZIP builder packages frozen decision contracts, core mechanics, and compact outputs. |

## 12. Content safeguards specific to this paper

Maintain the following boundaries in every revision:

- Do **not** claim that multiverse analysis itself is new to eye tracking.
- Do **not** interpret 92/144 positive specifications as a 63.9% probability that the effect is positive.
- Do **not** call a viewing-distance branch a measured participant distance; 65 cm is a midpoint sensitivity assumption and 60/70 cm are source-grounded bounds.
- Do **not** say I-VT 40 or I-DT is the correct detector because they agree more closely.
- Do **not** claim MCFW replicates the SRL Prompt contrast; it supplies independent measurement-level generalization.
- Do **not** infer that two datasets establish the prevalence of measurement fragility across HCI.
- Prefer **Prompt contrast/estimate** over causal wording unless the source design and assignment mechanism explicitly justify causal interpretation.
- Keep sample-definition sensitivities (quality/cohort) distinct from strictly measurement-definition alternatives.
- Keep specification frequency separate from statistical probability and interval-level inference.

## 13. Final pre-upload gate

Immediately before a PCS submission or resubmission, an author should complete this sequence against the exact files to be uploaded:

1. Confirm the submission exists and is eligible for the current CHI 2027 stage.
2. Confirm exact title, complete frozen author list, affiliations, and PCS email/profile metadata.
3. Confirm no related concurrent submission is undisclosed.
4. Confirm official one-column anonymous ACM class and inspect the compiled PDF visually.
5. Run a fresh anonymity search on PDF text and metadata, the single supplement ZIP, and every external link.
6. Run a PDF accessibility Full Check and remediate tags, figure alt text, language/title, reading order, table headers, and tab order as needed.
7. Confirm the main paper stands alone and all central numerical claims match frozen outputs.
8. Confirm AI-use disclosure and the secondary-data ethics note are present in the required locations.
9. Upload at most one non-video supplementary ZIP with README, plus any permitted video separately.
10. Select focused CHI 2027 reviewer descriptors and fill the four review-responsibility slots with qualified paper authors (or an approved exemption).
11. Open the uploaded files from PCS and check them again; do not rely on the local copies alone.

This ledger is a compliance aid, not a substitute for the live CHI 2027 website or PCS validation.
