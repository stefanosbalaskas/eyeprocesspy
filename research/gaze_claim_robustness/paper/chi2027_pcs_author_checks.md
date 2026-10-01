# CHI 2027 PCS author-only completeness checklist

This checklist complements `chi2027_submission_compliance.md`. It records CHI 2027 requirements that **cannot be verified from the repository or manuscript build** and must be checked by the authors in PCS/ACM systems against the live submission.

Official CHI 2027 sources checked on 2026-10-01 include the Papers page, the reviewer-identification guidance, the anonymization policy, and the publication-format guidance. When this checklist conflicts with live PCS or an official CHI notice, the live official requirement wins.

## Submission-stage eligibility

- [ ] Confirm that this paper already has a valid CHI 2027 Papers submission in PCS. The initial Papers deadline was **2026-09-10 AoE**, so a new initial CHI 2027 Papers submission cannot be created on 2026-10-01.
- [ ] Confirm the exact current stage shown in PCS and follow only the actions permitted for that stage.

## Frozen author and title metadata

- [ ] Confirm the complete author list matches the list frozen at the original CHI 2027 deadline. Do not add or remove authors after the deadline unless the Papers Chairs explicitly authorize it.
- [ ] Confirm every author's submission affiliation is correct under the CHI/ACM affiliation policy.
- [ ] Confirm the paper title exactly matches the title in PCS. CHI 2027 states that post-deadline title changes are only permitted when specifically requested by the Papers Chairs.
- [ ] Confirm contact email and other required PCS profile fields are current.

## ORCID and DBLP completeness

- [ ] **Every author has an ORCID recorded in the required profile/submission metadata.**
- [ ] **Every author has the DBLP profile field completed with their DBLP URL, or `n/a` when no DBLP profile exists.**
- [ ] Run/inspect the CHI/PCS submission-completeness check after updating author profiles; do not assume profile edits have propagated until PCS accepts the submission as complete.

## Conflicts and reviewer matching

- [ ] Each author's conflict information is current and complete enough for reviewer matching.
- [ ] Select a focused set of reviewer-expertise descriptors that actually exist in PCS and accurately cover the paper's content (eye tracking/HCI evaluation, quantitative or robustness methodology, and methodology as the primary contribution where available).
- [ ] Avoid adding broad descriptors merely to increase reviewer coverage; choose descriptors the authors can defend from the manuscript.

## Review responsibility

- [ ] Fill all **four** CHI 2027 review-responsibility slots with qualified **authors of this paper**, unless the submission has an approved exemption.
- [ ] If one author fills more than one slot, verify that PCS permits and records that assignment correctly.
- [ ] Do not nominate a non-author solely to satisfy the review-responsibility requirement.

## Ethics / secondary-data note

- [ ] Enter the short ethics/reviewer-context note requested by PCS, accurately describing this work as secondary analysis of previously released public research datasets with no new recruitment, intervention, participant contact, or new identifiable participant-data collection.
- [ ] Verify the wording against the source-study ethics/consent documentation; do not invent a new ethics approval for this secondary analysis.

Suggested wording, subject to the exact PCS field and source documentation:

> This paper reports secondary analyses of two previously released public eye-tracking research datasets. The present study involved no new participant recruitment, intervention, or participant contact and did not collect new identifiable human-subject data. Ethics approval and consent procedures for the original data collections are those reported by the respective source studies. The review supplement contains only derived summaries, analysis code, and prespecified decision records; it does not redistribute participant-level source archives.

## Originality and related/concurrent submissions

- [ ] Confirm that this exact manuscript is not concurrently under review at another venue in a way prohibited by CHI/ACM policy.
- [ ] Audit all closely related current submissions, software/artifact papers, and manuscripts using the same study, dataset, or directly overlapping contribution.
- [ ] Where CHI requires disclosure, upload the requested anonymized copy in the concurrent-submissions field rather than relying on a prose assurance alone.
- [ ] Confirm all closely related published/prior work is cited normally and in the third person; do not replace self-citations with `Anonymous`.

## Generative-AI responsibility

- [ ] Keep the Methods disclosure describing material ChatGPT use in methodological planning and software/test implementation unless the live ACM/CHI policy or chairs instruct a different location.
- [ ] Human authors perform the final verification of citations, code, numerical claims, methodological choices, and submitted files.
- [ ] Do not list an AI system as an author.

## Exact upload files

- [ ] Upload the official anonymous single-column ACM review PDF built from `\\documentclass[manuscript,review,anonymous]{acmart}`.
- [ ] Upload no more than one non-video supplementary ZIP, with a README, plus any separately permitted video if applicable.
- [ ] Ensure every author-controlled external link visible during review is anonymous. Public third-party dataset/publisher links may remain when they identify the source rather than the submitting authors.
- [ ] Run an accessibility Full Check on the **exact PDF that will be uploaded** and remediate document tags, figure alternative text, title/language metadata, reading order, table headers, and tab order as necessary.
- [ ] Re-run anonymity checks on PDF text/metadata and inside the exact supplementary ZIP.
- [ ] Open the files from PCS after upload and inspect the uploaded copies rather than relying only on local files.

## Final completeness sign-off

Do not treat the repository's green CI state as proof that the PCS submission is complete. Final CHI 2027 readiness requires both:

1. the repository/artifact gates in `chi2027_submission_compliance.md`; and
2. every applicable author/PCS checkbox above to be completed against the live submission.
