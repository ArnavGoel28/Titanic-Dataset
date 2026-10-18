# Codex Operating Instructions

This file is the binding project guide for Codex. Read this file and the linked specification before making changes.

## Mission
Implement a complete, reproducible, scientifically defensible comparison of **Gaussian Naive Bayes** and **Logistic Regression** on ONE documented, real-world binary classification dataset. Requirements originate from the teacher's assignment and supplied Bishop chapters plus the two supplementary notes. See `docs/ASSIGNMENT_SPEC.md` and `docs/SOURCE_ALIGNMENT.md`.

## Rules
1. Never fabricate data, measurements, experiment results, plots, citations, file contents, test outcomes, or commit status.
2. Make a decision on the dataset only after examining its source, features, license/access, class distribution, and data-quality issues. See `docs/DATASET_DECISION.md`. If multiple candidates remain comparably suitable, provide a brief evidence-based recommendation and ask the user to choose. Do not stall on routine implementation choices.
3. Preserve a genuine, untouched holdout test set. Split before learning imputers, scalers, encoders, feature selectors, samplers, or other data-dependent transformations. Fit those **inside training-only pipelines / cross-validation folds**. No resampling before splitting. Check related/duplicate records for leakage.
4. Provide a clear, minimal baseline and train both requested models on the **same underlying train/test partitions**, using model-appropriate preprocessing. No neural network or extra classifier is needed unless explicitly approved.
5. Report both validation and final test results honestly, with appropriate metrics and uncertainty/limitations. Choose the positive class explicitly. Do not tune on final test performance.
6. Prefer straightforward Python, pandas, numpy, matplotlib, scikit-learn and pytest. Install imbalanced-learn only if resampling is justified; do not add heavyweight infrastructure unnecessarily. Use local CPU.
7. Do NOT modify, delete or commit any unrelated user changes. Inspect `git status` before and after each milestone; stage explicit paths (`git add path...`), never blindly stage all files.
8. Follow `docs/GIT_WORKFLOW.md`: complete a major milestone, run its checks, update status/decision docs, create one meaningful commit, and record the SHA. No empty, cosmetic, speculative, or failing milestone commits.
9. Do not assume remote GitHub access, credentials, or permission to push. **Local commits are authorized by this project workflow; remote creation, pushes, PRs, or changes to remote settings require the user's explicit authorization**. Never amend or rewrite published history without user approval.
10. Document setup blockers and external inputs precisely. Continue unblocked work. Ask for only required user choices or account access; never request secrets in chat, files, or Git history.
11. Keep notebook output reproducible and consistent with report tables. Generated files must be regenerable; avoid committing large data, credentials, caches, models, or incidental artifacts. Respect dataset licensing.
12. Update `docs/PROGRESS.md` after each milestone with evidence (commands/results), unresolved issues, and next action. If blocked, do not claim the milestone is complete.

## Execution sequence
Read in order: `docs/ASSIGNMENT_SPEC.md`, `docs/DATASET_DECISION.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/EVALUATION_PLAN.md`, `docs/MILESTONES.md`, `docs/GIT_WORKFLOW.md`, `docs/REPORT_GUIDE.md`, `docs/SOURCE_ALIGNMENT.md`, `docs/PROGRESS.md`.

Before each implementation change: write a short current-milestone plan, check repository status, and inspect relevant code. Work through M0–M8 in order unless there is a concrete dependency reason to vary it. Keep complexity proportional to a college DL assignment.

## Final acceptance
A fresh environment can install pinned dependencies and run the documented workflow. Data provenance is clear; preprocessing and validation are leakage-safe; metrics and charts are generated from real results; a concise analysis discusses when NB and LR agree/disagree and why; tests and GitHub Actions succeed (if CI is available); major completed milestones are committed; and the submission report does not overclaim.
