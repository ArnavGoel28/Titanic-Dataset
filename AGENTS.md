# Project guide

This repository compares Gaussian Naive Bayes and Logistic Regression on the Titanic survival dataset. Keep the implementation straightforward and the published findings reproducible.

- Read `docs/DATASET.md` and `docs/METHODOLOGY.md` before changing the analysis.
- Fit imputation, category definitions and scaling only on training records. Reuse the fixed train/test membership for both models.
- Preserve the raw input files. Do not fabricate measurements, citations, test outcomes or commit status. Preserve the stated limitations of the evaluation.
- Keep `titanic.py` and `notebooks/titanic.ipynb` synchronized, including executed notebook outputs and generated report values.
- Verify relevant changes with `python -m pytest -q` and a notebook run when its code changes. Review `git diff --check` and the intended file list.
- Update `docs/PROGRESS.md` with actual verification evidence. Stage explicit paths and make meaningful commits.
- Completed milestones are authorized to be pushed to `origin/main` at https://github.com/ArnavGoel28/Titanic-Dataset. Preserve published history; do not force-push or modify unrelated remote settings.
- Keep raw data, credentials, caches and `.local/` notes out of commits. Leave unrelated local files untouched.
