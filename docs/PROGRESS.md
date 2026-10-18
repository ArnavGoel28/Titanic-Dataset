# Project progress

## Verified model results
Fixed split: 712 training records / 179 test records. Default GaussianNB accuracy: 0.7877; default LogisticRegression accuracy: 0.8045. Both identify 49 of the 68 test survivors. Generated results are in `reports/results.csv` and `reports/summary.json`.

## Published milestones
- Data preparation: `41b00b4`.
- Explicit Naive Bayes training: `6e2eba5`.
- Separate Logistic Regression training: `97183cb`.
- Detailed model findings: `f448cdc`.

## Current milestone: results and project overview
Plan: present setup and usage before the visual results in README; synchronize descriptive code/notebook/report paths; retain dataset provenance and evaluation limitations. Preserve measured results and existing Git history. Run the notebook, consistency tests, text/link checks and Git diff checks before publishing.

Previous verification: 25 notebook cells executed, 3 local tests passed, and GitHub Actions passed for f448cdc. Current verification: 25 notebook cells executed with zero errors; 3 tests passed in 1.90s; syntax and Git diff checks passed. All 18 public text files and notebook outputs were reviewed, and local Markdown links resolve. Metrics CSV, prediction summary, split membership and missing-value counts exactly match the previous published version. Source CSV checksum remains unchanged. README presents setup, two generated figures, metric tables and error analysis. The current milestone will be published with its GitHub Actions check.
