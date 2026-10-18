# Current scope: beginner assignment

2026-10-18: The user explicitly clarified that the goal is a beginner college assignment, not production code or maximum accuracy. This supersedes the earlier expanded implementation plan where they conflict.

## Submission
One explained notebook and matching plain script: data loading, quality checks, feature selection, fixed train/test records, missing-value preparation, category encoding, numeric scaling, GaussianNB and LogisticRegression, metrics, two confusion matrices, a short conclusion/report.

No parameter search, cross-validation machinery, bootstrap analysis, model registry, deployment, neural network or separate modeling package is needed. Default model parameters are fixed by this scope decision, not chosen from new test results. LogisticRegression(max_iter=1000) only allows convergence.

## Preserved scientific basics
Same partitions for both models; no target in predictors; preprocessing fit only on training records; no use of Kaggle's unlabeled test file or example predictions as labels; results computed from real runs. Original raw bytes are preserved. The existing group-separated partition is reused to avoid searching for a better split.

## Evaluation history
Earlier work selected parameters with CV and evaluated the 179-passenger holdout. Those artifacts and modules were subsequently removed from the current tree to simplify the project; the commits preserve an honest development history. The current notebook uses defaults rather than those selected parameters. Its test set is not described as a newly untouched sample, and scores must not be tuned further. No claim of statistically significant superiority is made.

## Git and original files
Keep the supplied AGENTS.md and original project guides as historical instructions; the user's newer scope controls the final deliverable. Keep the unrelated GAN notebook and raw datasets unchanged and uncommitted. Local milestone commits remain authorized. The user subsequently authorized pushing every completed major step to the specified GitHub repository on main; the existing history and separate NB/LR steps have been published. No history rewriting or unrelated remote settings change is authorized.
