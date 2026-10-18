# Initial Prompt to Send to Codex

You are implementing my Deep Learning college assignment in this repository. Read `AGENTS.md` and every file referenced by its execution sequence before editing. Treat these documents as your implementation contract.

The task: choose and document a legitimate binary classification dataset, perform only justified cleaning/EDA/sampling/preprocessing, implement Gaussian Naive Bayes and Logistic Regression in leakage-safe sklearn pipelines, compare on common partitions via appropriate stratified or group/time-aware validation, evaluate an untouched test set with suitable classification and probability metrics, and produce a concise evidence-based report and reproducible code/notebook.

**Git requirement:** work in coherent tested milestones (M0–M8). Before each change inspect Git state; after each completed milestone verify tests, inspect the staged diff, and make a meaningful Conventional Commit. Keep `docs/PROGRESS.md` current; do not invent SHAs or test results. Do not push to GitHub, create remote repositories, change permissions, or alter remote branch settings without first asking me.

First, assess the existing project without overwriting unrelated work. Tell me (briefly) any essential decisions you need from me, especially dataset choice if material tradeoffs remain or external GitHub access before pushing. If not blocked, complete the next milestone and commit it after verifying. Continue through unblocked milestones efficiently, using simple well-tested code and preventing leakage. Don't prematurely produce simulated results. At the end of each work session, summarize exactly what changed, tests run (actual outcomes), latest commit SHA, next milestone, and any required user action.
