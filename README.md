# Naive Bayes vs Logistic Regression — Deep Learning Assignment

A reproducible empirical comparison of generative **Gaussian Naive Bayes** and discriminative **Logistic Regression** on a documented binary-classification dataset.

> **Status:** Scaffolding only. No dataset, trained model, or experimental result is claimed yet.

## Assignment objective
Select a legitimate dataset; inspect, clean, sample/split and preprocess appropriately; train and compare Naive Bayes with Logistic Regression; validate on unseen data; analyze performance and limitations using defensible metrics and plots.

## Study context
Based on the supplied Bishop *Deep Learning* Chapters 1–5 and teacher handouts *Generative and Discriminative Classifiers* and *Transforming Densities, Measuring Information*. See `docs/SOURCE_ALIGNMENT.md`.

## For Codex
Start with [`AGENTS.md`](AGENTS.md) and [`docs/CODEX_START_PROMPT.md`](docs/CODEX_START_PROMPT.md). Work milestone by milestone. Do not invent observations, experimental findings, or missing dataset metadata.

## Expected artifacts (to be added during development)
- `notebooks/assignment.ipynb` — clear, executable narrative notebook
- `src/` — lightweight reusable data/model/evaluation functions as necessary
- `tests/` — leakage, data, metric and smoke tests
- `reports/` — concise final report and regenerated figures/tables
- `data/README.md` — dataset source, license, download steps and hash if appropriate
- `requirements.txt` — pinned versions used for the final run

## Reproducibility
Exact environment setup, dataset download, run commands, results and citation details will be filled in **after verification**, never guessed. The repository should run on a standard CPU laptop.

## Project guide
See the specification, experiment protocol, metrics plan, milestones, Git/CI instructions and progress tracker in [`docs/`](docs/).
