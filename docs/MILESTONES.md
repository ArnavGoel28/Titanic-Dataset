# Milestones, Checks and Commit Gates

Each milestone = coherent set of changes + verification + one logical Git commit (if changed). On failure, fix before committing. An internal code checkpoint is NOT necessarily a Git commit until validated.

| ID | Deliverable | Acceptance check | Suggested Conventional Commit |
|---|---|---|---|
| M0 | README, AGENTS, docs, CI, ignore rules, environment skeleton | Docs consistent; YAML parses; secrets excluded; initial smoke command | `chore: initialize assignment repository` |
| M1 | Dataset source/version/label/access verified | Dataset decision doc completed, license noted, loader smoke tested | `docs: record dataset selection and provenance` |
| M2 | Inspection & training-safe EDA functions/notebook | Reproduced shapes, missingness, class counts and visualizations; tests | `feat(data): add dataset inspection and EDA` |
| M3 | Split + preprocessing pipeline | No test fit or split overlap; conditional handling documented; tests | `feat(data): implement leakage-safe preprocessing` |
| M4 | Dummy, GaussianNB, LogisticRegression baselines | Both fit/predict; fixed seed, positive class set, same partitions | `feat(models): train naive bayes and logistic regression` |
| M5 | CV and small tuning, if warranted | Only training CV used; fold scores recorded; no test peeking | `feat(eval): add cross-validation and model tuning` |
| M6 | Holdout metrics, figures, comparative discussion | Metric formulas/label directions validated; figures regenerated | `feat(eval): add model comparison and diagnostics` |
| M7 | Concise verified report and final README | Report mirrors computed outputs and cites dataset/methods | `docs: add experimental report and reproducibility guide` |
| M8 | Clean-run, environment pinning, final QA | `pytest`, notebook execute, CI-local equivalent, no secrets, clean status | `chore: finalize reproducible assignment submission` |

## Gates applicable to EVERY milestone
- Review `git diff --check`, `git diff`, `git status --short`; verify current branch and intended file list.
- Run appropriate tests/CLI or notebook smoke checks; don't invent successful commands.
- Ensure no accidental data, secrets, tokens, API keys, PII, huge files or notebook execution noise.
- Update `docs/PROGRESS.md` with actual evidence and limitations.
- Stage ONLY relevant paths, then commit with concise imperative subject.
- Record resulting SHA in progress tracker **after** commit; recording it can be done in the next milestone or a separate bookkeeping update, to avoid an infinite commit loop.
- If blocked, mark pending with exact next step and do not make a success-labelled commit.

## Iteration policy
A meaningful later correction may have its own `fix(...)` or `refactor(...)` commit. Do not force exactly nine commits if actual work warrants more/fewer. No fake filler commits, no resetting user work, no force pushes.
