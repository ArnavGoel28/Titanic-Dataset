# Execution Progress and Decision Log

**Initial status:** Scaffold received. Implementation began 2026-10-18. No remote pushes are authorized or claimed.

| Milestone | Status | Evidence / tests | Commit SHA | Blockers / next steps |
|---|---|---|---|---|
| M0 Repository scaffold | Complete | Python 3.12 venv; sklearn/pandas/numpy/matplotlib/pytest imports; both YAML files parse; ignore checks pass | c2073ea | Proceed to verified data loader |
| M1 Dataset selection | Complete | Exact archive CSV checksum/shape verified; pytest data tests: 4 passed; compileall passed | 1a19b67 | Freeze protocol and add training-only EDA |
| M2 EDA and data quality | Complete | Audit reproduced; two training-only figures generated and visually checked; six tests pass | 41b00b4 | Model baselines |
| M3 Preprocessing | Complete | 712 train / 179 holdout, 671 groups; row/group isolation and fold-local median tests pass | 41b00b4 | Model baselines |
| M4 Models | Complete | All three pipelines fit and return valid probabilities; 10 tests passed | Recorded next entry | Holdout metrics |
| M5 Validation/tuning | Complete | 5 shared group folds, 8 candidates; frozen GNB smoothing=.01 / LR C=.1 | Same combined milestone | First holdout evaluation |
| M6 Test evaluation | Pending | | | |
| M7 Report | Pending | | | |
| M8 Final QA | Pending | | | |

## Decision entries
Use append-only entries:

2026-10-18 — M0 in progress: initialized the intended project root on main. Python 3.12 isolated environment created; dependency installation is running. CI target aligned to 3.12. User-supplied CSV/ZIP and unrelated GAN notebook excluded from commits. The supplied archive's train.csv has the same SHA-256 as the user's existing CSV. Original textbook/handout PDFs are absent; do not invent page references.

2026-10-18 — M1 plan: document supplied archive and official source/terms, require exact train.csv hash, validate schema and construct target-free connected travel groups. Test malformed input and transitive grouping before committing. M0 commit: c2073ea.

2026-10-18 — M1 complete: `.venv/Scripts/python.exe -m pytest -q tests/test_data.py`: 4 passed. Initial test setup failed on host temp permissions, then a missing temp parent; corrected to ignored root `.pytest_tmp` and reran successfully. `compileall -q src tests` passed. Supplied archive left untouched. No models fitted or holdout results viewed.

2026-10-18 — M2/M3 plan: implement target-free group partitions before training-only EDA, preserving milestone dependency safety. Freeze protocol/config before fitting any model; generate audit and membership manifest, test no group/row overlap and fold-local imputation. Combining M2/M3 is justified because EDA must use the reserved development records. M1 commit: 1a19b67.

2026-10-18 — M2/M3 complete: `python -m pytest -q`: 6 passed; `python -m src.inspection` reproduced 891 rows, 671 connected groups (largest 8), 712 development / 179 untouched holdout. Numeric ranges/missingness audited. 127 repeated predictor vectors are distinct passenger IDs, not duplicate source rows; retain them. Fifteen zero fares retained without evidence of invalidity. Both EDA figures visually inspected; compileall/diff checks passed. No model fitting or holdout predictions yet.

2026-10-18 — M4/M5 plan: add three minimal pipeline factories and shared-fold GridSearchCV; save all candidate fold scores and selected parameters. Convergence failures are errors. Combine baseline fitting and CV in one coherent training-only implementation, avoiding unused baseline artifacts. Commit selected parameters before first holdout prediction. M2/M3 commit: 41b00b4.

2026-10-18 — M4/M5 complete: `python -m pytest -q`: 10 passed. `python -m src.experiment --stage cv` selected var_smoothing=0.01 and C=0.1; survival F1 means/sample SDs 0.7281/0.0367 (NB), 0.7300/0.0272 (LR), 0/0 (dummy). All candidate fold scores saved; no convergence errors. Compileall/diff checks pass. Selection-CV estimates may be optimistic after tuning; holdout still unevaluated.

`YYYY-MM-DD — Decision — Alternatives — Evidence/rationale — Implications`.

## Progress format
At the end of each work block: Completed; verified commands and results; changed files; commit SHA (if created); risks; next step; user inputs needed. Never mark pending work complete.
