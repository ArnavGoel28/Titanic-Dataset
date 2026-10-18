# Execution Progress and Decision Log

**Current scope:** Beginner notebook, matching script, short generated report and basic checks. The later user clarification replaces the extended M7/M8 plan. See `BEGINNER_SCOPE.md`.

**Initial status:** Scaffold received. Implementation began 2026-10-18. No remote pushes are authorized or claimed.

| Milestone | Status | Evidence / tests | Commit SHA | Blockers / next steps |
|---|---|---|---|---|
| M0 Repository scaffold | Complete | Python 3.12 venv; sklearn/pandas/numpy/matplotlib/pytest imports; both YAML files parse; ignore checks pass | c2073ea | Proceed to verified data loader |
| M1 Dataset selection | Complete | Exact archive CSV checksum/shape verified; pytest data tests: 4 passed; compileall passed | 1a19b67 | Freeze protocol and add training-only EDA |
| M2 EDA and data quality | Complete | Audit reproduced; two training-only figures generated and visually checked; six tests pass | 41b00b4 | Model baselines |
| M3 Preprocessing | Complete | 712 train / 179 holdout, 671 groups; row/group isolation and fold-local median tests pass | 41b00b4 | Model baselines |
| M4 Models | Complete | All three pipelines fit and return valid probabilities; 10 tests passed | 99580c8 | Holdout metrics |
| M5 Validation/tuning | Complete | 5 shared group folds, 8 candidates; frozen GNB smoothing=.01 / LR C=.1 | 99580c8 | First holdout evaluation |
| M6 Test evaluation | Historical complete | First advanced evaluation completed; later retired for beginner scope | 1fc782e | Simplified final submission |
| M7 Beginner notebook/report | Complete | 23-cell notebook executed with 2 figures and zero errors; short report regenerated from real results | Final scope-change commit (see Git log) | Final basic QA |
| M8 Final QA | Complete for revised beginner scope | 3 tests pass; compileall and pip check pass; notebook/script ASTs match; plots visually checked | Final scope-change commit (see Git log) | No remote CI or push claimed |

## Decision entries
Use append-only entries:

2026-10-18 — M0 in progress: initialized the intended project root on main. Python 3.12 isolated environment created; dependency installation is running. CI target aligned to 3.12. User-supplied CSV/ZIP and unrelated GAN notebook excluded from commits. The supplied archive's train.csv has the same SHA-256 as the user's existing CSV. Original textbook/handout PDFs are absent; do not invent page references.

2026-10-18 — M1 plan: document supplied archive and official source/terms, require exact train.csv hash, validate schema and construct target-free connected travel groups. Test malformed input and transitive grouping before committing. M0 commit: c2073ea.

2026-10-18 — M1 complete: `.venv/Scripts/python.exe -m pytest -q tests/test_data.py`: 4 passed. Initial test setup failed on host temp permissions, then a missing temp parent; corrected to ignored root `.pytest_tmp` and reran successfully. `compileall -q src tests` passed. Supplied archive left untouched. No models fitted or holdout results viewed.

2026-10-18 — M2/M3 plan: implement target-free group partitions before training-only EDA, preserving milestone dependency safety. Freeze protocol/config before fitting any model; generate audit and membership manifest, test no group/row overlap and fold-local imputation. Combining M2/M3 is justified because EDA must use the reserved development records. M1 commit: 1a19b67.

2026-10-18 — M2/M3 complete: `python -m pytest -q`: 6 passed; `python -m src.inspection` reproduced 891 rows, 671 connected groups (largest 8), 712 development / 179 untouched holdout. Numeric ranges/missingness audited. 127 repeated predictor vectors are distinct passenger IDs, not duplicate source rows; retain them. Fifteen zero fares retained without evidence of invalidity. Both EDA figures visually inspected; compileall/diff checks passed. No model fitting or holdout predictions yet.

2026-10-18 — M4/M5 plan: add three minimal pipeline factories and shared-fold GridSearchCV; save all candidate fold scores and selected parameters. Convergence failures are errors. Combine baseline fitting and CV in one coherent training-only implementation, avoiding unused baseline artifacts. Commit selected parameters before first holdout prediction. M2/M3 commit: 41b00b4.

2026-10-18 — M4/M5 complete: `python -m pytest -q`: 10 passed. `python -m src.experiment --stage cv` selected var_smoothing=0.01 and C=0.1; survival F1 means/sample SDs 0.7281/0.0367 (NB), 0.7300/0.0272 (LR), 0/0 (dummy). All candidate fold scores saved; no convergence errors. Compileall/diff checks pass. Selection-CV estimates may be optimistic after tuning; holdout still unevaluated.

2026-10-18 — M6 plan: independently test confusion/metric formulas, class-probability column selection and group-bootstrap integrity; then evaluate frozen pipelines on holdout once. Generate aggregate comparisons, coefficients and diagnostic plots without changing model decisions. M4/M5 commit: 99580c8.

2026-10-18 — M6 first evaluation: `python -m pytest -q`: 13 passed (single-class bootstrap diagnostic warnings prompted a reporting-only guard). `python -m src.experiment --stage full`: 179 holdout passengers, 68 survived, 133 travel groups. NB accuracy/F1/AUC/log-loss: .7989/.7313/.8404/.6388; LR .7933/.7176/.8592/.4554; dummy accuracy .6201, F1 0. Models agree on 172/179; NB alone correct 4, LR alone correct 3. Paired F1 difference interval includes zero. Both holdout figures visually inspected. Model selection and protocol unchanged; later reruns are fixed-protocol reproduction only. Optimized bootstrap to compute only requested interval metrics; dummy PR curve displayed as prevalence baseline rather than interpolated endpoints.

`YYYY-MM-DD — Decision — Alternatives — Evidence/rationale — Implications`.

## Progress format
At the end of each work block: Completed; verified commands and results; changed files; commit SHA (if created); risks; next step; user inputs needed. Never mark pending work complete.

2026-10-18 — User scope correction: beginner assignment only. Plan: replace advanced src modules, CV/bootstrap outputs and extended configuration with one sequential script and matching notebook. Reuse fixed train/test IDs; defaults chosen by requested simplicity, not test performance. Keep raw files, original guides and unrelated GAN notebook unchanged. A short Markdown report replaces the planned six-page PDF. Preserve earlier work in Git history.

2026-10-18 — Revised M7/M8 complete: advanced modules/search/bootstrap artifacts removed from current tree, preserved in prior commits. The user-supplied files and original AGENTS.md remain untouched. Added assignment.py, executed 23-cell notebook, direct tested dependency pins, concise README, current-scope doc, 3 small tests, generated report/CSV/missingness/summary and 2 figures. NotebookClient executed all cells with the .venv interpreter: 2 displayed figures, 0 error outputs. `python -m pytest -q`: 3 passed in 1.82s; `compileall -q assignment.py tests`: passed; `pip check`: no broken requirements; `git diff --check`: passed. Default NB accuracy/F1/log-loss: .7877/.7206/.6546; default LR .8045/.7368/.4342. Both use 712 train / 179 test; majority baseline .6201. Models agree on 168/179. No tuning in simplified run. Both new figures visually checked. Kaggle raw files ignored, original GAN notebook ignored, no credentials or model files added. Remote CI not available without a push; no push or remote configuration change made.

2026-10-18 — Scope-change commit e279bfd. A staged diff check then exposed trailing spaces in pandas-rendered report text; PowerShell continued past that check. Corrected the report generator to trim line endings and synchronized notebook code. Rerun the three tests and staged diff check with explicit failure guards before the correction commit. No model/preprocessing choices changed.
