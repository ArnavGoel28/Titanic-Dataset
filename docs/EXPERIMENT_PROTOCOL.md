# Experiment Protocol — Leakage-Safe by Design

> Historical extended protocol below. The user subsequently requested a beginner assignment. The current fixed-default notebook/script and `BEGINNER_SCOPE.md` supersede tuning, CV, bootstrap and module requirements here. The original test membership is preserved.

## Historical frozen Titanic protocol (2026-10-18, before any model fitting)
Executable values are in `config/experiment.json`. Positive label 1 means survived. Raw predictors: Age, SibSp, Parch, Fare, Pclass, Sex, Embarked. PassengerId is identity only; Name and Ticket define travel groups only; Cabin is an excluded high-cardinality identifier. Exclusions are semantic choices made before model results. No feature-selection search, clipping, family-size predictor or target-informed engineering.

Travel groups are connected components under normalized shared ticket or (surname, reported family size) with size > 1. Use StratifiedGroupKFold(5, shuffle=True, random_state=42), outer fold 0 as the approximately 20% holdout, chosen by index without evaluating candidates. Remaining records form development training. Use the same five inner group-aware folds for all models and every candidate. Group stratification is approximate; do not search seeds for attractive balance/results. If a partition lacks a class, fail explicitly.

Numeric predictors: training-fold median imputation then StandardScaler. Categorical predictors: training-fold most-frequent imputation, dense one-hot encoding with first category dropped and unknowns ignored. Same raw predictors and encoding for both algorithms; numeric scaling benefits LR regularization and makes GaussianNB's variance floor less dependent on units. One-hot indicators and counts are not Gaussian: GaussianNB is an explicitly misspecified approximation, and dropping redundant dummy columns only partially reduces dependence. Do not describe this as a mixed-likelihood NB model.

Baseline: DummyClassifier(strategy='prior'), threshold 0.5. GaussianNB var_smoothing candidates [1e-9, 1e-5, 0.01, 0.1]. LogisticRegression C [0.1, 1, 10], lbfgs solver, default L2 regularization, max_iter=3000. Select highest mean inner-CV survival F1; ties use listed order. GridSearchCV refits selected pipeline on development training. Report candidate fold scores and selected fold mean/sample SD. These are selection-CV estimates (potentially optimistic), not nested-CV performance estimates.

No resampling or class weighting: 38.4% positives is moderate imbalance. Fixed probability threshold 0.5. Primary metric survival F1; also report accuracy, precision, recall, specificity, balanced accuracy, ROC-AUC, AP, log-loss and Brier score. AP is average precision, not trapezoidal PR area. Undefined precision/recall use zero_division=0. No probability calibration or threshold tuning.

Freeze parameters before one holdout evaluation. Compute 1,000 paired bootstrap replicates by resampling whole holdout travel groups with replacement, seed 2026; 95% percentile intervals for F1/accuracy/ROC-AUC/log-loss and LR-minus-NB differences. Skip single-class replicates for ROC-AUC and report accepted count. These conditional, approximate intervals do not include training/model-selection variability or prove population superiority. Reproduction runs repeat this same fixed experiment; never revise choices based on holdout results.

## Pre-register the approach
Record dataset, target, positive class, split strategy, validation folds, primary metric and candidate tuning ranges BEFORE viewing test results. Do not retroactively edit the rationale to match favourable results.

## Order of operations
1. Inspect metadata and rows; verify target definition, units, invalid values, leakage and dependent records. Descriptive EDA on full raw data may be used to understand schema/target balance, but **don't use full-data statistical patterns to choose transforms/features or fit estimators**.
2. Remove only proven target leakage and invalid identifiers. Decide handling of groups/time/duplicates before partitioning.
3. Create untouched test partition (default stratified 80/20, `random_state=42` for independent binary records). For grouped/temporal data use group-aware or chronological split instead; explain why.
4. On train only, fit imputation, encoders, scalers, clipping thresholds, feature selectors and resamplers via `Pipeline`/`ColumnTransformer`. Numerical median imputation and `StandardScaler` are reasonable LR defaults; GaussianNB preprocessing must be justified separately. Use different preprocessing pipelines when needed, while retaining same raw feature columns/records.
5. Compare a `DummyClassifier` baseline and both requested models on identical folds (default stratified 5-fold CV; reduce folds when minority count is insufficient). If groups/time exist, use corresponding CV.
6. Set tuning grid **small** and explicit; e.g. GaussianNB `var_smoothing`, LogisticRegression `C`. Restrict model selection and any threshold tuning to training CV. Track solver settings, convergence, and seed.
7. Freeze selected choices; fit on full training partition; perform **one** final test evaluation. Never iteratively adapt based on the test outcomes.
8. Explain assumptions and limitations, especially GaussianNB's conditional independence and per-class Gaussian feature assumptions, and LR's linear log-odds and regularization.

## Class imbalance / sampling
- Report class frequencies, majority-class baseline and positive label.
- Do not equate stratified splitting with synthetic resampling; use class weighting/oversampling/SMOTE only if justified.
- If resampling, place sampler INSIDE training-fold pipeline (`imblearn.pipeline.Pipeline`) and ensure validation/test remain unchanged.
- Class weights apply differently to LR and GaussianNB; do not force identically named parameters when not equivalent.

## Rules for evaluating probabilities
- Use `predict_proba` scores corresponding to the chosen positive label for ROC-AUC, log-loss, Brier and PR analysis; no probabilities from binary hard labels.
- Use fixed threshold 0.5 as main default; only tune decision threshold on validation data if motivated, and label threshold-specific results separately.
- Note that test confidence intervals based on limited sample sizes may be wide; do not claim significant superiority from tiny metric differences.

## Reproducibility
- Pin dependency versions only after actual tested installation.
- Store script/config with seed, dataset source, split logic and run commands.
- Avoid committing large original data if terms forbid redistribution; provide secure reproduction instructions instead.
- Test no NaN/inf reaches estimators, no target in features, no fit occurs on holdout, no accidental row overlap, no unbounded manual copy-paste results.
