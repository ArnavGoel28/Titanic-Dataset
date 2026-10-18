# Experiment Protocol — Leakage-Safe by Design

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
