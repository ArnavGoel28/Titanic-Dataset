# Evaluation and Comparative Analysis

## Before experiment
Record the chosen positive class, class counts, primary selection metric, and loss trade-off. Suggested default: **stratified CV F1** for balanced emphasis on precision/recall; change if dataset stakes/imbalance justify ROC-AUC or average precision. Do not choose after seeing test results.

## Core metrics (test set)
- Confusion matrix with clear rows/columns and TP/FP/TN/FN.
- Accuracy = (TP+TN)/N.
- Precision = TP/(TP+FP), recall = TP/(TP+FN), specificity = TN/(TN+FP).
- F1 = 2 PR/(P+R); specify undefined-metric convention (`zero_division`).
- ROC-AUC from continuous positive-class probability.
- Log-loss (binary cross-entropy) from probabilities.
- Optionally balanced accuracy, average precision (AP; distinguish from trapezoidal PR-AUC), Brier score, calibration curve, ROC/PR curves and inference/training time.
- Cross-validation: mean and standard deviation of chosen metrics across the *training* folds, for both algorithms and dummy baseline.

## Interpretation questions
1. Which model beats a reasonable baseline? By which metrics and by how much?
2. Does ranking quality (ROC-AUC/AP) agree with the threshold-based F1/accuracy? Explain discrepancies.
3. Does one model produce confidently wrong probabilities? Compare log-loss/calibration and show examples if ethical/appropriate.
4. Are there highly correlated features **within classes**, making naive conditional independence questionable? Use appropriate training-only exploratory evidence. Correlation does not alone prove causal explanation for a performance gap.
5. Do LR coefficient signs and magnitudes remain interpretable after scaling/encoding? Note collinearity and conditioning.
6. Are results stable across folds? Consider dataset size, imbalance, and uncertainty.
7. Are differences practically material? Do not claim statistical significance without a justified test/interval.
8. What would you change if given more data or time?

## Output artifact checklist
- Dataset overview (features, labels, split sizes).
- Data-quality before/after summary; exact pipeline transformations.
- Fold-level model comparison table and held-out test comparison table.
- Two clearly labelled confusion matrices, one overlay ROC plot and optional PR/calibration plots.
- Table comparing assumptions, tuning, complexity, failure modes.
- Plain-language conclusion supported by measurements, not blanket claims.

## Reporting honesty
Do not present cross-validation and holdout metrics interchangeably. Do not report training accuracy as final accuracy. If tuning is omitted, note it; if a metric is undefined or unavailable, say why rather than inventing it.
