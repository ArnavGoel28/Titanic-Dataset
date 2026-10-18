# Report Writing Guide

## Preferred report format
Approximately 5–8 focused pages unless teacher provides a fixed limit. A longer report is not automatically better. Produce a PDF from a source format once completed; do not create generic unsupported prose.

## Sections
1. Title / objective / model comparison question.
2. Dataset: official source, citation/license, shape, label semantics, selection rationale.
3. Data quality and EDA: missingness, duplicates, invalid values, outliers, class proportions, meaningful plots.
4. Preprocessing/sampling: exact transforms, train/test split, CV scheme, leakage protections, class-imbalance rationale.
5. Models: NB generative formulation and conditional independence; Gaussian densities; LR sigmoid and log-loss; model hyperparameters and settings.
6. Validation: cross-validation on training, baseline, held-out evaluation, positive-class choice and exact metric definitions.
7. Results: clear numerical comparison tables, confusion matrices, ROC/PR plots where meaningful, optional calibration.
8. Interpretation: performance trade-offs, disagreements, correlated features, failure modes, data-size limitations; no causal claims unsupported by tests.
9. Conclusion and reproducibility: honest takeaway, limitations, versioned commands, citations.

## Requirements
- Generate result tables and plots from code; check final report values against stored outputs after clean rerun.
- Label plot axes, class direction, sample sizes, units and uncertainty when applicable.
- State which operations were *not* necessary and why (e.g. no missing values or no resampling).
- No copying textbook paragraphs, invented citations, fabricated data or unsupported statements that one approach is always superior.
- State when only one holdout partition was used; acknowledge limited external validity.
