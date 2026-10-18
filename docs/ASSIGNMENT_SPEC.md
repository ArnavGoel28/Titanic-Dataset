# Assignment Specification

## Teacher's wording (user-provided paraphrase)
> Take any dataset from anywhere; perform the necessary preprocessing, data cleaning, sampling, etc.; apply Naive Bayes and Logistic Regression; analyze and validate results using appropriate measures.

This is the assignment requirement; the interpretations below are **project recommendations**, not additional confirmed teacher requirements.

## Core deliverables (MUST)
- One lawful, traceable dataset; clearly defined features, prediction target and positive class.
- Auditable schema, missingness, duplicate, invalid-value, outlier, class-balance and leakage assessment. Apply corrections only if evidence justifies them; document when not needed.
- Data splitting/sampling rationale and reproducible seed. Handle severe imbalance if justified, without leaking test data.
- Model-appropriate preprocessing; identical train/test records for both models; Gaussian Naive Bayes and Logistic Regression trained, compared and explained.
- Holdout evaluation and meaningful metrics (confusion matrix, accuracy, precision, recall, F1, ROC-AUC and log-loss, subject to class/score availability). Strongly prefer stratified CV on training data.
- Comparisons with discussion of model assumptions, errors, limitations and possible reasons for performance differences.
- Runnable source/notebook, final report, installation/use guide, tests and meaningful Git history.

## Nice-to-have (ONLY when justified)
- Accuracy baseline (DummyClassifier), balanced accuracy, average precision, Brier score/calibration, ROC/PR plots, sensitivity to feature correlations, confidence intervals or repeated CV, small parameter search.
- Categorical encoders, winsorization, custom feature engineering, feature reduction, resampling or SMOTE ONLY when the dataset warrants them.

## Non-goals
- Fabricating dirty data to show artificial cleaning; unsupported claims that one model is universally better.
- Large hyperparameter sweeps, deep learning architectures, explainability frameworks, dashboards, deployment, extra classifiers, live APIs, heavyweight MLOps, unrequested cloud services.

## Unknowns to confirm if necessary
- Submission deadline; notebook vs scripts vs report required; allowed collaboration; preferred dataset/domain; any prescribed split or metrics; whether an online GitHub repo should be created/pushed. If absent, proceed with local preparation, note assumptions and ask before external actions.

## Definition of done
All mandatory outputs exist, clean-run succeeds, tests pass, results are generated (not typed from memory), and the report's figures/tables agree with the executable output.
