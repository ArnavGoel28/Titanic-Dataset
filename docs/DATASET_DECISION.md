# Dataset Selection and Provenance

**Status: selected and inspected on 2026-10-18.** Current beginner implementation: `assignment.py` / `notebooks/assignment.ipynb`; see `BEGINNER_SCOPE.md`.

## Verified selection
- Dataset: Titanic - Machine Learning from Disaster, Kaggle competition, user-supplied archive `data/titanic.zip`.
- Source: https://www.kaggle.com/competitions/titanic/data ; terms: https://www.kaggle.com/competitions/titanic/rules . Metadata lists **Subject to Competition Rules**, not a blanket open-source data license. Raw files remain ignored; download independently after accepting Kaggle terms. Archive provenance is user supplied, not authenticated by a publisher signature.
- Selected bytes: archive member `train.csv`, 61,194 bytes; SHA-256 `7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f`. This exactly matches the earlier CSV from https://www.kaggle.com/datasets/yasserh/titanic-dataset . Its mirror license need not be inferred.
- 891 records, 12 columns: 11 non-target fields, of which 7 will be predictors. Target `Survived`: 1 = survived (positive), 0 = did not survive. Counts 342 / 549 (38.38% positive).
- Numeric predictors: Age (years), SibSp, Parch, Fare (recorded fare units). Categorical: Pclass, Sex, Embarked. See Kaggle data dictionary for family-count definitions.
- Exclude PassengerId, Name, Ticket, Cabin as raw identifiers/high-cardinality fields; retain Name/Ticket solely for target-free travel-group construction. No post-outcome fields are predictors. Sex and class are sensitive/social attributes; results describe this historical sample, not causal effects or contemporary allocation policy.
- Missing Age 177, Cabin 687, Embarked 2; other columns 0. No exact duplicate rows or duplicate PassengerId. Zero fares exist; no evidence establishes them as erroneous, so do not replace automatically. Plausible extreme fares/ages stay in the data.
- 134 shared-ticket groups cover 344 rows before normalization. Dependence can span tickets: use connected components of normalized shared ticket OR (surname, reported family size) when family size > 1. This imperfect proxy can miss companions or overmerge unrelated names; report that limitation.
- Split and CV must keep these components together. No arbitrary record deduplication or synthetic resampling.
- Archive `test.csv`: 418 unlabeled records; `gender_submission.csv`: 418 example predictions, not ground truth. Neither is used for evaluation, fitting, imputation, encoding, tuning, or EDA.
- User explicitly supplied the Titanic files and identified the official Kaggle source. Choose Titanic over clean numeric alternatives because it supports meaningful missingness, categorical handling and dependence analysis within the requested scope.
- Retrieval: sign in to Kaggle, accept competition rules if prompted, Download All, save archive as `data/titanic.zip`. Optional Kaggle CLI: `kaggle competitions download -c titanic -p data` (CLI/account setup is optional and outside the experiment dependencies).
- Loader: `assignment.py` reads archive member train.csv with zipfile/pandas and asserts its exact SHA-256. No account access or download is needed at experiment runtime once the user supplies the file.

## Selection criteria
1. Binary classification with explicit semantic positive class and ethically appropriate usage.
2. Manageable on CPU; ideally several hundred to a few thousand independent records.
3. Numeric features or feasible categorical handling compatible with GaussianNB and LogisticRegression.
4. Published source/attribution/license or access terms and stable retrieval steps.
5. Genuine analytical value: class overlap, feature correlations, missing/invalid values, imbalance, etc. (not all are mandatory).
6. No direct leakage feature, impossible-to-obtain-at-inference variable, duplicated sample leakage, or unjustified post-outcome information.

## Candidates to verify
| Candidate | Why consider | Caution |
|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | Clean numeric, binary labels, GaussianNB vs Logistic Regression assumptions | Limited genuine cleaning; avoid any medical conclusions |
| Titanic survival | Missing values, mixed feature types, interpretable preprocessing | Mixed categorical features may require modelling decisions |
| Pima Indians Diabetes | Numeric, placeholder-zero analysis can matter | Check values, origin, restrictions and fairness of interpretation |
| Banknote authentication | Numeric, accessible classroom demo | Possibly little cleaning effort |
| Bank marketing | Realistic, potential imbalance and categoricals | Potentially more complex/time consuming |

## Original decision checklist (covered by verified selection above)
- Dataset name and exact version:
- Official source URL, download path, accessed date:
- License/usage permission:
- Citation or attribution:
- Number of rows / features:
- Target name, positive label and explanation:
- Feature types / possible predictors unavailable at inference:
- Missing / placeholder / duplicate / invalid records:
- Class counts and ratio:
- Independence, group or temporal structure affecting split:
- Risks / sensitive attributes / limits:
- Exact retrieval command, checksum if stored:
- Reason for choosing over alternatives:
- Approved by user? (if a preference or ambiguous tradeoff needs approval):

Do not automatically remove duplicates: legitimate repeated feature rows with different labels can reflect aleatoric uncertainty. Investigate row-level identity and timing.
