# Dataset and provenance

Source: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data), Kaggle. Data access is subject to [competition rules](https://www.kaggle.com/competitions/titanic/rules). Download independently after accepting the applicable terms; no blanket redistribution license is assumed.

## Input version
The project reads only `train.csv` directly from `data/titanic.zip`.

- 891 passengers and 12 columns; 7 fields are used as predictors.
- Target: `Survived`, where 1 means survived and 0 means did not survive.
- Class counts: 342 survived, 549 did not survive.
- CSV size: 61,194 bytes.
- SHA-256: `7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f`.
- Local bytes verified on 2026-10-18. The checksum fixes the input version; it is not a publisher signature.

The archive also contains `test.csv` (418 unlabeled passengers) and `gender_submission.csv` (example predictions). Neither is used as evaluation truth or in model preparation.

## Features and quality
Numeric predictors: Age (years), SibSp (sibling/spouse count), Parch (parent/child count), Fare (recorded fare units).
Categorical predictors: Pclass, Sex, Embarked. See Kaggle's data dictionary for the detailed family-count definitions.

PassengerId is used only to identify split membership. Name and Ticket were used only to define travel groups. PassengerId, Name, Ticket and Cabin are excluded from the model predictors.

Missing values: Age 177, Cabin 687, Embarked 2; other fields 0. No exact duplicate rows or duplicate passenger IDs were found. Fifteen zero fares and plausible extreme fares are retained because their presence alone does not establish an error. Repeated feature combinations belonging to distinct passengers are retained.

Sex and passenger class describe historical associations. This project does not infer causal effects or recommend contemporary allocation decisions.

## Acquisition
Sign in to Kaggle, accept terms if prompted, choose Download All, and place the ZIP at `data/titanic.zip`. Raw CSV/ZIP files are excluded from Git. No credentials are required by the analysis after the data is downloaded.
