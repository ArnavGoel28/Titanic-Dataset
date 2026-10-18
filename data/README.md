# Dataset Acquisition and Provenance

Use Kaggle's [Titanic competition data](https://www.kaggle.com/competitions/titanic/data), subject to [competition rules](https://www.kaggle.com/competitions/titanic/rules). Sign in, accept terms if required, download all files, and save as `data/titanic.zip`. Do not add account credentials to the project.

The experiment reads only `train.csv` directly from the ZIP; it never extracts arbitrary archive paths. Expected: 891 rows, 12 columns, 61,194 bytes, SHA-256 `7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f`. Verified 2026-10-18 from user-supplied bytes. The earlier `Titanic-Dataset.csv` is byte-identical.

`test.csv` has no survival labels. `gender_submission.csv` is a sample prediction file, never evaluation truth. Raw CSV/ZIP files are intentionally ignored by Git; no redistribution license is assumed. See `docs/DATASET_DECISION.md` for full provenance and quality assessment.

The beginner notebook/script reads `train.csv` directly using Python zipfile and pandas. No src package is needed. `reports/split_manifest.json` contains only the previously fixed train/test passenger IDs.
