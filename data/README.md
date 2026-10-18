# Titanic input data

Download the archive from [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data), accepting the [competition rules](https://www.kaggle.com/competitions/titanic/rules) if prompted. Save it as `data/titanic.zip`.

The analysis reads only its labeled `train.csv`: 891 passengers, 12 columns, 61,194 bytes. Expected SHA-256: `7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f`.

`test.csv` is unlabeled; `gender_submission.csv` contains example predictions. Neither supplies ground truth for this comparison. Raw files are ignored by Git and remain unchanged. See [dataset notes](../docs/DATASET.md) for provenance and quality details.
