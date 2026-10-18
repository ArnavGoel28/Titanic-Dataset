# Titanic Survival: Naive Bayes vs Logistic Regression

[![Titanic checks](https://github.com/ArnavGoel28/Titanic-Dataset/actions/workflows/ci.yml/badge.svg)](https://github.com/ArnavGoel28/Titanic-Dataset/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)

A comparison of two classic classifiers on Titanic passenger data, from missing-value preparation to predictions, probability quality, and error analysis. The code and notebook reproduce the published tables and figures.

## Get started

### 1. Clone the repository

```bash
git clone https://github.com/ArnavGoel28/Titanic-Dataset.git
cd Titanic-Dataset
```

### 2. Install the dependencies

Use Python 3.12, the tested version.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 3. Download the data

Open [Kaggle Titanic data](https://www.kaggle.com/competitions/titanic/data), sign in and accept the competition rules if prompted. Choose **Download All** and save the archive as **`data/titanic.zip`**. Keep its contents unchanged.

Only the labeled `train.csv` is used. Kaggle's `test.csv` is unlabeled, and `gender_submission.csv` contains example predictions. Raw data is excluded from Git; see [dataset provenance](docs/DATASET.md).

### 4. Run the analysis

On Windows:

```powershell
.venv/Scripts/python.exe titanic.py
```

On macOS / Linux:

```bash
.venv/bin/python titanic.py
```

For a guided walkthrough, open [the executed notebook](notebooks/titanic.ipynb) in VS Code or an existing Jupyter environment, select the `.venv` Python kernel, and run the cells in order. You can also browse its saved outputs directly on GitHub.

Running the script regenerates the [comparison report](reports/model_comparison.md), [metrics CSV](reports/results.csv), [prediction summary](reports/summary.json), and figures below.

## Explore the data

The dataset contains **891 passengers**, including **342 survivors**. Survival is the positive class (`Survived = 1`). The two models share **712 training records**, **179 test records**, and the same seven predictors.

| Feature type | Predictors | Preparation |
| --- | --- | --- |
| Numeric | Age, SibSp, Parch, Fare | Training-median fill, then numeric standardization |
| Categorical | Pclass, Sex, Embarked | Training-mode fill and one-hot encoding |

Age has 177 missing entries, Cabin 687, and Embarked 2. PassengerId, Name, Ticket and Cabin are excluded from the predictors. No exact duplicate passenger records were found. Zero and extreme fares are retained because they are not proven errors.

![Training class counts and observed age distribution](reports/figures/data_overview.png)

*The plots show training passengers only; the age histogram uses observed values. Missing-value fill values, category definitions and scaling statistics are learned exclusively from training records.*

## Model results

Both classifiers use default model settings, with `max_iter=1000` allowing Logistic Regression to converge. They use survival probability **0.5** as the prediction threshold. Precision, recall and F1 below refer to survivors.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Log-loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes | 78.77% | 0.7206 | 0.7206 | 0.7206 | 0.8409 | 0.6546 |
| Logistic Regression | 80.45% | 0.7538 | 0.7206 | 0.7368 | 0.8650 | 0.4342 |

The training-majority reference predicts no survival and scores **62.01% accuracy** (111/179 correct). Higher accuracy, F1 and ROC-AUC are better; lower log-loss is better.

### Where the models make mistakes

![Naive Bayes and Logistic Regression test confusion matrices](reports/figures/confusion_matrices.png)

*Rows are true labels and columns are predictions. The bottom row contains actual survivors.*

| Test outcome | Naive Bayes | Logistic Regression |
| --- | ---: | ---: |
| Correct predictions | 141 / 179 | 144 / 179 |
| Survivors correctly identified | 49 / 68 | 49 / 68 |
| Survivors missed | 19 | 19 |
| Non-survivors predicted to survive | 19 | 16 |

### What the results show

**Logistic Regression performs slightly better on this split.** It classifies three additional passengers correctly, a **1.68 percentage-point** accuracy difference. Both models identify 49 of the 68 survivors; the difference comes from Logistic Regression making fewer false-positive survival predictions (16 versus 19).

**The probability metrics add another perspective.** Logistic Regression has higher ROC-AUC (0.865 versus 0.841) and lower log-loss (0.434 versus 0.655). Its probability rankings and probability error are better in this run, alongside the small accuracy gain.

**Agreement includes shared errors.** The models agree on 168 of 179 passengers: both are correct for 137 and wrong for 31. Among the 11 disagreements, Logistic Regression alone is correct 7 times and Naive Bayes alone 4 times.

## Method and limitations

The [saved split](reports/split_manifest.json) keeps identified ticket/family travel groups together. Its membership was fixed before preprocessing and is reused for both models. See [methodology](docs/METHODOLOGY.md) for the grouping rule, preparation steps and metric definitions.

Gaussian Naive Bayes assumes conditionally independent Gaussian features; counts and one-hot indicators violate that approximation. Logistic Regression models linear log-odds. These assumptions help interpret the comparison, but this run does not isolate the cause of the performance difference.

Results describe a small historical sample and one fixed partition, which was inspected during earlier development. They are not an independent external evaluation or evidence that one classifier is universally superior. No settings are adjusted to improve these test scores, and historical associations are not causal conclusions.

## Project files

| Path | Purpose |
| --- | --- |
| [`titanic.py`](titanic.py) | Run the complete analysis as a sequential Python script |
| [`notebooks/titanic.ipynb`](notebooks/titanic.ipynb) | Walk through the same code with explanations and executed outputs |
| [`reports/model_comparison.md`](reports/model_comparison.md) | Detailed findings and model comparison |
| [`reports/results.csv`](reports/results.csv) | Machine-readable evaluation metrics |
| [`docs/DATASET.md`](docs/DATASET.md) | Source, checksum, feature definitions and data-quality notes |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Split design, preparation and evaluation details |
| [`tests/test_titanic.py`](tests/test_titanic.py) | Notebook synchronization, result consistency and training-only preparation checks |

## Checks and reproducibility

Run `.venv/Scripts/python.exe -m pytest -q` on Windows, or `.venv/bin/python -m pytest -q` on macOS / Linux.

[GitHub Actions](https://github.com/ArnavGoel28/Titanic-Dataset/actions) checks syntax and notebook/result consistency on each push. The real-data test runs locally when the Kaggle ZIP is present; CI skips that test because raw data is not redistributed. Tested dependency versions are pinned in [`requirements.txt`](requirements.txt).

## Sources

- [Kaggle Titanic data and dictionary](https://www.kaggle.com/competitions/titanic/data) and [competition rules](https://www.kaggle.com/competitions/titanic/rules)
- [scikit-learn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [scikit-learn: Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
