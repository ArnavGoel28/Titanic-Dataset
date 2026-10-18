# Titanic: Naive Bayes and Logistic Regression

A beginner college assignment: inspect and prepare the data, train two classifiers, and explain the results.

## Start here
- **`notebooks/assignment.ipynb`**: the complete assignment, explained step by step, with executed outputs.
- **`assignment.py`**: the same code as a plain Python script.
- **`reports/assignment_report.md`**: a short generated report.
- **`reports/results.csv`**: the actual model comparison.

The submission uses default GaussianNB and LogisticRegression models. There is no hyperparameter search, bootstrap analysis, deployment, or advanced framework. Earlier advanced work remains in Git history; it is not needed for this submission.

## Data
Download all files from [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data), accepting its rules if required. Place `titanic.zip` in `data/`. The code reads only its labeled `train.csv` (891 passengers). Raw files are kept out of Git. See `data/README.md` for provenance.

We reuse the already fixed split: 712 training and 179 test passengers. The saved IDs in `reports/split_manifest.json` keep identified travel groups together. Both models use these same records. Missing-value fill values, category definitions and scaling statistics come only from training passengers.

## Run on Windows
Use Python 3.12 (the tested version). In the project folder:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe assignment.py
```

The existing project `.venv` is already configured on this computer. The script regenerates the report, table and two figures. Open the notebook in VS Code or Jupyter, choose the `.venv` Python kernel and run cells from top to bottom for the same work with explanations.

Optional checks:

```powershell
.venv/Scripts/python.exe -m pytest -q
```

CI runs consistency checks without downloading account-gated data. The real-data check runs when the ZIP is present locally. GitHub Actions runs after each push; see the repository Actions tab for its current status.

## What to explain in class
1. What the columns and survival labels mean.
2. Why missing values are filled and categories encoded.
3. Why preprocessing learns only from the training set.
4. The basic assumptions of Naive Bayes and Logistic Regression.
5. Accuracy, precision, recall, F1 and the confusion matrices.

ROC-AUC and log-loss are included as short probability-based comparisons. The report does not claim that these results apply to every dataset. This test partition was evaluated in earlier development; the simpler notebook reports a default-model comparison and does not claim a new independent holdout.

## Project history
The original `docs/` guides describe a larger suggested workflow. The user's later instruction explicitly reduced the submission to a beginner assignment; see `docs/BEGINNER_SCOPE.md` and `docs/PROGRESS.md`. The unrelated user-supplied GAN notebook and original data files remain untouched and ignored.

## GitHub and milestone history
Published repository: https://github.com/ArnavGoel28/Titanic-Dataset

- Data preparation: [41b00b4](https://github.com/ArnavGoel28/Titanic-Dataset/commit/41b00b4).
- Explicit Naive Bayes training: [6e2eba5](https://github.com/ArnavGoel28/Titanic-Dataset/commit/6e2eba5).
- Separate Logistic Regression training: [97183cb](https://github.com/ArnavGoel28/Titanic-Dataset/commit/97183cb).
- Final findings: the current report and generated results in this commit.

[Full commit history](https://github.com/ArnavGoel28/Titanic-Dataset/commits/main) preserves the earlier development and simplification. The user authorized pushing each completed major step to main. See `docs/GIT_WORKFLOW.md`.
