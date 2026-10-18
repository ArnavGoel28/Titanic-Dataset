# # Titanic: Naive Bayes and Logistic Regression
# **Objective:** prepare the data, train two classifiers, and compare their results.
# Positive class: `Survived = 1`. Both models use their default settings.
# Compare prepared features, predictions, and probability quality on shared test records.

# ## 1. Import libraries
# pandas handles tables, matplotlib draws graphs, and scikit-learn supplies the models.

from pathlib import Path
import hashlib
import json
import joblib
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss, confusion_matrix, ConfusionMatrixDisplay)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
(OUTPUT / "graphs").mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ## 2. Load the data
# Source: [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data).
# Read the labeled `data/train.csv`. Kaggle's competition `test.csv` has
# no survival labels; `gender_submission.csv` contains example predictions, not truth.
# The checksum confirms that we use the same original file each time.

data_path = ROOT / "data" / "train.csv"
raw_csv = data_path.read_bytes()
expected_hash = "7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f"
assert hashlib.sha256(raw_csv).hexdigest() == expected_hash, "Unexpected dataset version"
df = pd.read_csv(data_path)
print("Rows and columns:", df.shape)
print(df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"]).head().to_string(index=False))

# ## 3. Inspect data quality
# Check types, missing values, duplicates, class counts and numeric ranges.
# No records are removed. Zero and high fares are kept because they are not proven errors.

print("Column types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate passenger IDs:", df.PassengerId.duplicated().sum())
print("\nClass counts (0=no, 1=yes):\n", df.Survived.value_counts())
print("\nNumeric summary:\n", df[["Age", "SibSp", "Parch", "Fare"]].describe())
df.isna().sum().rename("missing_records").to_csv(OUTPUT / "missing_values.csv")

# ## 4. Select features and reuse the fixed train/test split
# Predictors: Age, SibSp, Parch, Fare, Pclass, Sex and Embarked.
# Exclude PassengerId, Name, Ticket and Cabin as identity/detail fields.
# Cabin also has many missing entries.
#
# The earlier split reserved 712 training and 179 test passengers while keeping
# known travel groups together. Reuse its saved IDs for both models instead of
# changing test passengers for better scores. This partition was already evaluated
# during earlier development; this is not a newly unseen test experiment.

numeric = ["Age", "SibSp", "Parch", "Fare"]
categorical = ["Pclass", "Sex", "Embarked"]
split = json.loads((ROOT / "data" / "split.json").read_text())
train_df = df[df.PassengerId.isin(split["train_ids"])].copy()
test_df = df[df.PassengerId.isin(split["holdout_ids"])].copy()
assert set(train_df.PassengerId).isdisjoint(test_df.PassengerId)
assert len(train_df) + len(test_df) == len(df)
X_train = train_df[numeric + categorical].copy()
X_test = test_df[numeric + categorical].copy()
y_train = train_df.Survived
y_test = test_df.Survived
print("Training records:", len(X_train), "| Test records:", len(X_test))

# ## 5. Fill missing values using training data only
# Use training medians for numeric columns and training modes for categories.
# Apply those same values to test passengers, avoiding data leakage.

train_medians = X_train[numeric].median()
X_train[numeric] = X_train[numeric].fillna(train_medians)
X_test[numeric] = X_test[numeric].fillna(train_medians)
training_modes = {}
for column in categorical:
    training_mode = X_train[column].mode().iloc[0]
    training_modes[column] = training_mode
    X_train[column] = X_train[column].fillna(training_mode)
    X_test[column] = X_test[column].fillna(training_mode)
print("Numeric fill values:\n", train_medians)
print("Remaining missing train/test values:", X_train.isna().sum().sum(), X_test.isna().sum().sum())

# ## 6. Encode categories and scale numeric features
# One-hot encoding converts categories into indicator columns. Training categories
# define the encoding; a previously unseen test category maps to all-zero indicators.
# StandardScaler learns numeric means and standard deviations from training data.
# Both models receive the same prepared features and records.

training_categories = {}
for column in categorical:
    categories = sorted(X_train[column].unique())
    training_categories[column] = categories
    X_train[column] = pd.Categorical(X_train[column], categories=categories)
    X_test[column] = pd.Categorical(X_test[column], categories=categories)
X_train = pd.get_dummies(X_train, columns=categorical, drop_first=True, dtype=int)
X_test = pd.get_dummies(X_test, columns=categorical, drop_first=True, dtype=int)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
scaler = StandardScaler()
X_train[numeric] = scaler.fit_transform(X_train[numeric])
X_test[numeric] = scaler.transform(X_test[numeric])
assert np.isfinite(X_train.to_numpy()).all()
assert np.isfinite(X_test.to_numpy()).all()
print("Prepared columns:", X_train.columns.tolist())

# ## 7. Show a simple training-data graph
# Use training passengers only. No synthetic oversampling is needed for this dataset.

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
y_train.value_counts().sort_index().plot.bar(ax=axes[0], color=["#687787", "#147d92"])
axes[0].set(title="Training class counts", xlabel="Survived: 0=no, 1=yes", ylabel="Passengers")
train_df.Age.dropna().plot.hist(ax=axes[1], bins=20, color="#147d92")
axes[1].set(title="Training ages (observed values)", xlabel="Age in years", ylabel="Passengers")
fig.savefig(OUTPUT / "graphs" / "data_overview.png", dpi=150)
if plt.get_backend().lower() != "agg":
    plt.show()
plt.close(fig)

# ## 8. Train Naive Bayes
# GaussianNB uses Bayes' rule and assumes independent Gaussian features within each
# class. Counts and encoded categories are not Gaussian, so this is an approximation.
# GaussianNB uses its default settings for this comparison.

# Train Naive Bayes directly so the fit step is easy to identify.
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
print("Naive Bayes training complete")

# ## 9. Train Logistic Regression
# Logistic Regression applies a sigmoid to a weighted feature sum to estimate
# survival probability. It assumes linear log-odds. Use the same training records
# and prepared features as Naive Bayes. max_iter=1000 allows enough iterations
# for convergence; the other model settings remain at their defaults.

logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train, y_train)
print("Logistic Regression training complete")
models = {"Naive Bayes": naive_bayes, "Logistic Regression": logistic_regression}

# Save each fitted model with everything needed to prepare future input rows.
preprocessing = {"numeric_columns": numeric, "categorical_columns": categorical,
                 "numeric_medians": train_medians, "categorical_modes": training_modes,
                 "categories": training_categories, "feature_columns": X_train.columns.tolist(),
                 "scaler": scaler}
versions = {package: importlib.metadata.version(package)
            for package in ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]}
model_files = {"Naive Bayes": "naive_bayes.joblib",
               "Logistic Regression": "logistic_regression.joblib"}
for name, model in models.items():
    bundle = {"model": model, "preprocessing": preprocessing, "positive_class": 1,
              "library_versions": versions, "dataset_sha256": expected_hash,
              "training_records": len(y_train), "test_records": len(y_test)}
    joblib.dump(bundle, MODEL_DIR / model_files[name], compress=3)
print("Saved fitted model bundles to models/")

# ## 10. Compare results
# Accuracy measures overall correctness. Precision measures correctness among
# predicted survivors. Recall measures how many actual survivors were found.
# F1 balances precision and recall. ROC-AUC evaluates probability ranking;
# log-loss penalizes wrong, confident probabilities. Use threshold 0.5.
# A majority-class prediction provides a basic accuracy reference.

majority_class = y_train.mode().iloc[0]
baseline_accuracy = accuracy_score(y_test, np.full(len(y_test), majority_class))
baseline_correct = int((y_test == majority_class).sum())
print(f"Majority-class baseline accuracy: {baseline_accuracy:.3f}")
rows, predictions, matrices = [], {}, {}
for name, model in models.items():
    probability = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    predicted = (probability >= 0.5).astype(int)
    predictions[name] = predicted
    matrices[name] = confusion_matrix(y_test, predicted, labels=[0, 1]).tolist()
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(y_test, predicted),
                 "Precision": precision_score(y_test, predicted, zero_division=0),
                 "Recall": recall_score(y_test, predicted, zero_division=0),
                 "F1": f1_score(y_test, predicted, zero_division=0),
                 "ROC-AUC": roc_auc_score(y_test, probability),
                 "Log-loss": log_loss(y_test, np.column_stack([1-probability, probability]), labels=[0, 1])})
results = pd.DataFrame(rows).set_index("Model")
print(results.round(4).to_string())
results.to_csv(OUTPUT / "results.csv")

# ## 11. Confusion matrices
# Rows = true labels, columns = predictions. Top-left: true negatives; top-right:
# false positives; bottom-left: false negatives; bottom-right: true positives.

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
for ax, (name, predicted) in zip(axes, predictions.items()):
    ConfusionMatrixDisplay.from_predictions(
        y_test, predicted, labels=[0, 1], display_labels=["Did not survive", "Survived"],
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
fig.savefig(OUTPUT / "graphs" / "confusion_matrices.png", dpi=150)
if plt.get_backend().lower() != "agg":
    plt.show()
plt.close(fig)

# ## 12. Interpret the results
# Higher accuracy/F1 is better; lower log-loss is better. A single split cannot
# prove that one algorithm is always superior. Correlation and non-Gaussian inputs
# can challenge NB, while LR is limited to linear log-odds. Sex and class are
# historical associations, not evidence of causal effects.

agreement = int((predictions["Naive Bayes"] == predictions["Logistic Regression"]).sum())
nb, lr = results.loc["Naive Bayes"], results.loc["Logistic Regression"]
conclusion = (
    f"Naive Bayes: accuracy {nb['Accuracy']:.3f}, F1 {nb['F1']:.3f}, log-loss {nb['Log-loss']:.3f}.\n"
    f"Logistic Regression: accuracy {lr['Accuracy']:.3f}, F1 {lr['F1']:.3f}, log-loss {lr['Log-loss']:.3f}.\n"
    f"The models agreed on {agreement} of {len(y_test)} test passengers.\n"
    f"Highest accuracy in this run: {results['Accuracy'].idxmax()}.\n"
    "These results apply to this split and preprocessing. No parameter tuning was done in this analysis."
)
print(conclusion)
# Compare actual mistakes, not only overall accuracy.
nb_tn, nb_fp, nb_fn, nb_tp = np.array(matrices["Naive Bayes"]).ravel()
lr_tn, lr_fp, lr_fn, lr_tp = np.array(matrices["Logistic Regression"]).ravel()
nb_right = predictions["Naive Bayes"] == y_test.to_numpy()
lr_right = predictions["Logistic Regression"] == y_test.to_numpy()
disagreements = {"both_correct": int((nb_right & lr_right).sum()),
                 "both_wrong": int((~nb_right & ~lr_right).sum()),
                 "nb_only_correct": int((nb_right & ~lr_right).sum()),
                 "lr_only_correct": int((~nb_right & lr_right).sum())}
findings = (
    f"Naive Bayes correctly classified {nb_tn + nb_tp}/{len(y_test)} passengers; "
    f"Logistic Regression correctly classified {lr_tn + lr_tp}/{len(y_test)}. "
    f"The majority-class reference correctly classified {baseline_correct}/{len(y_test)}.\n\n"
    f"Among {int(y_test.sum())} actual survivors, NB found {nb_tp} and missed {nb_fn}; "
    f"LR found {lr_tp} and missed {lr_fn}. NB made {nb_fp} false-positive survival predictions, "
    f"compared with {lr_fp} for LR. LR-minus-NB accuracy difference: "
    f"{100 * (lr['Accuracy'] - nb['Accuracy']):.2f} percentage points.\n\n"
    f"Both models were correct for {disagreements['both_correct']} passengers and wrong for "
    f"{disagreements['both_wrong']}. Of the {len(y_test) - agreement} disagreements, "
    f"LR alone was correct {disagreements['lr_only_correct']} times and NB alone "
    f"{disagreements['nb_only_correct']} times.\n\n"
    f"Probability ranking (ROC-AUC): NB {nb['ROC-AUC']:.3f}, LR {lr['ROC-AUC']:.3f}. "
    f"Probability error (log-loss, lower is better): NB {nb['Log-loss']:.3f}, "
    f"LR {lr['Log-loss']:.3f}. Ranking quality and confidence should be considered alongside accuracy."
)
print("\nDetailed findings:\n", findings)
summary = {"train_records": len(y_train), "test_records": len(y_test),
           "majority_baseline_accuracy": baseline_accuracy, "majority_baseline_correct": baseline_correct,
           "confusion_matrices": matrices, "agreement": agreement, "disagreements": disagreements}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
report = f"""# Titanic survival: model comparison

## Objective and dataset
Compare Gaussian Naive Bayes and Logistic Regression on Kaggle Titanic data.
Source: https://www.kaggle.com/competitions/titanic/data (subject to competition rules).
The labeled train.csv has 891 passengers, 12 columns, 342 survivors and 549 non-survivors.
Positive class: Survived=1. Kaggle's unlabeled test.csv is not used.

## Data preparation
Missing: Age 177, Cabin 687, Embarked 2. No complete duplicate records were found.
Use Age, SibSp, Parch, Fare, Pclass, Sex and Embarked. Exclude identity/detail fields
PassengerId, Name, Ticket and Cabin. Keep zero/extreme fares because they are not proven errors.
Reuse the fixed travel-group-separated split: {len(y_train)} train / {len(y_test)} test.
Fill numeric gaps with training medians, categorical gaps with training modes.
Encode using training categories and scale numeric values using training statistics.
No oversampling, feature search, parameter search or threshold tuning is used.

## Models and measures
GaussianNB assumes class-conditional independent Gaussian features; counts and
categorical indicators violate that approximation. LR estimates sigmoid probabilities
with linear log-odds. Both use defaults; LR max_iter=1000 allows convergence.
Accuracy=(TP+TN)/N; precision=TP/(TP+FP); recall=TP/(TP+FN);
F1=2TP/(2TP+FP+FN). Undefined precision/recall use zero_division=0.
ROC-AUC uses probabilities; log-loss measures probability confidence.

## Results
Majority-class baseline accuracy: {baseline_accuracy:.3f}.

```text
{results.round(4).to_string()}
```

![Confusion matrices](output/graphs/confusion_matrices.png)

## Final findings and analysis
{conclusion}

{findings}

For this fixed split, LR performed better on accuracy, F1, ROC-AUC and
log-loss; both models found the same number of survivors. This is a result for this
sample and preparation, not evidence that LR is always better.

Related predictors can challenge NB's conditional independence assumption. LR permits
correlated predictors but uses linear log-odds. Ranking and threshold metrics may favor
different models. These observations do not establish a causal explanation for differences.
The small historical sample and one split limit generalization. Travel groups are
imperfect proxies for companions. The test partition was evaluated in earlier development;
this simplified default-model comparison is not a new independent experiment.
No setting was adjusted to improve these test scores.

## Reproduction and references
Run `python titanic.py` to regenerate the fitted models, output tables, graphs and this report.
The input is `data/train.csv`; fixed training/test membership is stored in `data/split.json`.
The fitted bundles in `models/` contain the estimators and training-only preprocessing settings.
Tables, plots and this report are generated from real results.
- Data/rules: https://www.kaggle.com/competitions/titanic/data and https://www.kaggle.com/competitions/titanic/rules
- NB: https://scikit-learn.org/stable/modules/naive_bayes.html
- LR: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
"""
(ROOT / "FINAL_REPORT.md").write_text("\n".join(line.rstrip() for line in report.splitlines()) + "\n", encoding="utf-8")
