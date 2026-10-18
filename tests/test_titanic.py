"""Small consistency checks; the analysis is in the notebook/script."""
import ast
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest


def test_notebook_code_matches_script():
    notebook = json.loads(Path("notebooks/titanic.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert ast.dump(ast.parse(code)) == ast.dump(ast.parse(Path("titanic.py").read_text()))


def test_report_numbers_match_confusion_matrices():
    results = pd.read_csv("reports/results.csv", index_col="Model")
    summary = json.loads(Path("reports/summary.json").read_text())
    for name, matrix in summary["confusion_matrices"].items():
        (tn, fp), (fn, tp) = matrix
        assert tn + fp + fn + tp == summary["test_records"]
        assert results.loc[name, "Accuracy"] == pytest.approx((tn + tp) / summary["test_records"])
        assert results.loc[name, "F1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert summary["majority_baseline_accuracy"] == pytest.approx(summary["majority_baseline_correct"] / summary["test_records"])
    comparisons = summary["disagreements"]
    assert comparisons["both_correct"] + comparisons["both_wrong"] == summary["agreement"]
    assert sum(comparisons.values()) == summary["test_records"]
    nb_matrix = summary["confusion_matrices"]["Naive Bayes"]
    lr_matrix = summary["confusion_matrices"]["Logistic Regression"]
    assert comparisons["both_correct"] + comparisons["nb_only_correct"] == nb_matrix[0][0] + nb_matrix[1][1]
    assert comparisons["both_correct"] + comparisons["lr_only_correct"] == lr_matrix[0][0] + lr_matrix[1][1]


@pytest.mark.skipif(not Path("data/titanic.zip").exists(), reason="Download Kaggle data locally")
def test_real_run_keeps_preparation_training_only():
    import matplotlib
    matplotlib.use("Agg")
    run = runpy.run_path("titanic.py")
    train, test = run["train_df"], run["test_df"]
    assert set(train.PassengerId).isdisjoint(test.PassengerId)
    np.testing.assert_allclose(run["train_medians"], train[run["numeric"]].median())
    filled_train = train[run["numeric"]].fillna(run["train_medians"])
    np.testing.assert_allclose(run["scaler"].mean_, filled_train.mean())
    assert run["models"]["Naive Bayes"].class_count_.sum() == len(train)
    assert run["X_train"].columns.tolist() == run["X_test"].columns.tolist()
    assert run["models"]["Naive Bayes"].var_smoothing == 1e-9
