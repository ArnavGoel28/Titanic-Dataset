import numpy as np
import pytest

from src.evaluation import bootstrap, group_bootstrap_indices, metrics, positive_probability


def test_metric_formulas_and_label_direction():
    p = np.array([.1, .6, .4, .9])
    y = np.array([0, 0, 1, 1])
    result = metrics(y, p)
    assert [result[k] for k in ["tn", "fp", "fn", "tp"]] == [1, 1, 1, 1]
    for metric in ["accuracy", "precision", "recall", "specificity", "f1"]:
        assert result[metric] == .5
    assert result["roc_auc"] == .75
    assert result["log_loss"] == pytest.approx(-np.log([.9, .4, .4, .9]).mean())
    assert metrics(y, np.full(4, .2))["precision"] == 0
    with pytest.raises(ValueError):
        metrics(y, np.array([.1, np.nan, .2, .3]))


def test_probability_column_uses_positive_label():
    class Reverse:
        classes_ = np.array([1, 0])
        def predict_proba(self, x):
            return np.array([[.8, .2], [.3, .7]])
    np.testing.assert_array_equal(positive_probability(Reverse(), None), [.8, .3])


def test_bootstrap_preserves_groups_and_is_reproducible():
    groups = np.array([1, 1, 2, 3, 3, 3])
    rng = np.random.default_rng(9)
    for _ in range(20):
        idx = group_bootstrap_indices(groups, rng)
        multiplicities = np.bincount(idx, minlength=len(groups))
        assert multiplicities[0] == multiplicities[1]
        assert multiplicities[3] == multiplicities[4] == multiplicities[5]
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([.1, .7, .2, .8, .6, .3])
    config = {"bootstrap_seed": 2026, "bootstrap_repeats": 10, "threshold": .5}
    probabilities = {"Dummy": np.full(6, .4), "GaussianNB": p, "LogisticRegression": p}
    result = bootstrap(y, probabilities, groups, config)
    assert result == bootstrap(y, probabilities, groups, config)
    assert result["lr_minus_nb"]["f1"]["low"] == 0
    assert result["lr_minus_nb"]["f1"]["high"] == 0
