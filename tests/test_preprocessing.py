import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.data import FEATURES, load_data
from src.inspection import quality_summary
from src.preprocessing import partitions, preprocessor, read_config


def test_fit_statistics_exclude_holdout_and_unknown_categories():
    fit = pd.DataFrame({"Age": [10., np.nan, 30.], "SibSp": [0, 1, 0], "Parch": [0, 0, 1],
                        "Fare": [5., 10., 15.], "Pclass": [1, 2, 3],
                        "Sex": ["male", "female", "male"], "Embarked": ["S", "C", "S"]})
    holdout = fit.iloc[:1].assign(Age=9999., Embarked="UNSEEN")
    prep = preprocessor().fit(fit)
    imputer = prep.named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[0] == 20.
    before = imputer.statistics_.copy()
    with pytest.warns(UserWarning, match="unknown categories"):
        transformed = prep.transform(holdout)
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(before, imputer.statistics_)
    encoder = prep.named_transformers_["categorical"].named_steps["encode"]
    assert "UNSEEN" not in encoder.categories_[2]


def test_real_partitions_and_fold_local_statistics():
    from pathlib import Path
    if not Path("data/titanic.zip").exists():
        pytest.skip("Kaggle data not distributed")
    data = load_data()
    train, test, folds, groups = partitions(data, read_config())
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(range(len(data)))
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
    assert data.PassengerId.iloc[train].isin(data.PassengerId.iloc[test]).sum() == 0
    repeated = partitions(data, read_config())
    np.testing.assert_array_equal(train, repeated[0])
    validated = []
    for fit, valid in folds:
        assert set(groups.iloc[train[fit]]).isdisjoint(groups.iloc[train[valid]])
        assert set(train[fit]).isdisjoint(test)
        assert set(train[valid]).isdisjoint(test)
        prep = clone(preprocessor()).fit(data.iloc[train[fit]][FEATURES])
        expected = data.iloc[train[fit]][["Age", "SibSp", "Parch", "Fare"]].median().to_numpy()
        actual = prep.named_transformers_["numeric"].named_steps["impute"].statistics_
        np.testing.assert_allclose(actual, expected)
        assert np.isfinite(prep.transform(data.iloc[train[valid]][FEATURES])).all()
        validated.extend(valid)
    assert sorted(validated) == list(range(len(train)))
    assert quality_summary(data)["duplicate_ids"] == 0
