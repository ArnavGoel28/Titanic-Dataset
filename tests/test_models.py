import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.data import FEATURES
from src.models import model_pipeline
from src.preprocessing import read_config
from src.validation import select_models


@pytest.fixture
def synthetic():
    # Synthetic data is for software tests only; never used for assignment results.
    rng = np.random.default_rng(7)
    n = 60
    return pd.DataFrame({"PassengerId": np.arange(n), "Age": rng.uniform(5, 70, n),
                         "SibSp": rng.integers(0, 3, n), "Parch": rng.integers(0, 3, n),
                         "Fare": rng.uniform(1, 90, n), "Pclass": rng.integers(1, 4, n),
                         "Sex": np.tile(["female", "male"], n // 2),
                         "Embarked": np.tile(["S", "C", "Q"], n // 3),
                         "Survived": np.tile([1, 0, 1, 0, 0, 1], n // 6)})


@pytest.mark.parametrize("name", ["Dummy", "GaussianNB", "LogisticRegression"])
def test_model_probabilities_are_finite_and_ordered(name, synthetic):
    model = model_pipeline(name, read_config()).fit(synthetic[FEATURES], synthetic.Survived)
    p = model.predict_proba(synthetic[FEATURES])
    assert model.classes_.tolist() == [0, 1]
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)


def test_cv_records_candidates_and_guards_frozen_selection(synthetic, tmp_path):
    cfg = read_config()
    folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(synthetic, synthetic.Survived))
    fitted, selected = select_models(synthetic, folds, cfg, tmp_path)
    assert set(fitted) == {"Dummy", "GaussianNB", "LogisticRegression"}
    assert pd.read_csv(tmp_path / "cv_folds.csv").shape[0] == 9
    assert pd.read_csv(tmp_path / "cv_candidates.csv").shape[0] == (1 + 4 + 3) * 3 * 6
    cfg["seed"] = 99
    with pytest.raises(ValueError, match="Stored selection differs"):
        select_models(synthetic, folds, cfg, tmp_path)
