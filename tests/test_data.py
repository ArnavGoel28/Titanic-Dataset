from pathlib import Path

import pandas as pd
import pytest

from src.data import load_data, travel_groups, validate_schema


@pytest.fixture
def passengers():
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4], "Survived": [0, 1, 0, 1],
        "Pclass": [3, 1, 2, 3], "Name": ["A, One", "B, Two", "B, Three", "C, Four"],
        "Sex": ["male", "female", "male", "female"], "Age": [22., None, 30., 9.],
        "SibSp": [1, 1, 1, 0], "Parch": [0, 0, 0, 0], "Ticket": ["X", "X", "Y", "Z"],
        "Fare": [7., 30., 12., 8.], "Cabin": [None] * 4, "Embarked": ["S", "C", "Q", None],
    })


def test_group_connections_are_transitive_and_target_free(passengers):
    groups = travel_groups(passengers)
    assert groups.iloc[0] == groups.iloc[1] == groups.iloc[2]
    assert groups.iloc[3] != groups.iloc[0]
    changed = passengers.assign(Survived=1 - passengers.Survived)
    pd.testing.assert_series_equal(groups, travel_groups(changed))


def test_schema_rejects_invalid_identity_target_and_numbers(passengers):
    validate_schema(passengers)
    for col, value in [("PassengerId", 1), ("Survived", 2), ("Fare", -1), ("SibSp", 0.5)]:
        bad = passengers.copy()
        bad[col] = bad[col].astype(float) if col == "SibSp" else bad[col]
        bad.loc[1, col] = value
        with pytest.raises(ValueError):
            validate_schema(bad)


def test_unknown_data_version_rejected(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("PassengerId,Survived\n1,1\n")
    with pytest.raises(ValueError, match="checksum"):
        load_data(path)


@pytest.mark.skipif(not Path("data/titanic.zip").exists(), reason="Kaggle data supplied locally only")
def test_real_data_checksum_and_shape():
    frame = load_data()
    assert frame.shape == (891, 12)
    assert frame.Survived.value_counts().to_dict() == {0: 549, 1: 342}
