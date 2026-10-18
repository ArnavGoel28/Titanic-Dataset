"""Deterministic grouped partitions and preprocessing fitted within each fold."""
import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL, NUMERIC, travel_groups


def read_config(path="config/experiment.json"):
    return json.loads(Path(path).read_text())


def assert_partition(train, valid, labels, groups):
    if set(train) & set(valid):
        raise ValueError("Overlapping row indices.")
    if set(groups.iloc[train]) & set(groups.iloc[valid]):
        raise ValueError("Travel group crosses partition boundary.")
    if len(np.unique(labels.iloc[train])) != 2 or len(np.unique(labels.iloc[valid])) != 2:
        raise ValueError("Both classes required in every partition.")


def partitions(frame, config):
    groups = travel_groups(frame)
    y = frame.Survived
    outer = StratifiedGroupKFold(n_splits=config["outer_folds"], shuffle=True,
                                 random_state=config["seed"])
    train, test = list(outer.split(frame, y, groups))[config["holdout_fold"]]
    assert_partition(train, test, y, groups)
    inner = StratifiedGroupKFold(n_splits=config["cv_folds"], shuffle=True,
                                 random_state=config["seed"])
    train_y, train_g = y.iloc[train], groups.iloc[train]
    folds = list(inner.split(frame.iloc[train], train_y, train_g))
    for fit, valid in folds:
        assert_partition(fit, valid, train_y, train_g)
    return train, test, folds, groups


def preprocessor():
    # Standardization also makes GaussianNB's variance floor less dependent on
    # Fare units. The scaler and imputer learn exclusively from each fit subset.
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(drop="first", handle_unknown="ignore",
                                                     sparse_output=False))])
    return ColumnTransformer([("numeric", numeric, NUMERIC),
                              ("categorical", categorical, CATEGORICAL)],
                             remainder="drop", sparse_threshold=0)
