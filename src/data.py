"""Read the supplied Kaggle archive without extracting or changing raw files."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

TRAIN_SHA256 = "7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f"
COLUMNS = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp",
           "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
NUMERIC = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Pclass", "Sex", "Embarked"]
FEATURES = NUMERIC + CATEGORICAL


def load_data(path="data/titanic.zip"):
    """Require the recorded labeled CSV, including its exact byte checksum."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. See data/README.md for Kaggle download steps.")
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            raw = archive.read("train.csv")
    else:
        raw = path.read_bytes()
    if sha256(raw).hexdigest() != TRAIN_SHA256:
        raise ValueError("train.csv checksum differs from the documented version; investigate before use.")
    frame = pd.read_csv(BytesIO(raw))
    validate_schema(frame)
    return frame


def validate_schema(frame):
    if list(frame.columns) != COLUMNS:
        raise ValueError("Unexpected schema: require labeled Kaggle train.csv.")
    if frame["PassengerId"].isna().any() or frame["PassengerId"].duplicated().any():
        raise ValueError("PassengerId must be present and unique.")
    for col, allowed in [("Survived", {0, 1}), ("Pclass", {1, 2, 3}),
                         ("Sex", {"male", "female"}), ("Embarked", {"C", "Q", "S"})]:
        if not set(frame[col].dropna()).issubset(allowed):
            raise ValueError(f"Invalid category in {col}.")
    if frame["Survived"].isna().any() or frame["Survived"].nunique() != 2:
        raise ValueError("Both target classes must be present without missing labels.")
    for col in NUMERIC:
        values = frame[col].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Negative or nonfinite values in {col}.")
    for col in ["SibSp", "Parch"]:
        if frame[col].isna().any() or (frame[col] % 1 != 0).any():
            raise ValueError(f"{col} must contain nonnegative integer counts.")


def travel_groups(frame):
    """Connected components of shared tickets or surname + family-size keys.

    Target-free, deterministic grouping only; these keys are never predictors.
    The family key is restricted to reported family size > 1. It is a proxy,
    not verified genealogy, and may merge unrelated same-name families.
    """
    parent = list(range(len(frame)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = root(i), root(j)
        parent[max(a, b)] = min(a, b)

    seen = {}
    for i, row in enumerate(frame.itertuples(index=False)):
        ticket = "".join(str(row.Ticket).upper().split()) if pd.notna(row.Ticket) else ""
        size = int(row.SibSp + row.Parch + 1)
        surname = str(row.Name).split(",", 1)[0].strip().upper()
        keys = [("ticket", ticket)] if ticket else []
        if size > 1 and surname:
            keys.append(("family", surname, size))
        for key in keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    ids = frame["PassengerId"].to_numpy()
    return pd.Series([int(ids[root(i)]) for i in range(len(frame))], index=frame.index, name="group")
