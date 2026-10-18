"""Schema-level audit and training-only exploratory figures; never fit a model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import FEATURES, NUMERIC, load_data
from src.preprocessing import partitions, read_config


def quality_summary(frame):
    return {
        "rows": len(frame), "columns": list(frame.columns),
        "missing": {c: int(n) for c, n in frame.isna().sum().items()},
        "class_counts": {str(c): int(n) for c, n in frame.Survived.value_counts().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_ids": int(frame.PassengerId.duplicated().sum()),
        "duplicate_predictor_rows": int(frame.duplicated(FEATURES).sum()),
        "zero_fares": int((frame.Fare == 0).sum()),
        "numeric_ranges": {c: {"min": float(frame[c].min()), "max": float(frame[c].max())}
                           for c in NUMERIC},
        "actions": "No rows removed; no outlier clipping or zero-fare replacement. Imputation happens in training folds.",
    }


def inspect_data(data, out="reports"):
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    frame = load_data(data)
    train, test, folds, groups = partitions(frame, read_config())
    audit = quality_summary(frame)
    audit["train_rows"], audit["test_rows"] = len(train), len(test)
    audit["travel_groups"] = int(groups.nunique())
    audit["largest_group"] = int(groups.value_counts().max())
    audit["training_missing"] = quality_summary(frame.iloc[train])["missing"]
    (out / "data_quality.json").write_text(json.dumps(audit, indent=2) + "\n")
    # Record membership without names, ticket strings, or raw predictor values.
    manifest = {
        "train_ids": frame.PassengerId.iloc[train].astype(int).tolist(),
        "holdout_ids": frame.PassengerId.iloc[test].astype(int).tolist(),
        "folds": [{"fit_ids": frame.PassengerId.iloc[train[fit]].astype(int).tolist(),
                   "validation_ids": frame.PassengerId.iloc[train[valid]].astype(int).tolist()}
                  for fit, valid in folds],
        "group_overlap": 0,
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    training = frame.iloc[train]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    training.Survived.value_counts().sort_index().plot.bar(ax=axes[0], color=["#687787", "#147d92"])
    axes[0].set(title=f"Training labels (n={len(train)})", xlabel="Survived: 0=no, 1=yes", ylabel="Passengers")
    training[FEATURES].isna().sum().plot.bar(ax=axes[1], color="#147d92")
    axes[1].set(title="Training predictor missingness", ylabel="Missing records")
    fig.savefig(out / "figures" / "training_overview.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    correlations = {}
    for label in [0, 1]:
        subset = training.loc[training.Survived == label, NUMERIC]
        corr = subset.corr()  # pairwise observed values, no full-data imputation
        correlations[str(label)] = json.loads(corr.to_json())
        ax = axes[0, label]
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set(xticks=range(4), xticklabels=NUMERIC, yticks=range(4), yticklabels=NUMERIC,
               title=f"Within-class correlations: y={label}")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(corr.iloc[i,j]) > .65 else "black")
        fig.colorbar(im, ax=ax, shrink=.7)
    for label, color in [(0, "#687787"), (1, "#147d92")]:
        for ax, col in zip(axes[1], ["Age", "Fare"]):
            ax.hist(training.loc[training.Survived == label, col].dropna(), bins=20,
                    alpha=.5, label=f"y={label}", color=color)
            ax.set(xlabel="Age (years)" if col == "Age" else "Fare (recorded units)", ylabel="Passengers",
                   title=f"Observed training {col}")
            ax.legend()
    fig.savefig(out / "figures" / "training_diagnostics.png", dpi=160)
    plt.close(fig)
    (out / "training_correlations.json").write_text(json.dumps(correlations, indent=2) + "\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/titanic.zip")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    print(json.dumps(inspect_data(args.data, args.out), indent=2))
