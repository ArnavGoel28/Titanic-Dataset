"""Probability-aware holdout metrics and paired group-bootstrap uncertainty."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src.data import FEATURES, TRAIN_SHA256

INTERVAL_METRICS = ["accuracy", "f1", "roc_auc", "log_loss"]


def positive_probability(model, features):
    index = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(index) != 1:
        raise ValueError("Expected positive class 1.")
    return model.predict_proba(features)[:, index[0]]


def metrics(y, probability, threshold=.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    if y.shape != p.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probability shape/range/finiteness error.")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2
                                 else float(accuracy_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
            "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def group_bootstrap_indices(groups, rng):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    draw = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == group) for group in draw])


def bootstrap(y, probabilities, groups, config):
    y = np.asarray(y)
    rng = np.random.default_rng(config["bootstrap_seed"])
    values = {name: {metric: [] for metric in INTERVAL_METRICS} for name in probabilities}
    differences = {metric: [] for metric in INTERVAL_METRICS}
    for _ in range(config["bootstrap_repeats"]):
        sample = group_bootstrap_indices(groups, rng)
        for name, p in probabilities.items():
            sample_y, sample_p = y[sample], p[sample]
            sample_pred = sample_p >= config["threshold"]
            scores = {"accuracy": float(np.mean(sample_y == sample_pred)),
                      "f1": float(f1_score(sample_y, sample_pred, zero_division=0)),
                      "roc_auc": float(roc_auc_score(sample_y, sample_p))
                                 if len(np.unique(sample_y)) == 2 else None,
                      "log_loss": float(log_loss(sample_y, np.column_stack([1-sample_p, sample_p]), labels=[0, 1]))}
            for metric in INTERVAL_METRICS:
                if scores[metric] is not None:
                    values[name][metric].append(scores[metric])
        for metric in INTERVAL_METRICS:
            # Both models use the same resampled groups; AUC availability matches.
            if len(values["LogisticRegression"][metric]) > len(differences[metric]):
                differences[metric].append(values["LogisticRegression"][metric][-1]
                                           - values["GaussianNB"][metric][-1])

    def interval(items):
        if not items:
            return {"low": None, "high": None, "replicates": 0}
        low, high = np.percentile(items, [2.5, 97.5])
        return {"low": float(low), "high": float(high), "replicates": len(items)}

    return {"method": "Paired whole-travel-group percentile bootstrap; conditional on fitted models",
            "models": {n: {m: interval(v) for m, v in row.items()} for n, row in values.items()},
            "lr_minus_nb": {m: interval(v) for m, v in differences.items()}}


def plots(y, probabilities, config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    for ax, name in zip(axes, ["GaussianNB", "LogisticRegression"]):
        ConfusionMatrixDisplay.from_predictions(y, probabilities[name] >= config["threshold"],
                                                labels=[0, 1], display_labels=["Did not survive", "Survived"],
                                                ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name} (n={len(y)})")
    fig.savefig(out / "confusion_matrices.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    colors = {"Dummy": "#687787", "GaussianNB": "#be7131", "LogisticRegression": "#147d92"}
    for name, p in probabilities.items():
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        observed, predicted = calibration_curve(y, p, n_bins=5, strategy="uniform")
        axes[0].plot(fpr, tpr, label=name, color=colors[name])
        if name == "Dummy":
            axes[1].axhline(np.mean(y), label=name, color=colors[name])
        else:
            axes[1].plot(recall, precision, label=name, color=colors[name])
        axes[2].plot(predicted, observed, "o-", label=name, color=colors[name])
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=.7)
    axes[1].axhline(np.mean(y), color="black", linestyle="--", linewidth=.7)
    axes[2].plot([0, 1], [0, 1], "k--", linewidth=.7)
    for ax, title, xlabel, ylabel in zip(axes, ["Holdout ROC", "Holdout precision-recall", "Calibration (5 bins)"],
                                       ["False positive rate", "Recall", "Mean predicted survival"],
                                       ["True positive rate", "Precision", "Observed survival fraction"]):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=7, loc="best")
    fig.savefig(out / "probability_diagnostics.png", dpi=160)
    plt.close(fig)


def evaluate(fitted, training, holdout, groups, config, out):
    out = Path(out)
    y = holdout.Survived.to_numpy()
    probability = {n: positive_probability(m, holdout[FEATURES]) for n, m in fitted.items()}
    scores = {n: metrics(y, p, config["threshold"]) for n, p in probability.items()}
    intervals = bootstrap(y, probability, groups, config)
    nb, lr = [(probability[n] >= config["threshold"]).astype(int)
              for n in ["GaussianNB", "LogisticRegression"]]
    disagreement = {"agree": int((nb == lr).sum()), "disagree": int((nb != lr).sum()),
                    "both_correct": int(((nb == y) & (lr == y)).sum()),
                    "both_wrong": int(((nb != y) & (lr != y)).sum()),
                    "nb_only_correct": int(((nb == y) & (lr != y)).sum()),
                    "lr_only_correct": int(((lr == y) & (nb != y)).sum())}
    fitted_details = {}
    for name, model in fitted.items():
        names = model.named_steps["preprocess"].get_feature_names_out().tolist()
        fitted_details[name] = {"transformed_features": names}
        if name == "LogisticRegression":
            fitted_details[name]["coefficients"] = dict(zip(names, model.named_steps["model"].coef_[0].tolist()))
            fitted_details[name]["intercept"] = float(model.named_steps["model"].intercept_[0])
        p = probability[name]
        losses = -(y * np.log(np.clip(p, 1e-15, 1)) + (1-y) * np.log(np.clip(1-p, 1e-15, 1)))
        fitted_details[name]["largest_holdout_losses"] = [
            {"passenger_id": int(holdout.PassengerId.iloc[i]), "true_label": int(y[i]),
             "probability": float(p[i]), "log_loss": float(losses[i])}
            for i in np.argsort(losses)[-3:][::-1]]
    result = {"dataset_sha256": TRAIN_SHA256, "config": config,
              "train_n": len(training), "holdout_n": len(holdout),
              "train_positive": int(training.Survived.sum()), "holdout_positive": int(y.sum()),
              "holdout_groups": int(groups.nunique()), "metrics": scores, "uncertainty": intervals,
              "disagreement": disagreement, "fitted_details": fitted_details}
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(scores).T.rename_axis("model").to_csv(out / "holdout_metrics.csv")
    plots(y, probability, config, out / "figures")
    print(pd.DataFrame(scores).T[["accuracy", "f1", "roc_auc", "log_loss"]].to_string())
    return result
