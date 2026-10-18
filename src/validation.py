"""Select parameters using development records only; never receive test rows."""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from src.data import FEATURES, TRAIN_SHA256
from src.models import model_pipeline, parameter_grids

SCORING = {"f1": make_scorer(f1_score, zero_division=0), "accuracy": "accuracy",
           "precision": make_scorer(precision_score, zero_division=0),
           "recall": make_scorer(recall_score, zero_division=0),
           "roc_auc": "roc_auc", "neg_log_loss": "neg_log_loss"}


def signature(training, folds, config):
    payload = {"dataset_sha256": TRAIN_SHA256, "config": config,
               "training_ids": training.PassengerId.astype(int).tolist(),
               "folds": [(a.tolist(), b.tolist()) for a, b in folds]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def select_models(training, folds, config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fitted, selected, candidates, fold_rows, summary = {}, {}, [], [], []
    for name, grid in parameter_grids(config).items():
        search = GridSearchCV(model_pipeline(name, config), grid, scoring=SCORING,
                              refit=config["primary_metric"], cv=folds, n_jobs=1,
                              error_score="raise", return_train_score=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            search.fit(training[FEATURES], training.Survived)
        fitted[name] = search.best_estimator_
        selected[name] = search.best_params_
        results = search.cv_results_
        for i, params in enumerate(results["params"]):
            record = {"model": name, "params": json.dumps(params, sort_keys=True)}
            for metric in SCORING:
                scores = [float(results[f"split{f}_test_{metric}"][i])
                          for f in range(len(folds))]
                record[f"mean_{metric}"] = float(np.mean(scores))
                record[f"sd_{metric}"] = float(np.std(scores, ddof=1))
                for fold, score in enumerate(scores):
                    candidates.append({"model": name, "params": record["params"],
                                       "fold": fold, "metric": metric, "score": score})
            if i == search.best_index_:
                summary.append(record)
                for f in range(len(folds)):
                    fold_rows.append({"model": name, "fold": f,
                                      **{m: float(results[f"split{f}_test_{m}"][i]) for m in SCORING}})
    selection = {"signature": signature(training, folds, config), "parameters": selected,
                 "selection_metric": config["primary_metric"],
                 "note": "Selection CV, not nested CV; uncertainty from selection is not estimated."}
    existing = out / "selection.json"
    if existing.exists() and json.loads(existing.read_text()) != selection:
        raise ValueError("Stored selection differs. Investigate before any holdout evaluation.")
    existing.write_text(json.dumps(selection, indent=2) + "\n")
    pd.DataFrame(summary).to_csv(out / "cv_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "cv_folds.csv", index=False)
    pd.DataFrame(candidates).to_csv(out / "cv_candidates.csv", index=False)
    return fitted, selection
