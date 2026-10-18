"""Run the frozen development-only model selection stage."""
import argparse
import json
from pathlib import Path

from src.data import load_data
from src.preprocessing import partitions, read_config
from src.validation import select_models


def run(data="data/titanic.zip", out="reports", stage="cv"):
    config = read_config()
    frame = load_data(data)
    train, test, folds, groups = partitions(frame, config)
    fitted, selection = select_models(frame.iloc[train], folds, config, out)
    print(json.dumps(selection, indent=2))
    if stage == "full":
        from src.evaluation import evaluate
        return evaluate(fitted, frame.iloc[train], frame.iloc[test], groups.iloc[test], config, out)
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/titanic.zip")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--stage", choices=["cv", "full"], default="cv")
    args = parser.parse_args()
    run(args.data, args.out, args.stage)
