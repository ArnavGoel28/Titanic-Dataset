"""Only the two requested classifiers and a minimal prior-probability baseline."""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src.preprocessing import preprocessor


def model_pipeline(name, config):
    models = {
        "Dummy": DummyClassifier(strategy="prior", random_state=config["seed"]),
        "GaussianNB": GaussianNB(),
        "LogisticRegression": LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000,
                                                  random_state=config["seed"]),
    }
    return Pipeline([("preprocess", preprocessor()), ("model", models[name])])


def parameter_grids(config):
    return {"Dummy": {},
            "GaussianNB": {"model__var_smoothing": config["gnb_var_smoothing"]},
            "LogisticRegression": {"model__C": config["lr_C"]}}
