"""Candidate models, each wrapped in the preprocessing it needs."""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import Config
from .features import make_linear_preprocessor, make_tree_preprocessor

LINEAR_KINDS = {"dummy", "logreg", "mlp"}

MODEL_NOTES = {
    "dummy": "Predicts the base rate for everyone. The floor every model must beat.",
    "logreg": "Linear in the features; every coefficient can be read directly.",
    "tree": "One readable tree, but unstable and prone to overfitting.",
    "rf": "Averages many trees; robust, but large to store and slower to score.",
    "hgb": "Gradient boosting in scikit-learn; strong and fast on tabular data.",
    "lgbm": "Gradient boosting in LightGBM; usually the strongest on tabular data. Explained with SHAP.",
    "mlp": "Small neural network; needs scaling, hardest to interpret.",
}


def base_estimator(kind: str, cfg: Config, **params):
    """The bare estimator for ``kind`` with fixed seeds and thread counts."""
    seed, jobs = cfg.seed, cfg.models.n_jobs
    if kind == "dummy":
        return DummyClassifier(strategy="prior")
    if kind == "logreg":
        return LogisticRegression(max_iter=3000, **params)
    if kind == "tree":
        return DecisionTreeClassifier(random_state=seed, **{"min_samples_leaf": 20, **params})
    if kind == "rf":
        defaults = {"n_estimators": 200, "min_samples_leaf": 5, "max_features": "sqrt"}
        return RandomForestClassifier(random_state=seed, n_jobs=jobs, **{**defaults, **params})
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=seed, early_stopping=True, **params)
    if kind == "lgbm":
        defaults = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63}
        return LGBMClassifier(objective="binary", random_state=seed, n_jobs=jobs, deterministic=True,
                              force_row_wise=True, verbose=-1, **{**defaults, **params})
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                             max_iter=200, random_state=seed, **params)
    raise ValueError(f"unknown model kind: {kind}")


def make_model(kind: str, cats: list[str], nums: list[str], cfg: Config, **params) -> Pipeline:
    """Preprocessing plus estimator. Linear models and the MLP get one-hot and scaling."""
    minf = cfg.features.min_category_frequency
    prep = (make_linear_preprocessor(cats, nums, minf) if kind in LINEAR_KINDS
            else make_tree_preprocessor(cats, nums, minf))
    return Pipeline(prep + [("model", base_estimator(kind, cfg, **params))])


CANDIDATE_KINDS = ["dummy", "logreg", "tree", "rf", "hgb", "lgbm", "mlp"]
