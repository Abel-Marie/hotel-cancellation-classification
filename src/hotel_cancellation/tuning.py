"""Hyperparameter search: Grid, Random and Bayesian (Optuna TPE) behind one interface.

All searches use cross-validation on training data only, fixed trial counts
and seeded samplers, so repeated runs give the same result.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import optuna
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split

from .config import Config
from .models import make_model


def stratified_subsample(X: pd.DataFrame, y: pd.Series, n_rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """A stratified sample of at most ``n_rows`` rows, used to keep searches affordable."""
    if len(X) <= n_rows:
        return X, y
    keep, _ = train_test_split(np.arange(len(X)), train_size=n_rows, stratify=y, random_state=seed)
    keep = np.sort(keep)
    return X.iloc[keep], y.iloc[keep]


def cv_splitter(cfg: Config) -> StratifiedKFold:
    return StratifiedKFold(n_splits=cfg.tuning.cv_folds, shuffle=True, random_state=cfg.seed)


def search_result(name: str, model: str, scores: list[float], params: list[dict], seconds: float, folds: int) -> dict:
    """Common summary for every strategy: history in evaluation order and the best point."""
    best = int(np.argmax(scores))
    return {
        "strategy": name, "model": model, "configs_tried": len(scores), "fits": len(scores) * folds,
        "seconds": round(seconds, 1), "best_score": float(scores[best]), "best_params": params[best],
        "history": [float(s) for s in scores],
        "best_so_far": [float(v) for v in np.maximum.accumulate(scores)],
    }


def strip_prefix(params: dict) -> dict:
    return {k.split("__", 1)[1]: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


def run_grid_search(kind: str, grid: dict, X, y, cats, nums, cfg: Config) -> dict:
    start = time.time()
    search = GridSearchCV(make_model(kind, cats, nums, cfg), {f"model__{k}": v for k, v in grid.items()},
                          scoring="roc_auc", cv=cv_splitter(cfg), n_jobs=1, refit=False)
    search.fit(X, y)
    res = search.cv_results_
    return search_result("grid", kind, list(res["mean_test_score"]),
                         [strip_prefix(p) for p in res["params"]], time.time() - start, cfg.tuning.cv_folds)


def run_random_search(kind: str, distributions: dict, n_iter: int, X, y, cats, nums, cfg: Config) -> dict:
    start = time.time()
    search = RandomizedSearchCV(make_model(kind, cats, nums, cfg),
                                {f"model__{k}": v for k, v in distributions.items()}, n_iter=n_iter,
                                scoring="roc_auc", cv=cv_splitter(cfg), n_jobs=1, refit=False,
                                random_state=cfg.seed)
    search.fit(X, y)
    res = search.cv_results_
    return search_result("random", kind, list(res["mean_test_score"]),
                         [strip_prefix(p) for p in res["params"]], time.time() - start, cfg.tuning.cv_folds)


def lgbm_distributions() -> dict:
    """Random-search space for LightGBM; the Optuna space below covers the same ranges."""
    return {
        "num_leaves": randint(15, 256), "learning_rate": loguniform(0.01, 0.2),
        "n_estimators": randint(100, 900), "min_child_samples": randint(5, 100),
        "subsample": uniform(0.5, 0.5), "subsample_freq": [1], "colsample_bytree": uniform(0.5, 0.5),
        "reg_lambda": loguniform(1e-3, 10.0),
    }


def lgbm_trial_params(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 899),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 99),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def run_optuna_search(n_trials: int, X, y, cats, nums, cfg: Config, start_from: list[dict] | None = None,
                      name: str = "bayesian") -> dict:
    """Bayesian optimisation with a seeded TPE sampler and a single worker (reproducible).

    ``start_from`` lists settings to evaluate first; the sampler then learns from them.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    start = time.time()
    cv = cv_splitter(cfg)

    def objective(trial: optuna.Trial) -> float:
        model = make_model("lgbm", cats, nums, cfg, **lgbm_trial_params(trial))
        return float(np.mean(cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=1)))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    for params in start_from or []:
        study.enqueue_trial({k: v for k, v in params.items() if k != "subsample_freq"})
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    scores = [t.value for t in study.trials]
    params = [{**t.params, "subsample_freq": 1} for t in study.trials]
    return search_result(name, "lgbm", scores, params, time.time() - start, cfg.tuning.cv_folds)


# LightGBM's settings when nothing is tuned (the "lgbm" candidate in Section 5), written in the
# search space's terms so the refinement can start from them.
LGBM_UNTUNED = {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 400, "min_child_samples": 20,
                "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1e-3}


def refine_lgbm(tuning: dict, X, y, cats, nums, cfg: Config) -> dict:
    """Second stage: Bayesian search on ALL training rows.

    The first stage compares the three strategies cheaply on a subsample, but the best settings
    for 20,000 rows are not the best for the full training set (smaller data favours smaller,
    slower-learning models). This stage starts from the untuned settings and from the winner of
    each first-stage search, so its result can only match or beat them in cross-validation, and
    then lets the sampler explore from there.
    """
    starts = [dict(LGBM_UNTUNED)]
    for key in ("lgbm_grid", "lgbm_random", "lgbm_bayesian"):
        best = {**LGBM_UNTUNED, **{k: v for k, v in tuning[key]["best_params"].items() if k in LGBM_UNTUNED}}
        # keep the starting point inside the search space (a grid may contain values outside it)
        best["n_estimators"] = int(min(max(best["n_estimators"], 100), 899))
        best["num_leaves"] = int(min(max(best["num_leaves"], 15), 255))
        best["learning_rate"] = float(min(max(best["learning_rate"], 0.01), 0.2))
        starts.append(best)
    result = run_optuna_search(max(cfg.tuning.refine_trials, len(starts)), X, y, cats, nums, cfg,
                               start_from=starts, name="bayesian, all training rows")
    result["rows"] = int(len(X))
    result["started_from"] = ["untuned settings", "best of grid search", "best of random search",
                              "best of Bayesian search"]
    result["start_scores"] = result["history"][:len(starts)]
    return result


def rf_distributions() -> dict:
    return {"n_estimators": randint(100, 301), "max_depth": [None, 12, 20, 30],
            "min_samples_leaf": randint(1, 11), "max_features": ["sqrt", 0.3, 0.5]}


def grid_size(grid: dict) -> int:
    return len(list(itertools.product(*grid.values())))


def tune_all(X, y, cats, nums, cfg: Config) -> dict:
    """Grid search on logistic regression, random search on the random forest, and all
    three strategies on LightGBM with the same budget. Runs on a stratified subsample."""
    Xs, ys = stratified_subsample(X, y, cfg.tuning.subsample_rows, cfg.seed)
    budget = cfg.tuning.lgbm_budget
    if grid_size(cfg.tuning.lgbm_grid) != budget:
        raise ValueError("lgbm_grid must contain exactly lgbm_budget combinations for a fair comparison")
    tuning = {
        "subsample_rows": int(len(Xs)),
        "logreg_grid": run_grid_search("logreg", cfg.tuning.logreg_grid, Xs, ys, cats, nums, cfg),
        "rf_random": run_random_search("rf", rf_distributions(), cfg.tuning.rf_random_iterations,
                                       Xs, ys, cats, nums, cfg),
        "lgbm_grid": run_grid_search("lgbm", cfg.tuning.lgbm_grid, Xs, ys, cats, nums, cfg),
        "lgbm_random": run_random_search("lgbm", lgbm_distributions(), budget, Xs, ys, cats, nums, cfg),
        "lgbm_bayesian": run_optuna_search(budget, Xs, ys, cats, nums, cfg),
    }
    tuning["lgbm_refine"] = refine_lgbm(tuning, X, y, cats, nums, cfg)
    return tuning


def best_lgbm_search(tuning: dict) -> dict:
    """The LightGBM settings to refit: the second-stage search on all training rows.

    Its cross-validated scores are not comparable with the first-stage ones (more rows per fold),
    which is why it is not simply the maximum over all searches.
    """
    return tuning["lgbm_refine"]
