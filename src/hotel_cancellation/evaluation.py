"""Metrics, confidence intervals, calibration, thresholds and diagnostic curves."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, validation_curve

from .config import Config
from .models import MODEL_NOTES, make_model
from .tuning import cv_splitter, stratified_subsample


def classification_metrics(y, proba, threshold: float) -> dict:
    """Threshold metrics at ``threshold`` plus ranking and probability metrics."""
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
    }


def bootstrap_ci(y, proba, threshold: float, n_samples: int, seed: int) -> dict:
    """95% percentile intervals from resampling rows with replacement."""
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_samples):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        for k, v in classification_metrics(y[idx], proba[idx], threshold).items():
            if k != "threshold":
                draws.setdefault(k, []).append(v)
    return {k: {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))}
            for k, v in draws.items()}


def evaluate_candidates(X_train, y_train, X_val, y_val, cats, nums, cfg: Config) -> pd.DataFrame:
    """Cross-validation on a training subsample, then a full-train fit scored on validation."""
    import pickle

    Xs, ys = stratified_subsample(X_train, y_train, cfg.tuning.subsample_rows, cfg.seed)
    rows = []
    for kind in MODEL_NOTES:
        model = make_model(kind, cats, nums, cfg)
        cv = cross_validate(model, Xs, ys, cv=cv_splitter(cfg), scoring=["roc_auc", "average_precision"])
        start = time.time()
        model.fit(X_train, y_train)
        fit_seconds = time.time() - start
        start = time.time()
        proba = model.predict_proba(X_val)[:, 1]
        per_1k = (time.time() - start) / len(X_val) * 1000
        val = classification_metrics(y_val, proba, 0.5)
        rows.append({
            "model": kind, "cv_roc_auc_mean": float(np.mean(cv["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(cv["test_roc_auc"])),
            "cv_pr_auc_mean": float(np.mean(cv["test_average_precision"])),
            "val_roc_auc": val["roc_auc"], "val_pr_auc": val["pr_auc"], "val_accuracy_at_0_5": val["accuracy"],
            "val_precision_at_0_5": val["precision"], "val_recall_at_0_5": val["recall"],
            "val_f1_at_0_5": val["f1"], "val_brier": val["brier"], "fit_seconds": round(fit_seconds, 2),
            "predict_ms_per_1k_rows": round(per_1k * 1000, 2),
            "size_kb": round(len(pickle.dumps(model)) / 1024, 1), "notes": MODEL_NOTES[kind],
        })
    return pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False).reset_index(drop=True)


def fit_calibration(val_proba, y_val, cfg: Config) -> dict:
    """Decide whether isotonic calibration helps, using out-of-fold predictions on validation.

    The mapping is fitted on all validation rows only if the out-of-fold Brier score
    improves by at least ``calibration_min_brier_gain``. Stored as breakpoints so the
    API can apply it with a plain interpolation.
    """
    p = np.asarray(val_proba, dtype=float)
    y = np.asarray(y_val)
    folds = StratifiedKFold(cfg.evaluation.calibration_folds, shuffle=True, random_state=cfg.seed)
    oof = np.zeros_like(p)
    for fit_idx, eval_idx in folds.split(p.reshape(-1, 1), y):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[fit_idx], y[fit_idx])
        oof[eval_idx] = iso.predict(p[eval_idx])
    raw_brier, cal_brier = brier_score_loss(y, p), brier_score_loss(y, oof)
    use = (raw_brier - cal_brier) >= cfg.evaluation.calibration_min_brier_gain
    result = {"method": "isotonic", "used": bool(use), "brier_raw": float(raw_brier),
              "brier_calibrated_out_of_fold": float(cal_brier), "x": [], "y": []}
    if use:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        result["x"] = [float(v) for v in iso.X_thresholds_]
        result["y"] = [float(v) for v in iso.y_thresholds_]
    return result


def apply_calibration(proba, calibration: dict) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if not calibration["used"]:
        return proba
    return np.interp(proba, calibration["x"], calibration["y"])


def cost_curve(y, proba, fn_cost: float, fp_cost: float, thresholds=None) -> pd.DataFrame:
    """Average cost per booking at each threshold."""
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        pred = proba >= t
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        rows.append({"threshold": float(t), "false_positives": fp, "false_negatives": fn,
                     "cost_per_booking": (fp * fp_cost + fn * fn_cost) / len(y)})
    return pd.DataFrame(rows)


def choose_threshold(curve: pd.DataFrame) -> float:
    return float(curve.loc[curve["cost_per_booking"].idxmin(), "threshold"])


def confusion(y, proba, threshold: float) -> list[list[int]]:
    return confusion_matrix(np.asarray(y), (np.asarray(proba) >= threshold).astype(int)).tolist()


def learning_curve_data(model, X, y, cfg: Config) -> dict:
    sizes, train_scores, val_scores = learning_curve(
        clone(model), X, y, train_sizes=cfg.diagnostics.learning_curve_sizes, cv=cv_splitter(cfg),
        scoring="roc_auc", n_jobs=1, shuffle=True, random_state=cfg.seed)
    return {"train_sizes": sizes.tolist(), "train_mean": train_scores.mean(1).tolist(),
            "train_std": train_scores.std(1).tolist(), "cv_mean": val_scores.mean(1).tolist(),
            "cv_std": val_scores.std(1).tolist()}


def validation_curve_data(params: dict, X, y, cats, nums, cfg: Config) -> dict:
    """Effect of ``num_leaves`` (model capacity) with the other tuned settings fixed.

    Run on all training rows, like the search that chose the settings: the best capacity
    depends on how much data there is, so a subsample would answer a different question.
    """
    fixed = {k: v for k, v in params.items() if k != "num_leaves"}
    values = cfg.diagnostics.validation_curve_num_leaves
    train_scores, val_scores = validation_curve(
        make_model("lgbm", cats, nums, cfg, **fixed), X, y, param_name="model__num_leaves",
        param_range=values, cv=cv_splitter(cfg), scoring="roc_auc", n_jobs=1)
    return {"param": "num_leaves", "values": list(values), "chosen": params.get("num_leaves"),
            "train_mean": train_scores.mean(1).tolist(), "cv_mean": val_scores.mean(1).tolist(),
            "cv_std": val_scores.std(1).tolist()}


def lead_time_band(days: pd.Series) -> pd.Series:
    return pd.cut(days, [-1, 7, 30, 90, 180, 365, 10_000],
                  labels=["0-7", "8-30", "31-90", "91-180", "181-365", "365+"]).astype(str)


def segment_report(raw: pd.DataFrame, y, proba, threshold: float, by: str) -> pd.DataFrame:
    """Where the model is wrong: error rate and AUC per segment on validation data."""
    frame = pd.DataFrame({"segment": lead_time_band(raw["lead_time"]) if by == "lead_time_band" else raw[by],
                          "y": np.asarray(y), "p": np.asarray(proba)}, index=raw.index)
    rows = []
    for seg, part in frame.groupby("segment", observed=True):
        auc = roc_auc_score(part["y"], part["p"]) if part["y"].nunique() == 2 else float("nan")
        rows.append({"by": by, "segment": str(seg), "rows": int(len(part)),
                     "actual_cancel_rate": float(part["y"].mean()),
                     "flagged_share": float((part["p"] >= threshold).mean()),
                     "error_rate": float(((part["p"] >= threshold).astype(int) != part["y"]).mean()),
                     "roc_auc": float(auc)})
    return pd.DataFrame(rows).sort_values("rows", ascending=False)


def best_f1_threshold(y, proba) -> float:
    """The threshold with the highest F1 on the given (validation) data."""
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def operating_points(y_val, val_proba, cost_threshold: float, cfg: Config) -> list[dict]:
    """Three thresholds a hotel might act on, every one of them chosen on validation data.

    The cost-based one is the deployed decision rule. The other two are reported so the
    threshold metrics are not judged at a single, deliberately cautious operating point.
    """
    c = cfg.costs
    return [
        {"name": "cost-based (deployed)", "threshold": float(cost_threshold),
         "chosen_by": f"lowest expected cost: missed cancellation = {c.false_negative:g}, "
                      f"false alarm = {c.false_positive:g}"},
        {"name": "default 0.5", "threshold": 0.5, "chosen_by": "convention"},
        {"name": "best F1", "threshold": best_f1_threshold(y_val, val_proba), "chosen_by": "highest F1 on validation"},
    ]


def metrics_at_points(y, proba, points: list[dict]) -> pd.DataFrame:
    """Accuracy, precision, recall and F1 at each operating point, plus the share of bookings flagged."""
    proba = np.asarray(proba, dtype=float)
    rows = []
    for point in points:
        m = classification_metrics(y, proba, point["threshold"])
        rows.append({"operating_point": point["name"], "threshold": point["threshold"],
                     "flagged_share": float(np.mean(proba >= point["threshold"])),
                     **{k: m[k] for k in ("accuracy", "precision", "recall", "f1")}})
    return pd.DataFrame(rows)


def cost_sensitivity(y_val, val_proba, y_eval, eval_proba, cfg: Config) -> pd.DataFrame:
    """How the decision changes with the cost of a false alarm (a missed cancellation costs 1).

    For each cost the threshold is chosen on validation and then scored on the evaluation set.
    ``cost_without_model`` is the cheaper of the two rules that need no model: act on nobody
    or act on everybody. ``saving`` is the share of that cost the model removes.
    """
    y_eval = np.asarray(y_eval)
    eval_proba = np.asarray(eval_proba, dtype=float)
    fn = cfg.costs.false_negative
    rate = float(np.mean(y_eval))
    rows = []
    for fp in cfg.evaluation.cost_sensitivity_false_positive:
        threshold = choose_threshold(cost_curve(y_val, val_proba, fn, fp))
        cost = float(cost_curve(y_eval, eval_proba, fn, fp, thresholds=[threshold])["cost_per_booking"].iloc[0])
        without = min(rate * fn, (1 - rate) * fp)
        m = classification_metrics(y_eval, eval_proba, threshold)
        rows.append({"false_alarm_cost": float(fp), "threshold": threshold,
                     "flagged_share": float(np.mean(eval_proba >= threshold)), "precision": m["precision"],
                     "recall": m["recall"], "f1": m["f1"], "cost_per_booking": cost,
                     "cost_without_model": float(without), "saving": float(1 - cost / without)})
    return pd.DataFrame(rows)


def gains_table(y, proba, groups: int = 10) -> pd.DataFrame:
    """Bookings sorted from highest to lowest predicted risk and cut into equal groups.

    ``cumulative_share_of_cancellations`` answers: if the hotel contacted the riskiest
    10%, 20%, ... of bookings, what share of all cancellations would it reach?
    """
    y = np.asarray(y)
    order = np.argsort(-np.asarray(proba, dtype=float), kind="stable")
    parts = np.array_split(y[order], groups)
    base, total = float(y.mean()), float(y.sum())
    rows, reached = [], 0.0
    for i, part in enumerate(parts, start=1):
        reached += float(part.sum())
        rows.append({"risk_group": i, "bookings": int(len(part)), "cancel_rate": float(part.mean()),
                     "lift": float(part.mean() / base), "cumulative_share_of_bookings": i / groups,
                     "cumulative_share_of_cancellations": reached / total})
    return pd.DataFrame(rows)


def capacity_report(raw: pd.DataFrame, y, proba, base_rate: float) -> dict:
    """Does the model forecast HOW MANY bookings will cancel, per hotel and arrival week?

    Overbooking is decided for a hotel and a date range, not for one booking: the useful
    quantity is the expected number of cancellations, which is the sum of the predicted
    probabilities. It is compared with the forecast that needs no model (bookings times
    the overall cancellation rate seen in training). WAPE is the summed absolute error
    divided by the summed actual count; lower is better.
    """
    arrival = pd.to_datetime(raw["arrival_date_year"].astype(str) + "-" + raw["arrival_date_month"] + "-"
                             + raw["arrival_date_day_of_month"].astype(str), format="%Y-%B-%d")
    monday = arrival - pd.to_timedelta(arrival.dt.weekday, unit="D")
    frame = pd.DataFrame({"hotel": raw["hotel"].to_numpy(), "week": monday.dt.strftime("%Y-%m-%d").to_numpy(),
                          "actual": np.asarray(y, dtype=float), "predicted": np.asarray(proba, dtype=float)})
    weekly = frame.groupby(["hotel", "week"], as_index=False).agg(
        bookings=("actual", "size"), actual=("actual", "sum"), predicted=("predicted", "sum"))
    weekly["base_rate_forecast"] = weekly["bookings"] * base_rate
    total = float(weekly["actual"].sum())

    def wape(column: str) -> float:
        return float((weekly[column] - weekly["actual"]).abs().sum() / total)

    return {"weeks": int(len(weekly)), "bookings_per_week_median": float(weekly["bookings"].median()),
            "total_actual": total, "total_predicted": float(weekly["predicted"].sum()),
            "bias": float(weekly["predicted"].sum() / total - 1),
            "wape_model": wape("predicted"), "wape_base_rate": wape("base_rate_forecast"),
            "correlation": float(np.corrcoef(weekly["predicted"], weekly["actual"])[0, 1]),
            "by_week": weekly}
