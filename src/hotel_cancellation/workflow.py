"""Steps shared by the pipeline stages and the notebook.

The pipeline wraps each step with disk input/output; the notebook calls the
same steps in narrative order. Because both run this code, they produce the
same results, and ``tests/parity/test_notebook_vs_pipeline.py`` checks it.
"""

from __future__ import annotations

import dataclasses
import pickle

import pandas as pd

from .config import Config, deployed_sources
from .evaluation import (
    apply_calibration,
    bootstrap_ci,
    capacity_report,
    choose_threshold,
    classification_metrics,
    confusion,
    cost_curve,
    cost_sensitivity,
    fit_calibration,
    gains_table,
    learning_curve_data,
    metrics_at_points,
    operating_points,
    segment_report,
    validation_curve_data,
)
from .features import build_feature_frame, split_feature_types
from .models import make_model
from .tuning import best_lgbm_search

TREE_KINDS = {"lgbm", "rf", "hgb", "tree"}


def build_model_frames(clean: pd.DataFrame, splits: dict, cfg: Config) -> dict:
    """Deployed feature set for each split, plus the raw rows for error analysis."""
    sources = deployed_sources(cfg)
    X = build_feature_frame(clean, sources)
    y = clean[cfg.target]
    cats, nums = split_feature_types(list(X.columns))
    frames = {"sources": sources, "cats": cats, "nums": nums}
    for name, pos in splits.items():
        frames[f"X_{name}"], frames[f"y_{name}"], frames[f"raw_{name}"] = X.iloc[pos], y.iloc[pos], clean.iloc[pos]
    return frames


def fit_tuned_models(tuning: dict, frames: dict, cfg: Config) -> dict:
    """Refit the best setting of each tuned family on all training rows; score on validation."""
    picks = {"logreg": tuning["logreg_grid"], "rf": tuning["rf_random"], "lgbm": best_lgbm_search(tuning)}
    out = {}
    for kind, search in picks.items():
        model = make_model(kind, frames["cats"], frames["nums"], cfg, **search["best_params"])
        model.fit(frames["X_train"], frames["y_train"])
        proba = model.predict_proba(frames["X_validation"])[:, 1]
        out[kind] = {"model": model, "params": search["best_params"], "strategy": search["strategy"],
                     "cv_roc_auc": search["best_score"],
                     "val_roc_auc": classification_metrics(frames["y_validation"], proba, 0.5)["roc_auc"],
                     "size_kb": round(len(pickle.dumps(model)) / 1024, 1)}
    return out


def pick_champion(fitted: dict, cfg: Config) -> dict:
    """Best validation ROC-AUC among tree models (the ones the service can run), with a tie rule.

    Models within ``selection.tie_tolerance`` of the best score are treated as tied, because a
    difference that small is below run-to-run noise; among tied models the smallest is chosen,
    since it deploys and scores faster. Decided on validation data only.
    """
    trees = {k: v for k, v in fitted.items() if k in TREE_KINDS}
    top = max(v["val_roc_auc"] for v in trees.values())
    tied = {k: v for k, v in trees.items() if top - v["val_roc_auc"] <= cfg.selection.tie_tolerance}
    name = min(tied, key=lambda k: tied[k]["size_kb"])
    return {"name": name, "params": fitted[name]["params"], "strategy": fitted[name]["strategy"],
            "val_roc_auc": fitted[name]["val_roc_auc"], "best_val_roc_auc": top, "tied_models": sorted(tied),
            "rule": f"best validation ROC-AUC; models within {cfg.selection.tie_tolerance} of it are tied and "
                    "the smallest tied model wins",
            "comparison": [{"model": k, "strategy": v["strategy"], "cv_roc_auc": v["cv_roc_auc"],
                            "val_roc_auc": v["val_roc_auc"], "size_kb": v["size_kb"], "params": v["params"]}
                           for k, v in fitted.items()]}


def threshold_for(model, frames: dict, calibration: dict, cfg: Config) -> tuple[float, pd.DataFrame]:
    """Cost-minimising threshold on validation probabilities (after calibration)."""
    proba = apply_calibration(model.predict_proba(frames["X_validation"])[:, 1], calibration)
    curve = cost_curve(frames["y_validation"], proba, cfg.costs.false_negative, cfg.costs.false_positive)
    return choose_threshold(curve), curve


def run_diagnostics(model, champion: dict, frames: dict, cfg: Config) -> dict:
    """Calibration, threshold, validation metrics, learning and validation curves, error analysis."""
    y_val = frames["y_validation"]
    raw = model.predict_proba(frames["X_validation"])[:, 1]
    calibration = fit_calibration(raw, y_val, cfg)
    proba = apply_calibration(raw, calibration)
    threshold, curve = threshold_for(model, frames, calibration, cfg)
    segments = pd.concat([segment_report(frames["raw_validation"], y_val, proba, threshold, by)
                          for by in ["hotel", "market_segment", "lead_time_band"]], ignore_index=True)
    train_auc = classification_metrics(frames["y_train"], model.predict_proba(frames["X_train"])[:, 1],
                                       0.5)["roc_auc"]
    points = operating_points(y_val, proba, threshold, cfg)
    return {
        "calibration": calibration, "threshold": threshold, "cost_curve": curve,
        "val_metrics": classification_metrics(y_val, proba, threshold),
        "val_confusion": confusion(y_val, proba, threshold),
        "operating_points": points, "val_operating_metrics": metrics_at_points(y_val, proba, points),
        "val_cost_sensitivity": cost_sensitivity(y_val, proba, y_val, proba, cfg),
        "val_gains": gains_table(y_val, proba),
        "val_capacity": capacity_report(frames["raw_validation"], y_val, proba, float(frames["y_train"].mean())),
        "train_roc_auc": train_auc, "val_proba_raw": raw, "val_proba": proba,
        "learning_curve": learning_curve_data(model, frames["X_train"], frames["y_train"], cfg),
        "validation_curve": (validation_curve_data(champion["params"], frames["X_train"], frames["y_train"],
                                                   frames["cats"], frames["nums"], cfg)
                             if champion["name"] == "lgbm" else None),
        "segments": segments,
    }


def final_test_pass(test_models: dict, champion_calibration: dict, frames: dict, cfg: Config) -> dict:
    """The only function that reads the test set. Scores the pre-declared models once.

    Each model's threshold is chosen on validation before any test row is
    scored; only the champion is calibrated. Nothing is refitted here.
    """
    if list(test_models) != list(cfg.test_pass_models):
        raise ValueError(f"test pass must score exactly {cfg.test_pass_models}, got {list(test_models)}")
    results = {}
    for name, model in test_models.items():
        calibration = champion_calibration if name == "champion" else {"used": False}
        threshold, _ = threshold_for(model, frames, calibration, cfg)
        proba = apply_calibration(model.predict_proba(frames["X_test"])[:, 1], calibration)
        curve = cost_curve(frames["y_test"], proba, cfg.costs.false_negative, cfg.costs.false_positive,
                           thresholds=[threshold])
        results[name] = {"metrics": classification_metrics(frames["y_test"], proba, threshold),
                         "metrics_at_0_5": classification_metrics(frames["y_test"], proba, 0.5),
                         "cost_per_booking": float(curve["cost_per_booking"].iloc[0]),
                         "confusion": confusion(frames["y_test"], proba, threshold), "proba": proba}
    champ = results["champion"]
    champ["ci"] = bootstrap_ci(frames["y_test"], champ["proba"], champ["metrics"]["threshold"],
                               cfg.evaluation.bootstrap_samples, cfg.seed)
    # The same analyses as on validation (Section 6), with every threshold taken from validation.
    val_proba = apply_calibration(test_models["champion"].predict_proba(frames["X_validation"])[:, 1],
                                  champion_calibration)
    points = operating_points(frames["y_validation"], val_proba, champ["metrics"]["threshold"], cfg)
    champ["operating_metrics"] = metrics_at_points(frames["y_test"], champ["proba"], points)
    champ["cost_sensitivity"] = cost_sensitivity(frames["y_validation"], val_proba, frames["y_test"],
                                                 champ["proba"], cfg)
    champ["gains"] = gains_table(frames["y_test"], champ["proba"])
    champ["capacity"] = capacity_report(frames["raw_test"], frames["y_test"], champ["proba"],
                                        float(frames["y_train"].mean()))
    met = champ["metrics"]["roc_auc"] >= cfg.success.roc_auc
    beats = all(champ["metrics"]["roc_auc"] > results[m]["metrics"]["roc_auc"] for m in results if m != "champion")
    return {"models": results, "success": {
        "criterion": f"test ROC-AUC >= {cfg.success.roc_auc} and above every baseline",
        "roc_auc_target_met": bool(met), "beats_baselines": bool(beats),
        "ci_lower_bound_above_target": bool(champ["ci"]["roc_auc"]["low"] >= cfg.success.roc_auc)}}


def strip_arrays(obj):
    """Drop fitted models and probability arrays, and turn DataFrames into records, before writing JSON."""
    if isinstance(obj, dict):
        return {k: strip_arrays(v) for k, v in obj.items()
                if k not in {"proba", "val_proba", "val_proba_raw", "eval_proba"} and not hasattr(v, "fit")}
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    if isinstance(obj, list):
        return [strip_arrays(v) for v in obj]
    return obj


def build_metrics_report(parts: dict, cfg: Config) -> dict:
    """The single results file. Docs and slides read their numbers from here."""
    return strip_arrays({
        "project": cfg.project, "fast_mode": cfg.is_fast, "seed": cfg.seed,
        "data": {"audit": parts["audit"], "cleaning": parts["cleaning_log"],
                 "rows_after_cleaning": parts["rows_after_cleaning"],
                 "duplicate_rows_cancel_rate": parts["duplicate_rows_cancel_rate"],
                 "split": parts["split_summary"]},
        "features": {"sources": parts["frames"]["sources"], "categorical": parts["frames"]["cats"],
                     "numeric": parts["frames"]["nums"]},
        "leakage": {"ablation": parts["leakage"], "dedup_sensitivity": parts["dedup"],
                    "decisions": {c: {"decision": d.decision, "reason": d.reason}
                                  for c, d in cfg.leakage.tier3_decisions.items()}},
        "candidates": parts["candidates"],
        "experiments": parts.get("experiments"),
        "tuning": {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "history"})
                   for k, v in parts["tuning"].items()},
        "champion": parts["champion"],
        "diagnostics": {k: v for k, v in parts["diagnostics"].items() if k not in {"cost_curve", "segments"}},
        "quality_audit": parts.get("quality_audit"),
        "duplicates": parts.get("duplicates"),
        "first_test_evaluation": dataclasses.asdict(cfg.first_test_evaluation),
        "temporal_check": parts["temporal"],
        "explain": {"permutation_top10": parts["permutation"].head(10),
                    "shap_top10": parts["shap_importance"].head(10)},
        "test": parts["test"],
        "deployment": parts.get("deployment"),
    })
