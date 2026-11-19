"""Pipeline stages. Each reads its inputs from disk and writes its outputs to disk,
so any stage can be re-run on its own and an interrupted run can resume.

The logic lives in the shared modules (mostly ``workflow.py``); this file only
moves data between stages and saves figures.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .artifacts import RunContext, load_json, save_json, write_manifest
from .cleaning import clean_bookings
from .config import Config, deployed_sources
from .data import download_raw, load_raw
from .evaluation import apply_calibration, evaluate_candidates, lead_time_band
from .experiments import compare_encodings, compare_feature_sets, compare_imbalance
from .explain import permutation_importance_frame, shap_explanation, shap_importance_frame
from .export import build_metadata, category_vocabulary, export_onnx, service_style_proba
from .features import build_feature_frame, split_feature_types
from .leakage import leakage_ablation, rate_table, tier3_evidence
from .log import LOGGER
from .models import make_model
from .monitoring.drift import drift_report
from .monitoring.latency import replay_requests
from .plotting import (
    finish_figure,
    plot_calibration,
    plot_candidates,
    plot_capacity,
    plot_cleaning_log,
    plot_confusion,
    plot_correlation,
    plot_cost_curve,
    plot_drift,
    plot_duplicate_decomposition,
    plot_duplicate_effect,
    plot_experiments,
    plot_gains,
    plot_importance,
    plot_latency,
    plot_leakage_ablation,
    plot_learning_curve,
    plot_missingness,
    plot_monthly_rate,
    plot_numeric_by_outcome,
    plot_operating_points,
    plot_outliers,
    plot_pca,
    plot_rate_by_category,
    plot_roc_pr,
    plot_search_comparison,
    plot_segments,
    plot_shap_summary,
    plot_shap_waterfall,
    plot_temporal,
    plot_test_results,
    plot_tier3_evidence,
    plot_validation_curve,
    set_plot_style,
)
from .robustness import dedup_sensitivity, duplicate_decomposition, temporal_check
from .split import sample_for_fast_mode, split_summary, stratified_split
from .tuning import tune_all
from .validation import quality_audit, validate_raw
from .workflow import (
    build_metrics_report,
    build_model_frames,
    final_test_pass,
    fit_tuned_models,
    pick_champion,
    run_diagnostics,
    strip_arrays,
)

STAGES = ["download", "validate", "clean", "split", "eda", "leakage-audit", "train", "tune", "select", "diagnose",
          "evaluate", "explain", "export", "report"]


def raw_path(ctx: RunContext) -> Path:
    return ctx.data_dir / "raw" / "hotels.csv"


def load_clean_and_splits(ctx: RunContext) -> tuple[pd.DataFrame, dict]:
    clean = pd.read_pickle(ctx.data_dir / "interim" / "clean.pkl")
    splits = {k: np.array(v) for k, v in load_json(ctx.data_dir / "processed" / "splits.json").items()}
    return clean, splits


def train_val_rows(clean: pd.DataFrame, splits: dict) -> pd.DataFrame:
    """Training and validation rows only: the data EDA and audits may look at."""
    return clean.iloc[np.sort(np.concatenate([splits["train"], splits["validation"]]))]


def stage_download(cfg: Config, ctx: RunContext) -> dict:
    info = download_raw(cfg, ctx.data_dir / "raw", repo_root=ctx.root)
    save_json(info, ctx.data_dir / "raw" / "source.json")
    return info


def stage_validate(cfg: Config, ctx: RunContext) -> dict:
    raw = load_raw(raw_path(ctx))
    audit = validate_raw(raw, cfg)
    save_json(audit, ctx.metrics_dir / "data_audit.json")
    save_json(quality_audit(raw, cfg), ctx.metrics_dir / "quality_audit.json")
    return audit


def stage_clean(cfg: Config, ctx: RunContext) -> dict:
    raw = load_raw(raw_path(ctx))
    clean, log = clean_bookings(raw, cfg)
    data_cols = [c for c in raw.columns if c != "row_id"]
    dup_rate = float(raw.loc[raw.duplicated(subset=data_cols), cfg.target].mean())
    rows_after = len(clean)
    clean = sample_for_fast_mode(clean, cfg)
    clean.to_pickle(ctx.data_dir / "interim" / "clean.pkl")
    out = {"cleaning_log": log, "rows_after_cleaning": rows_after, "rows_used": len(clean),
           "duplicate_rows_cancel_rate": dup_rate}
    save_json(out, ctx.metrics_dir / "cleaning.json")
    return out


def stage_split(cfg: Config, ctx: RunContext) -> dict:
    clean = pd.read_pickle(ctx.data_dir / "interim" / "clean.pkl")
    splits = stratified_split(clean, cfg.target, cfg.split.val_size, cfg.split.test_size, cfg.seed)
    save_json({k: v.tolist() for k, v in splits.items()}, ctx.data_dir / "processed" / "splits.json")
    summary = split_summary(clean, splits, cfg.target)
    save_json(summary, ctx.metrics_dir / "split.json")
    return summary


def stage_eda(cfg: Config, ctx: RunContext) -> dict:
    raw = load_raw(raw_path(ctx))
    clean, splits = load_clean_and_splits(ctx)
    trval = train_val_rows(clean, splits)
    cleaning = load_json(ctx.metrics_dir / "cleaning.json")
    feats = build_feature_frame(trval, deployed_sources(cfg))
    t = cfg.target
    finish_figure(plot_missingness(raw.drop(columns="row_id")), "eda_missingness", ctx)
    finish_figure(plot_outliers(raw), "eda_outliers", ctx)
    finish_figure(plot_cleaning_log(cleaning["cleaning_log"]), "eda_cleaning_rules", ctx)
    finish_figure(plot_duplicate_effect(cleaning["duplicate_rows_cancel_rate"], float(trval[t].mean())),
                  "eda_duplicates_are_cancellations", ctx)
    finish_figure(plot_rate_by_category(trval, ["hotel", "deposit_type", "market_segment", "customer_type"], t),
                  "eda_rate_by_category", ctx)
    finish_figure(plot_numeric_by_outcome(trval, t), "eda_numeric_by_outcome", ctx)
    finish_figure(plot_monthly_rate(trval, t), "eda_monthly_rate", ctx)
    finish_figure(plot_correlation(feats, trval[t]), "eda_correlation", ctx)
    finish_figure(plot_pca(feats, trval[t], cfg.seed), "eda_pca", ctx)
    # Summary tables behind the charts (training and validation rows), for the documents and slides.
    banded = trval.assign(lead_time_band=lead_time_band(trval["lead_time"]))
    months = pd.to_datetime(trval["arrival_date_year"].astype(str) + "-" + trval["arrival_date_month"] + "-01",
                            format="%Y-%B-%d").dt.strftime("%Y-%m")
    monthly = trval.assign(month=months).groupby("month")[t].agg(cancel_rate="mean", bookings="size").reset_index()
    order = ["0-7", "8-30", "31-90", "91-180", "181-365", "365+"]
    by_band = {r["value"]: r for r in rate_table(banded, "lead_time_band", t)}
    save_json({
        "rows": int(len(trval)), "cancel_rate": float(trval[t].mean()),
        "rate_by_category": {c: rate_table(trval, c, t, top=8)
                             for c in ["hotel", "deposit_type", "market_segment", "customer_type"]},
        "rate_by_lead_time": [by_band[b] for b in order if b in by_band],
        "lead_time_median": {"cancelled": float(trval.loc[trval[t] == 1, "lead_time"].median()),
                             "kept": float(trval.loc[trval[t] == 0, "lead_time"].median())},
        "monthly": monthly.to_dict("records"),
    }, ctx.metrics_dir / "eda.json")
    return {"figures": 9}


def stage_leakage_audit(cfg: Config, ctx: RunContext) -> dict:
    clean, splits = load_clean_and_splits(ctx)
    train, val = clean.iloc[splits["train"]], clean.iloc[splits["validation"]]
    evidence = tier3_evidence(train_val_rows(clean, splits), cfg)
    ablation = leakage_ablation(train, val, cfg)
    raw = load_raw(raw_path(ctx))
    if cfg.is_fast:
        raw = raw.sample(n=min(len(raw), cfg.sample_rows * 2), random_state=cfg.seed)
    dedup = dedup_sensitivity(raw, cfg, ablation["ladder"][1]["val_roc_auc"])
    duplicates = duplicate_decomposition(raw, clean, splits, cfg, repeats=1 if cfg.is_fast else 3)
    save_json({"evidence": evidence, "ablation": ablation, "dedup": dedup, "duplicates": duplicates},
              ctx.metrics_dir / "leakage.json")
    finish_figure(plot_tier3_evidence(evidence), "leakage_tier3_evidence", ctx)
    finish_figure(plot_leakage_ablation(ablation, dedup), "leakage_ablation", ctx)
    finish_figure(plot_duplicate_decomposition(duplicates), "leakage_duplicates", ctx)
    return {"ablation": ablation, "dedup": dedup, "duplicates": duplicates}


def frames_from_disk(cfg: Config, ctx: RunContext) -> dict:
    clean, splits = load_clean_and_splits(ctx)
    return build_model_frames(clean, splits, cfg)


def stage_train(cfg: Config, ctx: RunContext) -> dict:
    f = frames_from_disk(cfg, ctx)
    table = evaluate_candidates(f["X_train"], f["y_train"], f["X_validation"], f["y_validation"],
                                f["cats"], f["nums"], cfg)
    save_json(table.to_dict("records"), ctx.metrics_dir / "candidates.json")
    finish_figure(plot_candidates(table), "models_candidates", ctx)
    encodings = compare_encodings(f["X_train"], f["y_train"], f["cats"], f["nums"], cfg)
    imbalance = compare_imbalance(f["X_train"], f["y_train"], f["X_validation"], f["y_validation"],
                                  f["cats"], f["nums"], cfg)
    clean, splits = load_clean_and_splits(ctx)
    additions = compare_feature_sets(clean.iloc[splits["train"]], cfg)
    save_json({"encodings": encodings.to_dict("records"), "imbalance": imbalance.to_dict("records"),
               "feature_additions": additions.to_dict("records")}, ctx.metrics_dir / "experiments.json")
    finish_figure(plot_experiments(encodings, imbalance), "models_experiments", ctx)
    return {"best": table.iloc[0]["model"]}


def stage_tune(cfg: Config, ctx: RunContext) -> dict:
    f = frames_from_disk(cfg, ctx)
    tuning = tune_all(f["X_train"], f["y_train"], f["cats"], f["nums"], cfg)
    save_json(tuning, ctx.metrics_dir / "tuning.json")
    finish_figure(plot_search_comparison(tuning), "tuning_strategies", ctx)
    return {k: v["best_score"] for k, v in tuning.items() if isinstance(v, dict)}


def stage_select(cfg: Config, ctx: RunContext) -> dict:
    """Refit each family's best setting on all training rows and choose the champion on validation."""
    f = frames_from_disk(cfg, ctx)
    tuning = load_json(ctx.metrics_dir / "tuning.json")
    fitted = fit_tuned_models(tuning, f, cfg)
    champion = pick_champion(fitted, cfg)
    dummy = make_model("dummy", f["cats"], f["nums"], cfg).fit(f["X_train"], f["y_train"])
    test_models = {"dummy": dummy, "logreg": fitted["logreg"]["model"], "champion": fitted[champion["name"]]["model"]}
    joblib.dump(test_models, ctx.artifacts_dir / "test_models.joblib")
    save_json(champion, ctx.metrics_dir / "champion.json")
    return {k: v for k, v in champion.items() if k != "comparison"}


def stage_diagnose(cfg: Config, ctx: RunContext) -> dict:
    clean, splits = load_clean_and_splits(ctx)
    f = build_model_frames(clean, splits, cfg)
    champion = load_json(ctx.metrics_dir / "champion.json")
    model = joblib.load(ctx.artifacts_dir / "test_models.joblib")["champion"]
    diag = run_diagnostics(model, champion, f, cfg)
    temporal = temporal_check(train_val_rows(clean, splits), champion["params"] if champion["name"] == "lgbm" else {},
                              cfg)
    save_json(strip_arrays(diag), ctx.metrics_dir / "diagnostics.json")
    save_json(strip_arrays(temporal), ctx.metrics_dir / "temporal.json")
    y = f["y_validation"]
    finish_figure(plot_learning_curve(diag["learning_curve"]), "diag_learning_curve", ctx)
    if diag["validation_curve"]:
        finish_figure(plot_validation_curve(diag["validation_curve"]), "diag_validation_curve", ctx)
    finish_figure(plot_calibration(y, diag["val_proba_raw"], diag["val_proba"]), "diag_calibration", ctx)
    finish_figure(plot_cost_curve(diag["cost_curve"], diag["threshold"], cfg.costs.false_negative,
                                  cfg.costs.false_positive), "diag_threshold_cost", ctx)
    finish_figure(plot_confusion(diag["val_confusion"], "Validation confusion matrix"), "diag_confusion_val", ctx)
    finish_figure(plot_roc_pr({"champion": (y, diag["val_proba"])}, "Champion on validation"), "diag_roc_pr_val", ctx)
    finish_figure(plot_operating_points(y, diag["val_proba"], diag["val_operating_metrics"],
                                        "Operating points (validation)"), "diag_operating_points", ctx)
    finish_figure(plot_gains(diag["val_gains"], "Risk groups (validation)"), "diag_gains", ctx)
    finish_figure(plot_capacity(diag["val_capacity"], "Weekly cancellations per hotel (validation)"),
                  "diag_capacity", ctx)
    finish_figure(plot_segments(diag["segments"]), "diag_segments", ctx)
    finish_figure(plot_temporal(temporal), "diag_temporal", ctx)
    return {"threshold": diag["threshold"], "calibrated": diag["calibration"]["used"]}


def stage_evaluate(cfg: Config, ctx: RunContext) -> dict:
    f = frames_from_disk(cfg, ctx)
    models = joblib.load(ctx.artifacts_dir / "test_models.joblib")
    calibration = load_json(ctx.metrics_dir / "diagnostics.json")["calibration"]
    test = final_test_pass(models, calibration, f, cfg)
    save_json(strip_arrays(test), ctx.metrics_dir / "test.json")
    y = f["y_test"]
    finish_figure(plot_roc_pr({n: (y, r["proba"]) for n, r in test["models"].items()}, "Test set"),
                  "test_roc_pr", ctx)
    finish_figure(plot_confusion(test["models"]["champion"]["confusion"], "Test confusion matrix (champion)"),
                  "test_confusion", ctx)
    finish_figure(plot_test_results(test), "test_results", ctx)
    champ = test["models"]["champion"]
    finish_figure(plot_operating_points(y, champ["proba"], champ["operating_metrics"], "Operating points (test)"),
                  "test_operating_points", ctx)
    finish_figure(plot_gains(champ["gains"], "Risk groups (test)"), "test_gains", ctx)
    finish_figure(plot_capacity(champ["capacity"], "Weekly cancellations per hotel (test)"), "test_capacity", ctx)
    return strip_arrays(test["success"])


def stage_explain(cfg: Config, ctx: RunContext) -> dict:
    f = frames_from_disk(cfg, ctx)
    model = joblib.load(ctx.artifacts_dir / "test_models.joblib")["champion"]
    perm = permutation_importance_frame(model, f["X_validation"], f["y_validation"], cfg)
    values = shap_explanation(model, f["X_validation"], cfg)
    shap_imp = shap_importance_frame(values)
    save_json({"permutation": perm.to_dict("records"), "shap": shap_imp.to_dict("records")},
              ctx.metrics_dir / "explain.json")
    finish_figure(plot_importance(perm, "importance_mean", "importance_std",
                                  "Permutation importance (drop in validation ROC-AUC)"), "explain_permutation", ctx)
    finish_figure(plot_shap_summary(values), "explain_shap_summary", ctx)
    finish_figure(plot_shap_waterfall(values, int(np.argmax(values.values.sum(1)))), "explain_shap_waterfall", ctx)
    return {"top_feature": perm.iloc[0]["feature"]}


def stage_export(cfg: Config, ctx: RunContext) -> dict:
    clean, splits = load_clean_and_splits(ctx)
    f = build_model_frames(clean, splits, cfg)
    model = joblib.load(ctx.artifacts_dir / "test_models.joblib")["champion"]
    champion = load_json(ctx.metrics_dir / "champion.json")
    diag = load_json(ctx.metrics_dir / "diagnostics.json")
    test = load_json(ctx.metrics_dir / "test.json")
    out_dir = ctx.artifacts_dir / "export"
    onnx_path = export_onnx(model, len(f["cats"]) + len(f["nums"]), out_dir / "model.onnx")
    metadata = build_metadata(model, f["cats"], f["nums"], f["sources"], diag["calibration"], diag["threshold"],
                              test["models"]["champion"]["metrics"], champion, onnx_path, cfg)
    save_json(metadata, out_dir / "metadata.json")
    sample = f["raw_validation"].head(200)
    expected = apply_calibration(model.predict_proba(f["X_validation"].head(200))[:, 1], diag["calibration"])
    got = service_style_proba(sample, metadata, onnx_path)
    parity = float(np.max(np.abs(expected - got)))
    sample.to_pickle(out_dir / "parity_rows.pkl")
    np.save(out_dir / "parity_expected.npy", expected)
    deployment = {"format": "onnx", "model_kb": round(onnx_path.stat().st_size / 1024, 1),
                  "model_version": metadata["model_version"], "parity_rows": len(sample),
                  "parity_max_abs_diff": parity, "parity_ok": parity <= 1e-4,
                  "vocabulary_sizes": {k: len(v) for k, v in category_vocabulary(model, f["cats"]).items()}}
    save_json(deployment, ctx.metrics_dir / "deployment.json")
    if parity > 1e-4:
        raise RuntimeError(f"export parity failed: max difference {parity:.2e} > 1e-4")
    return deployment


def stage_report(cfg: Config, ctx: RunContext) -> dict:
    m = ctx.metrics_dir
    clean, splits = load_clean_and_splits(ctx)
    cleaning, leakage, explain = load_json(m / "cleaning.json"), load_json(m / "leakage.json"), load_json(m / "explain.json")
    parts = {
        "audit": load_json(m / "data_audit.json"), "cleaning_log": cleaning["cleaning_log"],
        "rows_after_cleaning": cleaning["rows_after_cleaning"],
        "duplicate_rows_cancel_rate": cleaning["duplicate_rows_cancel_rate"],
        "split_summary": load_json(m / "split.json"), "frames": build_model_frames(clean, splits, cfg),
        "leakage": leakage["ablation"], "dedup": leakage["dedup"], "candidates": load_json(m / "candidates.json"),
        "quality_audit": load_json(m / "quality_audit.json"), "duplicates": leakage["duplicates"],
        "experiments": load_json(m / "experiments.json") if (m / "experiments.json").exists() else None,
        "tuning": load_json(m / "tuning.json"), "champion": load_json(m / "champion.json"),
        "diagnostics": load_json(m / "diagnostics.json"), "temporal": load_json(m / "temporal.json"),
        "permutation": pd.DataFrame(explain["permutation"]), "shap_importance": pd.DataFrame(explain["shap"]),
        "test": load_json(m / "test.json"),
        "deployment": load_json(m / "deployment.json") if (m / "deployment.json").exists() else None,
    }
    report = build_metrics_report(parts, cfg)
    save_json(report, m / "metrics.json")
    return {"metrics_file": str(m / "metrics.json")}


STAGE_FUNCTIONS = {
    "download": stage_download, "validate": stage_validate, "clean": stage_clean, "split": stage_split,
    "eda": stage_eda, "leakage-audit": stage_leakage_audit, "train": stage_train, "tune": stage_tune, "select": stage_select,
    "diagnose": stage_diagnose, "evaluate": stage_evaluate, "explain": stage_explain, "export": stage_export,
    "report": stage_report,
}


def run_stage(name: str, cfg: Config, ctx: RunContext) -> dict:
    set_plot_style()
    ctx.ensure()
    start = time.time()
    LOGGER.info("stage %s: start%s", name, " (fast mode)" if cfg.is_fast else "")
    result = STAGE_FUNCTIONS[name](cfg, ctx)
    seconds = round(time.time() - start, 1)
    LOGGER.info("stage %s: done in %.1fs", name, seconds)
    write_manifest(ctx, name, dataclasses.asdict(cfg), {"seconds": seconds, "result": strip_arrays(result)})
    return result


def run_all(cfg: Config, ctx: RunContext) -> None:
    for name in STAGES:
        run_stage(name, cfg, ctx)


def run_latency(cfg: Config, ctx: RunContext, url: str, n_rows: int = 300) -> dict:
    """Replay validation bookings against a running API; compare with the exported model offline."""
    out_dir = ctx.artifacts_dir / "export"
    metadata = load_json(out_dir / "metadata.json")
    clean, splits = load_clean_and_splits(ctx)
    rows = clean.iloc[splits["validation"]].head(n_rows)
    expected = service_style_proba(rows, metadata, out_dir / "model.onnx")
    result = replay_requests(url, rows, expected)
    save_json(result, ctx.metrics_dir / "monitoring.json")
    set_plot_style()
    finish_figure(plot_latency(result), "monitor_latency", ctx)
    return result


def run_drift(cfg: Config, ctx: RunContext) -> dict:
    """Feature drift between arrivals up to the cut-off year and the year after (no labels used)."""
    clean, _ = load_clean_and_splits(ctx)
    sources = deployed_sources(cfg)
    feats = build_feature_frame(clean, sources)
    year = clean["arrival_date_year"].to_numpy()
    cut = cfg.split.temporal_train_until_year
    cats, _ = split_feature_types(list(feats.columns))
    report = drift_report(feats[year <= cut], feats[year == cut + 1], cats)
    save_json(report.to_dict("records"), ctx.metrics_dir / "drift.json")
    set_plot_style()
    finish_figure(plot_drift(report, f"Feature drift: arrivals up to {cut} vs {cut + 1}"), "monitor_drift", ctx)
    return {"largest_shift": report.iloc[0].to_dict()}
