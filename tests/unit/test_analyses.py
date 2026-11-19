"""The analyses added around the champion: operating points, costs, risk groups, weekly counts,
the data-quality audit, the duplicate decomposition, and the record of the first test evaluation."""

import json

import numpy as np
import pandas as pd

from hotel_cancellation.cleaning import clean_bookings
from hotel_cancellation.data import download_raw, gunzip
from hotel_cancellation.evaluation import (
    best_f1_threshold,
    capacity_report,
    cost_sensitivity,
    gains_table,
    metrics_at_points,
    operating_points,
)
from hotel_cancellation.robustness import BOOKING_TIME_COLUMNS, duplicate_decomposition
from hotel_cancellation.rowlevel import derive_features
from hotel_cancellation.split import stratified_split
from hotel_cancellation.validation import quality_audit


def scored(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Labels and informative, roughly calibrated probabilities."""
    rng = np.random.default_rng(seed)
    p = rng.beta(1.2, 3.0, n)
    return (rng.random(n) < p).astype(int), p


def test_operating_points_are_chosen_on_the_data_given(cfg):
    y, p = scored()
    points = operating_points(y, p, 0.61, cfg)
    assert [pt["name"] for pt in points] == ["cost-based (deployed)", "default 0.5", "best F1"]
    assert points[2]["threshold"] == best_f1_threshold(y, p)
    table = metrics_at_points(y, p, points).set_index("operating_point")
    assert table.loc["best F1", "f1"] >= table["f1"].max() - 1e-12
    assert table.loc["cost-based (deployed)", "flagged_share"] < table.loc["best F1", "flagged_share"]


def test_dearer_false_alarms_raise_the_threshold_and_lower_recall(cfg):
    y, p = scored()
    table = cost_sensitivity(y, p, y, p, cfg)
    assert list(table["false_alarm_cost"]) == sorted(cfg.evaluation.cost_sensitivity_false_positive)
    assert table["threshold"].is_monotonic_increasing
    assert table["recall"].is_monotonic_decreasing
    assert (table["cost_per_booking"] <= table["cost_without_model"] + 1e-12).all()


def test_gains_table_concentrates_cancellations_in_the_first_groups():
    y, p = scored()
    gains = gains_table(y, p)
    assert len(gains) == 10 and gains["bookings"].sum() == len(y)
    assert gains.iloc[0]["cancel_rate"] > gains.iloc[-1]["cancel_rate"]
    assert abs(gains.iloc[-1]["cumulative_share_of_cancellations"] - 1.0) < 1e-12
    assert gains.iloc[0]["lift"] > 1


def test_capacity_report_is_exact_for_a_perfect_forecast(sample_raw):
    rows = sample_raw.head(400)
    y = rows["is_canceled"].to_numpy()
    perfect = capacity_report(rows, y, y.astype(float), base_rate=float(y.mean()))
    assert perfect["wape_model"] == 0 and abs(perfect["bias"]) < 1e-12
    assert perfect["wape_base_rate"] > 0
    assert perfect["by_week"]["bookings"].sum() == len(rows)


def test_quality_audit_counts_match_the_cleaning_rules(sample_raw, cfg):
    audit = quality_audit(sample_raw, cfg)
    found = {r["check"]: r for r in audit["inconsistencies"]}
    _, log = clean_bookings(sample_raw, cfg, drop_duplicates=False)
    removed = {r["rule"]: r["rows_removed"] for r in log}
    assert found["Booking for zero guests"]["rows"] == removed["drop_zero_guests"]
    assert {r["action"] for r in audit["inconsistencies"]} >= {"removed", "kept"}
    assert all(r["rows_above_fence"] >= 0 and r["decision"] for r in audit["outliers"])


def test_duplicate_decomposition_never_scores_a_test_booking(sample_raw, cfg):
    clean, _ = clean_bookings(sample_raw, cfg)
    splits = stratified_split(clean, cfg.target, cfg.split.val_size, cfg.split.test_size, cfg.seed)
    result = duplicate_decomposition(sample_raw, clean, splits, cfg, repeats=1)
    kept, _ = clean_bookings(sample_raw, cfg, drop_duplicates=False)
    test_inputs = clean.iloc[splits["test"]][BOOKING_TIME_COLUMNS].drop_duplicates()
    shared = kept.merge(test_inputs, on=BOOKING_TIME_COLUMNS, how="inner")
    assert result["rows"] == len(kept) - len(shared)
    for key in ("roc_auc_random_split", "roc_auc_grouped_split", "roc_auc_grouped_one_per_record"):
        assert 0.5 < result[key] <= 1.0
    assert abs(result["memorisation"] - (result["roc_auc_random_split"] - result["roc_auc_grouped_split"])) < 1e-12


def test_candidate_features_use_the_booking_date(sample_raw):
    row = {**sample_raw.iloc[0].to_dict(), "arrival_date_year": 2016, "arrival_date_month": "March",
           "arrival_date_day_of_month": 10, "lead_time": 9, "company": float("nan")}
    feats = derive_features(row, ["candidate"])
    assert feats["booking_weekday"] == 1.0  # 1 March 2016 was a Tuesday
    assert feats["booking_day_of_year"] == 61.0 and feats["arrival_day_of_month"] == 10.0
    assert feats["company_id"] == "none"


def test_first_test_evaluation_matches_the_archived_results(cfg, root):
    archived = json.loads((root / "reports" / "metrics" / "history" / "2026-11-10_first_test_evaluation.json").read_text())
    champ = archived["models"]["champion"]
    first = cfg.first_test_evaluation
    for key in ("roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"):
        assert abs(getattr(first, key) - champ["metrics"][key]) < 5e-5, key
    assert abs(first.roc_auc_ci[0] - champ["ci"]["roc_auc"]["low"]) < 5e-5
    assert abs(first.roc_auc_ci[1] - champ["ci"]["roc_auc"]["high"]) < 5e-5
    first_champion = json.loads((root / "reports" / "metrics" / "history" / "2026-11-10_first_champion.json").read_text())
    assert abs(first.validation_roc_auc - first_champion["val_roc_auc"]) < 5e-5


def test_download_reuses_a_verified_copy_without_the_network(cfg, root, tmp_path):
    gunzip(root / cfg.data.mirror_path, tmp_path / "hotels.csv")
    info = download_raw(cfg, tmp_path, repo_root=None)
    assert info["source_kind"] == "cache" and info["sha256"] == cfg.data.sha256
    assert len(pd.read_csv(info["path"], usecols=["hotel"])) == cfg.data.expected_rows
