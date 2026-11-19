import numpy as np

from hotel_cancellation.cleaning import clean_bookings
from hotel_cancellation.split import rows_hash, split_summary, stratified_split


def test_cleaning_rules(sample_raw, cfg):
    df = sample_raw.copy()
    df.loc[0, ["adults", "children", "babies"]] = [0, 0, 0]
    df.loc[1, "adr"] = -5
    df.loc[2, "adr"] = 5000
    dup = df.iloc[[10]].copy()
    dup["row_id"] = 999_999
    df = __import__("pandas").concat([df, dup], ignore_index=True)
    clean, log = clean_bookings(df, cfg)
    removed = {entry["rule"]: entry["rows_removed"] for entry in log}
    assert removed["drop_negative_price"] == 1 and removed["drop_extreme_price"] == 1
    assert removed["drop_zero_guests"] >= 1 and removed["drop_exact_duplicates"] >= 1
    assert clean["children"].isna().sum() == 0 and clean["country"].isna().sum() == 0
    assert "Undefined" not in set(clean["meal"])
    assert log[-1]["rows_after"] == len(clean)


def test_split_is_disjoint_stratified_and_deterministic(sample_raw, cfg):
    clean, _ = clean_bookings(sample_raw, cfg)
    a = stratified_split(clean, cfg.target, 0.15, 0.15, 42)
    b = stratified_split(clean, cfg.target, 0.15, 0.15, 42)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not set(a["train"]) & set(a["test"]) and not set(a["validation"]) & set(a["test"])
    assert sum(len(v) for v in a.values()) == len(clean)
    s = split_summary(clean, a, cfg.target)
    assert abs(s["train"]["positive_rate"] - s["test"]["positive_rate"]) < 0.03
    assert rows_hash([3, 1, 2]) == rows_hash([1, 2, 3])
