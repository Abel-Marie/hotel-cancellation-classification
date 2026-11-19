"""Cleaning rules. Each rule is logged with the number of rows it removed or changed.

Only fixed, documented rules live here; nothing is learned from the data, so
cleaning can safely run before the split.
"""

from __future__ import annotations

import pandas as pd

from .config import Config


def clean_bookings(df: pd.DataFrame, cfg: Config, drop_duplicates: bool | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the cleaning rules in order and return the cleaned frame and a log."""
    drop_dups = cfg.cleaning.drop_duplicates if drop_duplicates is None else drop_duplicates
    out = df.copy()
    log: list[dict] = []

    def record(rule: str, before: int, why: str, changed: int | None = None) -> None:
        log.append({"rule": rule, "rows_before": before, "rows_after": len(out),
                    "rows_removed": before - len(out), "values_changed": changed, "why": why})

    data_cols = [c for c in out.columns if c != "row_id"]
    if drop_dups:
        before = len(out)
        out = out[~out.duplicated(subset=data_cols, keep="first")]
        record("drop_exact_duplicates", before,
               "Identical rows would land in both train and test and inflate scores.")

    before = len(out)
    guests = out["adults"] + out["children"].fillna(0) + out["babies"]
    out = out[guests > 0]
    record("drop_zero_guests", before, "A booking for nobody is a data-entry error.")

    before = len(out)
    out = out[out["adr"] >= 0]
    record("drop_negative_price", before, "A negative nightly rate is impossible.")

    before = len(out)
    out = out[out["adr"] <= cfg.cleaning.max_valid_adr]
    record("drop_extreme_price", before,
           f"Rates above {cfg.cleaning.max_valid_adr:.0f} per night are entry errors in this data.")

    before = len(out)
    n = int(out["children"].isna().sum())
    out["children"] = out["children"].fillna(0)
    record("fill_children_missing_as_zero", before, "Missing children count means none.", n)

    n = int(out["country"].isna().sum())
    out["country"] = out["country"].fillna("UNK")
    record("fill_country_missing_as_UNK", len(out), "Kept as its own category.", n)

    n = int((out["meal"] == "Undefined").sum())
    out["meal"] = out["meal"].replace({"Undefined": cfg.cleaning.fill_meal_undefined_as})
    record("merge_meal_undefined_into_SC", len(out), "Dataset paper: both codes mean no meal package.", n)

    record("keep_agent_company_missing", len(out),
           "Missing agent or company means the booking had none; handled as a feature.", 0)
    return out.reset_index(drop=True), log
