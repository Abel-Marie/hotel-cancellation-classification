"""Sanity checks on the raw data. Hard failures raise; everything else is reported."""

from __future__ import annotations

import pandas as pd

from .config import Config

RAW_NUMERIC_COLUMNS = [
    "is_canceled", "lead_time", "arrival_date_year", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights", "adults",
    "children", "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "agent", "company",
    "days_in_waiting_list", "adr", "required_car_parking_spaces", "total_of_special_requests",
]
RAW_TEXT_COLUMNS = [
    "hotel", "arrival_date_month", "meal", "country", "market_segment", "distribution_channel",
    "reserved_room_type", "assigned_room_type", "deposit_type", "customer_type",
    "reservation_status", "reservation_status_date",
]


def validate_raw(df: pd.DataFrame, cfg: Config) -> dict:
    """Check shape, columns, types and the target; return a structural audit report."""
    data_cols = [c for c in df.columns if c != "row_id"]
    problems = []
    if len(df) != cfg.data.expected_rows:
        problems.append(f"expected {cfg.data.expected_rows} rows, found {len(df)}")
    if len(data_cols) != cfg.data.expected_columns:
        problems.append(f"expected {cfg.data.expected_columns} columns, found {len(data_cols)}")
    missing_cols = sorted(set(RAW_NUMERIC_COLUMNS + RAW_TEXT_COLUMNS) - set(data_cols))
    if missing_cols:
        problems.append(f"missing columns: {missing_cols}")
    for col in RAW_NUMERIC_COLUMNS:
        if col in df and not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"{col} should be numeric")
    if set(df[cfg.target].dropna().unique()) - {0, 1}:
        problems.append("target is not binary")
    if problems:
        raise ValueError("raw data failed validation: " + "; ".join(problems))
    missing = df[data_cols].isna().sum()
    return {
        "rows": int(len(df)),
        "columns": int(len(data_cols)),
        "missing_values": {k: int(v) for k, v in missing[missing > 0].items()},
        "exact_duplicate_rows": int(df.duplicated(subset=data_cols).sum()),
        "target_rate": float(df[cfg.target].mean()),
        "arrival_years": sorted(int(y) for y in df["arrival_date_year"].unique()),
    }


def quality_audit(df: pd.DataFrame, cfg: Config) -> dict:
    """Evidence behind the cleaning rules: logical inconsistencies and outliers per numeric column.

    Structural only (no row is judged by its outcome), so it can look at every row.
    """
    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    history = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    limit = cfg.cleaning.max_valid_adr
    checks = [
        ("Booking for zero guests", guests == 0, "removed", "A booking for nobody is a data-entry error."),
        ("Negative price per night", df["adr"] < 0, "removed", "Impossible."),
        (f"Price above {limit:.0f} per night", df["adr"] > limit, "removed",
         "A single row, far above every other price in the data."),
        ("Meal code 'Undefined'", df["meal"] == "Undefined", "merged into SC",
         "The dataset paper says both codes mean no meal package."),
        ("Children count missing", df["children"].isna(), "set to 0", "Missing means none."),
        ("Country missing", df["country"].isna(), "own category",
         "Country is not a deployed feature; it is used only in the leakage audit."),
        ("Zero nights booked", nights == 0, "kept", "Day-use rooms and same-day departures exist in hotels."),
        ("Price of exactly 0", df["adr"] == 0, "kept", "Complimentary stays and staff bookings are real."),
        ("No adults, only children or babies", (df["adults"] == 0) & (guests > 0), "kept",
         "Usually an extra room attached to a family booking."),
        ("Market segment or channel 'Undefined'",
         (df["market_segment"] == "Undefined") | (df["distribution_channel"] == "Undefined"), "kept",
         "A handful of rows; the encoder groups rare categories together."),
        ("Not flagged as repeated guest, but has earlier bookings", (df["is_repeated_guest"] == 0) & (history > 0),
         "kept", "The two fields are recorded separately; the model sees both."),
        ("Flagged as repeated guest, but no earlier bookings", (df["is_repeated_guest"] == 1) & (history == 0),
         "kept", "Same reason."),
    ]
    inconsistencies = [{"check": name, "rows": int(mask.sum()), "share": float(mask.mean()), "action": action,
                        "why": why} for name, mask, action, why in checks]
    decisions = {
        "lead_time": "kept: bookings a year or more ahead are real (groups, events); the model uses log(1 + lead time)",
        "adr": f"kept up to {limit:.0f}; the single value above it is removed",
        "stays_in_week_nights": "kept: long stays are real",
        "stays_in_weekend_nights": "kept: long stays are real",
        "adults": "kept: large parties are group bookings",
        "children": "kept",
        "babies": "kept",
        "previous_cancellations": "kept: a strong, legitimate signal",
        "previous_bookings_not_canceled": "kept: a strong, legitimate signal",
    }
    outliers = []
    for col, decision in decisions.items():
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        fence = q3 + 1.5 * (q3 - q1)
        outliers.append({"column": col, "median": float(s.median()), "p99": float(s.quantile(0.99)),
                         "max": float(s.max()), "upper_fence": float(fence),
                         "rows_above_fence": int((s > fence).sum()), "share_above_fence": float((s > fence).mean()),
                         "decision": decision})
    return {"inconsistencies": inconsistencies, "outliers": outliers}
