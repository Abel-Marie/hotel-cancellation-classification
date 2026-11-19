import ast
import math
from pathlib import Path

import pytest

from hotel_cancellation import rowlevel
from hotel_cancellation.rowlevel import CATEGORICAL_FEATURES, derive_features

RECORD = {
    "hotel": "City Hotel", "lead_time": 99, "arrival_date_year": 2016, "arrival_date_month": "January",
    "arrival_date_day_of_month": 4, "stays_in_weekend_nights": 2, "stays_in_week_nights": 2, "adults": 2,
    "children": 1, "babies": 0, "meal": "BB", "market_segment": "Online TA", "distribution_channel": "TA/TO",
    "is_repeated_guest": 0, "previous_cancellations": 1, "previous_bookings_not_canceled": 3,
    "reserved_room_type": "A", "deposit_type": "No Deposit", "agent": 9.0, "company": float("nan"),
    "customer_type": "Transient", "adr": 90.0,
}


def test_base_features():
    f = derive_features(RECORD, ["base", "adr"])
    assert f["log_lead_time"] == pytest.approx(math.log1p(99))
    assert f["arrival_weekday"] == 0.0  # 4 January 2016 was a Monday
    assert f["month_sin"] == pytest.approx(0.0) and f["month_cos"] == pytest.approx(1.0)
    assert f["total_nights"] == 4 and f["weekend_share"] == 0.5
    assert f["total_guests"] == 3 and f["is_family"] == 1.0
    assert f["prior_cancel_ratio"] == pytest.approx(0.25) and f["has_booking_history"] == 1.0
    assert f["agent_id"] == "9" and f["has_company"] == 0.0
    assert f["adr_per_person"] == pytest.approx(30.0)


def test_missing_values_and_edge_cases():
    r = {**RECORD, "agent": None, "stays_in_weekend_nights": 0, "stays_in_week_nights": 0,
         "previous_cancellations": 0, "previous_bookings_not_canceled": 0, "adults": 0, "children": 0}
    f = derive_features(r, ["base", "adr"])
    assert f["agent_id"] == "none" and f["weekend_share"] == 0.0 and f["prior_cancel_ratio"] == 0.0
    assert f["adr_per_person"] == 90.0  # divided by at least one paying guest


def test_groups_are_stable_and_typed():
    f = derive_features(RECORD, ["base", "adr"])
    assert list(f) == list(derive_features(RECORD, ["base", "adr"]))
    for name, value in f.items():
        assert isinstance(value, str) if name in CATEGORICAL_FEATURES else isinstance(value, float), name


def test_rowlevel_uses_only_the_standard_library():
    tree = ast.parse(Path(rowlevel.__file__).read_text())
    modules = {n.module if isinstance(n, ast.ImportFrom) else a.name
               for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
               for a in (n.names if isinstance(n, ast.Import) else [n])}
    assert modules <= {"__future__", "math", "datetime"}
