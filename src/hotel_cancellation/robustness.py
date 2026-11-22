"""Checks that the headline numbers are not artefacts: duplicates and time.

Both use training and validation rows only; the test set is not touched.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from .cleaning import clean_bookings
from .config import Config, deployed_sources
from .features import build_feature_frame, split_feature_types
from .models import make_model
from .split import stratified_split, temporal_positions


def dedup_sensitivity(raw: pd.DataFrame, cfg: Config, deduplicated_score: float) -> dict:
    """Validation ROC-AUC when duplicates are kept, compared with the deduplicated run.

    With duplicates kept, copies of training rows appear in validation, so the
    score is inflated. The gap is the size of that illusion.
    """
    kept, _ = clean_bookings(raw, cfg, drop_duplicates=False)
    splits = stratified_split(kept, cfg.target, cfg.split.val_size, cfg.split.test_size, cfg.seed)
    sources = deployed_sources(cfg)
    train, val = kept.iloc[splits["train"]], kept.iloc[splits["validation"]]
    Xtr, Xva = build_feature_frame(train, sources), build_feature_frame(val, sources)
    cats, nums = split_feature_types(list(Xtr.columns))
    model = make_model("lgbm", cats, nums, cfg).fit(Xtr, train[cfg.target])
    with_dups = float(roc_auc_score(val[cfg.target], model.predict_proba(Xva)[:, 1]))
    val_rows_seen_in_train = int(val.drop(columns="row_id").merge(
        train.drop(columns="row_id").drop_duplicates(), how="inner").shape[0])
    return {"val_roc_auc_deduplicated": deduplicated_score, "val_roc_auc_duplicates_kept": with_dups,
            "inflation": with_dups - deduplicated_score,
            "share_of_val_rows_also_in_train_when_kept": val_rows_seen_in_train / len(val)}


# What the hotel knows when the booking is made: the raw columns behind the deployed features.
BOOKING_TIME_COLUMNS = [
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "meal",
    "market_segment", "distribution_channel", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type", "deposit_type", "agent", "company",
    "customer_type", "adr",
]


def residual_overlap(clean: pd.DataFrame, splits: dict) -> dict:
    """Held-out rows whose booking-time inputs also occur in the training set.

    Exact duplicates are removed on every column, so two rows that differ only in a column the
    model never sees are both kept and can land on different sides of the split.
    """
    train = clean.iloc[splits["train"]][BOOKING_TIME_COLUMNS].drop_duplicates().assign(in_training=True)
    result = {"columns_compared": len(BOOKING_TIME_COLUMNS)}
    for part in ("validation", "test"):
        held_out = clean.iloc[splits[part]][BOOKING_TIME_COLUMNS]
        shared = int(held_out.merge(train, on=BOOKING_TIME_COLUMNS, how="left")["in_training"].notna().sum())
        result[part] = {"rows": int(len(held_out)), "rows_sharing_inputs_with_training": shared,
                        "share": shared / len(held_out)}
    return result


def duplicate_decomposition(raw: pd.DataFrame, clean: pd.DataFrame, splits: dict, cfg: Config,
                            repeats: int = 3) -> dict:
    """Why does keeping the duplicates raise the score? Memorisation, or a different population?

    Uses training and validation bookings only: every row that shares its booking-time inputs
    with a test booking is removed first. Duplicates are then kept, and LightGBM (default
    settings, deployed features) is scored three ways, averaged over ``repeats`` splits:

    1. random split: copies of a booking can sit on both sides (what a careless pipeline does);
    2. grouped split: all rows with the same booking-time inputs stay on one side, so nothing
       can be memorised, and every row is scored;
    3. the same grouped split, scoring one row per distinct record (the deduplicated view).

    1 minus 2 is memorisation. 2 minus 3 is the effect of counting the repeated rows, which are
    a different and easier population. The last score drops the non-refundable bookings, which
    are almost always cancelled, to show where that easy signal sits.
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    t = cfg.target
    kept, _ = clean_bookings(raw, cfg, drop_duplicates=False)
    record_cols = [c for c in kept.columns if c != "row_id"]
    kept = kept.assign(inputs=kept.groupby(BOOKING_TIME_COLUMNS, dropna=False, sort=False).ngroup())
    test_inputs = clean.iloc[splits["test"]][BOOKING_TIME_COLUMNS].drop_duplicates().merge(
        kept[BOOKING_TIME_COLUMNS + ["inputs"]].drop_duplicates(), on=BOOKING_TIME_COLUMNS, how="left")["inputs"]
    pool = kept[~kept["inputs"].isin(set(test_inputs))].reset_index(drop=True)
    copies = pool.groupby(record_cols, dropna=False, sort=False)[t].transform("size").to_numpy()
    repeated, single = pool[copies > 1], pool[copies == 1]
    sources = deployed_sources(cfg)
    X, y = build_feature_frame(pool, sources), pool[t]
    cats, nums = split_feature_types(list(X.columns))
    holdout = cfg.split.val_size / (1 - cfg.split.test_size)
    positions = pd.RangeIndex(len(pool)).to_numpy()
    scores: dict[str, list[float]] = {"random": [], "grouped": [], "grouped_one_per_record": [],
                                      "grouped_without_non_refundable": []}
    for i in range(repeats):
        seed = cfg.seed + i
        fit, ev = train_test_split(positions, test_size=holdout, stratify=y, random_state=seed)
        model = make_model("lgbm", cats, nums, cfg).fit(X.iloc[fit], y.iloc[fit])
        scores["random"].append(roc_auc_score(y.iloc[ev], model.predict_proba(X.iloc[ev])[:, 1]))
        fit, ev = next(GroupShuffleSplit(1, test_size=holdout, random_state=seed).split(positions, groups=pool["inputs"]))
        model = make_model("lgbm", cats, nums, cfg).fit(X.iloc[fit], y.iloc[fit])
        proba, part = model.predict_proba(X.iloc[ev])[:, 1], pool.iloc[ev]
        truth = y.iloc[ev].to_numpy()
        first = ~part.duplicated(subset=record_cols).to_numpy()
        refundable = (part["deposit_type"] != "Non Refund").to_numpy()
        scores["grouped"].append(roc_auc_score(truth, proba))
        scores["grouped_one_per_record"].append(roc_auc_score(truth[first], proba[first]))
        scores["grouped_without_non_refundable"].append(roc_auc_score(truth[refundable], proba[refundable]))
    mean = {k: float(sum(v) / len(v)) for k, v in scores.items()}

    def profile(frame: pd.DataFrame) -> dict:
        return {"rows": int(len(frame)), "cancel_rate": float(frame[t].mean()),
                "share_groups_segment": float((frame["market_segment"] == "Groups").mean()),
                "share_non_refundable": float((frame["deposit_type"] == "Non Refund").mean())}

    non_refundable = pool[pool["deposit_type"] == "Non Refund"]
    return {"rows": int(len(pool)), "repeats": repeats,
            "roc_auc_random_split": mean["random"], "roc_auc_grouped_split": mean["grouped"],
            "roc_auc_grouped_one_per_record": mean["grouped_one_per_record"],
            "roc_auc_grouped_without_non_refundable": mean["grouped_without_non_refundable"],
            "memorisation": mean["random"] - mean["grouped"],
            "population": mean["grouped"] - mean["grouped_one_per_record"],
            "rows_with_copies": profile(repeated), "rows_without_copies": profile(single),
            "non_refundable": {"rows": int(len(non_refundable)), "cancel_rate": float(non_refundable[t].mean())}}


def temporal_check(trval: pd.DataFrame, params: dict, cfg: Config) -> dict:
    """Fit on earlier arrivals, score the next year, month by month."""
    fit_pos, eval_pos = temporal_positions(trval, cfg.split.temporal_train_until_year)
    sources = deployed_sources(cfg)
    early, late = trval.iloc[fit_pos], trval.iloc[eval_pos]
    Xe, Xl = build_feature_frame(early, sources), build_feature_frame(late, sources)
    cats, nums = split_feature_types(list(Xe.columns))
    model = make_model("lgbm", cats, nums, cfg, **params).fit(Xe, early[cfg.target])
    proba = model.predict_proba(Xl)[:, 1]
    frame = pd.DataFrame({"month": late["arrival_date_month"].to_numpy(), "y": late[cfg.target].to_numpy(),
                          "p": proba})
    order = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                         "August", "September", "October", "November", "December"])}
    by_month = []
    for month, part in sorted(frame.groupby("month"), key=lambda kv: order[kv[0]]):
        by_month.append({"month": month, "rows": int(len(part)), "cancel_rate": float(part["y"].mean()),
                         "roc_auc": float(roc_auc_score(part["y"], part["p"]))})
    return {"fit_years": f"<= {cfg.split.temporal_train_until_year}",
            "eval_year": cfg.split.temporal_train_until_year + 1,
            "fit_rows": int(len(early)), "eval_rows": int(len(late)),
            "eval_roc_auc": float(roc_auc_score(late[cfg.target], proba)), "by_month": by_month,
            "model": model, "eval_proba": proba}
