"""Count the held-out bookings whose booking-time inputs also occur in the training set.

    python scripts/residual_overlap.py

Exact duplicates are removed on all 32 columns before the split. Two rows that differ only in a
column the model never sees (the assigned room, say) are therefore both kept, and can land on
different sides of the split with identical model inputs. This script counts them and writes
reports/metrics/pipeline/overlap.json. Needs the cleaned data and the split of a pipeline run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hotel_cancellation.artifacts import RunContext, save_json
from hotel_cancellation.pipeline import load_clean_and_splits
from hotel_cancellation.robustness import BOOKING_TIME_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def shared_with_training(clean: pd.DataFrame, splits: dict, part: str) -> dict:
    """Rows of one held-out part that have the same booking-time inputs as at least one training row."""
    train = clean.iloc[splits["train"]][BOOKING_TIME_COLUMNS].drop_duplicates()
    held_out = clean.iloc[splits[part]][BOOKING_TIME_COLUMNS]
    merged = held_out.merge(train.assign(in_training=True), on=BOOKING_TIME_COLUMNS, how="left")
    shared = int(merged["in_training"].notna().sum())
    return {"rows": int(len(held_out)), "rows_sharing_inputs_with_training": shared, "share": shared / len(held_out)}


def main() -> None:
    ctx = RunContext(ROOT).ensure()
    clean, splits = load_clean_and_splits(ctx)
    result = {"columns_compared": len(BOOKING_TIME_COLUMNS),
              "validation": shared_with_training(clean, splits, "validation"),
              "test": shared_with_training(clean, splits, "test")}
    save_json(result, ctx.metrics_dir / "overlap.json")
    for part in ("validation", "test"):
        r = result[part]
        print(f"{part}: {r['rows_sharing_inputs_with_training']:,} of {r['rows']:,} rows ({r['share']:.1%}) share all "
              f"{result['columns_compared']} booking-time inputs with a training row")


if __name__ == "__main__":
    main()
