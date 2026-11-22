"""Count the held-out bookings whose booking-time inputs also occur in the training set.

    python scripts/residual_overlap.py

Exact duplicates are removed on all 32 columns before the split. Two rows that differ only in a
column the model never sees (the assigned room, say) are therefore both kept, and can land on
different sides of the split with identical model inputs. This script counts them and writes
reports/metrics/pipeline/overlap.json. Needs the cleaned data and the split of a pipeline run.
"""

from __future__ import annotations

from pathlib import Path

from hotel_cancellation.artifacts import RunContext, save_json
from hotel_cancellation.pipeline import load_clean_and_splits
from hotel_cancellation.robustness import residual_overlap

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ctx = RunContext(ROOT).ensure()
    clean, splits = load_clean_and_splits(ctx)
    result = residual_overlap(clean, splits)
    save_json(result, ctx.metrics_dir / "overlap.json")
    for part in ("validation", "test"):
        r = result[part]
        print(f"{part}: {r['rows_sharing_inputs_with_training']:,} of {r['rows']:,} rows ({r['share']:.1%}) share all "
              f"{result['columns_compared']} booking-time inputs with a training row")


if __name__ == "__main__":
    main()
