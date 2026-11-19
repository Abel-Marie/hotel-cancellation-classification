"""Prove the notebook runs when it is the only file present (e.g. uploaded alone to Colab).

Copies the .ipynb into an empty temporary folder, outside the repository, and
executes it there with ML_FAST=1. Any error fails the check.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import execute  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    failures = 0
    for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
        with tempfile.TemporaryDirectory(prefix="standalone-") as tmp:
            copy = Path(tmp) / nb.name
            shutil.copy2(nb, copy)
            assert sorted(p.name for p in Path(tmp).iterdir()) == [nb.name]
            try:
                execute(copy, Path(tmp), {"ML_FAST": "1", "OUTPUT_DIR": str(Path(tmp) / "outputs")}, timeout=1800)
                print(f"PASS {nb.name} ran alone in {tmp}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {nb.name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
