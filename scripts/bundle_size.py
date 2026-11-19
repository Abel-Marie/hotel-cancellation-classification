"""Install service/requirements.txt into a temporary folder and report the size Vercel would bundle."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_MB = 250


def folder_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--target", tmp, "-r",
                               str(ROOT / "service" / "requirements.txt")])
        deps = folder_mb(Path(tmp))
    own = folder_mb(ROOT / "service")
    total = deps + own
    print(f"dependencies {deps:.1f} MB + service files {own:.1f} MB = {total:.1f} MB (target < {LIMIT_MB} MB)")
    return 0 if total < LIMIT_MB else 1


if __name__ == "__main__":
    sys.exit(main())
