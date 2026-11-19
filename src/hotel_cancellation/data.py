"""Download and load the raw bookings file."""

from __future__ import annotations

import gzip
import shutil
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .artifacts import file_sha256
from .config import Config
from .log import LOGGER


def fetch_url(url: str, dest: Path, timeout: int = 30, max_seconds: float | None = 120) -> None:
    """Download ``url`` to ``dest``.

    ``timeout`` only limits each read, so a connection that keeps trickling data never trips it.
    ``max_seconds`` is a deadline for the whole download; past it the caller moves on to the next source.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "hotel-cancellation/1.0"})
    start = time.monotonic()
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as fh:
            while chunk := resp.read(1 << 16):
                fh.write(chunk)
                if max_seconds is not None and time.monotonic() - start > max_seconds:
                    raise TimeoutError(f"download not finished after {max_seconds:.0f} s")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


def gunzip(source: Path, dest: Path) -> None:
    """Byte-exact decompression, so the result has the same checksum as the original file."""
    with gzip.open(source, "rb") as src, open(dest, "wb") as fh:
        shutil.copyfileobj(src, fh)


def download_raw(cfg: Config, dest_dir: Path, repo_root: Path | None = None) -> dict:
    """Obtain the raw CSV and verify its checksum, whatever the source.

    Order: a verified copy already on disk; the public URLs; the optional mirror URL; the
    compressed mirror inside the repository (when run from a clone). If every source failed (for
    example a slow connection cut off by the download deadline, with no mirror at hand), the
    first URL is retried without a deadline. Returns where the data came from.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "hotels.csv"
    if target.exists() and file_sha256(target) == cfg.data.sha256:
        LOGGER.info("raw data already on disk (%s), checksum verified", target.name)
        return {"path": str(target), "source_kind": "cache", "source": target.name, "sha256": cfg.data.sha256}
    attempts = [("url", u, 60.0) for u in cfg.data.sources]
    if cfg.data.mirror_url:
        attempts.append(("mirror_url", cfg.data.mirror_url, 60.0))
    if repo_root is not None and (Path(repo_root) / cfg.data.mirror_path).exists():
        attempts.append(("repo_mirror", str(Path(repo_root) / cfg.data.mirror_path), None))
    attempts.append(("url", cfg.data.sources[0], None))  # last resort: no deadline
    errors = []
    for kind, where, deadline in attempts:
        try:
            if kind == "repo_mirror":
                gunzip(Path(where), target)
            elif where.endswith(".gz"):
                tmp = dest_dir / "hotels.csv.gz"
                fetch_url(where, tmp, max_seconds=deadline)
                gunzip(tmp, target)
                tmp.unlink()
            else:
                fetch_url(where, target, max_seconds=deadline)
            digest = file_sha256(target)
            if digest != cfg.data.sha256:
                target.unlink()
                raise ValueError(f"checksum mismatch: {digest}")
            shown = where if kind != "repo_mirror" else cfg.data.mirror_path
            LOGGER.info("raw data from %s (%s), checksum verified", kind, shown)
            return {"path": str(target), "source_kind": kind, "source": shown, "sha256": digest}
        except Exception as exc:  # noqa: BLE001 - try the next source, report all failures
            errors.append(f"{kind} {where}: {exc}")
            LOGGER.warning("data source failed: %s", errors[-1])
    raise RuntimeError("Could not obtain the dataset from any source:\n" + "\n".join(errors))


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV and add ``row_id``, the row's position in the original file."""
    df = pd.read_csv(path)
    df.insert(0, "row_id", range(len(df)))
    return df
