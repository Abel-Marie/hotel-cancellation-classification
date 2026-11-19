"""Train / validation / test splits."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config


def stratified_split(df: pd.DataFrame, target: str, val_size: float, test_size: float,
                     seed: int) -> dict[str, np.ndarray]:
    """Stratified train/validation/test positions (fractions are of the whole frame)."""
    positions = np.arange(len(df))
    rest, test = train_test_split(positions, test_size=test_size, stratify=df[target], random_state=seed)
    train, val = train_test_split(rest, test_size=val_size / (1 - test_size),
                                  stratify=df[target].iloc[rest], random_state=seed)
    return {"train": np.sort(train), "validation": np.sort(val), "test": np.sort(test)}


def rows_hash(row_ids) -> str:
    """Fingerprint of a set of original rows, used to prove two runs used the same split."""
    joined = ",".join(str(i) for i in sorted(int(x) for x in row_ids))
    return hashlib.sha256(joined.encode()).hexdigest()


def split_summary(df: pd.DataFrame, splits: dict[str, np.ndarray], target: str) -> dict:
    out = {}
    for name, pos in splits.items():
        part = df.iloc[pos]
        out[name] = {"rows": int(len(part)), "positive_rate": float(part[target].mean())}
    out["test_rows_sha256"] = rows_hash(df.iloc[splits["test"]]["row_id"])
    return out


def temporal_positions(df: pd.DataFrame, train_until_year: int) -> tuple[np.ndarray, np.ndarray]:
    """Earlier arrivals for fitting, the following year for evaluation."""
    years = df["arrival_date_year"].to_numpy()
    return np.flatnonzero(years <= train_until_year), np.flatnonzero(years == train_until_year + 1)


def sample_for_fast_mode(clean: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """In fast mode only, keep a stratified sample so smoke runs finish in minutes."""
    if not cfg.sample_rows or len(clean) <= cfg.sample_rows:
        return clean
    keep, _ = train_test_split(np.arange(len(clean)), train_size=cfg.sample_rows,
                               stratify=clean[cfg.target], random_state=cfg.seed)
    return clean.iloc[np.sort(keep)].reset_index(drop=True)
