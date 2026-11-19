"""Leakage audit: evidence for timing-uncertain columns and an ablation by tier.

Every number here is computed on training and validation rows only.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import Config, deployed_sources
from .features import build_feature_frame, split_feature_types
from .models import make_model


def rate_table(frame: pd.DataFrame, column: str, target: str, top: int | None = None,
               by_size: bool = True) -> list[dict]:
    """Cancellation rate and row count per value of ``column``."""
    table = frame.groupby(column, observed=True)[target].agg(cancel_rate="mean", rows="size").reset_index()
    if by_size:
        table = table.sort_values("rows", ascending=False)
    if top:
        table = table.head(top)
    table[column] = table[column].astype(str)
    return table.rename(columns={column: "value"}).to_dict("records")


def tier3_evidence(trval: pd.DataFrame, cfg: Config) -> dict:
    """The tables behind each Tier 3 decision."""
    t = cfg.target
    bands = pd.qcut(trval["adr"], 10, duplicates="drop")
    labels = {iv: f"{max(iv.left, 0):.0f}-{iv.right:.0f}" for iv in bands.cat.categories}
    priced = trval.assign(price_band=bands.cat.rename_categories(labels))
    return {
        "required_car_parking_spaces": rate_table(trval, "required_car_parking_spaces", t),
        "total_of_special_requests": rate_table(trval, "total_of_special_requests", t),
        "country": rate_table(trval, "country", t, top=10),
        "adr": rate_table(priced, "price_band", t, by_size=False),
    }


def fit_and_score(sources: list[str], train: pd.DataFrame, val: pd.DataFrame, cfg: Config) -> float:
    """Validation ROC-AUC of LightGBM (default settings) on the given feature groups."""
    Xtr, Xva = build_feature_frame(train, sources), build_feature_frame(val, sources)
    cats, nums = split_feature_types(list(Xtr.columns))
    model = make_model("lgbm", cats, nums, cfg).fit(Xtr, train[cfg.target])
    return float(roc_auc_score(val[cfg.target], model.predict_proba(Xva)[:, 1]))


def leakage_ablation(train: pd.DataFrame, val: pd.DataFrame, cfg: Config) -> dict:
    """ROC-AUC as each tier of suspect columns is added back, plus each Tier 3 column alone."""
    f = cfg.features
    deployed = deployed_sources(cfg)
    steps = [
        ("Base features only", list(f.safe_sources)),
        ("Deployed (base + kept Tier 3)", deployed),
        ("+ all Tier 3 (timing uncertain)", list(f.safe_sources) + list(f.tier3_sources)),
        ("+ Tier 2 (known after booking)", list(f.safe_sources) + list(f.tier3_sources) + list(f.tier2_sources)),
        ("+ Tier 1 (the outcome itself)",
         list(f.safe_sources) + list(f.tier3_sources) + list(f.tier2_sources) + list(f.tier1_sources)),
    ]
    ladder = [{"step": name, "sources": src, "val_roc_auc": fit_and_score(src, train, val, cfg)}
              for name, src in steps]
    base_score = ladder[0]["val_roc_auc"]
    single = []
    for source in f.tier3_sources:
        score = fit_and_score(list(f.safe_sources) + [source], train, val, cfg)
        single.append({"column": source, "decision": cfg.leakage.tier3_decisions[source].decision,
                       "val_roc_auc": score, "gain_over_base": score - base_score})
    return {"ladder": ladder, "tier3_single": single}
