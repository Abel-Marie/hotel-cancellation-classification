import dataclasses

import pytest
import yaml

from hotel_cancellation.config import apply_fast_mode, config_from_dict, deployed_sources


def test_loads_and_resolves_deployed_features(cfg):
    assert cfg.seed == 42
    assert deployed_sources(cfg) == ["base", "adr"]
    assert cfg.test_pass_models == ["dummy", "logreg", "champion"]


def test_unknown_key_is_rejected(root):
    raw = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    raw["split"]["typo_key"] = 1
    with pytest.raises(ValueError, match="unknown"):
        config_from_dict(raw)


def test_bad_leakage_decision_is_rejected(root):
    raw = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    raw["leakage"]["tier3_decisions"]["adr"]["decision"] = "maybe"
    with pytest.raises(ValueError, match="keep or exclude"):
        config_from_dict(raw)


def test_fast_mode_only_shrinks_budgets(cfg):
    fast = apply_fast_mode(cfg)
    assert fast.is_fast and fast.sample_rows
    assert fast.tuning.cv_folds < cfg.tuning.cv_folds
    assert dataclasses.replace(fast, is_fast=False).leakage == cfg.leakage
