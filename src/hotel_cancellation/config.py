"""Typed configuration objects, loaded from ``configs/config.yaml``.

Loading fails loudly when a key is missing or unexpected, so a typo in the
YAML file cannot silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    name: str
    citation: str
    license: str
    sources: list[str]
    mirror_path: str
    mirror_url: str
    sha256: str
    expected_rows: int
    expected_columns: int


@dataclass(frozen=True)
class CleaningConfig:
    drop_duplicates: bool
    max_valid_adr: float
    fill_meal_undefined_as: str


@dataclass(frozen=True)
class SplitConfig:
    val_size: float
    test_size: float
    temporal_train_until_year: int


@dataclass(frozen=True)
class FeatureConfig:
    min_category_frequency: int
    safe_sources: list[str]
    tier3_sources: list[str]
    candidate_sources: list[str]
    tier2_sources: list[str]
    tier1_sources: list[str]


@dataclass(frozen=True)
class LeakageDecision:
    decision: str
    reason: str


@dataclass(frozen=True)
class LeakageConfig:
    tier1_columns: list[str]
    tier2_columns: list[str]
    never_columns: list[str]
    tier3_decisions: dict[str, LeakageDecision]


@dataclass(frozen=True)
class CostConfig:
    false_negative: float
    false_positive: float


@dataclass(frozen=True)
class RiskBandConfig:
    low: float
    high: float


@dataclass(frozen=True)
class ModelsConfig:
    n_jobs: int


@dataclass(frozen=True)
class TuningConfig:
    subsample_rows: int
    cv_folds: int
    logreg_grid: dict[str, list[Any]]
    rf_random_iterations: int
    lgbm_budget: int
    lgbm_grid: dict[str, list[Any]]
    refine_trials: int


@dataclass(frozen=True)
class DiagnosticsConfig:
    learning_curve_sizes: list[float]
    validation_curve_num_leaves: list[int]


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_samples: int
    calibration_folds: int
    calibration_min_brier_gain: float
    cost_sensitivity_false_positive: list[float]


@dataclass(frozen=True)
class ExplainConfig:
    permutation_repeats: int
    shap_rows: int


@dataclass(frozen=True)
class SuccessConfig:
    roc_auc: float


@dataclass(frozen=True)
class ServiceConfig:
    url: str


@dataclass(frozen=True)
class SelectionConfig:
    tie_tolerance: float


@dataclass(frozen=True)
class FirstTestEvaluation:
    """Record of the first time the test set was scored (see configs/config.yaml)."""

    date: str
    champion: str
    validation_roc_auc: float
    roc_auc: float
    roc_auc_ci: list[float]
    pr_auc: float
    threshold: float
    precision: float
    recall: float
    f1: float
    what_changed_afterwards: str


@dataclass(frozen=True)
class Config:
    project: str
    seed: int
    data: DataConfig
    target: str
    cleaning: CleaningConfig
    split: SplitConfig
    features: FeatureConfig
    leakage: LeakageConfig
    costs: CostConfig
    risk_bands: RiskBandConfig
    models: ModelsConfig
    tuning: TuningConfig
    diagnostics: DiagnosticsConfig
    evaluation: EvaluationConfig
    explain: ExplainConfig
    success: SuccessConfig
    service: ServiceConfig
    selection: SelectionConfig
    test_pass_models: list[str]
    first_test_evaluation: FirstTestEvaluation
    fast: dict[str, Any]
    sample_rows: int | None = None
    is_fast: bool = False


SECTION_TYPES = {
    "data": DataConfig,
    "cleaning": CleaningConfig,
    "split": SplitConfig,
    "features": FeatureConfig,
    "costs": CostConfig,
    "risk_bands": RiskBandConfig,
    "models": ModelsConfig,
    "tuning": TuningConfig,
    "diagnostics": DiagnosticsConfig,
    "evaluation": EvaluationConfig,
    "explain": ExplainConfig,
    "success": SuccessConfig,
    "service": ServiceConfig,
    "selection": SelectionConfig,
    "first_test_evaluation": FirstTestEvaluation,
}


def build_section(cls: type, values: dict[str, Any], section: str) -> Any:
    """Instantiate one config dataclass, rejecting missing or unknown keys."""
    fields = dataclasses.fields(cls)
    allowed = {f.name for f in fields}
    required = {f.name for f in fields
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING}
    missing, unknown = required - set(values), set(values) - allowed
    if missing or unknown:
        raise ValueError(f"config section '{section}': missing={sorted(missing)} unknown={sorted(unknown)}")
    return cls(**values)


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Convert the parsed YAML mapping into a validated :class:`Config`."""
    raw = dict(raw)
    sections = {name: build_section(cls, raw.pop(name), name) for name, cls in SECTION_TYPES.items()}
    leak = dict(raw.pop("leakage"))
    leak["tier3_decisions"] = {
        col: build_section(LeakageDecision, val, f"leakage.tier3_decisions.{col}")
        for col, val in leak["tier3_decisions"].items()
    }
    for col, dec in leak["tier3_decisions"].items():
        if dec.decision not in {"keep", "exclude"}:
            raise ValueError(f"leakage decision for {col} must be keep or exclude")
    sections["leakage"] = build_section(LeakageConfig, leak, "leakage")
    return build_section(Config, {**raw, **sections}, "top level")


def load_config_text(text: str) -> Config:
    """Parse configuration from a YAML string (used by the self-contained notebook)."""
    return config_from_dict(yaml.safe_load(text))


def load_config(path: str | Path) -> Config:
    """Parse configuration from a YAML file."""
    return load_config_text(Path(path).read_text(encoding="utf-8"))


def apply_fast_mode(cfg: Config) -> Config:
    """Return a copy with tiny budgets for smoke tests. Never used for reported results."""
    fast = cfg.fast
    return dataclasses.replace(
        cfg,
        is_fast=True,
        sample_rows=fast["sample_rows"],
        tuning=dataclasses.replace(
            cfg.tuning,
            subsample_rows=fast["subsample_rows"],
            cv_folds=fast["cv_folds"],
            rf_random_iterations=fast["rf_random_iterations"],
            lgbm_budget=fast["lgbm_budget"],
            refine_trials=fast["refine_trials"],
            lgbm_grid={"num_leaves": [15, 31], "learning_rate": [0.1], "n_estimators": [50, 100]},
        ),
        diagnostics=dataclasses.replace(
            cfg.diagnostics, learning_curve_sizes=[0.3, 1.0], validation_curve_num_leaves=[8, 32]
        ),
        evaluation=dataclasses.replace(cfg.evaluation, bootstrap_samples=fast["bootstrap_samples"]),
        explain=ExplainConfig(permutation_repeats=fast["permutation_repeats"], shap_rows=fast["shap_rows"]),
    )


def deployed_sources(cfg: Config) -> list[str]:
    """Feature groups used by the deployed model: safe groups plus Tier 3 groups marked keep."""
    kept = [s for s in cfg.features.tier3_sources if cfg.leakage.tier3_decisions[s].decision == "keep"]
    return list(cfg.features.safe_sources) + kept

