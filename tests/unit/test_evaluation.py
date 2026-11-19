import numpy as np
import pytest

from hotel_cancellation.evaluation import (
    apply_calibration,
    bootstrap_ci,
    choose_threshold,
    classification_metrics,
    cost_curve,
)
from hotel_cancellation.monitoring.drift import population_stability_index


def test_metrics_on_a_known_case():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    m = classification_metrics(y, p, 0.5)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(0.5)


def test_cost_curve_prefers_fewer_expensive_errors():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.2, 0.4, 0.6, 0.7, 0.9])
    curve = cost_curve(y, p, fn_cost=1.0, fp_cost=5.0)
    assert choose_threshold(curve) > 0.6  # a false alarm costs 5x, so avoid flagging the 0.6 row


def test_bootstrap_interval_contains_the_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    point = classification_metrics(y, p, 0.5)["roc_auc"]
    ci = bootstrap_ci(y, p, 0.5, 200, seed=1)["roc_auc"]
    assert ci["low"] <= point <= ci["high"]


def test_calibration_identity_when_unused():
    p = np.array([0.1, 0.5])
    assert np.array_equal(apply_calibration(p, {"used": False}), p)
    mapped = apply_calibration(p, {"used": True, "x": [0.0, 1.0], "y": [0.0, 0.5]})
    assert mapped.tolist() == pytest.approx([0.05, 0.25])


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert population_stability_index(a, rng.normal(0, 1, 5000)) < 0.05
    assert population_stability_index(a, rng.normal(1.5, 1, 5000)) > 0.25
