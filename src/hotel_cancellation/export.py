"""Export the champion for serving: an ONNX model plus a metadata file.

The service reproduces the pipeline exactly from these two files:
row-level features (shared code) -> category codes (vocabulary below) ->
ONNX model -> calibration breakpoints -> threshold.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from .config import Config
from .evaluation import apply_calibration
from .rowlevel import derive_features


def category_vocabulary(model, cats: list[str]) -> dict[str, dict[str, float]]:
    """The exact code the fitted encoder gives each training category.

    Computed by passing every known category through the fitted encoder, so
    grouped rare categories get the same shared code as in training. Categories
    the encoder never saw are absent and map to the unknown code (-1).
    """
    if not cats:
        return {}
    encoder = model.named_steps["prep"].named_transformers_["cat"]
    vocab = {}
    for j, col in enumerate(cats):
        known = list(encoder.categories_[j])
        probe = pd.DataFrame({c: [encoder.categories_[i][0]] * len(known) for i, c in enumerate(cats)})
        probe[col] = known
        codes = encoder.transform(probe)[:, j]
        vocab[col] = {str(k): float(v) for k, v in zip(known, codes)}
    return vocab


# The converters name the graph with a random ID unless told otherwise. A fixed name makes the file, and
# therefore the model version (a fingerprint of the file), the same every time the same model is exported.
ONNX_GRAPH_NAME = "hotel_cancellation_model"


def export_onnx(model, n_features: int, path: Path) -> Path:
    """Convert the fitted estimator (after preprocessing) to ONNX, float32 input."""
    estimator = model.named_steps["model"]
    if type(estimator).__name__ == "LGBMClassifier":
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType as LgbmFloat
        onx = convert_lightgbm(estimator, name=ONNX_GRAPH_NAME, initial_types=[("input", LgbmFloat([None, n_features]))],
                               zipmap=False, target_opset=15)
    else:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        onx = convert_sklearn(estimator, name=ONNX_GRAPH_NAME, initial_types=[("input", FloatTensorType([None, n_features]))],
                              options={id(estimator): {"zipmap": False}}, target_opset=15)
    # The converter lists the operator sets in an order that depends on Python's per-process hash seed.
    # Sorting them makes the file byte-for-byte reproducible, so its fingerprint identifies the model.
    opsets = sorted(onx.opset_import, key=lambda o: o.domain)
    del onx.opset_import[:]
    onx.opset_import.extend(opsets)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onx.SerializeToString())
    return path


def onnx_probabilities(path: Path, X32: np.ndarray) -> np.ndarray:
    """Positive-class probabilities from an exported ONNX model."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: X32.astype(np.float32)})
    return np.asarray(outputs[1])[:, 1]


def build_metadata(model, cats: list[str], nums: list[str], sources: list[str], calibration: dict,
                   threshold: float, test_metrics: dict, champion: dict, onnx_path: Path, cfg: Config) -> dict:
    """Everything the service needs, plus a model card summary for ``/api/v1/model``."""
    prep = model.named_steps["prep"]
    order = list(prep.get_feature_names_out())
    digest = hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()[:12]
    return {
        "model_version": f"{champion['name']}-{digest}",
        "trained_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": "binary classification: will this booking be cancelled?",
        "estimator": champion["name"],
        "params": champion["params"],
        "feature_sources": sources,
        "input_order": order,
        "categorical_features": cats,
        "numeric_features": nums,
        "unknown_category_code": -1.0,
        "category_codes": category_vocabulary(model, cats),
        "calibration": calibration,
        "threshold": float(threshold),
        "risk_bands": {"low": cfg.risk_bands.low, "high": cfg.risk_bands.high},
        "costs": {"false_negative": cfg.costs.false_negative, "false_positive": cfg.costs.false_positive},
        "test_metrics": test_metrics,
        "excluded_columns": {
            "outcome": cfg.leakage.tier1_columns,
            "known_only_after_booking": cfg.leakage.tier2_columns,
            "timing_uncertain": [c for c, d in cfg.leakage.tier3_decisions.items() if d.decision == "exclude"],
            "never_used": cfg.leakage.never_columns,
        },
        "data": {"name": cfg.data.name, "citation": cfg.data.citation, "license": cfg.data.license},
    }


def service_style_proba(rows: pd.DataFrame, metadata: dict, onnx_path: Path) -> np.ndarray:
    """Reproduce the service's computation from the exported files (pipeline-side check)."""
    codes, unknown = metadata["category_codes"], metadata["unknown_category_code"]
    vectors = []
    for r in rows.to_dict("records"):
        feats = derive_features(r, metadata["feature_sources"])
        vectors.append([codes[c].get(str(feats[c]), unknown) if c in codes else float(feats[c])
                        for c in metadata["input_order"]])
    raw = onnx_probabilities(onnx_path, np.asarray(vectors, dtype=np.float32))
    return apply_calibration(raw, metadata["calibration"])
