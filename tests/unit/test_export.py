"""The exported model file must be reproducible, because its fingerprint is the model version."""

import hashlib

import numpy as np
import onnx
import pandas as pd

from hotel_cancellation.export import ONNX_GRAPH_NAME, export_onnx
from hotel_cancellation.models import make_model


def tiny_model(cfg):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"segment": rng.choice(["a", "b", "c"], 400), "x1": rng.normal(size=400), "x2": rng.normal(size=400)})
    y = (X["x1"] + (X["segment"] == "a") + rng.normal(scale=0.5, size=400) > 0.5).astype(int)
    return make_model("lgbm", ["segment"], ["x1", "x2"], cfg, n_estimators=20, num_leaves=7).fit(X, y)


def test_export_has_a_fixed_name_and_sorted_operator_sets(cfg, tmp_path):
    model = tiny_model(cfg)
    exported = onnx.load(str(export_onnx(model, 3, tmp_path / "a.onnx")))
    assert exported.graph.name == ONNX_GRAPH_NAME
    domains = [o.domain for o in exported.opset_import]
    assert domains == sorted(domains)


def test_exporting_the_same_model_twice_gives_the_same_bytes(cfg, tmp_path):
    model = tiny_model(cfg)
    first = export_onnx(model, 3, tmp_path / "a.onnx").read_bytes()
    second = export_onnx(model, 3, tmp_path / "b.onnx").read_bytes()
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
