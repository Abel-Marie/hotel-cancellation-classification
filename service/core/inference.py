"""Load the exported model once and turn bookings into predictions.

Steps, identical to the training pipeline: shared feature code -> category
codes from metadata.json -> ONNX model -> calibration breakpoints -> threshold.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .features import derive_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class CancellationModel:
    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        m = self.metadata
        self.order, self.codes, self.unknown = m["input_order"], m["category_codes"], m["unknown_category_code"]
        self.sources, self.threshold = m["feature_sources"], m["threshold"]
        cal = m["calibration"]
        self.calibration = (np.asarray(cal["x"]), np.asarray(cal["y"])) if cal["used"] else None
        self.bands = m["risk_bands"]

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    def known_categories(self, list_up_to: int = 30) -> dict:
        """The category values the model was trained on; long lists (agent IDs) are given as a count."""
        return {name: sorted(codes) if len(codes) <= list_up_to else {"known_values": len(codes)}
                for name, codes in self.codes.items()}

    def vectorize(self, records: list[dict]) -> tuple[np.ndarray, list[list[str]]]:
        rows, unseen = [], []
        for record in records:
            feats = derive_features(record, self.sources)
            row, missing = [], []
            for name in self.order:
                if name in self.codes:
                    code = self.codes[name].get(str(feats[name]))
                    if code is None:
                        missing.append(name)
                        code = self.unknown
                    row.append(code)
                else:
                    row.append(float(feats[name]))
            rows.append(row)
            unseen.append(missing)
        return np.asarray(rows, dtype=np.float32), unseen

    def probabilities(self, records: list[dict]) -> tuple[np.ndarray, list[list[str]]]:
        X, unseen = self.vectorize(records)
        raw = np.asarray(self.session.run(None, {self.input_name: X})[1])[:, 1].astype(float)
        if self.calibration is not None:
            raw = np.interp(raw, *self.calibration)
        if not np.all(np.isfinite(raw)):  # never observed; a NaN must not reach a client as a "probability"
            raise ValueError("the model returned a non-finite probability")
        return raw, unseen

    def band(self, p: float) -> str:
        return "high" if p >= self.bands["high"] else "medium" if p >= self.bands["low"] else "low"

    def predict(self, records: list[dict]) -> list[dict]:
        probs, unseen = self.probabilities(records)
        return [{"prediction_id": uuid.uuid4().hex, "cancellation_probability": round(float(p), 6),
                 "will_cancel": bool(p >= self.threshold), "risk_band": self.band(float(p)),
                 "threshold": self.threshold, "unseen_categories": u, "model_version": self.version}
                for p, u in zip(probs, unseen)]
