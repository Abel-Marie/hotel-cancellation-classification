"""The deployed service must reproduce the pipeline's predictions within 1e-4.

Goes all the way through the API's input schema: data row -> JSON payload ->
Booking -> raw record -> shared features -> ONNX -> calibration.
"""

import numpy as np
import pandas as pd
import pytest

from hotel_cancellation.monitoring.latency import to_api_payload


@pytest.mark.needs_artifacts
def test_service_matches_pipeline(root, service_app):
    export = root / "artifacts" / "export"
    if not (export / "parity_rows.pkl").exists():
        pytest.skip("run `make pipeline`")
    from core.schemas import Booking

    rows = pd.read_pickle(export / "parity_rows.pkl")
    expected = np.load(export / "parity_expected.npy")
    records = [Booking(**to_api_payload(r)).to_record() for r in rows.to_dict("records")]
    got, unseen = service_app.MODEL.probabilities(records)
    assert len(rows) >= 200
    assert np.max(np.abs(got - expected)) <= 1e-4
