"""The service's feature code and model files must be exact copies of the pipeline's."""

import hashlib

import pytest


def test_feature_code_is_identical(root):
    service_file = root / "service" / "core" / "features.py"
    if not service_file.exists():
        pytest.skip("run `make sync-service`")
    body = "".join(service_file.read_text().splitlines(keepends=True)[2:])
    assert body == (root / "src" / "hotel_cancellation" / "rowlevel.py").read_text(), \
        "service/core/features.py is stale: run `make sync-service`"


@pytest.mark.needs_artifacts
@pytest.mark.parametrize("name", ["model.onnx", "metadata.json"])
def test_model_files_are_identical(root, name):
    exported, served = root / "artifacts" / "export" / name, root / "service" / "models" / name
    if not exported.exists() or not served.exists():
        pytest.skip("run `make pipeline sync-service`")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    assert digest(exported) == digest(served), f"service/models/{name} is stale: run `make sync-service`"
