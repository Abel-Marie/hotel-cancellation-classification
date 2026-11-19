"""The notebook and the pipeline must use the same data and reach the same results.

Step 1 (exact): rows removed by each cleaning rule, split sizes, test-row fingerprint.
Step 2 (tolerance): headline test metrics. If step 1 passes and step 2 fails,
the difference is in modelling, which narrows the search.
"""

import json

import pytest

NOTEBOOK = "01_hotel_cancellation_classification.json"


@pytest.fixture(scope="module")
def reports(root):
    nb = root / "reports" / "metrics" / "notebook" / NOTEBOOK
    pl = root / "reports" / "metrics" / "pipeline" / "metrics.json"
    if not nb.exists() or not pl.exists():
        pytest.skip("run `make pipeline notebooks` first")
    return json.loads(nb.read_text()), json.loads(pl.read_text())


def test_same_data(reports):
    nb, pl = reports
    assert [r["rows_removed"] for r in nb["data"]["cleaning"]] == [r["rows_removed"] for r in pl["data"]["cleaning"]]
    for part in ("train", "validation", "test"):
        assert nb["data"]["split"][part]["rows"] == pl["data"]["split"][part]["rows"]
    assert nb["data"]["split"]["test_rows_sha256"] == pl["data"]["split"]["test_rows_sha256"]


def test_same_champion_and_metrics(reports):
    nb, pl = reports
    assert nb["champion"]["name"] == pl["champion"]["name"]
    a, b = nb["test"]["models"]["champion"]["metrics"], pl["test"]["models"]["champion"]["metrics"]
    assert abs(a["roc_auc"] - b["roc_auc"]) <= 0.005
    assert abs(a["f1"] - b["f1"]) <= 0.005
