"""End to end: every stage in fast mode, in a scratch copy of the project."""

import json
import shutil

import pytest

from hotel_cancellation.cli import main


@pytest.mark.slow
def test_all_stages_fast(root, tmp_path):
    shutil.copytree(root / "configs", tmp_path / "configs")
    shutil.copytree(root / "data" / "mirror", tmp_path / "data" / "mirror")
    main(["all", "--fast", "--root", str(tmp_path)])
    report = json.loads((tmp_path / "reports" / "metrics" / "pipeline" / "metrics.json").read_text())
    assert report["fast_mode"] is True
    assert set(report["test"]["models"]) == {"dummy", "logreg", "champion"}
    assert report["deployment"]["parity_ok"] is True
    assert len(list((tmp_path / "reports" / "figures").glob("*.png"))) >= 20
