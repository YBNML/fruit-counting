import json
from pathlib import Path

import numpy as np
from PIL import Image

from counting.data.diagnostics import DiagnosticsReport, diagnose_directory


def _make(path: Path, size, fill):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0], 3), fill, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_diagnose_produces_report(tmp_path):
    _make(tmp_path / "a.jpg", (64, 48), 128)
    _make(tmp_path / "b.jpg", (128, 96), 200)

    report_dir = tmp_path / "out"
    report = diagnose_directory(tmp_path, report_dir=report_dir)

    assert isinstance(report, DiagnosticsReport)
    assert report.image_count == 2
    assert {"width", "height"} <= set(report.resolution.keys())
    assert (report_dir / "diagnostics.json").exists()
    assert (report_dir / "diagnostics_report.html").exists()

    data = json.loads((report_dir / "diagnostics.json").read_text())
    assert data["image_count"] == 2


def test_blur_metric_flags_flat_image(tmp_path):
    _make(tmp_path / "flat.jpg", (32, 32), 128)

    report = diagnose_directory(tmp_path, report_dir=tmp_path / "out")
    assert report.blur["min"] <= 1.0
    assert report.blur["low_blur_ratio"] >= 0.5


def test_empty_directory_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        diagnose_directory(tmp_path, report_dir=tmp_path / "out")
