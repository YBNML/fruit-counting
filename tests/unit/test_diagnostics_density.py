import numpy as np
from PIL import Image

from counting.data.diagnostics import (
    DiagnosticsReport,
    diagnose_directory,
)


def test_density_section_absent_without_sam(tmp_path):
    # SAM weights not present; diagnose should still run and simply omit density
    Image.new("RGB", (16, 16), (128, 128, 128)).save(tmp_path / "a.jpg")
    report = diagnose_directory(tmp_path, report_dir=tmp_path / "out")
    assert isinstance(report, DiagnosticsReport)
    assert not getattr(report, "density", None)
