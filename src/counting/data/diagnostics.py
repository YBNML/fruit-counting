"""Dataset diagnostics: resolution, blur, exposure. HTML + JSON report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np

from counting.data.formats.imagefolder import ImageFolderDataset

_BLUR_THRESHOLD = 100.0       # Laplacian variance; < threshold ⇒ "blurry"
_LOW_EXPOSURE = 40            # mean intensity < ⇒ underexposed
_HIGH_EXPOSURE = 215          # mean intensity > ⇒ overexposed


@dataclass
class DiagnosticsReport:
    image_count: int
    resolution: dict[str, Any]
    blur: dict[str, Any]
    exposure: dict[str, Any]
    per_image: list[dict[str, Any]] = field(default_factory=list)


def diagnose_directory(image_dir: str | Path, *, report_dir: str | Path) -> DiagnosticsReport:
    ds = ImageFolderDataset(image_dir)
    widths, heights, blurs, means = [], [], [], []
    per_image = []

    for rec in ds:
        arr = rec.read_rgb()
        h, w = arr.shape[:2]
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean = float(gray.mean())

        widths.append(w); heights.append(h); blurs.append(lap_var); means.append(mean)
        per_image.append({
            "relpath": rec.relpath,
            "width": w, "height": h,
            "blur_var": round(lap_var, 3),
            "mean_intensity": round(mean, 2),
        })

    low_blur_ratio = sum(1 for b in blurs if b < _BLUR_THRESHOLD) / len(blurs)
    under = sum(1 for m in means if m < _LOW_EXPOSURE) / len(means)
    over = sum(1 for m in means if m > _HIGH_EXPOSURE) / len(means)

    report = DiagnosticsReport(
        image_count=len(ds),
        resolution={
            "width": {"min": min(widths), "max": max(widths), "median": median(widths)},
            "height": {"min": min(heights), "max": max(heights), "median": median(heights)},
        },
        blur={
            "min": round(min(blurs), 3),
            "max": round(max(blurs), 3),
            "median": round(float(np.median(blurs)), 3),
            "threshold": _BLUR_THRESHOLD,
            "low_blur_ratio": round(low_blur_ratio, 3),
        },
        exposure={
            "underexposed_ratio": round(under, 3),
            "overexposed_ratio": round(over, 3),
            "low_threshold": _LOW_EXPOSURE,
            "high_threshold": _HIGH_EXPOSURE,
        },
        per_image=per_image,
    )

    _write_reports(report, Path(report_dir))
    return report


def _write_reports(report: DiagnosticsReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    (out_dir / "diagnostics_report.html").write_text(_render_html(report), encoding="utf-8")


def _render_html(report: DiagnosticsReport) -> str:
    rows = "\n".join(
        f"<tr><td>{p['relpath']}</td><td>{p['width']}×{p['height']}</td>"
        f"<td>{p['blur_var']}</td><td>{p['mean_intensity']}</td></tr>"
        for p in report.per_image
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Diagnostics</title>
<style>body{{font-family:system-ui;margin:24px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 8px}}</style>
</head><body>
<h1>Dataset Diagnostics</h1>
<p>images: {report.image_count}</p>
<h2>Resolution</h2><pre>{json.dumps(report.resolution, indent=2)}</pre>
<h2>Blur (Laplacian variance)</h2><pre>{json.dumps(report.blur, indent=2)}</pre>
<h2>Exposure (gray mean)</h2><pre>{json.dumps(report.exposure, indent=2)}</pre>
<h2>Per image</h2>
<table><tr><th>relpath</th><th>size</th><th>blur_var</th><th>mean</th></tr>
{rows}
</table></body></html>
"""
