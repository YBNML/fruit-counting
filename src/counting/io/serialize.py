"""JSON/CSV I/O for CountingResult."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from counting.io.results import CountingResult, CropMeta, StageTiming

SCHEMA_VERSION = 1

_CSV_HEADERS = [
    "image_path", "raw_count", "verified_count",
    "device", "config_hash", "error", "timings_ms_total",
]


def write_batch_json(results: Iterable[CountingResult], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "results": [asdict(r) for r in results],
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_batch_json(path: str | Path) -> list[CountingResult]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version in {p}")
    return [_result_from_dict(d) for d in data["results"]]


def write_batch_csv(results: Iterable[CountingResult], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_HEADERS)
        w.writeheader()
        for r in rows:
            w.writerow({
                "image_path": r.image_path,
                "raw_count": r.raw_count,
                "verified_count": r.verified_count,
                "device": r.device,
                "config_hash": r.config_hash,
                "error": r.error or "",
                "timings_ms_total": round(sum(t.ms for t in r.timings_ms), 2),
            })


def read_batch_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _result_from_dict(d: dict) -> CountingResult:
    return CountingResult(
        image_path=d["image_path"],
        raw_count=int(d["raw_count"]),
        verified_count=int(d["verified_count"]),
        points=[tuple(x) for x in d.get("points", [])],
        boxes=[tuple(x) for x in d.get("boxes", [])],
        crops=[CropMeta(**c) for c in d.get("crops", [])],
        timings_ms=[StageTiming(**t) for t in d.get("timings_ms", [])],
        device=d.get("device", "cpu"),
        config_hash=d.get("config_hash", ""),
        error=d.get("error"),
    )
