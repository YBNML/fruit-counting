import json

from counting.io.results import CountingResult, CropMeta, StageTiming
from counting.io.serialize import (
    read_batch_csv,
    read_batch_json,
    write_batch_csv,
    write_batch_json,
)


def _sample(image_path="x.jpg", count=3):
    return CountingResult(
        image_path=image_path,
        raw_count=count,
        verified_count=count,
        points=[(1.0, 2.0), (3.0, 4.0)],
        boxes=[(0.0, 0.0, 10.0, 10.0)],
        crops=[CropMeta(bbox=(0, 0, 10, 10), score=0.9, is_bag=True)],
        timings_ms=[StageTiming(stage="pseco", ms=12.3)],
        device="cpu",
        config_hash="abc123",
        error=None,
    )


def test_json_roundtrip(tmp_path):
    results = [_sample("a.jpg"), _sample("b.jpg", count=1)]
    p = tmp_path / "out.json"
    write_batch_json(results, p)
    out = read_batch_json(p)
    assert len(out) == 2
    assert out[0].image_path == "a.jpg"
    assert out[0].raw_count == 3
    raw = json.loads(p.read_text())
    assert raw["schema_version"] == 1


def test_csv_roundtrip(tmp_path):
    results = [_sample("a.jpg"), _sample("b.jpg", count=1)]
    p = tmp_path / "out.csv"
    write_batch_csv(results, p)
    rows = read_batch_csv(p)
    assert [r["image_path"] for r in rows] == ["a.jpg", "b.jpg"]
    assert rows[0]["raw_count"] == "3"


def test_error_preserved_in_json(tmp_path):
    r = _sample()
    r = r.__class__(**{**r.__dict__, "error": "pseco failed"})
    p = tmp_path / "out.json"
    write_batch_json([r], p)
    assert read_batch_json(p)[0].error == "pseco failed"
