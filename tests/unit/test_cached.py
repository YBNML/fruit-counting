import dataclasses
import json
import os
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import counting.detection.cached as cached
from counting.data.formats.orchard_synth import SynthRun
from counting.detection.base import Detections
from counting.detection.cached import (
    CACHE_FORMAT, CachedDetector, cache_digest, cache_path, detect_to_cache, file_sha256, postprocess, read_cache,
    rgb_fingerprint, write_cache,
)
from tests.unit.synthfake import write_run

META = {"run_id": "apple_s5_0123abcd", "fingerprint": "f" * 64, "weights_sha256": "a" * 64}


def _dets(rows):
    """Detections from (x0, y0, x1, y1, score, label) rows."""
    a = np.array(rows, np.float64).reshape(-1, 6)
    return Detections(a[:, :4].astype(np.float32), a[:, 4].astype(np.float32), a[:, 5].astype(np.int8))


def test_cache_round_trip_keeps_dtypes_order_and_meta(tmp_path):
    dets = {7: _dets([]), 3: _dets([(50, 5, 60, 15, 0.7, 1), (10, 5, 20, 15, 0.9, 0), (10, 5, 20, 15, 0.4, 1)])}
    digest = write_cache(tmp_path / "c.npz", dets, META)
    meta, back = read_cache(tmp_path / "c.npz")
    assert meta == {**META, "format": CACHE_FORMAT} and digest == cache_digest(tmp_path / "c.npz")
    assert sorted(back) == [3, 7] and len(back[7]) == 0
    d = back[3]
    assert (d.boxes.dtype, d.scores.dtype, d.labels.dtype, d.depth_m) == (np.float32, np.float32, np.int8, None)
    assert d.boxes.tolist() == [[10, 5, 20, 15], [10, 5, 20, 15], [50, 5, 60, 15]]
    assert d.scores.tolist() == pytest.approx([0.9, 0.4, 0.7]) and d.labels.tolist() == [0, 1, 1]
    assert cache_path(tmp_path, "apple_s5_0123abcd") == tmp_path / "apple_s5_0123abcd.npz"


def test_cache_digest_follows_content_not_file_bytes(tmp_path):
    row = (10, 5, 20, 15, 0.9, 0)
    a = write_cache(tmp_path / "a.npz", {0: _dets([row])}, META)
    b = write_cache(tmp_path / "sub" / "b.npz", {0: _dets([row])}, META)
    variants = [({0: _dets([(10, 5, 20, 15, 0.8, 0)])}, META), ({0: _dets([(11, 5, 20, 15, 0.9, 0)])}, META),
                ({0: _dets([(10, 5, 20, 15, 0.9, 1)])}, META), ({1: _dets([row])}, META),
                ({0: _dets([row])}, {**META, "weights_sha256": "b" * 64})]
    digests = [write_cache(tmp_path / f"v{i}.npz", d, m) for i, (d, m) in enumerate(variants)]
    assert a == b and len({a, *digests}) == 6 and not list(tmp_path.rglob("*.tmp"))


def test_write_cache_refuses_overwrite_depth_and_missing_identity(tmp_path):
    dets = {0: _dets([(10, 5, 20, 15, 0.9, 0)])}
    write_cache(tmp_path / "c.npz", dets, META)
    with pytest.raises(FileExistsError, match="cache exists"):
        write_cache(tmp_path / "c.npz", dets, META)
    deep = {0: dataclasses.replace(dets[0], depth_m=np.ones(1, np.float32))}
    with pytest.raises(ValueError, match="a detection cache stores no depth"):
        write_cache(tmp_path / "d.npz", deep, META)
    with pytest.raises(ValueError, match="cache meta needs run_id and fingerprint"):
        write_cache(tmp_path / "e.npz", dets, {"run_id": "x"})


def test_read_cache_refuses_malformed_files(tmp_path):
    good = dict(frame_idx=np.array([0], np.int32), offsets=np.array([0, 1], np.int64),
                boxes=np.array([[10, 5, 20, 15]], np.float32), scores=np.array([0.9], np.float32),
                labels=np.array([0], np.int8), meta=np.array(json.dumps({**META, "format": CACHE_FORMAT})))
    cases = {"keys": {k: v for k, v in good.items() if k != "labels"},
             "format": {**good, "meta": np.array(json.dumps({**META, "format": "other/1"}))},
             "offsets": {**good, "offsets": np.array([0, 2], np.int64)},
             "labels": {**good, "labels": np.array([2], np.int8)},
             "dtype": {**good, "boxes": good["boxes"].astype(np.float64)},
             "pickle": {**good, "meta": np.array([{"a": 1}], dtype=object)},
             "meta": {**good, "meta": np.array(json.dumps([1, 2]))},
             "order": {**good, "frame_idx": np.array([3, 1], np.int32), "offsets": np.array([0, 1, 1], np.int64)},
             "finite": {**good, "boxes": np.array([[10, 5, np.nan, 15]], np.float32)}}
    for name, arrays in cases.items():
        p = tmp_path / f"{name}.npz"
        np.savez(p, **arrays)
        with pytest.raises(ValueError, match="bad detection cache"):
            read_cache(p)
    whole = tmp_path / "whole.npz"
    write_cache(whole, {0: _dets([(10, 5, 20, 15, 0.9, 0)])}, META)
    (tmp_path / "cut.npz").write_bytes(whole.read_bytes()[:-40])
    with pytest.raises(ValueError, match="bad detection cache"):
        read_cache(tmp_path / "cut.npz")


def test_corrupt_compressed_members_are_bad_caches(tmp_path):
    whole = tmp_path / "whole.npz"
    write_cache(whole, {0: _dets([(10, 5, 20, 15, 0.9, 0)])}, META)
    data = whole.read_bytes()
    with zipfile.ZipFile(whole) as z:
        info = z.getinfo("boxes.npy")
    off = info.header_offset           # the local header's own name and extra lengths (the central ones may differ)
    start = off + 30 + int.from_bytes(data[off + 26:off + 28], "little") + int.from_bytes(data[off + 28:off + 30], "little")
    flip = tmp_path / "flip.npz"
    for k in range(info.compress_size):
        b = bytearray(data)
        b[start + k] ^= 0xFF
        flip.write_bytes(bytes(b))
        for read in (read_cache, cache_digest):
            with pytest.raises(ValueError, match="bad detection cache"):
                read(flip)
    part, one = tmp_path / "part.npz", tmp_path / "one.npz"
    np.savez(part, meta=np.array("{}"), boxes=np.zeros((0, 4), np.float32))     # arrays missing
    with open(one, "wb") as fh:                                                  # one .npy array named .npz
        np.save(fh, np.zeros(3))
    for p in (part, one):
        for read in (read_cache, cache_digest):
            with pytest.raises(ValueError, match=re.escape(f"{p}: bad detection cache")):
                read(p)


def test_postprocess_clips_snaps_and_suppresses_across_classes():
    raw = _dets([(0.8, 10, 40, 50, 0.8, 0),         # the same object as a fruit (IoU 0.97, lower score): suppressed
                 (1, 10, 41, 50, 0.9, 1),           # kept as a bag, snapped to the left border
                 (600, 340, 639.5, 361, 0.7, 0),    # clipped to the bottom, snapped to the right border
                 (100, 100, 100, 120, 0.6, 0),      # zero width: dropped
                 (200, 200, 220, 220, 0.5, 1)])
    out = postprocess(raw, width=640, height=360)
    assert out.boxes.tolist() == [[0, 10, 41, 50], [200, 200, 220, 220], [600, 340, 640, 360]]
    assert out.labels.tolist() == [1, 1, 0] and out.scores.tolist() == pytest.approx([0.9, 0.5, 0.7])
    assert out.depth_m is None and out.boxes.dtype == np.float32
    loose = postprocess(raw, width=640, height=360, nms_iou=0.99, edge_snap_px=0.0)
    np.testing.assert_allclose(loose.boxes, [[0.8, 10, 40, 50], [1, 10, 41, 50], [200, 200, 220, 220],
                                             [600, 340, 639.5, 360]], atol=1e-6)
    assert len(postprocess(_dets([]), width=640, height=360)) == 0


def test_postprocess_boundary_rules_are_strict():
    raw = _dets([(0, 0, 10, 10, 0.9, 0), (0, 0, 10, 5, 0.8, 0),        # IoU exactly 0.5: both stay
                 (2, 100, 30, 130, 0.7, 0), (1.5, 200, 30, 230, 0.6, 0),  # 2.0 from the border stays, 1.5 snaps
                 (600, 300, 638, 330, 0.5, 1), (600, 200, 638.5, 230, 0.4, 1)])
    out = postprocess(raw, width=640, height=360, nms_iou=0.5, edge_snap_px=2.0)
    assert sorted(map(tuple, out.boxes.tolist())) == [(0, 0, 10, 5), (0, 0, 10, 10), (0, 200, 30, 230),
                                                      (2, 100, 30, 130), (600, 200, 640, 230), (600, 300, 638, 330)]


def test_postprocess_drops_small_boxes_off_the_border():
    raw = _dets([(0, 10, 8, 30, 0.9, 0), (100, 100, 108, 130, 0.9, 0), (200, 200, 215, 215, 0.9, 1),
                 (300, 352.5, 340, 359, 0.9, 0), (400, 100, 412, 140, 0.9, 0)])
    assert len(postprocess(raw, width=640, height=360)) == 5
    kept = postprocess(raw, width=640, height=360, min_short_px=12.0)
    assert sorted(map(tuple, kept.boxes.tolist())) == [(0, 10, 8, 30), (200, 200, 215, 215), (300, 352.5, 340, 360),
                                                       (400, 100, 412, 140)]
    assert sorted(map(tuple, postprocess(raw, width=640, height=360, min_short_px=16.0).boxes.tolist())) == \
        [(0, 10, 8, 30), (300, 352.5, 340, 360)]


def test_cached_detector_refuses_foreign_or_incomplete_caches(tmp_path, apple_run):
    run = SynthRun.open(apple_run)
    frames = run.frames[:3]
    path = cache_path(tmp_path, run.meta.run_id)
    write_cache(path, {f.idx: _dets([(10, 10, 30, 30, 0.9, 0)]) for f in frames[:2]},
                {"run_id": run.meta.run_id, "fingerprint": run.meta.fingerprint})
    det = CachedDetector(path, fingerprint=run.meta.fingerprint)
    assert det.name == "cached" and det.digest == cache_digest(path)
    assert det.detect(frames[0]).boxes.tolist() == [[10, 10, 30, 30]]
    assert det.raw(frames[0].idx).boxes.tolist() == [[10, 10, 30, 30]]
    with pytest.raises(ValueError, match="cache was made for bundle fingerprint"):
        CachedDetector(path, fingerprint="0" * 64)
    with pytest.raises(ValueError, match=f"cache is for run {run.meta.run_id}, frame is from run other"):
        det.detect(dataclasses.replace(frames[0], run_id="other"))
    with pytest.raises(ValueError, match=f"no detections stored for frame {frames[2].idx}"):
        det.detect(frames[2])


def test_cached_detector_reads_once_and_checks_the_pinned_digest(tmp_path, apple_run, monkeypatch):
    import counting.detection.cached as cached

    run = SynthRun.open(apple_run)
    f = run.frames[0]
    path = cache_path(tmp_path, run.meta.run_id)
    digest = write_cache(path, {f.idx: _dets([(100, 100, 108, 130, 0.9, 0), (10, 10, 30, 30, 0.8, 0)])},
                         {"run_id": run.meta.run_id, "fingerprint": run.meta.fingerprint})
    opened, real = [], cached._open_npz
    monkeypatch.setattr(cached, "_open_npz", lambda p: opened.append(p) or real(p))
    det = CachedDetector(path, fingerprint=run.meta.fingerprint, digest=digest, min_short_px=12.0)
    assert opened == [path] and det.digest == digest and det.min_short_px == 12.0
    assert det.detect(f).boxes.tolist() == [[10, 10, 30, 30]]
    with pytest.raises(ValueError, match=re.escape(f"{path}: cache digest {digest[:16]} differs from the pinned "
                                                   f"{'0' * 16}")):
        CachedDetector(path, digest="0" * 64)


class _Stub:
    name = "stub"

    def __init__(self, sizes):
        self.sizes = sizes

    def detect(self, frame):
        return _dets([(10 * i, 0, 10 * i + 8, 8, 0.9, 0) for i in range(self.sizes[frame.idx])])


def test_detect_to_cache_counts_frames_at_the_cap(tmp_path, apple_run, monkeypatch):
    run = SynthRun.open(apple_run)
    frames = run.frames[:2]
    path = cache_path(tmp_path, run.meta.run_id)
    meta = {"run_id": run.meta.run_id, "fingerprint": run.meta.fingerprint}
    stub = _Stub({frames[0].idx: 3, frames[1].idx: 5})
    moves = []

    def replace(src, dst):
        moves.append((Path(src).name, Path(dst).name))
        os.replace(src, dst)

    monkeypatch.setattr(cached, "os", SimpleNamespace(replace=replace, fsync=os.fsync))
    summary = detect_to_cache(frames, stub, path, meta=meta, max_dets=5)
    monkeypatch.undo()
    rid = run.meta.run_id                         # the cache and its sidecar each have their own temporary file
    assert moves == [(f"{rid}.npz.tmp", f"{rid}.npz"), (f"{rid}.json.tmp", f"{rid}.json")]
    assert summary["saturated"] == [frames[1].idx] and (summary["n_dets"], summary["n_frames"]) == (8, 2)
    assert summary["digest"] == cache_digest(path)
    side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert side["digest"] == summary["digest"] and side["meta"]["run_id"] == run.meta.run_id
    with pytest.raises(FileExistsError, match="cache exists"):
        detect_to_cache(frames, stub, path, meta=meta, max_dets=5)
    path.unlink()                                 # a sidecar without its cache still blocks a rewrite
    with pytest.raises(FileExistsError, match="cache exists"):
        detect_to_cache(frames, stub, path, meta=meta, max_dets=5)


def test_writers_refuse_what_the_reader_refuses(tmp_path, apple_run):
    for name, row, phrase in (("label", (10, 5, 20, 15, 0.9, 2), "labels must be 0 (fruit) or 1 (bag)"),
                              ("nan", (10, 5, np.nan, 15, 0.9, 0), "boxes must be finite")):
        p = tmp_path / f"{name}.npz"
        with pytest.raises(ValueError, match=re.escape(f"{p}: bad detection cache: {phrase}")):
            write_cache(p, {0: _dets([row])}, META)
        assert not p.exists() and not p.with_name(p.name + ".tmp").exists()
    f = SynthRun.open(apple_run).frames[0]
    p = tmp_path / "twice.npz"
    with pytest.raises(ValueError, match=re.escape(f"{p}: frame {f.idx} is listed twice")):
        detect_to_cache([f, f], _Stub({f.idx: 1}), p, meta=META, max_dets=10)
    assert not p.exists() and not p.with_suffix(".json").exists()


def test_rgb_fingerprint_and_file_sha(tmp_path):
    frames = SynthRun.open(write_run(tmp_path / "b", crop="apple", pitches=(30.0,), max_frames=2, images=True)).frames
    before = rgb_fingerprint(frames)
    assert before == rgb_fingerprint(list(reversed(frames)))
    p = frames[0].rgb_path
    assert len(file_sha256(p)) == 64
    p.write_bytes(p.read_bytes() + b"\0")
    assert rgb_fingerprint(frames) != before
    p.unlink()
    with pytest.raises(ValueError, match="no RGB image"):
        rgb_fingerprint(frames)
