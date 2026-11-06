"""The on-disk detection cache: a torch-free writer/reader and `CachedDetector`.

A cache holds one bundle's raw detector output, one `.npz` file per run id. `write_cache`
and `read_cache` round-trip it; `cache_digest` hashes its content (not its file bytes, so a
rewrite with the same arrays and meta digests the same). `postprocess` turns raw per-frame
detections into the detector's actual output (clip, drop degenerate boxes, class-agnostic
NMS, edge-snap, an optional minimum box size off the border); `CachedDetector` is a
`Detector` that reads a cache once and applies it.

Only `FrameInput` is imported from `counting.data.formats.orchard_synth` (no `SceneTruth`),
and only `Detections`/`Detector`/`box_iou` from `counting.detection.base`: this module never
imports torch or `counting.evaluation`, so it stays cheap to import at eval time.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
import zlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from counting.data.formats.orchard_synth import FrameInput
from counting.detection.base import Detections, Detector, box_iou

CACHE_FORMAT = "counting-detections/1"
NMS_IOU = 0.5
EDGE_SNAP_PX = 2.0
CACHE_ARRAYS = ("frame_idx", "offsets", "boxes", "scores", "labels")


def file_sha256(path: Path) -> str:
    """The hex sha256 of the file at `path`, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rgb_fingerprint(frames: Sequence[FrameInput]) -> str:
    """sha256 over (rgb file name, sha256) of every frame's RGB image, in idx order.

    Order-independent in `frames` (sorted by idx first), so it matches regardless of how the
    frames were listed. A missing RGB image raises `ValueError` naming its path.
    """
    pairs = []
    for f in sorted(frames, key=lambda fr: fr.idx):
        if not f.rgb_path.is_file():
            raise ValueError(f"{f.rgb_path}: no RGB image")
        pairs.append((f.rgb_path.name, file_sha256(f.rgb_path)))
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode("utf-8")).hexdigest()


def cache_path(cache_dir: Path, run_id: str) -> Path:
    return Path(cache_dir) / f"{run_id}.npz"


def raw_order(dets: Detections) -> np.ndarray:
    """The storage/output order: x0, y0, x1, y1, label ascending, score descending."""
    x0, y0, x1, y1 = dets.boxes[:, 0], dets.boxes[:, 1], dets.boxes[:, 2], dets.boxes[:, 3]
    return np.lexsort((-dets.scores, dets.labels, y1, x1, y0, x0))


def _exists_error(path: Path) -> FileExistsError:
    return FileExistsError(
        f"{path}: cache exists; a cache is written once per weights, so remove it first to make it again"
    )


def _bad(path: Path, reason: str) -> ValueError:
    return ValueError(f"{path}: bad detection cache: {reason}")


def write_cache(path: Path, dets: Mapping[int, Detections], meta: Mapping) -> str:
    """Write `dets` (one `Detections` per frame idx) and `meta` to `path`; return `cache_digest(path)`.

    `path` must not exist yet. No `Detections` may carry `depth_m` (a detection cache stores no
    depth), and `meta` must have `run_id` and `fingerprint` (the identity a reader checks against).
    The arrays pass the reader's `_check_arrays` before anything is written, so a cache the reader
    would refuse is never written. The write is atomic: a sibling `<file name>.tmp` is written,
    fsynced and `os.replace`d onto `path`.
    """
    path = Path(path)
    if path.exists():
        raise _exists_error(path)
    if any(d.depth_m is not None for d in dets.values()):
        raise ValueError("a detection cache stores no depth")
    if "run_id" not in meta or "fingerprint" not in meta:
        raise ValueError("cache meta needs run_id and fingerprint")

    idxs = sorted(dets)
    ordered = [dets[i].subset(raw_order(dets[i])) for i in idxs]
    boxes = np.concatenate([d.boxes for d in ordered], axis=0) if ordered else np.zeros((0, 4), np.float32)
    scores = np.concatenate([d.scores for d in ordered], axis=0) if ordered else np.zeros(0, np.float32)
    labels = np.concatenate([d.labels for d in ordered], axis=0) if ordered else np.zeros(0, np.int8)
    frame_idx = np.array(idxs, dtype=np.int32)
    offsets = np.array([0, *np.cumsum([len(d) for d in ordered])], dtype=np.int64)
    _check_arrays(path, frame_idx, offsets, boxes, scores, labels)
    meta_json = json.dumps({**meta, "format": CACHE_FORMAT}, sort_keys=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, frame_idx=frame_idx, offsets=offsets, boxes=boxes, scores=scores,
                            labels=labels, meta=np.array(meta_json))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return cache_digest(path)


def _open_npz(path: Path) -> dict[str, np.ndarray]:
    """The arrays of the npz at `path`, as a plain dict, after every check on the file itself.

    A file `np.load` cannot read (not a zip, a corrupt compressed member, pickled data), a single
    `.npy` array under an `.npz` name, or an archive whose array set is not `CACHE_ARRAYS` plus
    `meta` raises `ValueError` naming `path`. `read_cache`, `cache_digest` and `CachedDetector`
    all read through here, so they refuse the same files.
    """
    try:
        obj = np.load(path, allow_pickle=False)
        if isinstance(obj, np.lib.npyio.NpzFile):      # a `.npy` comes back as one array, not a context manager
            with obj as data:
                arrays = {k: data[k] for k in data.files}
    except (zipfile.BadZipFile, EOFError, OSError, zlib.error) as e:
        raise _bad(path, f"not a readable npz ({e})") from None
    except ValueError as e:
        raise _bad(path, str(e)) from None
    if not isinstance(obj, np.lib.npyio.NpzFile):
        raise _bad(path, "not a readable npz (a single array, not an archive)")
    if set(arrays) != {*CACHE_ARRAYS, "meta"}:
        raise _bad(path, f"expected arrays {{{', '.join((*CACHE_ARRAYS, 'meta'))}}}, got {sorted(arrays)}")
    return arrays


def _check_arrays(path: Path, frame_idx: np.ndarray, offsets: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                  labels: np.ndarray) -> None:
    """Refuse cache arrays a reader cannot trust; raises `ValueError` naming `path`.

    Checks every array's dtype and shape, that `offsets` starts at 0, never decreases and ends at
    the detection count, that `frame_idx` strictly increases, that `labels` is 0 or 1 and that
    every box is finite.
    """
    for name, arr, dtype in (("frame_idx", frame_idx, np.int32), ("offsets", offsets, np.int64),
                             ("boxes", boxes, np.float32), ("scores", scores, np.float32),
                             ("labels", labels, np.int8)):
        if arr.dtype != dtype:
            raise _bad(path, f"{name} has dtype {arr.dtype}, expected {np.dtype(dtype)}")
    if frame_idx.ndim != 1 or offsets.ndim != 1 or scores.ndim != 1 or labels.ndim != 1:
        raise _bad(path, "frame_idx, offsets, scores and labels must be 1-D")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise _bad(path, f"boxes has shape {boxes.shape}, expected (N, 4)")

    n_frames, n_dets = len(frame_idx), len(boxes)
    if offsets.shape != (n_frames + 1,):
        raise _bad(path, f"offsets has shape {offsets.shape}, expected ({n_frames + 1},)")
    if len(scores) != n_dets or len(labels) != n_dets:
        raise _bad(path, "scores and labels must match boxes in length")
    if offsets[0] != 0 or offsets[-1] != n_dets:
        raise _bad(path, f"offsets must start at 0 and end at {n_dets}, got {offsets[0]} .. {offsets[-1]}")
    if np.any(np.diff(offsets) < 0):
        raise _bad(path, "offsets must not decrease")
    if n_frames > 1 and np.any(np.diff(frame_idx) <= 0):
        raise _bad(path, "frame_idx must strictly increase")
    if not np.all(np.isin(labels, (0, 1))):
        raise _bad(path, "labels must be 0 (fruit) or 1 (bag)")
    if not np.all(np.isfinite(boxes)):
        raise _bad(path, "boxes must be finite")


def _parse(path: Path, data: Mapping[str, np.ndarray]) -> tuple[dict, dict[int, Detections]]:
    """(meta, {frame idx: Detections}) from the arrays `_open_npz` returned for `path`."""
    try:
        meta = json.loads(data["meta"].item())
    except (ValueError, TypeError) as e:
        raise _bad(path, f"meta is not a JSON string ({e})") from None
    if not isinstance(meta, dict):
        raise _bad(path, "meta is not a JSON object")
    if meta.get("format") != CACHE_FORMAT:
        raise _bad(path, f"format is {meta.get('format')!r}, expected {CACHE_FORMAT!r}")

    frame_idx, offsets, boxes, scores, labels = (data[k] for k in CACHE_ARRAYS)
    _check_arrays(path, frame_idx, offsets, boxes, scores, labels)

    out: dict[int, Detections] = {}
    for i, idx in enumerate(frame_idx.tolist()):
        s, e = int(offsets[i]), int(offsets[i + 1])
        out[idx] = Detections(boxes[s:e].copy(), scores[s:e].copy(), labels[s:e].copy())
    return meta, out


def _digest(data: Mapping[str, np.ndarray]) -> str:
    """sha256 over the meta JSON, then each `CACHE_ARRAYS` array's (name, dtype, shape, bytes)."""
    h = hashlib.sha256()
    h.update(str(data["meta"].item()).encode("utf-8"))
    for name in CACHE_ARRAYS:
        arr = data[name]
        h.update(f"\n{name}:{arr.dtype.str}:{arr.shape}:".encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def read_cache(path: Path) -> tuple[dict, dict[int, Detections]]:
    """Read and validate a detection cache; return (meta, {frame idx: Detections}).

    Checks the file and its array set (`_open_npz`), the format tag, every array's dtype and
    shape, that `offsets` starts at 0, never decreases and ends at the detection count, that
    `labels` is 0 or 1, that every box is finite, and that `frame_idx` strictly increases. Any
    violation raises `ValueError`; the reader and the writers share `_check_arrays`.
    """
    path = Path(path)
    return _parse(path, _open_npz(path))


def cache_digest(path: Path) -> str:
    """sha256 of the cache's content: the meta JSON, then each `CACHE_ARRAYS` array's (name,
    dtype, shape, bytes). Independent of the file's own bytes (the npz/zip framing). It refuses
    what `read_cache` refuses about the file and its array set (`ValueError` naming `path`)."""
    path = Path(path)
    return _digest(_open_npz(path))


def postprocess(
    dets: Detections, *, width: int, height: int, nms_iou: float = NMS_IOU, edge_snap_px: float = EDGE_SNAP_PX,
    min_short_px: float = 0.0
) -> Detections:
    """Raw cache detections turned into a detector's actual output, for one frame of `width` x `height`.

    In order: clip boxes to the screen, drop boxes with zero width or height, run class-agnostic
    greedy NMS (score-descending, ties keep their original order; a box is dropped once its IoU
    with a kept, higher-scoring box exceeds `nms_iou`), snap near-edge sides onto the border
    (within `edge_snap_px`), drop boxes whose shorter side is under `min_short_px` unless they
    touch the border (a cut-off box is small because the screen cut it), then store in
    `raw_order`. With `min_short_px` 0 nothing is dropped by size. `depth_m` is always `None`.
    """
    boxes = dets.boxes.astype(np.float64, copy=True)
    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0.0, float(width))
    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0.0, float(height))
    keep = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    boxes, scores, labels = boxes[keep], dets.scores[keep], dets.labels[keep]

    order = np.argsort(-scores, kind="stable")
    boxes, scores, labels = boxes[order], scores[order], labels[order]
    alive = np.ones(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if not alive[i]:
            continue
        later = np.flatnonzero(alive[i + 1:]) + i + 1
        if len(later) == 0:
            continue
        ious = box_iou(boxes[i:i + 1], boxes[later])[0]
        alive[later[ious > nms_iou]] = False
    boxes, scores, labels = boxes[alive], scores[alive], labels[alive]

    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x0 = np.where(x0 < edge_snap_px, 0.0, x0)
    y0 = np.where(y0 < edge_snap_px, 0.0, y0)
    x1 = np.where(x1 > width - edge_snap_px, float(width), x1)
    y1 = np.where(y1 > height - edge_snap_px, float(height), y1)

    inside = (x0 > 0) & (y0 > 0) & (x1 < width) & (y1 < height)
    keep = ~(inside & (np.minimum(x1 - x0, y1 - y0) < min_short_px))
    x0, y0, x1, y1, scores, labels = x0[keep], y0[keep], x1[keep], y1[keep], scores[keep], labels[keep]

    out = Detections(np.stack([x0, y0, x1, y1], axis=1).astype(np.float32),
                     scores.astype(np.float32), labels.astype(np.int8))
    return out.subset(raw_order(out))


class CachedDetector:
    """A `Detector` that replays a written cache: raw boxes through `postprocess`.

    The file is read once: its meta, detections and `digest` come from the same read. A
    `fingerprint` must match the cache's bundle, and a pinned `digest` the cache's content.
    """

    name = "cached"

    def __init__(
        self, path: Path, *, fingerprint: str | None = None, digest: str | None = None, nms_iou: float = NMS_IOU,
        edge_snap_px: float = EDGE_SNAP_PX, min_short_px: float = 0.0
    ) -> None:
        self.path = Path(path)
        data = _open_npz(self.path)
        self.meta, self._dets = _parse(self.path, data)
        self.digest = _digest(data)
        if fingerprint is not None and self.meta.get("fingerprint") != fingerprint:
            raise ValueError(
                f"{self.path}: cache was made for bundle fingerprint {self.meta.get('fingerprint', '')[:12]}, "
                f"this bundle has {fingerprint[:12]}"
            )
        if digest is not None and self.digest != digest:
            raise ValueError(f"{self.path}: cache digest {self.digest[:16]} differs from the pinned {digest[:16]}")
        self.nms_iou = nms_iou
        self.edge_snap_px = edge_snap_px
        self.min_short_px = min_short_px

    def raw(self, idx: int) -> Detections:
        """The stored detections for frame `idx`, unprocessed."""
        if idx not in self._dets:
            raise ValueError(f"{self.path}: no detections stored for frame {idx}")
        return self._dets[idx]

    def detect(self, frame: FrameInput) -> Detections:
        if frame.run_id != self.meta.get("run_id"):
            raise ValueError(
                f"{self.path}: cache is for run {self.meta.get('run_id')}, frame is from run {frame.run_id}"
            )
        return postprocess(self.raw(frame.idx), width=frame.camera.width, height=frame.camera.height,
                           nms_iou=self.nms_iou, edge_snap_px=self.edge_snap_px, min_short_px=self.min_short_px)


def detect_to_cache(
    frames: Sequence[FrameInput], detector: Detector, path: Path, *, meta: Mapping, max_dets: int
) -> dict:
    """Run `detector` over `frames`, write the raw detections to `path`, and return a summary.

    Refuses to run if the cache file or its sidecar JSON record already exists (a cache is
    written once) or if a frame idx is listed twice; both are checked before the detector runs.
    A frame whose detector output reaches `max_dets` is listed in `saturated` (detections past
    the cap, if any, are simply what the detector returned). The sidecar
    `path.with_suffix(".json")` mirrors the returned summary, for reading without `numpy`.
    """
    path = Path(path)
    side_path = path.with_suffix(".json")
    if path.exists() or side_path.exists():
        raise _exists_error(path)
    seen: set[int] = set()
    for f in frames:
        if f.idx in seen:
            raise ValueError(f"{path}: frame {f.idx} is listed twice")
        seen.add(f.idx)

    start = time.perf_counter()
    dets: dict[int, Detections] = {}
    saturated = []
    n_dets = 0
    for f in frames:
        d = detector.detect(f)
        dets[f.idx] = d
        n_dets += len(d)
        if len(d) >= max_dets:
            saturated.append(f.idx)
    seconds = time.perf_counter() - start

    digest = write_cache(path, dets, meta)
    summary = {"file": path.name, "digest": digest, "meta": dict(meta), "n_frames": len(frames),
               "n_dets": n_dets, "saturated": sorted(saturated), "seconds": seconds}
    side_tmp = side_path.with_name(side_path.name + ".tmp")
    side_tmp.write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(side_tmp, side_path)
    return summary
