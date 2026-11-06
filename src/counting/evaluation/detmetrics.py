"""Detection metrics of a learned detector against orchard_synth ground truth (pure functions).

Per frame, `evaluate_frame` splits the records into positives (`is_positive` under the
truth variant) and ignored records (a `bbox` but not positive), and matches the
detections greedily (`greedy_match`): in score order, each detection takes the free
positive of highest IoU >= 0.5; one that takes none but overlaps an ignored record at
IoU >= 0.5 is left out of the TP/FP counts and AP (it is counted as `ignored` and feeds
`ignored_hit`; an ignored record may absorb several), and the rest are false positives.
`fp_kinds` sorts each false positive into a duplicate of a matched positive, a merge over
two or more positives, a part of one positive, an overlap with an ignored record, or
background.

`summarize` pools frames (frame order, then detection order, with stable sorts) into
AP50 and the counts and rates at the operating and low score thresholds and at the
F1-best one, with breakdowns by visibility, box size, row, label, image edge, match IoU
and box depth. `summarize_groups` does this for all frames and per camera pitch.
`detection_verdicts` judges the val pitches against the criteria table, and
`detection_rows` does the same for a ladder's rung L5 in the shape of the ladder
verdicts. Results are JSON-shaped dicts of plain Python values; a ratio with a zero
denominator, and a percentile of an empty set, is None.

`bundle_frames` evaluates one bundle's selected frames against a `CachedDetector`, and
`score_runs` groups them per bundle and pooled over all bundles; `render_detmetrics` and
`write_detmetrics` turn the `counting eval detmetrics` result into its folder.

This module never imports torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import yaml

from counting.config.eval_schema import EvalAppConfig, LiftConfig
from counting.data.formats.orchard_synth import FrameInput, GtRecord, SynthRun, read_depth_m, select_frames
from counting.detection.base import Detections, box_iou
from counting.detection.cached import CachedDetector
from counting.evaluation.report import ReproducibilityError, results_bytes
from counting.evaluation.runner import depth_error_stats
from counting.evaluation.truth import is_positive
from counting.evaluation.verdict import JUDGED_PITCHES, _ok
from counting.multiview.lift import box_far_depth, box_median_depth, edge_touch

# `schema_version` of a `counting eval detmetrics` result (`detmetrics.json`).
DETMETRICS_SCHEMA = 1

# Bins of a box's short side (px) and of a positive's visible fraction, each `[lo, hi)`.
SIZE_BINS = ((0.0, 16.0, "<16"), (16.0, 32.0, "16-32"), (32.0, 64.0, "32-64"), (64.0, math.inf, ">=64"))
VISIB_BINS = ((0.0, 0.25, "<0.25"), (0.25, 0.35, "0.25-0.35"), (0.35, 0.5, "0.35-0.5"), (0.5, math.inf, ">=0.5"))
# `split_rate_low_visib` counts the positives below this visible fraction, where occlusion splits a fruit.
SPLIT_VISIB_MAX = 0.6
# `ignored_hit` also reports the ignored records with at least this many visible pixels.
IGNORED_BIG_PX = 100
THRESHOLD_KEYS = ("operating", "low", "f1_max")
M_KEYS = ("score", "tp", "fp", "fp_dup", "fp_merge", "fp_part", "fp_ign", "fp_bg", "ignored", "fn", "precision",
          "recall", "f1", "fp_per_frame", "split_rate", "split_rate_low_visib", "frag_only_rate")

_FP_KINDS = ("dup", "merge", "part", "ign", "bg")
_LABELS = ("fruit", "bag")              # Detections.labels: 0 fruit, 1 bag
_IGNORED_HIT_IOU = 0.5
_NO_DEPTH = {"n": 0, "p50_m": None, "p90_m": None, "p95_m": None, "near_10cm": None, "far_10cm": None}


# --- matching -----------------------------------------------------------------------------

def _boxes(boxes) -> np.ndarray:
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def _area(b: np.ndarray) -> np.ndarray:
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def _short(b: np.ndarray) -> np.ndarray:
    return np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])


def _inter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection area between xyxy boxes `a` (N,4) and `b` (M,4)."""
    w = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    h = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    return np.clip(w, 0.0, None) * np.clip(h, 0.0, None)


def _share(inter: np.ndarray, area: np.ndarray) -> np.ndarray:
    """`inter / area` (`area` broadcast against `inter`), 0 where the area is not positive."""
    area = np.broadcast_to(area, inter.shape)
    return np.divide(inter, area, out=np.zeros_like(inter), where=area > 0)


def greedy_match(boxes, scores, pos, ign, iou_min: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily match detections (`boxes` (D,4), `scores` (D,)) to positive boxes `pos` (P,4).

    Detections are taken by descending score (ties in stored order). Each takes the
    still-free positive of largest IoU >= `iou_min` (ties to the lower index); one that
    takes none is ignored when it overlaps any ignored box of `ign` (I,4) at IoU >=
    `iou_min` (an ignored box may absorb several), else it is a false positive.

    Returns `kind` (D,) int8 (1 true positive, 0 false positive, -1 ignored), `tp_of` (D,)
    int64 (the matched positive's index, else -1) and `tp_iou` (D,) float64 (the match's
    IoU, else NaN).
    """
    boxes, pos, ign = _boxes(boxes), _boxes(pos), _boxes(ign)
    n = len(boxes)
    kind = np.zeros(n, np.int8)
    tp_of = np.full(n, -1, np.int64)
    tp_iou = np.full(n, np.nan)
    iou = box_iou(boxes, pos)
    on_ignored = (box_iou(boxes, ign) >= iou_min).any(axis=1)
    free = np.ones(len(pos), bool)
    for d in np.argsort(-np.asarray(scores, np.float64), kind="stable"):
        ok = free & (iou[d] >= iou_min)
        if ok.any():
            j = int(np.argmax(np.where(ok, iou[d], -np.inf)))
            kind[d], tp_of[d], tp_iou[d] = 1, j, iou[d, j]
            free[j] = False
        elif on_ignored[d]:
            kind[d] = -1
    return kind, tp_of, tp_iou


def fp_kinds(boxes, kind, tp_of, pos, ign, *, iou_min: float = 0.5,
             part_min: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """The kind of each false positive (`kind == 0`) and the positive it points at.

    The first that applies: `dup` (IoU >= `iou_min` with an already matched positive;
    points at the one of largest IoU), `merge` (two or more positives have
    `intersection / positive area >= part_min`; points at the one of largest share, ties
    to the lower index), `part` (some positive has `intersection / detection area >=
    part_min`; points at the one of largest share), `ign` (some ignored box has
    `intersection / detection area >= part_min`; points at nothing), else `bg`.

    Returns `fp_kind` (D,) `<U5` (`""` for a detection that is not a false positive) and
    `near_of` (D,) int64 (the positive's index, else -1).
    """
    boxes, pos, ign = _boxes(boxes), _boxes(pos), _boxes(ign)
    kind, tp_of = np.asarray(kind), np.asarray(tp_of)
    out = np.full(len(boxes), "", dtype="<U5")
    near = np.full(len(boxes), -1, np.int64)
    fp = np.flatnonzero(kind == 0)
    if fp.size == 0:
        return out, near
    b = boxes[fp]
    matched = np.zeros(len(pos), bool)
    matched[tp_of[kind == 1]] = True
    dup_iou = np.where(matched[None, :], box_iou(b, pos), -1.0)
    inter = _inter(b, pos)
    cover = _share(inter, _area(pos)[None, :])             # share of each positive the detection covers
    inside = _share(inter, _area(b)[:, None])              # share of the detection inside each positive
    in_ignored = _share(_inter(b, ign), _area(b)[:, None])

    is_dup = (dup_iou >= iou_min).any(axis=1)
    is_merge = ~is_dup & (np.count_nonzero(cover >= part_min, axis=1) >= 2)
    is_part = ~is_dup & ~is_merge & (inside >= part_min).any(axis=1)
    is_ign = ~is_dup & ~is_merge & ~is_part & (in_ignored >= part_min).any(axis=1)
    sub = np.full(fp.size, "bg", dtype="<U5")
    sub_near = np.full(fp.size, -1, np.int64)
    for mask, name, share in ((is_dup, "dup", dup_iou), (is_merge, "merge", cover), (is_part, "part", inside),
                              (is_ign, "ign", None)):
        sub[mask] = name
        if share is not None and mask.any():
            sub_near[mask] = np.argmax(share[mask], axis=1)
    out[fp], near[fp] = sub, sub_near
    return out, near


def average_precision(scores, kind, n_pos: int) -> float | None:
    """All-point AP of detections of `kind` (1 TP, 0 FP, -1 ignored and left out); None when `n_pos` is 0.

    Detections are ranked by descending score (stable). The precision envelope (running
    maximum from the end) is summed at each true positive and divided by `n_pos`.
    """
    if n_pos == 0:
        return None
    scores, kind = np.asarray(scores, np.float64), np.asarray(kind)
    keep = kind != -1
    hit = kind[keep][np.argsort(-scores[keep], kind="stable")] == 1
    if not hit.any():
        return 0.0
    precision = np.cumsum(hit) / np.arange(1, hit.size + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(envelope[hit].sum() / n_pos)


def _tracker_threshold(t: float) -> float:
    """`t` as the tracker compares it with the cached float32 scores.

    NumPy compares a float32 array with a Python float in float32 (NEP 50), that is against
    `float32(t)`. Widened back to float64, that value gives the same answer on the float64 copies
    of the same scores, so a box scoring exactly `float32(t)` counts here as it counts for the
    tracker, the score floor and the score-ordered matching.
    """
    return float(np.float32(t))


def cross_class_pairs(raw: Detections, *, nms_iou: float, score: float) -> tuple[int, int]:
    """Pairs of `raw` boxes with different labels and IoU > `nms_iou`, and how many of those
    have both scores below `score` but a sum at or above it (`score` compared as the tracker
    compares it, `_tracker_threshold`)."""
    score = _tracker_threshold(score)
    s = raw.scores.astype(np.float64)
    pair = (box_iou(raw.boxes, raw.boxes) > nms_iou) & (raw.labels[:, None] != raw.labels[None, :])
    i, j = np.nonzero(np.triu(pair, k=1))
    below = (s[i] < score) & (s[j] < score) & (s[i] + s[j] >= score)
    return int(i.size), int(np.count_nonzero(below))


# --- per frame ----------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FrameEval:
    """One frame's matched detections, positives and ignored records (see `evaluate_frame`)."""

    idx: int
    pitch_deg: float
    has_depth: bool
    # per detection (D,)
    scores: np.ndarray           # float64
    kind: np.ndarray             # int8: 1 true positive, 0 false positive, -1 ignored
    tp_of: np.ndarray            # int64: matched positive, else -1
    tp_iou: np.ndarray           # float64: match IoU, else NaN
    fp_kind: np.ndarray          # <U5: dup | merge | part | ign | bg, "" when not a false positive
    near_of: np.ndarray          # int64: positive a false positive points at, else -1
    det_short: np.ndarray        # float64: short side, px
    det_label: np.ndarray        # int8: 0 fruit, 1 bag
    det_edge: np.ndarray         # bool: touches an image border
    err_far: np.ndarray          # float64: a TP's box_far depth minus its positive's depth_med, else NaN
    err_median: np.ndarray       # float64: the same with box_median
    # per positive (P,), in record order
    pos_short: np.ndarray        # float64
    pos_visib: np.ndarray        # float64
    pos_other_row: np.ndarray    # bool: tree outside the target row
    pos_bagged: np.ndarray       # bool
    pos_edge: np.ndarray         # bool
    pos_err_far: np.ndarray      # float64: truth box's box_far depth minus depth_med, NaN without depth
    pos_err_median: np.ndarray   # float64: the same with box_median
    # per ignored record (I,), in record order
    ign_vis_px: np.ndarray       # int64
    ign_hit: np.ndarray          # float64: best score of an ignored detection on it (IoU >= 0.5), else NaN


def evaluate_frame(
    dets: Detections,
    recs: Sequence[GtRecord],
    *,
    idx: int,
    pitch_deg: float,
    count1: tuple[float, int],
    target_trees: frozenset[int],
    width: int,
    height: int,
    positives: str = "count1",
    bagged_ids: frozenset[int] = frozenset(),
    depth_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
) -> FrameEval:
    """Match one frame's `dets` against its records `recs` and keep what `summarize` needs.

    Positives are the records with `is_positive(r, positives, count1)`, in record order;
    ignored records are the other records with a `bbox`. `depth_fn(boxes)` returns the
    `(box_far, box_median)` depths of (N,4) boxes; it is called once on the detection
    boxes and once on the positive truth boxes. Without it, `has_depth` is False and
    every depth error is NaN.
    """
    flags = [is_positive(r, positives, count1) for r in recs]
    pos_recs = [r for r, f in zip(recs, flags) if f]
    ign_recs = [r for r, f in zip(recs, flags) if not f and r.bbox is not None]
    pos, ign = _boxes([r.bbox for r in pos_recs]), _boxes([r.bbox for r in ign_recs])
    boxes, scores = _boxes(dets.boxes), dets.scores.astype(np.float64)

    kind, tp_of, tp_iou = greedy_match(boxes, scores, pos, ign)
    fp_kind, near_of = fp_kinds(boxes, kind, tp_of, pos, ign)

    on_ignored = (kind == -1)[:, None] & (box_iou(boxes, ign) >= _IGNORED_HIT_IOU)
    best = np.where(on_ignored, scores[:, None], -np.inf).max(axis=0, initial=-np.inf)

    n, p = len(boxes), len(pos)
    err = {k: np.full(n, np.nan) for k in ("far", "median")}
    pos_err = {k: np.full(p, np.nan) for k in ("far", "median")}
    if depth_fn is not None:
        depth_med = np.array([np.nan if r.depth_med is None else r.depth_med for r in pos_recs], np.float64)
        tp = kind == 1
        for key, depth in zip(("far", "median"), depth_fn(dets.boxes)):
            err[key][tp] = np.asarray(depth, np.float64)[tp] - depth_med[tp_of[tp]]
        for key, depth in zip(("far", "median"), depth_fn(pos)):
            pos_err[key] = np.asarray(depth, np.float64) - depth_med

    return FrameEval(
        idx=idx, pitch_deg=pitch_deg, has_depth=depth_fn is not None,
        scores=scores, kind=kind, tp_of=tp_of, tp_iou=tp_iou, fp_kind=fp_kind, near_of=near_of,
        det_short=_short(boxes), det_label=dets.labels.astype(np.int8),
        det_edge=edge_touch(boxes, width, height).any(axis=1), err_far=err["far"], err_median=err["median"],
        pos_short=_short(pos), pos_visib=np.array([r.visib for r in pos_recs], np.float64),
        pos_other_row=np.array([r.tree not in target_trees for r in pos_recs], bool),
        pos_bagged=np.array([r.id in bagged_ids for r in pos_recs], bool),
        pos_edge=edge_touch(pos, width, height).any(axis=1),
        pos_err_far=pos_err["far"], pos_err_median=pos_err["median"],
        ign_vis_px=np.array([r.vis_px for r in ign_recs], np.int64),
        ign_hit=np.where(np.isfinite(best), best, np.nan),
    )


# --- pooled summaries ---------------------------------------------------------------------

def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def _f1(tp: int, fp: int, n_pos: int) -> float | None:
    """The harmonic mean of precision and recall, or None when either is None or both are 0."""
    if tp + fp == 0 or n_pos == 0 or tp == 0:
        return None
    return float(2 * tp / (tp + fp + n_pos))


def _percentile(xs: np.ndarray, q: float) -> float | None:
    return float(np.percentile(xs, q)) if xs.size else None


def _count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


# Typed empty arrays, so that pooling no frames still gives every array its dtype.
_NO_FRAME = evaluate_frame(Detections.empty(), [], idx=-1, pitch_deg=math.nan, count1=(0.0, 0),
                           target_trees=frozenset(), width=0, height=0)
_ARRAY_FIELDS = tuple(f.name for f in fields(FrameEval) if f.name not in ("idx", "pitch_deg", "has_depth"))


def _join(frames: Sequence[FrameEval]) -> FrameEval:
    """`frames` as one: each array joined in frame order, `tp_of`/`near_of` shifted to index the joined positives."""
    parts = [_NO_FRAME, *frames]
    offsets = np.cumsum([0] + [len(f.pos_visib) for f in parts])

    def column(name: str) -> np.ndarray:
        cols = [getattr(f, name) for f in parts]
        if name in ("tp_of", "near_of"):
            cols = [np.where(c >= 0, c + off, -1) for c, off in zip(cols, offsets)]
        return np.concatenate(cols)

    return FrameEval(idx=-1, pitch_deg=math.nan, has_depth=any(f.has_depth for f in frames),
                     **{name: column(name) for name in _ARRAY_FIELDS})


def _per_pos(j: FrameEval, mask: np.ndarray, of: np.ndarray) -> np.ndarray:
    return np.bincount(of[mask], minlength=len(j.pos_visib))


def _at(j: FrameEval, n_frames: int, t: float) -> dict:
    """The `M_KEYS` of the joined frames `j`, counting the detections that score at least `t`
    (compared as the tracker compares it, `_tracker_threshold`; `score` stays `t`)."""
    n_pos = len(j.pos_visib)
    above = j.scores >= _tracker_threshold(t)
    tp_m, fp_m = above & (j.kind == 1), above & (j.kind == 0)
    tp, fp = _count(tp_m), _count(fp_m)
    tp_per = _per_pos(j, tp_m, j.tp_of)
    part_per = _per_pos(j, fp_m & (j.fp_kind == "part"), j.near_of)
    split = tp_per + _per_pos(j, fp_m & (j.fp_kind == "dup"), j.near_of) + part_per >= 2
    low_visib = j.pos_visib < SPLIT_VISIB_MAX
    return {"score": float(t), "tp": tp, "fp": fp, **{f"fp_{k}": _count(fp_m & (j.fp_kind == k)) for k in _FP_KINDS},
            "ignored": _count(above & (j.kind == -1)), "fn": n_pos - tp,
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, n_pos), "f1": _f1(tp, fp, n_pos),
            "fp_per_frame": _ratio(fp, n_frames), "split_rate": _ratio(_count(split), n_pos),
            "split_rate_low_visib": _ratio(_count(split & low_visib), _count(low_visib)),
            "frag_only_rate": _ratio(_count((tp_per == 0) & (part_per >= 1)), n_pos)}


def _f1_best(j: FrameEval) -> float | None:
    """The detection score (ignored ones left out) whose threshold has the largest F1, ties to the
    higher score; None when no threshold has an F1."""
    keep = j.kind != -1
    s, hit = j.scores[keep], j.kind[keep] == 1
    cand = np.unique(s)                                                  # ascending
    n_ge = s.size - np.searchsorted(np.sort(s), cand, side="left")      # detections at or above each
    tp = np.concatenate([[0], np.cumsum(hit[np.argsort(-s, kind="stable")])])[n_ge]
    n_pos = len(j.pos_visib)
    if n_pos == 0 or not (tp > 0).any():
        return None
    f1 = np.where(tp > 0, 2 * tp / (n_ge + n_pos), -1.0)               # the same value `_f1` gives
    return float(cand[len(cand) - 1 - np.argmax(f1[::-1])])


def _cell(mask: np.ndarray, hit: np.ndarray, hit_key: str, rate_key: str) -> dict:
    n, k = _count(mask), _count(hit & mask)
    return {"n": n, hit_key: k, rate_key: _ratio(k, n)}


def _bins(values: np.ndarray, hit: np.ndarray, bins, rate_key: str) -> dict:
    return {name: _cell((values >= lo) & (values < hi), hit, "tp", rate_key) for lo, hi, name in bins}


def _depth_cell(err: np.ndarray) -> dict:
    finite = np.isfinite(err)
    return {**(depth_error_stats(err[finite]) or _NO_DEPTH), "nan_rate": _ratio(_count(~finite), err.size)}


def summarize(frames: Sequence[FrameEval], thresholds: Mapping[str, float]) -> dict:
    """Pool `frames` into AP50, the `M_KEYS` at each of `THRESHOLD_KEYS`, and breakdowns.

    `thresholds` is `{"operating": t_hi, "low": t_lo}`; a detection counts at a threshold
    when its score is at least it, compared as the tracker compares it (`_tracker_threshold`).
    `f1_max` is the detection score of largest F1 (every
    value None when no threshold has an F1). Recall by visibility is given at both
    thresholds; every other breakdown is at the operating one.
    """
    j = _join(frames)
    n_frames, n_pos = len(frames), len(j.pos_visib)
    t_hi, t_lo = float(thresholds["operating"]), float(thresholds["low"])
    best = _f1_best(j)
    at = {"operating": _at(j, n_frames, t_hi), "low": _at(j, n_frames, t_lo),
          "f1_max": {k: None for k in M_KEYS} if best is None else _at(j, n_frames, best)}

    def seen_at(t: float) -> np.ndarray:
        return _per_pos(j, (j.scores >= _tracker_threshold(t)) & (j.kind == 1), j.tp_of) > 0

    above = j.scores >= _tracker_threshold(t_hi)
    tp_m, judged = above & (j.kind == 1), above & (j.kind != -1)
    tp_pos = j.tp_of[tp_m]
    seen = seen_at(t_hi)
    truth_bag, det_bag = j.pos_bagged[tp_pos], j.det_label[tp_m] == 1
    truth_edge, det_edge = j.pos_edge[tp_pos], j.det_edge[tp_m]
    ign_hit = j.ign_hit >= _tracker_threshold(t_hi)
    depth = None
    if j.has_depth:
        depth = {"box_far": _depth_cell(j.err_far[tp_m]), "box_median": _depth_cell(j.err_median[tp_m]),
                 "truth_box_far": _depth_cell(j.pos_err_far), "truth_box_median": _depth_cell(j.pos_err_median)}
    return {
        "n_frames": n_frames, "n_pos": n_pos, "n_ignore": len(j.ign_vis_px),
        "ap50": average_precision(j.scores, j.kind, n_pos),
        "at": {k: at[k] for k in THRESHOLD_KEYS},
        "recall_by_visib": {name: _bins(j.pos_visib, seen_at(t), VISIB_BINS, "recall")
                            for name, t in (("operating", t_hi), ("low", t_lo))},
        "recall_by_size": _bins(j.pos_short, seen, SIZE_BINS, "recall"),
        "precision_by_size": _bins(j.det_short[judged], j.kind[judged] == 1, SIZE_BINS, "precision"),
        "recall_by_row": {"target": _cell(~j.pos_other_row, seen, "tp", "recall"),
                          "other": _cell(j.pos_other_row, seen, "tp", "recall")},
        "recall_by_edge": {"edge": _cell(j.pos_edge, seen, "tp", "recall"),
                           "inner": _cell(~j.pos_edge, seen, "tp", "recall")},
        "ignored_hit": {"all": _cell(np.ones_like(ign_hit), ign_hit, "hit", "rate"),
                        f"vis_px>={IGNORED_BIG_PX}": _cell(j.ign_vis_px >= IGNORED_BIG_PX, ign_hit, "hit", "rate")},
        "other_row_share": _ratio(_count(j.pos_other_row[tp_pos]), tp_pos.size),
        "label_confusion": {f"{_LABELS[a]}_{_LABELS[b]}": _count((truth_bag == a) & (det_bag == b))
                            for a in (0, 1) for b in (0, 1)},
        "edge": {"both": _count(truth_edge & det_edge), "truth_only": _count(truth_edge & ~det_edge),
                 "det_only": _count(~truth_edge & det_edge), "neither": _count(~truth_edge & ~det_edge)},
        "iou": {f"p{q}": _percentile(j.tp_iou[tp_m], q) for q in (10, 50, 90)},
        "depth": depth,
    }


def summarize_groups(frames: Sequence[FrameEval], thresholds: Mapping[str, float]) -> dict[str, dict]:
    """`summarize` over all `frames` (`"all"`), then over each camera pitch present (`"pitch_<deg>"`, ascending)."""
    groups = {"all": summarize(frames, thresholds)}
    for pitch in sorted({f.pitch_deg for f in frames}):
        groups[f"pitch_{pitch:g}"] = summarize([f for f in frames if f.pitch_deg == pitch], thresholds)
    return groups


def _detection_spec(criteria: Mapping | None, crop: str) -> Mapping | None:
    """`criteria["detection"]["L5"][crop]`, or None when any level is missing."""
    if not criteria:
        return None
    return ((criteria.get("detection") or {}).get("L5") or {}).get(crop)


def _judged_points(groups: Mapping[str, dict], spec: Mapping, crop: str) -> list[dict]:
    """One `{group, metric, op, limit, value, ok}` per `JUDGED_PITCHES[crop]` group and metric (`recall`,
    then `precision`) of `spec`, from the group's operating point; `value` and `ok` are None when the
    group is missing, `ok` also when the value is."""
    rows = []
    for pitch in JUDGED_PITCHES[crop]:
        name = f"pitch_{pitch:g}"
        point = groups[name]["at"]["operating"] if name in groups else None
        for metric in ("recall", "precision"):
            (op, limit), = spec[metric].items()
            value = None if point is None else point[metric]
            rows.append({"group": name, "metric": metric, "op": op, "limit": limit, "value": value,
                         "ok": _ok(value, op, limit)})
    return rows


def detection_verdicts(groups: Mapping[str, dict], criteria: Mapping | None, crop: str,
                       split: str) -> list[dict] | None:
    """Judge the operating-point recall and precision of the `JUDGED_PITCHES[crop]` groups.

    None unless `split` is `val` and `criteria["detection"]["L5"][crop]` exists. One row
    per pitch and metric (`recall`, then `precision`): `{group, metric, op, limit, value,
    ok}`; `value` and `ok` are None when the group is missing, `ok` also when the value is.
    """
    spec = _detection_spec(criteria, crop)
    if split != "val" or not spec:
        return None
    return _judged_points(groups, spec, crop)


def detection_rows(detection: Mapping, criteria: Mapping | None, crop: str) -> list[dict]:
    """The rows of `detection_verdicts`, judged on `detection["pooled"]` (`score_runs`' shape) in the
    shape of `verdict.ladder_verdicts`' rows, on any split.

    `[]` without `criteria["detection"]["L5"][crop]`. Per `JUDGED_PITCHES[crop]` group, a recall
    then a precision row: `{metric: "detection.<m>", group, op, limit, unit: "pooled", value,
    per_run, ok}`, `per_run` being `[{run_id, seed, value, spread: None, ok}]` from each bundle's
    own groups in `detection["runs"]` order.
    """
    spec = _detection_spec(criteria, crop)
    if not spec:
        return []
    scenes = [(entry["meta"], _judged_points(entry["groups"], spec, crop)) for entry in detection["runs"]]
    rows = []
    for i, point in enumerate(_judged_points(detection["pooled"]["groups"], spec, crop)):
        per_run = [{"run_id": meta["run_id"], "seed": meta["seed"], "value": points[i]["value"], "spread": None,
                    "ok": points[i]["ok"]} for meta, points in scenes]
        rows.append({"metric": f"detection.{point['metric']}", "group": point["group"], "op": point["op"],
                     "limit": point["limit"], "unit": "pooled", "value": point["value"], "per_run": per_run,
                     "ok": point["ok"]})
    return rows


# --- bundles ------------------------------------------------------------------------------

def describe_bundle(run: SynthRun, n_frames: int) -> dict:
    """Bundle identity for results (no paths): the keys of `runner.describe_run` plus the
    bundle's `depth_fingerprint`. `n_frames` counts the frames scored."""
    m = run.meta
    return {"run_id": m.run_id, "crop": m.crop, "preset": m.preset, "seed": m.seed, "samples": m.samples,
            "time": m.time, "n_frames": n_frames, "is_sequence": m.is_sequence, "base_step_m": m.base_step_m,
            "fingerprint": m.fingerprint, "synth_commit": m.synth_commit,
            "depth_fingerprint": run.depth_fingerprint()}


def _selected(run: SynthRun, cfg: EvalAppConfig, frames: Sequence[int] | None) -> list[FrameInput]:
    """The frames `cfg.passes` selects (`select_frames`), in idx order; only the idx in `frames` when given."""
    chosen = select_frames(run.frames, base_step_m=run.meta.base_step_m, pitches=cfg.passes.pitches,
                           sides=cfg.passes.sides, step_m=cfg.passes.step_m)
    if frames is None:
        return chosen
    keep = set(frames)
    return [f for f in chosen if f.idx in keep]


def _box_depths(depth: np.ndarray, lift: LiftConfig) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """`depth_fn` of `evaluate_frame` on one depth image (m): the `(box_far, box_median)` depths of
    (N,4) boxes, with `lift`'s parameters."""
    def depth_fn(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (box_far_depth(boxes, depth, lift.far_scale, lift.far_gap_m, lift.far_min_frac),
                box_median_depth(boxes, depth, lift.box_frac))
    return depth_fn


def bundle_frames(run: SynthRun, cache: CachedDetector, cfg: EvalAppConfig, *,
                  frames: Sequence[int] | None = None) -> list[FrameEval]:
    """`evaluate_frame` of `cache`'s detections on each frame of `run` that `cfg.passes` selects
    (only the idx in `frames`, when given), in idx order.

    Positives follow `cfg.detector.positives` and the target row is the layout's target-row
    trees. Each frame's depth16 image gives the box depths (`_box_depths` with `cfg.lift`).
    """
    truth = run.truth()
    target = frozenset(run.layout.tree_ids)
    out = []
    for f in _selected(run, cfg, frames):
        recs = truth.records[f.idx]
        bagged = frozenset(r.id for r in recs if r.vis_px > 0 and truth.fruits[r.id].bagged)
        out.append(evaluate_frame(
            cache.detect(f), recs, idx=f.idx, pitch_deg=f.pass_key.pitch_deg, count1=truth.count1,
            target_trees=target, width=f.camera.width, height=f.camera.height, positives=cfg.detector.positives,
            bagged_ids=bagged, depth_fn=_box_depths(read_depth_m(f.depth_path), cfg.lift)))
    return out


def _cross_class(pairs: int, split_score: int, n_frames: int) -> dict:
    return {"pairs": pairs, "pairs_per_frame": _ratio(pairs, n_frames), "split_score": split_score}


def score_runs(runs: Sequence[SynthRun], caches: Sequence[CachedDetector], cfg: EvalAppConfig, *,
               frames: Sequence[int] | None = None) -> dict:
    """Score each bundle of `runs` with the cache at the same position of `caches`, then all together.

    The thresholds are `{"operating": cfg.tracker.score_hi, "low": cfg.tracker.score_lo}`.
    Per bundle: `meta` (`describe_bundle`), `cache` (file name, content digest and the cache
    meta's `weights_sha256`), `groups` (`summarize_groups` of `bundle_frames`) and
    `cross_class`, the `cross_class_pairs` of the raw cached detections (at the cache's NMS
    IoU and the operating threshold) summed over the scored frames (`pairs_per_frame` is
    None without frames). `pooled` has the groups of every bundle's frames together and the
    summed `cross_class`.
    """
    if len(runs) != len(caches):
        raise ValueError(f"score_runs needs one cache per bundle, got {len(runs)} bundles and {len(caches)} caches")
    thresholds = {"operating": cfg.tracker.score_hi, "low": cfg.tracker.score_lo}
    entries, pooled = [], []
    for run, cache in zip(runs, caches):
        evals = bundle_frames(run, cache, cfg, frames=frames)
        pairs = split_score = 0
        for f in evals:
            n, below = cross_class_pairs(cache.raw(f.idx), nms_iou=cache.nms_iou, score=cfg.tracker.score_hi)
            pairs, split_score = pairs + n, split_score + below
        entries.append({"meta": describe_bundle(run, len(evals)),
                        "cache": {"file": cache.path.name, "digest": cache.digest,
                                  "weights_sha256": cache.meta.get("weights_sha256")},
                        "groups": summarize_groups(evals, thresholds),
                        "cross_class": _cross_class(pairs, split_score, len(evals))})
        pooled += evals
    total = _cross_class(sum(e["cross_class"]["pairs"] for e in entries),
                         sum(e["cross_class"]["split_score"] for e in entries), len(pooled))
    return {"runs": entries, "pooled": {"groups": summarize_groups(pooled, thresholds), "cross_class": total}}


# --- report and folder --------------------------------------------------------------------

_GROUP_HEADERS = ["묶음", "그룹", "프레임", "양성", "무시", "AP50", "P", "R", "F1", "오검출/프레임", "중복", "합침", "조각",
                  "무시 겹침", "배경", "쪼개짐", "조각만"]
_CROSS_HEADERS = ["묶음", "프레임", "쌍", "쌍/프레임", "둘 다 운영 문턱 아래, 합은 위"]
_VERDICT_HEADERS = ["그룹", "지표", "기준", "값", "판정"]
_OP_SYMBOL = {"le": "≤", "ge": "≥"}
_POOLED = "전체"


def _num(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _group_rows(label: str, groups: Mapping[str, dict]) -> list[list[str]]:
    rows = []
    for name, g in groups.items():
        op = g["at"]["operating"]
        rows.append([label, name, str(g["n_frames"]), str(g["n_pos"]), str(g["n_ignore"]), _num(g["ap50"]),
                     _num(op["precision"]), _num(op["recall"]), _num(op["f1"]), _num(op["fp_per_frame"]),
                     *(str(op[f"fp_{k}"]) for k in _FP_KINDS), _num(op["split_rate"]), _num(op["frag_only_rate"])])
    return rows


def _cross_row(label: str, n_frames: int, cc: dict) -> list[str]:
    return [label, str(n_frames), str(cc["pairs"]), _num(cc["pairs_per_frame"]), str(cc["split_score"])]


def render_detmetrics(result: dict) -> str:
    """The `report.md` of a `counting eval detmetrics` result: every bundle's groups at the
    operating threshold (and the pooled ones when there are several bundles), the
    cross-class overlap counts, and the verdicts when there are any."""
    runs = result["runs"]
    multi = len(runs) > 1
    t, post = result["thresholds"], result["postprocess"]
    lines = [f"# detmetrics {result['run_name']} {result['experiment_id']}",
             f"> 합성 렌더 · {result['crop']} {result['split']} · 운영 문턱 {t['operating']:g}, 낮은 문턱 {t['low']:g} · "
             f"NMS IoU {post['nms_iou']:g} · 가장자리 맞춤 {post['edge_snap_px']:g} px",
             ""]
    rows = [row for run in runs for row in _group_rows(run["meta"]["run_id"], run["groups"])]
    if multi:
        rows += _group_rows(_POOLED, result["pooled"]["groups"])
    lines.append(f"**운영 문턱({t['operating']:g})의 검출**")
    lines += _table(_GROUP_HEADERS, rows)
    lines.append("")

    cross = [_cross_row(run["meta"]["run_id"], run["meta"]["n_frames"], run["cross_class"]) for run in runs]
    if multi:
        cross.append(_cross_row(_POOLED, result["pooled"]["groups"]["all"]["n_frames"],
                                result["pooled"]["cross_class"]))
    lines.append("**종류가 다른 겹친 상자(NMS 전)**")
    lines += _table(_CROSS_HEADERS, cross)
    lines.append("")

    verdicts = result.get("verdicts")
    if verdicts:
        rows = [[v["group"], v["metric"], f"{_OP_SYMBOL[v['op']]}{v['limit']:g}", _num(v["value"]),
                 "-" if v["ok"] is None else ("OK" if v["ok"] else "FAIL")] for v in verdicts]
        lines.append("**판정**")
        lines += _table(_VERDICT_HEADERS, rows)
        lines.append(f"- 통과 {sum(1 for v in verdicts if v['ok'])} / 항목 {len(verdicts)} "
                     "(값이 없는 항목은 통과로 세지 않는다)")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_detmetrics(out_root: Path, *, cfg: EvalAppConfig, result: dict, record: dict) -> tuple[Path, bool]:
    """Write `out_root/<run_name>/<experiment_id>/`: `config.yaml` (`cfg`, as the eval experiment
    folders hold it), `detmetrics.json` (`results_bytes(result)`), `record.json` and `report.md`.

    If a different `detmetrics.json` is already there, raises `ReproducibilityError` before
    writing anything; an identical one makes `reproduced` True. Returns `(folder, reproduced)`.
    """
    folder = Path(out_root) / result["run_name"] / result["experiment_id"]
    new_bytes = results_bytes(result)
    path = folder / "detmetrics.json"
    reproduced = False
    if path.exists():
        if path.read_bytes() != new_bytes:
            raise ReproducibilityError(str(folder))
        reproduced = True
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=True, allow_unicode=True), encoding="utf-8")
    (folder / "record.json").write_bytes(results_bytes(record))
    path.write_bytes(new_bytes)
    (folder / "report.md").write_text(render_detmetrics(result), encoding="utf-8")
    return folder, reproduced
