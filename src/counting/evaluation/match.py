"""Matching detections to ground-truth source ids, and per-unit majority voting."""

from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from counting.data.formats.orchard_synth import GtRecord
from counting.detection.base import Detections, box_iou


def match_frame(
    dets: Detections,
    recs: Sequence[GtRecord],
    iou_min: float = 0.5,
    *,
    prefer: Callable[[GtRecord], bool] | None = None,
) -> np.ndarray:
    """Hungarian-match `dets` against every visible record in `recs` (positive and
    ignored, any row — the detector does not know rows).

    Returns, per detection, the matched record's `id`, or -1 for no match (background).
    Matching uses IoU on `bbox`; a pair is only accepted if the (unmodified) IoU is at
    least `iou_min`. Ties in IoU are broken, in order: `prefer(r)` when given, then the
    larger `vis_px`, then the smaller id — via a tiny cost bonus plus column order.
    """
    n = len(dets)
    result = np.full(n, -1, dtype=np.int64)
    visible = sorted((r for r in recs if r.bbox is not None), key=lambda r: r.id)
    if n == 0 or not visible:
        return result

    boxes = np.array([r.bbox for r in visible], dtype=np.float32)
    iou = box_iou(dets.boxes, boxes)

    max_vis_px = max((r.vis_px for r in recs), default=0)
    tie = np.array([
        (2.0 * bool(prefer(r)) if prefer is not None else 0.0) + r.vis_px / (1 + max_vis_px)
        for r in visible
    ])
    cost = (1.0 - iou) - 1e-6 * tie[None, :]

    row_ind, col_ind = linear_sum_assignment(cost)
    for i, j in zip(row_ind, col_ind):
        if iou[i, j] >= iou_min:
            result[i] = visible[j].id
    return result


def unit_majority(member_ids: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Majority source id and purity for each unit's observed ids (a track, a landmark, ...).

    The majority is the most frequent id in that unit; ties go to the smaller id (-1
    included). Purity is the majority's count divided by the unit's observation count.
    A unit with no observations raises `ValueError`; no units at all gives two empty arrays.
    """
    if len(member_ids) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    majority = np.empty(len(member_ids), dtype=np.int64)
    purity = np.empty(len(member_ids), dtype=np.float64)
    for i, ids in enumerate(member_ids):
        if len(ids) == 0:
            raise ValueError(f"unit_majority: unit {i} has no observations")
        counts = Counter(ids)
        best = max(counts.values())
        majority[i] = min(k for k, c in counts.items() if c == best)
        purity[i] = best / len(ids)
    return majority, purity
