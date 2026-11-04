"""Oracle detector: emits ground-truth boxes directly instead of reading the frame's PNG,
optionally with seeded detection noise."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from counting.config.eval_schema import DetNoise
from counting.data.formats.orchard_synth import FrameInput, SceneTruth
from counting.detection.base import Detections
from counting.evaluation.truth import is_positive
from counting.geometry.noise import noise_rng


def _output_order(boxes: np.ndarray, labels: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # np.lexsort treats the LAST key as 1st priority: x0 1st, y0 2nd. Fruit id is
    # never a sort key, so it cannot leak into detection output order.
    return np.lexsort((labels, depth_m, y1, x1, y0, x0))


def _poisson_quantile(lam: float, u: float) -> int:
    """The least k whose Poisson(`lam`) cumulative probability reaches `u`, for `lam` > 0.

    Terms are taken in log space so a large `lam` does not underflow. Past the mode the sum also
    stops once a term no longer changes it, so a `u` within rounding of 1 still ends.
    """
    log_lam, k, cdf = math.log(lam), 0, 0.0
    while True:
        term = math.exp(k * log_lam - lam - math.lgamma(k + 1))
        if cdf + term >= u or (k > lam and cdf + term == cdf):
            return k
        cdf += term
        k += 1


def _halves(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each xyxy box cut in two at the middle of its long side (x when width >= height)."""
    wide = boxes[:, 2] - boxes[:, 0] >= boxes[:, 3] - boxes[:, 1]
    mid_x, mid_y = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    first, second = boxes.copy(), boxes.copy()
    first[:, 2] = np.where(wide, mid_x, boxes[:, 2])
    first[:, 3] = np.where(wide, boxes[:, 3], mid_y)
    second[:, 0] = np.where(wide, mid_x, boxes[:, 0])
    second[:, 1] = np.where(wide, boxes[:, 1], mid_y)
    return first, second


class OracleDetector:
    """Detector that reads `truth` records for the frame; never reads `rgb_path`/`depth_path`.

    `positives` selects which records count as detections (`is_positive`).
    With `noise`, draws come from `noise_rng(*rng_key, frame.idx)` in the clean output order, so ids
    never steer the noise; with every factor at 0 the output equals the oracle's.
    """

    def __init__(self, truth: SceneTruth, positives: str = "count1", noise: DetNoise | None = None, *,
                 rng_key: Sequence[int] = ()):
        if noise is not None and len(rng_key) == 0:
            raise ValueError("detection noise needs an rng_key")
        self._truth = truth
        self._positives = positives
        self._noise = noise
        self._rng_key = tuple(rng_key)
        self.name = "oracle" if noise is None else "oracle_noisy"

    def detect(self, frame: FrameInput) -> Detections:
        recs = [
            r for r in self._truth.records.get(frame.idx, ())
            if is_positive(r, self._positives, self._truth.count1)
        ]
        n = len(recs)
        boxes = np.array([r.bbox for r in recs], dtype=np.float32).reshape(-1, 4)
        depth_m = np.array([r.depth_med for r in recs], dtype=np.float32)
        labels = np.array([self._truth.fruits[r.id].bagged for r in recs], dtype=np.int8)
        order = _output_order(boxes, labels, depth_m)
        if self._noise is None:
            return Detections(boxes[order], np.ones(n, np.float32), labels[order], depth_m[order])
        visib = np.array([r.visib for r in recs], dtype=np.float64)
        return self._noisy(frame, boxes[order].astype(np.float64), labels[order], depth_m[order], visib[order])

    def _noisy(self, frame: FrameInput, boxes: np.ndarray, labels: np.ndarray, depth_m: np.ndarray,
               visib: np.ndarray) -> Detections:
        """Noisy detections from the `n` clean boxes, given in the clean output order."""
        nz, n = self._noise, len(boxes)
        W, H = frame.camera.width, frame.camera.height
        rng = noise_rng(*self._rng_key, frame.idx)
        # Every draw is made whatever the factor values, so one key gives the same draws for every value.
        u_miss = rng.random(n)
        g = rng.standard_normal((n, 4))
        u_split = rng.random(n)
        u_score = rng.random(n)
        u_fp = rng.random()

        keep = u_miss >= nz.miss_p

        s = nz.jitter_frac
        w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        dx, dy = s * w * g[:, 0], s * h * g[:, 1]
        dw, dh = w * np.maximum(s * g[:, 2], -1.0), h * np.maximum(s * g[:, 3], -1.0)
        jit = np.stack([np.clip(boxes[:, 0] + dx - dw / 2, 0, W), np.clip(boxes[:, 1] + dy - dh / 2, 0, H),
                        np.clip(boxes[:, 2] + dx + dw / 2, 0, W), np.clip(boxes[:, 3] + dy + dh / 2, 0, H)],
                       axis=1)

        split = keep & (visib < nz.split_visib_max) & (u_split < nz.split_p)
        whole, cut = np.flatnonzero(keep & ~split), np.flatnonzero(split)
        first, second = _halves(jit[cut])
        out = np.concatenate([jit[whole], first, second])
        src = np.concatenate([whole, cut, cut])            # the clean box each output box comes from
        scores = np.ones(n) if nz.score == "fixed" else nz.score_min + (1 - nz.score_min) * u_score
        big = (out[:, 2] - out[:, 0] >= 1) & (out[:, 3] - out[:, 1] >= 1)
        out, src = out[big], src[big]

        n_fp = 0 if nz.fp_lambda == 0 or n == 0 else _poisson_quantile(nz.fp_lambda, u_fp)
        r = rng.random((n_fp, 4))                          # one row per false positive, drawn in turn
        j = np.minimum((r[:, 0] * n).astype(np.intp), n - 1)
        fx0, fy0 = r[:, 1] * (W - w[j]), r[:, 2] * (H - h[j])
        fp = np.stack([fx0, fy0, fx0 + w[j], fy0 + h[j]], axis=1)
        fp_scores = np.ones(n_fp) if nz.score == "fixed" else nz.score_min + (1 - nz.score_min) * r[:, 3]

        all_boxes = np.concatenate([out, fp]).astype(np.float32)
        all_scores = np.concatenate([scores[src], fp_scores]).astype(np.float32)
        all_labels = np.concatenate([labels[src], labels[j]]).astype(np.int8)
        all_depth = np.concatenate([depth_m[src], np.full(n_fp, np.nan)]).astype(np.float32)
        order = _output_order(all_boxes, all_labels, all_depth)
        return Detections(all_boxes[order], all_scores[order], all_labels[order], all_depth[order])
