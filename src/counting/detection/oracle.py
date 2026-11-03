"""Oracle detector: emits ground-truth boxes directly instead of reading the frame's PNG."""

from __future__ import annotations

import numpy as np

from counting.data.formats.orchard_synth import FrameInput, SceneTruth
from counting.detection.base import Detections
from counting.evaluation.truth import is_positive


class OracleDetector:
    """Detector that reads `truth` records for the frame; never reads `rgb_path`/`depth_path`.

    `positives` selects which records count as detections (`is_positive`).
    `noise` is not supported yet and raises `NotImplementedError`.
    """

    name = "oracle"

    def __init__(self, truth: SceneTruth, positives: str = "count1", noise: object | None = None):
        if noise is not None:
            raise NotImplementedError("detection noise is not supported yet")
        self._truth = truth
        self._positives = positives

    def detect(self, frame: FrameInput) -> Detections:
        recs = [
            r for r in self._truth.records.get(frame.idx, ())
            if is_positive(r, self._positives, self._truth.count1)
        ]
        n = len(recs)
        boxes = np.array([r.bbox for r in recs], dtype=np.float32).reshape(-1, 4)
        depth_m = np.array([r.depth_med for r in recs], dtype=np.float32)
        labels = np.array([self._truth.fruits[r.id].bagged for r in recs], dtype=np.int8)

        x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        # np.lexsort treats the LAST key as 1st priority: x0 1st, y0 2nd. Fruit id is
        # never a sort key, so it cannot leak into detection output order.
        order = np.lexsort((labels, depth_m, y1, x1, y0, x0))

        return Detections(boxes[order], np.ones(n, np.float32), labels[order], depth_m[order])
