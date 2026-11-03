"""Detection contract: the `Detections` container, the `Detector` protocol, and `box_iou`.

Only `FrameInput` is imported from `counting.data.formats.orchard_synth` here, so this
module stays on the prediction path (no `SceneTruth`/`GtRecord`/`FruitGt`; see
`counting.detection.oracle` for the detector that reads ground truth).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from counting.data.formats.orchard_synth import FrameInput


@dataclass
class Detections:
    """One frame's detections: boxes, scores and labels, with optional oracle depth.

    Carries no ground-truth ids — `counting.evaluation.match.match_frame` recovers
    source ids after the fact, the same way for the oracle and a trained detector.
    """

    boxes: np.ndarray            # (N,4) float32 xyxy, same convention as frame.json bbox
    scores: np.ndarray           # (N,) float32
    labels: np.ndarray           # (N,) int8: 0 fruit, 1 bag
    depth_m: np.ndarray | None = None   # oracle only: depth_med

    def __post_init__(self) -> None:
        n = len(self.boxes)
        self._check_field("boxes", self.boxes, (n, 4), np.float32)
        self._check_field("scores", self.scores, (n,), np.float32)
        self._check_field("labels", self.labels, (n,), np.int8)
        if self.depth_m is not None:
            self._check_field("depth_m", self.depth_m, (n,), np.float32)

    @staticmethod
    def _check_field(name: str, arr: np.ndarray, shape: tuple[int, ...], dtype: type) -> None:
        if arr.shape != shape or arr.dtype != dtype:
            raise ValueError(
                f"Detections.{name}: expected shape {shape} dtype {dtype}, "
                f"got shape {arr.shape} dtype {arr.dtype}"
            )

    def __len__(self) -> int:
        return len(self.boxes)

    @classmethod
    def empty(cls, with_depth: bool = False) -> "Detections":
        """An empty detection set, optionally with an empty `depth_m`."""
        depth = np.zeros(0, np.float32) if with_depth else None
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int8), depth)

    def subset(self, idx) -> "Detections":
        """A copy holding only the detections selected by `idx` (an index array or mask)."""
        depth = self.depth_m[idx] if self.depth_m is not None else None
        return Detections(self.boxes[idx], self.scores[idx], self.labels[idx], depth)


class Detector(Protocol):
    name: str

    def detect(self, frame: FrameInput) -> Detections: ...


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy boxes `a` (N,4) and `b` (M,4); box area is `(x1-x0)(y1-y0)`."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax0, ay0, ax1, ay1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    x0 = np.maximum(ax0[:, None], bx0[None, :])
    y0 = np.maximum(ay0[:, None], by0[None, :])
    x1 = np.minimum(ax1[:, None], bx1[None, :])
    y1 = np.minimum(ay1[:, None], by1[None, :])
    inter = np.clip(x1 - x0, 0.0, None) * np.clip(y1 - y0, 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
