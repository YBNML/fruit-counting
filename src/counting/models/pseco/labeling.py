"""Positive/negative target assignment for PseCo proposals.

Convention: a box `(x1, y1, x2, y2)` covers the half-open rectangle
`[x1, x2) × [y1, y2)`. A proposal is POSITIVE (target=1) iff at least one
ground-truth point lies within that rectangle.
"""

from __future__ import annotations

from typing import Sequence

import torch


def points_in_box(
    points: Sequence[tuple[float, float]],
    box: tuple[float, float, float, float],
) -> bool:
    """True iff any point lies in the half-open rectangle [x1,x2) × [y1,y2)."""
    x1, y1, x2, y2 = box
    for x, y in points:
        if x1 <= x < x2 and y1 <= y < y2:
            return True
    return False


def assign_targets_from_points(
    boxes_per_image: list[torch.Tensor],
    points_per_image: list[list[tuple[float, float]]],
) -> torch.Tensor:
    """Return a flat (B*K,) long tensor of pos/neg labels.

    Each `boxes_per_image[i]` has shape `(1, K, 4)` (matching PseCo's
    `ROIHeadMLP.forward`). `points_per_image[i]` is the list of GT points for
    image `i`. Output is flattened in row-major (image, proposal) order.
    """
    if len(boxes_per_image) != len(points_per_image):
        raise ValueError(
            f"length mismatch: boxes={len(boxes_per_image)} points={len(points_per_image)}"
        )

    labels: list[int] = []
    for boxes_tensor, gt_points in zip(boxes_per_image, points_per_image):
        boxes = boxes_tensor.reshape(-1, 4).tolist()
        for box in boxes:
            labels.append(1 if points_in_box(gt_points, tuple(box)) else 0)
    return torch.tensor(labels, dtype=torch.long)
