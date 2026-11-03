"""Lift detection boxes to 3D world points, completing edge-touching boxes first.

Edge-touching boxes (clipped by the image border) have an off-centre pixel
centre, so their centre is completed to a `max(w, h)` square along the
touched side before projecting.
"""

from __future__ import annotations

import warnings

import numpy as np

from counting.config.eval_schema import LiftConfig
from counting.data.formats.orchard_synth import FrameInput
from counting.detection.base import Detections
from counting.geometry.camera import Camera, box_centers


def edge_touch(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Which of the 4 image borders each box touches: columns are left, right, top, bottom."""
    boxes = np.asarray(boxes, dtype=float)
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    left = x0 <= 0
    right = x1 >= width
    top = y0 <= 0
    bottom = y1 >= height
    return np.stack([left, right, top, bottom], axis=1)


def complete_edge_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Grow a box touching only one side of an axis into a `max(w, h)` square on that axis.

    An axis touched on both sides, or not touched at all, is left unchanged.
    """
    boxes = np.asarray(boxes, dtype=float).copy()
    left, right, top, bottom = edge_touch(boxes, width, height).T
    side = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])

    x_left_only = left & ~right
    x_right_only = right & ~left
    y_top_only = top & ~bottom
    y_bottom_only = bottom & ~top

    boxes[x_left_only, 0] = boxes[x_left_only, 2] - side[x_left_only]
    boxes[x_right_only, 2] = boxes[x_right_only, 0] + side[x_right_only]
    boxes[y_top_only, 1] = boxes[y_top_only, 3] - side[y_top_only]
    boxes[y_bottom_only, 3] = boxes[y_bottom_only, 1] + side[y_bottom_only]
    return boxes


def box_median_depth(boxes: np.ndarray, depth_m: np.ndarray, box_frac: float) -> np.ndarray:
    """Median depth over the centre `box_frac` region of each (pre-completion) box.

    A box whose centre region has no valid (non-NaN, in-image) pixel gets NaN,
    without a warning.
    """
    boxes = np.asarray(boxes, dtype=float)
    height, width = depth_m.shape
    a = (1.0 - box_frac) / 2.0
    out = np.empty(len(boxes), dtype=float)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        w, h = x1 - x0, y1 - y0
        c0 = int(np.floor(x0 + a * w))
        c1 = max(c0 + 1, int(np.ceil(x1 - a * w)))
        r0 = int(np.floor(y0 + a * h))
        r1 = max(r0 + 1, int(np.ceil(y1 - a * h)))
        c0, c1 = max(c0, 0), min(c1, width)
        r0, r1 = max(r0, 0), min(r1, height)
        if c1 <= c0 or r1 <= r0:
            out[i] = np.nan
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out[i] = np.nanmedian(depth_m[r0:r1, c0:c1])
    return out


def box_far_depth(
    boxes: np.ndarray, depth_m: np.ndarray, scale: float, gap_m: float, min_frac: float
) -> np.ndarray:
    """Median depth of the farthest large depth cluster inside each (pre-completion) box's ellipse.

    The ellipse is inscribed in the box and scaled by `scale` (semi-axes at
    least 0.5 px); a pixel is inside when its centre is. The sorted finite
    depths inside are split where consecutive values differ by more than
    `gap_m`. The farthest cluster holding at least `min_frac` of those pixels
    is picked, or else the largest one (the nearer one on a tie), so a front
    leaf covering part of the box does not hide the fruit behind it. A box
    with no finite in-image pixel inside gets NaN, without a warning.
    """
    boxes = np.asarray(boxes, dtype=float)
    height, width = depth_m.shape
    out = np.empty(len(boxes), dtype=float)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        c0 = int(np.floor(x0))
        c1 = max(c0 + 1, int(np.ceil(x1)))
        r0 = int(np.floor(y0))
        r1 = max(r0 + 1, int(np.ceil(y1)))
        c0, c1 = max(c0, 0), min(c1, width)
        r0, r1 = max(r0, 0), min(r1, height)
        if c1 <= c0 or r1 <= r0:
            out[i] = np.nan
            continue
        cu, cv = (x0 + x1) / 2, (y0 + y1) / 2
        au = max(scale * (x1 - x0) / 2, 0.5)
        av = max(scale * (y1 - y0) / 2, 0.5)
        rows, cols = np.mgrid[r0:r1, c0:c1]
        inside = ((cols + 0.5 - cu) / au) ** 2 + ((rows + 0.5 - cv) / av) ** 2 <= 1.0
        v = depth_m[r0:r1, c0:c1][inside]
        v = np.sort(v[np.isfinite(v)])
        if len(v) == 0:
            out[i] = np.nan
            continue
        clusters = np.split(v, np.flatnonzero(np.diff(v) > gap_m) + 1)
        sizes = np.array([len(c) for c in clusters])
        enough = np.flatnonzero(sizes >= min_frac * len(v))
        k = int(enough[-1]) if len(enough) else int(np.argmax(sizes))
        out[i] = np.median(clusters[k])
    return out


def lift_points(
    boxes: np.ndarray, labels: np.ndarray, z: np.ndarray, camera: Camera, cfg: LiftConfig
) -> np.ndarray:
    """Back-project completed box centres at depth `z`, pushed along the ray by `cfg.offset`.

    A row whose `z` is NaN or <= 0 gets NaN.
    """
    full = complete_edge_boxes(boxes, camera.width, camera.height)
    uv = box_centers(full)
    z = np.asarray(z, dtype=float)
    offsets = np.array([cfg.offset(int(label)) for label in labels], dtype=float)
    X = camera.backproject(uv, z) + camera.ray_dir(uv) * offsets[:, None]
    X[~np.isfinite(z) | (z <= 0)] = np.nan
    return X


def lift(
    dets: Detections, frame: FrameInput, cfg: LiftConfig, depth_img: np.ndarray | None = None
) -> np.ndarray:
    """Lift one frame's detections to 3D, picking the depth source named by `cfg.depth`."""
    if cfg.depth == "det_depth":
        if dets.depth_m is None:
            raise ValueError("det_depth needs dets.depth_m")
        z = dets.depth_m
    else:
        if depth_img is None:
            raise ValueError(f"{cfg.depth} needs depth_img")
        if cfg.depth == "box_median":
            z = box_median_depth(dets.boxes, depth_img, cfg.box_frac)
        elif cfg.depth == "box_far":
            z = box_far_depth(dets.boxes, depth_img, cfg.far_scale, cfg.far_gap_m, cfg.far_min_frac)
        else:
            raise ValueError(f"unknown lift.depth={cfg.depth}")
    return lift_points(dets.boxes, dets.labels, z, frame.camera, cfg)
