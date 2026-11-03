from pathlib import Path

import numpy as np
import pytest

from counting.config.eval_schema import LiftConfig
from counting.data.formats.orchard_synth import FrameInput, PassKey
from counting.detection.base import Detections
from counting.geometry.camera import Camera
from counting.multiview.lift import box_median_depth, complete_edge_boxes, edge_touch, lift, lift_points
from tests.unit.synthfake import robot_matrix

CAM = Camera.from_blender(robot_matrix(0.0, 3.75, "A", 30.0, 1.75, 0.9), 640, 360, 69.0)
X = np.array([[0.3, 3.95, 1.6]])
EXACT = LiftConfig(depth="det_depth", center_offset_m={})


def _box_at(X):
    uv, z = CAM.project(X)
    c = uv + 0.5
    return np.c_[c - 5, c + 5].astype(np.float32), z


def test_box_median_uses_center_region():
    d = np.full((20, 20), 9.0)
    d[6:10, 6:10] = 2.0
    assert box_median_depth(np.array([[4, 4, 12, 12]], float), d, 0.5)[0] == 2.0


def test_box_median_tiny_box_uses_one_pixel():
    d = np.zeros((10, 10))
    d[3, 3] = 4.0
    assert box_median_depth(np.array([[3, 3, 4, 4]], float), d, 0.5)[0] == 4.0


def test_box_median_all_nan_gives_nan():
    d = np.full((20, 20), np.nan)
    d[0, 0] = 1.0
    assert np.isnan(box_median_depth(np.array([[4, 4, 12, 12]], float), d, 0.5)[0])


def test_edge_touch_and_complete():
    b = np.array([[0, 10, 6, 30], [630, 0, 640, 8], [100, 100, 120, 118]], float)
    assert edge_touch(b, 640, 360).tolist() == [[True, False, False, False], [False, True, True, False],
                                               [False, False, False, False]]
    np.testing.assert_allclose(complete_edge_boxes(b, 640, 360),
                               [[-14, 10, 6, 30], [630, -2, 640, 8], [100, 100, 120, 118]])


def test_lift_det_depth_recovers_point():
    boxes, z = _box_at(X)
    np.testing.assert_allclose(lift_points(boxes, np.zeros(1, np.int8), z, CAM, EXACT), X, atol=1e-5)


def test_lift_edge_box_recovers_point():
    Xe = CAM.backproject(np.array([[2.0, 180.0]]), np.array([2.0]))   # centre 2 px from the left border
    c = CAM.project(Xe)[0] + 0.5
    cut = np.clip(np.c_[c - 5, c + 5], 0, [640, 360, 640, 360]).astype(np.float32)
    assert cut[0, 0] == 0 and cut[0, 2] - cut[0, 0] < 10
    np.testing.assert_allclose(lift_points(cut, np.zeros(1, np.int8), np.array([2.0]), CAM, EXACT), Xe, atol=1e-5)


def test_lift_offset_pushes_along_ray():
    boxes, z = _box_at(X)
    cfg0, cfg1 = LiftConfig(center_offset_m={}), LiftConfig(center_offset_m={"bag": 0.04})
    a = lift_points(boxes, np.ones(1, np.int8), z, CAM, cfg0)
    b = lift_points(boxes, np.ones(1, np.int8), z, CAM, cfg1)
    np.testing.assert_allclose(b - a, 0.04 * CAM.ray_dir(CAM.project(X)[0]), atol=1e-6)


def test_lift_nan_for_missing_depth():
    boxes, _ = _box_at(X)
    out = lift_points(boxes, np.zeros(1, np.int8), np.array([np.nan]), CAM, LiftConfig())
    assert np.isnan(out).all()


def test_lift_mode_requirements():
    boxes, z = _box_at(X)
    frame = FrameInput("t", 0, PassKey(30.0, "A", 2), 0, CAM, 0.0, Path("r"), Path("d"))
    dets = Detections(boxes, np.ones(1, np.float32), np.zeros(1, np.int8))
    with pytest.raises(ValueError, match="depth_m"):
        lift(dets, frame, LiftConfig(depth="det_depth"))
    with pytest.raises(ValueError, match="depth_img"):
        lift(dets, frame, LiftConfig(depth="box_median"))
