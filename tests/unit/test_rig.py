"""The rig model: camera poses as one body pose times a known pitch extrinsic, and the body taken
from the reference camera."""

from math import cos, radians, sin

import numpy as np
import pytest

from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.posesource import model_cameras
from counting.evaluation.runner import evaluate
from counting.geometry.camera import Camera
from counting.geometry.rig import (believed_extrinsic, body_from_reference, interpolate_pose, pitch_extrinsic,
                                   pose_matrix, relative_extrinsic, rig_pose, rigid, with_pose)
from tests.unit.synthfake import fake_cfg, robot_matrix
from tests.unit.test_drift import _passes

pytestmark = pytest.mark.pose


def test_pitch_extrinsic_matches_the_render_cameras():
    for side, axes in (("A", np.diag([1.0, 1.0, 1.0])), ("B", np.diag([-1.0, -1.0, 1.0]))):
        for q in (10.0, 20.0, 30.0, 45.0, 50.0, 70.0):
            cam = Camera.from_blender(robot_matrix(3.0, 0.0, side, q, 2.0, 1.5), 640, 360, 69.0)
            body = pose_matrix(cam) @ np.linalg.inv(pitch_extrinsic(q))
            assert np.abs(body[:3, :3] - axes).max() < 1e-12 and np.allclose(body[:3, 3], cam.C, atol=0)


def test_relative_extrinsic_is_a_turn_about_camera_x(apple_run):
    a = radians(-20.0)
    turn = np.array([[1.0, 0.0, 0.0], [0.0, cos(a), -sin(a)], [0.0, sin(a), cos(a)]])
    rel = relative_extrinsic(30.0, 10.0)
    assert np.abs(rel[:3, :3] - turn).max() < 1e-12
    assert not rel[:3, 3].any() and np.array_equal(rel[3], [0.0, 0.0, 0.0, 1.0])
    spots: dict[tuple[str, int], dict[float, object]] = {}
    for f in SynthRun.open(apple_run).frames:
        spots.setdefault((f.pass_key.side, f.pos_k), {})[f.pass_key.pitch_deg] = f
    assert spots
    for spot, by_pitch in spots.items():
        assert set(by_pitch) == {10.0, 30.0}, spot
        moved = pose_matrix(by_pitch[30.0].camera) @ rel
        assert np.abs(pose_matrix(by_pitch[10.0].camera) - moved).max() < 1e-12, spot


def test_body_from_the_reference_camera_gives_the_two_camera_formula(apple_run):
    rng = np.random.default_rng(4)
    dimu = rigid(0.01 * rng.standard_normal(3), 0.003 * rng.standard_normal(3))
    drig = rigid(0.01 * rng.standard_normal(3), 0.003 * rng.standard_normal(3))
    first = next(f for f in SynthRun.open(apple_run).frames if f.pass_key.pitch_deg == 30.0)
    T_r = pose_matrix(first.camera)
    body = body_from_reference(T_r, believed_extrinsic(30.0, 30.0, dimu))
    got = rig_pose(body, believed_extrinsic(10.0, 30.0, dimu, drig))
    assert np.abs(got - T_r @ relative_extrinsic(30.0, 10.0) @ drig).max() < 1e-12   # dX_imu cancels
    X = believed_extrinsic(30.0, 30.0)
    assert np.abs(rig_pose(body_from_reference(T_r, X), X) - T_r).max() < 1e-12


def test_rigid_and_with_pose_round_trip():
    assert np.array_equal(rigid(np.zeros(3), np.zeros(3)), np.eye(4))
    cam = Camera.from_blender(robot_matrix(0.4, 3.75, "A", 30.0, 1.75, 0.9), 640, 360, 69.0)
    T = pose_matrix(cam)
    back = with_pose(cam, T)
    assert all(np.array_equal(getattr(back, n), getattr(cam, n)) for n in ("R_c2w", "C", "K"))
    assert (back.width, back.height) == (cam.width, cam.height)
    T[:3, :] = 0.0                                         # the camera holds copies of the pose
    assert np.array_equal(back.R_c2w, cam.R_c2w) and np.array_equal(back.C, cam.C)


def test_interpolate_pose_between_frames():
    times = np.array([0.0, 1.0])
    poses = np.stack([np.eye(4), rigid([0.0, 0.0, radians(10.0)], [1.0, 0.0, 0.0])])
    mid = interpolate_pose(times, poses, 0.5)
    a = radians(5.0)
    turn = np.array([[cos(a), -sin(a), 0.0], [sin(a), cos(a), 0.0], [0.0, 0.0, 1.0]])
    assert np.abs(mid[:3, 3] - [0.5, 0.0, 0.0]).max() < 1e-12 and np.abs(mid[:3, :3] - turn).max() < 1e-12
    assert np.array_equal(mid[3], [0.0, 0.0, 0.0, 1.0])
    assert np.array_equal(interpolate_pose(times, poses, -1.0), poses[0])
    assert np.array_equal(interpolate_pose(times, poses, 2.0), poses[1])


def _close(a, b, tol: float) -> bool:
    """`a` and `b` match: dicts key by key, lists item by item, floats within `tol`, the rest by `==`."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol
    return a == b


def test_truth_with_the_rig_matches_l3(apple_run, ctx_of):
    rig = evaluate(ctx_of["apple"], fake_cfg(apple_run, "apple", "pose.rig=true", "pose.rig_ref=30"))
    plain = evaluate(ctx_of["apple"], fake_cfg(apple_run, "apple"))
    for key in ("stages", "lift_depth", "y1", "y6"):
        assert _close(rig[key], plain[key], 1e-9), key


def test_rig_needs_the_reference_camera_at_every_position(apple_run):
    run = SynthRun.open(apple_run)
    cfg = fake_cfg(apple_run, "apple", "pose.rig=true", "pose.rig_ref=30")
    passes = _passes(run, cfg)

    def cams(frames):
        return model_cameras(frames, passes, cfg, None, scene=run.meta.seed, base_step_m=run.meta.base_step_m)

    assert cams(run.frames) is not None
    gap = [f for f in run.frames if not (f.pass_key.pitch_deg == 30.0 and f.pass_key.side == "A" and f.pos_k == 2)]
    with pytest.raises(ValueError, match="pose.rig_ref=30 has no frame at side A, pos_k 2"):
        cams(gap)
    no_b = [f for f in run.frames if not (f.pass_key.pitch_deg == 30.0 and f.pass_key.side == "B")]
    with pytest.raises(ValueError, match="pose.rig_ref=30 has no pass on side B"):
        cams(no_b)
