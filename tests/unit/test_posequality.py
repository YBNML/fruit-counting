"""Pose quality of the cameras a pose source gives against the true cameras: neighbour errors, the
three end errors with their body components, the misalignment between passes, the reference camera
frame that takes the rig calibration error out of the in-pass drift, and the per-pass truth fit."""

from dataclasses import replace
from math import radians

import numpy as np
import pytest

from counting.evaluation.posequality import apply_rigid, fit_truth_rigid, pose_quality
from counting.geometry.rig import pose_matrix, rigid, with_pose

pytestmark = pytest.mark.pose

E_KEYS = ("pos_m", "rot_deg", "along_m", "side_m", "height_m", "yaw_deg", "tilt_deg")
IN_PASS_KEYS = (*E_KEYS, "pos_frac")


def _items(ctx, change=lambda key, i, n, cam: cam):
    return [(k, [f.camera for f in fr], [change(k, i, len(fr), f.camera) for i, f in enumerate(fr)])
            for k, fr in ctx.passes.items()]


def _shift(cam, d):
    T = pose_matrix(cam)
    T[:3, 3] += d
    return with_pose(cam, T)


def _turn_z(cam, deg):
    """`cam` turned by `deg` about the world z axis (rotation multiplied on the left), centre kept."""
    T = pose_matrix(cam)
    T[:3, :3] = rigid([0.0, 0.0, radians(deg)], [0.0, 0.0, 0.0])[:3, :3] @ T[:3, :3]
    return with_pose(cam, T)


def _pass(q, pitch, side):
    return next(p for p in q["passes"] if (p["pitch"], p["side"]) == (pitch, side))


def test_known_offsets_give_known_values(ctx_of):
    ctx = ctx_of["apple"]
    side_a = lambda k, i, n, c: _shift(c, [0.0, 0.02, 0.0]) if k.side == "A" else c
    q = pose_quality(_items(ctx, side_a), ctx.layout)
    for p in q["passes"]:
        want = 0.02 if p["side"] == "A" else 0.0
        assert p["start"]["side_m"] == pytest.approx(want, abs=1e-12)
        assert p["world_end"]["side_m"] == pytest.approx(want, abs=1e-12)
        assert p["in_pass"]["pos_m"] == pytest.approx(0.0, abs=1e-12)
    assert q["rpe"]["trans_m"]["max"] == pytest.approx(0.0, abs=1e-12)
    assert q["interpass"]["same_side"]["p90_m"] == pytest.approx(0.0, abs=1e-12)
    assert q["interpass"]["both_sides"]["p50_m"] == pytest.approx(0.02, abs=1e-12)
    assert q["interpass"]["both_sides"]["p90_m"] == pytest.approx(0.02, abs=1e-12)


def test_truth_gives_zero_pose_error(ctx_of):
    ctx = ctx_of["apple"]
    q = pose_quality(_items(ctx), ctx.layout)
    errors = [q["rpe"][part][stat] for part in ("trans_m", "rot_deg") for stat in ("p50", "p90", "max")]
    for entries in (*q["passes"], q["summary"]):
        errors += [entries[name][k] for name in ("in_pass", "in_pass_cam") for k in IN_PASS_KEYS]
        errors += [entries[name][k] for name in ("start", "world_end") for k in E_KEYS]
    errors += [q["interpass"][pair][stat] for pair in ("same_side", "both_sides") for stat in ("p50_m", "p90_m")]
    assert all(abs(x) <= 1e-12 for x in errors), errors
    assert q["rpe"]["n"] == sum(p["n_frames"] - 1 for p in q["passes"])
    assert q["interpass"]["same_side"]["n"] > 0
    assert q["interpass"]["both_sides"]["n"] > 0


def test_last_frame_offset_is_in_pass_drift(ctx_of):
    ctx = ctx_of["apple"]
    last = lambda k, i, n, c: _shift(c, [0.03, 0.0, 0.0]) if (k.pitch_deg, k.side, i) == (30.0, "A", n - 1) else c
    q = pose_quality(_items(ctx, last), ctx.layout)
    p = _pass(q, 30.0, "A")
    assert p["in_pass"]["along_m"] == pytest.approx(0.03, abs=1e-12)
    assert p["in_pass"]["pos_m"] == pytest.approx(0.03, abs=1e-12)
    assert p["in_pass"]["pos_frac"] == pytest.approx(0.03 / p["length_m"], abs=1e-12)
    assert p["world_end"]["along_m"] == pytest.approx(0.03, abs=1e-12)
    assert p["start"]["pos_m"] == pytest.approx(0.0, abs=1e-12)
    assert q["rpe"]["trans_m"]["max"] == pytest.approx(0.03, abs=1e-12)


def test_yaw_error_components(ctx_of):
    ctx = ctx_of["apple"]
    last = lambda k, i, n, c: _turn_z(c, 0.5) if (k.pitch_deg, k.side, i) == (30.0, "A", n - 1) else c
    p = _pass(pose_quality(_items(ctx, last), ctx.layout), 30.0, "A")
    assert p["in_pass"]["yaw_deg"] == pytest.approx(0.5, abs=1e-9)
    assert p["in_pass"]["rot_deg"] == pytest.approx(0.5, abs=1e-9)
    assert p["in_pass"]["tilt_deg"] == pytest.approx(0.0, abs=1e-9)
    assert p["in_pass"]["pos_m"] == pytest.approx(0.0, abs=1e-9)


def test_fit_truth_rigid_undoes_a_rigid_error(ctx_of):
    frames = next(iter(ctx_of["apple"].passes.values()))
    G = rigid([0.01, -0.02, 0.03], [0.1, -0.05, 0.02])
    truth = [f.camera for f in frames]
    used = [with_pose(c, G @ pose_matrix(c)) for c in truth]
    A = fit_truth_rigid(truth, used)
    assert np.abs(A - np.linalg.inv(G)).max() <= 1e-9
    X = np.array([[1.0, 2.0, 1.5], [np.nan, np.nan, np.nan]])
    back = apply_rigid(A, apply_rigid(G, X))
    assert np.abs(back[0] - X[0]).max() <= 1e-9
    assert np.isnan(back[1]).all()


def test_pose_quality_handles_short_passes_and_one_side(ctx_of):
    ctx = ctx_of["apple"]
    items = [item for item in _items(ctx) if item[0].side == "A"]
    short, truth, used = items[0]
    items[0] = (short, truth[:1], used[:1])
    q = pose_quality(items, ctx.layout)
    p = _pass(q, short.pitch_deg, "A")
    assert p["n_frames"] == 1
    assert all(v is None for name in ("in_pass", "in_pass_cam") for v in p[name].values())
    assert p["start"]["pos_m"] is not None and p["world_end"]["pos_m"] is not None
    assert q["interpass"]["both_sides"] == {"n": 0, "p50_m": None, "p90_m": None}
    assert pose_quality(items, replace(ctx.layout, tree_x=()))["interpass"]["same_side"]["n"] == 0
    single = pose_quality([(k, t[:1], u[:1]) for k, t, u in items], ctx.layout)
    none = {"p50": None, "p90": None, "max": None}
    assert single["rpe"] == {"n": 0, "trans_m": none, "rot_deg": none}


def test_reference_frame_takes_out_rig_errors(ctx_of):
    ctx = ctx_of["apple"]
    dX = rigid([0.0, radians(0.5), 0.0], [0.005, 0.0, 0.0])
    items = _items(ctx, lambda k, i, n, c: with_pose(c, pose_matrix(c) @ dX) if k.pitch_deg == 10.0 else c)
    q = pose_quality(items, ctx.layout, ref_pitch=30, rig_errors={10.0: dX})
    assert {p["pitch"] for p in q["passes"]} == {10.0, 30.0}
    for p in q["passes"]:
        assert p["in_pass"]["pos_m"] == pytest.approx(0.0, abs=1e-9)
        assert p["in_pass"]["rot_deg"] == pytest.approx(0.0, abs=1e-9)
        if p["pitch"] == 10.0:
            assert p["in_pass_cam"]["pos_m"] > 1e-4
        else:
            assert p["in_pass_cam"] == pytest.approx(p["in_pass"], abs=1e-12)
    assert q["rpe"]["trans_m"]["max"] == pytest.approx(0.0, abs=1e-9)
    own_frame = pose_quality(items, ctx.layout)
    assert all(p["in_pass"]["pos_m"] > 1e-4 for p in own_frame["passes"] if p["pitch"] == 10.0)
