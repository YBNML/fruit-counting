"""The chained drift model: noisy relative motions keyed by position, chained from a start pose,
and the seeded start, rig, time and depth errors; and the cameras the truth pose source builds with it."""

import inspect
from math import cos, radians, sin, sqrt

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from counting.config.eval_schema import DRIFT_KEYS
from counting.data.formats.orchard_synth import SynthRun, select_frames
from counting.evaluation.posesource import drift_sigmas, model_cameras, rig_errors
from counting.evaluation.report import results_bytes
from counting.geometry.noise import (STEP_DEPTH_K, STEP_FLOW, STEP_RIG, STEP_START, STEP_TIME, STEP_WALK,
                                     DriftSigmas, FlowDraws, about_middle, chain_poses, depth_k_noise, flow_draws,
                                     noise_rng, noisy_increments, seeded_rigid, time_offset, vo_factor_sigma)
from counting.geometry.rig import pitch_extrinsic, pose_matrix, relative_extrinsic, rigid
from tests.unit.synthfake import fake_cfg

pytestmark = pytest.mark.pose

D_EXT_VO = ("noise.drift_scale_sigma=0.0045", "noise.drift_rate_deg_per_m=0.045", "noise.drift_step_trans_mm=2.1",
            "noise.drift_step_rot_deg=0.039")
# Every drift key at 0 that a config without the rig may hold.
ZERO_DRIFT = tuple(f"noise.{k}=0" for k in DRIFT_KEYS if not k.startswith("rig_extr"))


def _passes(run, cfg):
    out = {}
    for f in select_frames(run.frames, base_step_m=run.meta.base_step_m, pitches=cfg.passes.pitches,
                           sides=cfg.passes.sides, step_m=cfg.passes.step_m):
        out.setdefault(f.pass_key, []).append(f)
    return out


def _cams(run, cfg, ns):
    return model_cameras(run.frames, _passes(run, cfg), cfg, ns, scene=run.meta.seed, base_step_m=run.meta.base_step_m)


def _core(res):
    return results_bytes({k: res[k] for k in ("stages", "lift_depth", "y1", "y6")})


def _straight(n: int, step: float) -> np.ndarray:
    P = np.repeat(np.eye(4)[None], n, axis=0)
    P[:, 0, 3] = step * np.arange(n)
    return P


def _draws(n: int, s=0.0, w=(0.0, 0.0, 0.0), t=(0.0, 0.0, 0.0), r=(0.0, 0.0, 0.0)) -> FlowDraws:
    return FlowDraws(s=s, w=np.array(w), t=np.tile(t, (n - 1, 1)), r=np.tile(r, (n - 1, 1)))


def test_heading_error_moves_into_position():
    P = _straight(101, 0.15)
    C = chain_poses(P[0], noisy_increments(P, _draws(101, w=(0, 0, 1)), DriftSigmas(rate_deg_per_m=0.1), 0.15))
    a = radians(0.1) * 0.15
    assert C[-1][0, 3] == pytest.approx(sum(0.15 * cos(k * a) for k in range(100)), abs=1e-12)
    assert C[-1][1, 3] == pytest.approx(sum(0.15 * sin(k * a) for k in range(100)), abs=1e-12)
    assert Rotation.from_matrix(C[-1][:3, :3]).as_rotvec() == pytest.approx([0, 0, 100 * a], abs=1e-12)


def test_flow_draws_are_keyed_by_position():
    d = flow_draws(0, 2, 5, (0, 3), [0, 1, 2, 3])
    g = noise_rng(0, 2, STEP_FLOW, 5, 0, 3)
    assert d.s == g.standard_normal()
    assert np.array_equal(d.w, g.standard_normal(3))
    g = noise_rng(0, 2, STEP_WALK, 5, 0, 3, 2)
    assert np.array_equal(d.t[1], g.standard_normal(3))
    assert np.array_equal(d.r[1], g.standard_normal(3))
    assert d.t.shape == (3, 3) and d.r.shape == (3, 3)
    assert np.array_equal(flow_draws(0, 2, 5, (0, 3), [5, 3]).t[0], d.t[2])


def test_zero_sigmas_reproduce_the_true_motion():
    P = np.array([rigid([0.01 * i, -0.02 * i, 0.03 * i * i], [0.15 * i, 0.01 * i * i, -0.005 * i]) for i in range(5)])
    draws = _draws(5, s=0.7, w=(1.0, -2.0, 0.5), t=(0.3, -1.1, 2.0), r=(1.0, 0.4, -0.6))
    inc = noisy_increments(P, draws, DriftSigmas(), 0.15)
    assert inc.shape == (4, 4, 4)
    for i in range(1, 5):
        assert np.abs(inc[i - 1] - np.linalg.inv(P[i - 1]) @ P[i]).max() < 1e-15, i
    assert np.abs(chain_poses(P[0], inc) - P).max() < 1e-12


def test_scale_and_walk_move_the_end():
    P = _straight(101, 0.15)
    C = chain_poses(P[0], noisy_increments(P, _draws(101, s=1.0), DriftSigmas(scale=0.01), 0.15))
    assert C[-1][0, 3] == pytest.approx(15.15, abs=1e-12)
    assert C[-1][1, 3] == 0.0 and C[-1][2, 3] == 0.0
    C = chain_poses(P[0], noisy_increments(P, _draws(101, t=(0, 1, 0)), DriftSigmas(trans_m=0.002), 0.15))
    assert C[-1][1, 3] == pytest.approx(0.2, abs=1e-12)


def test_walk_sigma_scales_with_the_base_step():
    inc = noisy_increments(_straight(2, 0.30), _draws(2, t=(0, 1, 0), r=(0, 0, 1)),
                           DriftSigmas(trans_m=0.002, rot_deg=0.039), 0.30)
    assert inc.shape == (1, 4, 4)
    assert inc[0][1, 3] == pytest.approx(0.002 * sqrt(2), abs=1e-15)
    turn = Rotation.from_matrix(inc[0][:3, :3]).as_rotvec()
    assert turn == pytest.approx([0, 0, radians(0.039 * sqrt(2))], abs=1e-15)
    # On a turning true step the noise turn comes after the true turn: rel_R @ Exp(...).
    P = np.array([np.eye(4), rigid([0, 0, 0.5], [0.3, 0, 0])])
    inc = noisy_increments(P, _draws(2, r=(1, 0, 0)), DriftSigmas(rot_deg=0.039), 0.30)
    want = Rotation.from_rotvec([0, 0, 0.5]) * Rotation.from_rotvec([radians(0.039 * sqrt(2)), 0, 0])
    assert np.abs(inc[0][:3, :3] - want.as_matrix()).max() < 1e-12


def test_seeded_rigid_and_about_middle():
    key = (0, 1, STEP_START, 5, 0, 3)
    g = noise_rng(*key)
    t, r = g.standard_normal(3), g.standard_normal(3)
    error = seeded_rigid(key, 0.05, 0.5)
    assert np.abs(error[:3, 3] - 0.05 * t).max() < 1e-12
    assert np.abs(Rotation.from_matrix(error[:3, :3]).as_rotvec() - radians(0.5) * r).max() < 1e-12
    assert np.array_equal(seeded_rigid(key, 0.0, 0.0), np.eye(4))
    turn = rigid([0, 0, 0.01], [0, 0, 0])
    moved = about_middle(_straight(3, 1.5), turn)
    assert np.abs(moved[1][:3, 3] - [1.5, 0.0, 0.0]).max() < 1e-12
    assert np.abs(moved[2][:3, 3] - [1.5 + 1.5 * cos(0.01), 1.5 * sin(0.01), 0.0]).max() < 1e-12
    assert np.abs(moved[:, :3, :3] - turn[:3, :3]).max() < 1e-12
    # A shift and tilted cameras: every center also moves by t_e, every rotation becomes R_e @ R.
    tilted = _straight(3, 1.5)
    tilted[:, :3, :3] = Rotation.from_rotvec([0.2, 0, 0]).as_matrix()
    moved = about_middle(tilted, rigid([0, 0, 0.01], [0.02, -0.03, 0.04]))
    assert np.abs(moved[1][:3, 3] - [1.5 + 0.02, -0.03, 0.04]).max() < 1e-12
    assert np.abs(moved[2][:3, 3] - [1.5 + 1.5 * cos(0.01) + 0.02, 1.5 * sin(0.01) - 0.03, 0.04]).max() < 1e-12
    want = (Rotation.from_rotvec([0, 0, 0.01]) * Rotation.from_rotvec([0.2, 0, 0])).as_matrix()
    assert np.abs(moved[:, :3, :3] - want).max() < 1e-12


def test_depth_k_noise_and_time_offset():
    key = (0, 1, STEP_DEPTH_K, 5, 7)
    z = np.array([1.0, 2.0, np.nan])
    noise = depth_k_noise(key, z, 0.01)
    g = noise_rng(*key).standard_normal(3)
    assert noise.shape == (3,)
    assert noise[:2] == pytest.approx(0.01 * z[:2] ** 2 * g[:2], rel=1e-15)
    assert np.isnan(noise[2])
    key = (0, 1, STEP_TIME, 5, 30000, 0, 3)
    assert time_offset(key, 0.03) == pytest.approx(0.03 * noise_rng(*key).standard_normal(), rel=1e-15)


def test_vo_factor_sigma_values():
    sigmas = DriftSigmas(trans_m=0.0021, rot_deg=0.039, scale=0.0045, rate_deg_per_m=0.045)
    floors = {"floor_trans_m": 1e-4, "floor_rot_deg": 1e-3}
    assert vo_factor_sigma(sigmas, 0.15, **floors) == pytest.approx((0.002205816175477911, 0.039579824405876284),
                                                                    rel=1e-12)
    assert vo_factor_sigma(sigmas, 0.30, **floors) == pytest.approx((0.003262284475639732, 0.05678247969224311),
                                                                    rel=1e-12)
    assert vo_factor_sigma(DriftSigmas(), 0.15, **floors) == (1e-4, 1e-3)


def _same_cameras(a, b) -> bool:
    return all(np.array_equal(a[k][i].R_c2w, b[k][i].R_c2w) and np.array_equal(a[k][i].C, b[k][i].C)
               for k in a for i in a[k])


def _end_error(run, cams, pitch, side):
    """Camera minus truth position of the last frame of the (pitch, side) pass."""
    key = next(k for k in cams if k.pitch_deg == pitch and k.side == side)
    last = max(f.idx for f in run.frames if f.pass_key == key)
    truth = next(f for f in run.frames if f.idx == last)
    return cams[key][last].C - truth.camera.C


def test_drift_model_repeats_per_seed_and_changes_with_it(apple_run):
    run = SynthRun.open(apple_run)
    cfg = fake_cfg(apple_run, "apple", "pose.rig_ref=30", *D_EXT_VO)
    a, b, c = _cams(run, cfg, 0), _cams(run, cfg, 0), _cams(run, cfg, 1)
    assert set(a) == set(_passes(run, cfg))
    assert _same_cameras(a, b)
    assert not _same_cameras(a, c)
    truth = {f.idx: f.camera for f in run.frames}
    for key, cams in a.items():
        first = min(cams)
        assert np.abs(cams[first].C - truth[first].C).max() < 1e-12, key
        assert np.abs(cams[first].R_c2w - truth[first].R_c2w).max() < 1e-12, key
    assert np.linalg.norm(_end_error(run, a, 30.0, "A")) > 1e-3
    # The fusion setting does not reach the flow generator: the same sigmas and the same draws.
    fused = fake_cfg(apple_run, "apple", "pose.rig_ref=30", *D_EXT_VO, "pose.fuse=imu")
    assert drift_sigmas(fused.noise) == drift_sigmas(cfg.noise)
    got = drift_sigmas(cfg.noise)   # walk translation in m from mm
    assert (got.trans_m, got.rot_deg, got.scale, got.rate_deg_per_m) == pytest.approx((0.0021, 0.039, 0.0045, 0.045),
                                                                                       rel=1e-12)
    for fn in (flow_draws, noisy_increments):
        names = set(inspect.signature(fn).parameters)
        assert not any("fuse" in n or "cfg" in n for n in names), (fn.__name__, names)


def test_thinned_frames_keep_their_drift(apple_run):
    run = SynthRun.open(apple_run)
    full = _cams(run, fake_cfg(apple_run, "apple", "pose.rig_ref=30", *D_EXT_VO), 0)
    thin = _cams(run, fake_cfg(apple_run, "apple", "pose.rig_ref=30", *D_EXT_VO, "passes.step_m=0.3"), 0)
    shared = [(k, i) for k in thin for i in thin[k]]
    assert shared and len(shared) < sum(len(v) for v in full.values())
    for k, i in shared:
        assert np.array_equal(thin[k][i].R_c2w, full[k][i].R_c2w) and np.array_equal(thin[k][i].C, full[k][i].C)


def test_one_body_flow_per_side_under_the_rig(apple_run):
    run = SynthRun.open(apple_run)
    cfg = fake_cfg(apple_run, "apple", "pose.rig=true", "pose.rig_ref=30", *D_EXT_VO, "noise.rig_extr_trans_mm=2",
                   "noise.rig_extr_rot_deg=0.1")
    cams = _cams(run, cfg, 0)
    dX = seeded_rigid((0, 0, STEP_RIG, run.meta.seed, 10000), 0.002, 0.1)
    spots: dict[tuple[str, int], dict[float, np.ndarray]] = {}
    for key, by_idx in cams.items():
        for f in run.frames:
            if f.idx in by_idx:
                spots.setdefault((key.side, f.pos_k), {})[key.pitch_deg] = pose_matrix(by_idx[f.idx])
    assert spots
    for spot, by_pitch in spots.items():
        want = by_pitch[30.0] @ relative_extrinsic(30.0, 10.0) @ dX
        assert np.abs(by_pitch[10.0] - want).max() < 1e-12, spot
    assert np.abs(_end_error(run, cams, 30.0, "A") - _end_error(run, cams, 30.0, "B")).max() > 1e-6
    errors = rig_errors(cfg, 0, scene=run.meta.seed, pitches=[10.0, 30.0])
    assert list(errors) == [10.0] and np.array_equal(errors[10.0], dX)
    assert rig_errors(cfg, None, scene=run.meta.seed, pitches=[10.0, 30.0]) == {}
    plain_rig = fake_cfg(apple_run, "apple", "pose.rig=true", "pose.rig_ref=30", *D_EXT_VO)
    assert rig_errors(plain_rig, 0, scene=run.meta.seed, pitches=[10.0, 30.0]) == {}


def test_one_flow_per_pass_without_the_rig(apple_run):
    run = SynthRun.open(apple_run)
    cfg = fake_cfg(apple_run, "apple", "pose.rig_ref=30", *D_EXT_VO)
    cams = _cams(run, cfg, 0)
    for side in ("A", "B"):
        assert np.abs(_end_error(run, cams, 10.0, side) - _end_error(run, cams, 30.0, side)).max() > 1e-6, side
    pitched = fake_cfg(apple_run, "apple", "pose.rig_ref=30", *D_EXT_VO, "noise.pass_pitch_trans_m=0.02",
                       "noise.pass_pitch_rot_deg=0.25")
    assert not _same_cameras(cams, _cams(run, pitched, 0))
    # A side start error moves the first body pose of both passes of that side (body frame, right side).
    started = fake_cfg(apple_run, "apple", "pose.rig_ref=30", *D_EXT_VO, "noise.pass_side_trans_m=0.05",
                       "noise.pass_side_rot_deg=0.5")
    truth = {f.idx: f.camera for f in run.frames}
    for key, by_idx in _cams(run, started, 0).items():
        first, X = min(by_idx), pitch_extrinsic(key.pitch_deg)
        error = seeded_rigid((0, 0, STEP_START, run.meta.seed, 0 if key.side == "A" else 1, key.aisle), 0.05, 0.5)
        want = pose_matrix(truth[first]) @ np.linalg.inv(X) @ error @ X
        assert np.abs(pose_matrix(by_idx[first]) - want).max() < 1e-12, key


def test_zero_drift_gives_the_truth_cameras(apple_run):
    run = SynthRun.open(apple_run)
    cfg = fake_cfg(apple_run, "apple", "pose.rig_ref=30", *ZERO_DRIFT, "noise.depth_rel_sigma=0.02")
    passes = _passes(run, cfg)
    cams = model_cameras(run.frames, passes, cfg, 0, scene=run.meta.seed, base_step_m=run.meta.base_step_m)
    assert cams is not None and set(cams) == set(passes)
    assert all(cams[key][f.idx] is f.camera for key, frames in passes.items() for f in frames)


def test_drift_needs_frame_times_and_positions(apple_run):
    run = SynthRun.open(apple_run)
    timed = fake_cfg(apple_run, "apple", "pose.rig_ref=30", *D_EXT_VO, "noise.drift_time_offset_s=0.01")
    with pytest.raises(ValueError, match="needs frame times"):
        _cams(run, timed, 0)
    cfg = fake_cfg(apple_run, "apple", "pose.rig_ref=30", *D_EXT_VO)
    with pytest.raises(ValueError, match="needs a sequence bundle"):
        model_cameras(run.frames, _passes(run, cfg), cfg, 0, scene=run.meta.seed, base_step_m=None)
