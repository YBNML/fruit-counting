"""The fused pose sources: the truth source's body flow and simulated IMU in one graph per drive (with the
camera-IMU extrinsic error and row-end GPS fixes), the file source's pose caches and IMU files, the time
offsets they believe, the settings that stop a fused run before it starts, and what the results say."""

import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from typer.testing import CliRunner

import counting.evaluation.fusion as fusion_module
import counting.odometry.vio as vio_module
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.cli import eval_app
from counting.evaluation.fusion import fusion_inputs
from counting.evaluation.imusim import (
    AssumedMotion, ImuProfile, ImuSampling, frame_times, gps_end_fixes, imu_session, session_drives,
)
from counting.evaluation.posesource import check_pose_supported, drift_sigmas, model_cameras
from counting.evaluation.runner import _track_passes, evaluate, open_imu_file, prepare
from counting.geometry.noise import (
    STEP_GPS, STEP_IMU_EXTR, STEP_PITCH, STEP_START, STEP_TIME, flow_draws, noise_rng, noisy_increments,
    pass_key_code, seeded_rigid, side_code, time_offset, vo_factor_sigma,
)
from counting.geometry.posefile import PoseFile
from counting.geometry.rig import (
    believed_extrinsic, body_from_reference, interpolate_pose, pitch_extrinsic, pose_matrix, rigid,
)
from counting.odometry.cache import PassOdometry, load_pose_cache, pose_cache_path, write_pose_cache
from counting.odometry.imufile import GRAVITY, LAYOUTS, ImuFile, imu_file_path, write_imu_file
from counting.odometry.vio import body_motions
from tests.unit.posefake import write_truth_pose_cache
from tests.unit.synthfake import EVAL_CONFIGS, ROOT, fake_cfg, load_splits, write_run
from tests.unit.test_drift import D_EXT_VO, _passes
from tests.unit.test_pose_runner import _counts
from tests.unit.test_traj_adapter import _copy, _edit, _edit_npz
from tests.unit.test_vo import _first_frames, _moved

pytestmark = pytest.mark.pose
pytest.importorskip("gtsam")
runner = CliRunner()
SPLITS = str(ROOT / "configs" / "eval" / "splits_pose.yaml")
FUSE_FIXED = {"estimator": {"gyro_n": 7.85e-5, "acc_n": 9.81e-4, "gyro_w": 1.82e-6, "acc_w": 4.75e-5},
              "start_speed_mps": 0.5, "start_speed_sigma_mps": 0.2, "start_roll_pitch_sigma_deg": 5.0,
              "shake_share": {"side_m": 0.02, "height_m": 0.01, "along_m": 0.01, "yaw_deg": 0.5},
              "vo_sigma_floor_trans_mm": 0.1, "vo_sigma_floor_rot_deg": 0.001}
I_EXT = ("noise.imu_gyro_n=1.16e-3", "noise.imu_acc_n=4.97e-3", "noise.imu_gyro_w=1.82e-5", "noise.imu_acc_w=4.75e-4",
         "noise.imu_gyro_b0_dps=1.0", "noise.imu_acc_b0_mg=4.0", "noise.imu_temp_k=1.0")
BELIEF = ("noise.imu_time_offset_ms=0", "noise.imu_extr_trans_mm=10", "noise.imu_extr_rot_deg=0.3")
RIG = ("pose.rig=true", "pose.rig_ref=30", "pose.fuse=imu")


@pytest.fixture
def fixed(tmp_path, monkeypatch):
    path = tmp_path / "fixed" / "fuse_fixed.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(FUSE_FIXED), encoding="utf-8")
    monkeypatch.setattr(fusion_module, "FUSE_FIXED_PATH", path)
    return path


@pytest.fixture(scope="module")
def shake_root(tmp_path_factory):
    return write_run(tmp_path_factory.mktemp("shake"), crop="apple", size=(64, 36), shake=True)


@pytest.fixture(scope="module")
def small_root(tmp_path_factory):
    """A straight bundle at 64 x 36 for poses, IMU and results only: every count of it is 0."""
    return write_run(tmp_path_factory.mktemp("small"), crop="apple", size=(64, 36))


@pytest.fixture(scope="module")
def counted(tmp_path_factory):
    """A straight and a shake bundle at 160 x 90, small but with counts above 0."""
    where = tmp_path_factory.mktemp("counted")
    return {"straight": write_run(where / "straight", crop="apple", size=(160, 90)),
            "shake": write_run(where / "shake", crop="apple", size=(160, 90), shake=True)}


@pytest.fixture
def drives_seen(monkeypatch):
    """Every (VioDrive, fused bodies) the pose sources hand to the graph, in order."""
    seen, real = [], vio_module.fuse_drive

    def record(drive):
        out = real(drive)
        seen.append((drive, out))
        return out

    monkeypatch.setattr(vio_module, "fuse_drive", record)
    return seen


def _refs(run, side, pitch=30.0):
    return sorted((f for f in run.frames if f.pass_key.side == side and f.pass_key.pitch_deg == pitch),
                  key=lambda f: f.idx)


def _bodies(refs, pitch=30.0):
    return np.stack([body_from_reference(pose_matrix(f.camera), pitch_extrinsic(pitch)) for f in refs])


def _cams(run, cfg, noise_seed, *, pose_file=None, imu_file=None):
    return model_cameras(run.frames, _passes(run, cfg), cfg, noise_seed, scene=run.meta.seed,
                         base_step_m=run.meta.base_step_m, plans=run.plans(), pose_file=pose_file,
                         fusion=fusion_inputs(run, cfg, noise_seed, imu_file=imu_file))


def test_fused_truth_without_noise_gives_the_l3_counts(counted, fixed):
    for name, tol in (("straight", (1e-6, 1e-5)), ("shake", (5e-4, 0.005))):
        root = counted[name]
        ctx = prepare(SynthRun.open(root), fake_cfg(root, "apple"), load_splits())
        cfg = fake_cfg(root, "apple", *RIG)
        assert not cfg.is_noisy()
        staged = _track_passes(ctx, cfg, None)
        pos = max(np.abs(p.cameras[f.idx].C - f.camera.C).max() for p in staged for f in p.frames)
        rot = max(np.degrees(Rotation.from_matrix(p.cameras[f.idx].R_c2w.T @ f.camera.R_c2w).magnitude())
                  for p in staged for f in p.frames)
        assert pos < tol[0] and rot < tol[1], (name, pos, rot)
        l3 = _counts(evaluate(ctx, fake_cfg(root, "apple")))
        assert sum(p[0] for p in l3[0]) > 0, name                       # counts above 0, so the match means something
        fused = evaluate(ctx, cfg)
        assert _counts(fused) == l3, name
        assert fused["pose"]["start"] == "nominal_from_first_frame", name


def test_truth_factor_is_the_conjugate_of_the_body_flow(apple_run, shake_root, fixed, drives_seen):
    run = SynthRun.open(apple_run)
    cfg = fake_cfg(apple_run, "apple", *RIG, *D_EXT_VO, *I_EXT, *BELIEF)
    cams = _cams(run, cfg, 0)
    X, dX = pitch_extrinsic(30.0), seeded_rigid((0, 0, STEP_IMU_EXTR, 5, 30000), 0.010, 0.3)
    G = X @ dX @ np.linalg.inv(X)
    times = frame_times(run, AssumedMotion())
    memory = imu_session(session_drives(run, "rig", AssumedMotion()), ImuProfile.from_noise(cfg.noise), ImuSampling(),
                         imu_seed=0, noise_seed=0, scene=5)
    trans, rot = vo_factor_sigma(drift_sigmas(cfg.noise), 0.15, 1e-4, 1e-3)
    assert len(drives_seen) == 2
    for (drive, out), side, stream in zip(drives_seen, "AB", memory):
        refs = _refs(run, side)
        body = _bodies(refs)
        inc = noisy_increments(body, flow_draws(0, 0, 5, side_code(refs[0].pass_key), [f.pos_k for f in refs]),
                               drift_sigmas(cfg.noise), 0.15)
        (cam,) = drive.cameras
        np.testing.assert_allclose(body_motions(cam), G @ inc @ np.linalg.inv(G), rtol=0, atol=1e-12)
        np.testing.assert_allclose(cam.extrinsic, X @ dX, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(drive.times, [times[f.idx] for f in refs])
        np.testing.assert_array_equal(drive.start, body[0])                   # no start keys: the first true body
        assert drive.odometry_sigma == pytest.approx((trans, np.radians(rot)))
        np.testing.assert_allclose(drive.start_sigma, np.r_[np.radians([5.0, 5.0]), [1e-4] * 4], rtol=0, atol=1e-15)
        np.testing.assert_allclose(drive.start_velocity, body[0, :3, :3] @ [0.5, 0.0, 0.0], rtol=0, atol=1e-15)
        assert drive.imu.stream.gyro.tobytes() == stream.gyro.tobytes() and drive.gps == () and drive.standstill is None
        assert (drive.imu.gyro_n, drive.imu.acc_n, drive.imu.gyro_w, drive.imu.acc_w) == (7.85e-5, 9.81e-4, 1.82e-6, 4.75e-5)
        assert drive.imu.gyro_b0 == pytest.approx(np.radians(1.0)) and drive.imu.acc_b0 == pytest.approx(4e-3 * GRAVITY)
        node = {f.pos_k: k for k, f in enumerate(refs)}
        for f in _refs(run, side, 10.0):    # the other camera rides the fused body through its believed extrinsic
            want = out[node[f.pos_k]] @ believed_extrinsic(10.0, 30.0, dX, None)
            np.testing.assert_allclose(pose_matrix(cams[f.pass_key][f.idx]), want, rtol=0, atol=1e-12)
    drives_seen.clear()
    starts = ("noise.pass_side_trans_m=0.05", "noise.pass_side_rot_deg=0.5", "noise.pass_pitch_trans_m=0.02",
              "noise.pass_pitch_rot_deg=0.25")
    c1x = fake_cfg(apple_run, "apple", "pose.rig_ref=30", "pose.fuse=imu", *D_EXT_VO, *I_EXT, *starts)
    _cams(run, c1x, 0)
    drives = session_drives(run, "pass", AssumedMotion())
    made = imu_session(drives, ImuProfile.from_noise(c1x.noise), ImuSampling(), imu_seed=0, noise_seed=0, scene=5)
    streams = {(d.side, d.pitch_deg): s for d, s in zip(drives, made)}
    assert len(drives_seen) == 4
    for (drive, _), key in zip(drives_seen, _passes(run, c1x)):
        side = side_code(key)
        body = _bodies(_refs(run, key.side, key.pitch_deg), key.pitch_deg)
        start_error = seeded_rigid((0, 0, STEP_START, 5, *side), 0.05, 0.5)
        pitch_error = seeded_rigid((0, 0, STEP_PITCH, 5, *side, round(key.pitch_deg * 1000)), 0.02, 0.25)
        np.testing.assert_allclose(drive.start, body[0] @ start_error @ pitch_error, rtol=0, atol=1e-12)
        want = np.r_[np.radians([5.0, 5.0, np.hypot(0.5, 0.25)]), [np.hypot(0.05, 0.02)] * 3]
        np.testing.assert_allclose(drive.start_sigma, want, rtol=0, atol=1e-15)
        assert drive.imu.stream.acc.tobytes() == streams[(key.side, key.pitch_deg)].acc.tobytes()
    # A bundle rendered with body shake: each drive starts at the planned body of its first reference frame (times
    # its start errors), and the yaw and each position axis of the start sigma join the shake share.
    share = {"along_m": 0.01, "side_m": 0.02, "height_m": 0.03, "yaw_deg": 0.4}       # a different value on each axis
    fixed.write_text(yaml.safe_dump({**FUSE_FIXED, "shake_share": share}), encoding="utf-8")
    shaken = SynthRun.open(shake_root)
    plans, xyz = shaken.plans(), np.array([share["along_m"], share["side_m"], share["height_m"]])
    for sets in (RIG, ("pose.rig_ref=30", "pose.fuse=imu", *starts)):
        drives_seen.clear()
        cfg = fake_cfg(shake_root, "apple", *sets, *D_EXT_VO)
        _cams(shaken, cfg, 0)
        keys = [key for key in _passes(shaken, cfg) if not cfg.pose.rig or key.pitch_deg == 30.0]
        assert len(drives_seen) == len(keys)
        for (drive, _), key in zip(drives_seen, keys):
            side, q = side_code(key), key.pitch_deg
            refs = _refs(shaken, key.side, q)
            plan = body_from_reference(pose_matrix(plans[refs[0].idx].plan), pitch_extrinsic(q))
            assert np.abs(plan - _bodies(refs, q)[0]).max() > 5e-3             # the plan is not the first true body
            if cfg.pose.rig:                                                    # no start keys
                error, yaw, trans = np.eye(4), share["yaw_deg"], xyz
            else:
                error = (seeded_rigid((0, 0, STEP_START, 5, *side), 0.05, 0.5)
                         @ seeded_rigid((0, 0, STEP_PITCH, 5, *side, round(q * 1000)), 0.02, 0.25))
                yaw, trans = np.hypot(np.hypot(0.5, 0.25), share["yaw_deg"]), np.hypot(np.hypot(0.05, 0.02), xyz)
            np.testing.assert_allclose(drive.start, plan @ error, rtol=0, atol=1e-12)
            want = np.r_[np.radians([5.0, 5.0, yaw]), trans]
            np.testing.assert_allclose(drive.start_sigma, want, rtol=0, atol=1e-15)


def test_large_extrinsic_rotation_raises_the_world_end_not_the_in_pass_drift(small_root, fixed, monkeypatch):
    X = pitch_extrinsic(30.0)
    pitch_up = rigid(np.radians(2.0) * (X[:3, :3].T @ [0.0, 1.0, 0.0]), np.zeros(3))   # 2 deg of body pitch
    cfg = fake_cfg(small_root, "apple", *RIG, "noise.imu_extr_rot_deg=1")
    ctx = prepare(SynthRun.open(small_root), fake_cfg(small_root, "apple"), load_splits())

    def summary(error):
        monkeypatch.setattr(fusion_module, "imu_errors", lambda *args, **kwargs: {30.0: error})
        return evaluate(ctx, cfg, noise_seed=0)["pose_quality"]["summary"]

    tilted, level = summary(pitch_up), summary(np.eye(4))
    assert tilted["world_end"]["height_m"] > 0.10 and level["world_end"]["height_m"] < 0.01
    assert abs(tilted["in_pass"]["pos_m"] - level["in_pass"]["pos_m"]) < 1e-3


def test_gps_fixes_are_two_per_flow_and_leave_the_other_draws(apple_run, fixed, drives_seen):
    run = SynthRun.open(apple_run)
    plain = fake_cfg(apple_run, "apple", *RIG, *D_EXT_VO, *I_EXT, *BELIEF)
    gps = fake_cfg(apple_run, "apple", *RIG, *D_EXT_VO, *I_EXT, *BELIEF, "noise.gps_end_h_m=0.02",
                   "noise.gps_end_v_m=0.03")
    for cfg in (plain, gps):
        _cams(run, cfg, 0)
    (a, _), (b, _), (c, _), (d, _) = drives_seen
    for without, with_fixes in ((a, c), (b, d)):
        assert body_motions(without.cameras[0]).tobytes() == body_motions(with_fixes.cameras[0]).tobytes()
        assert without.start.tobytes() == with_fixes.start.tobytes()
        assert without.imu.stream.acc.tobytes() == with_fixes.imu.stream.acc.tobytes() and without.gps == ()
    for drive, side in ((c, "A"), (d, "B")):
        refs = _refs(run, side)
        body = _bodies(refs)
        flow = side_code(refs[0].pass_key)
        first, last = gps_end_fixes((0, 0, STEP_GPS, 5, *flow), body[0, :3, 3], body[-1, :3, 3], 0.02, 0.03)
        for end, (node, position) in enumerate(((0, first), (len(refs) - 1, last))):
            want = body[[0, -1][end], :3, 3] + noise_rng(0, 0, STEP_GPS, 5, *flow, end).standard_normal(3) * [0.02, 0.02, 0.03]
            np.testing.assert_array_equal(position, want)
            fix = drive.gps[end]
            assert fix.node == node and np.array_equal(fix.position, position)
            np.testing.assert_array_equal(fix.sigma, [0.02, 0.02, 0.03])


def test_fusion_stops_before_bundles_without_gtsam_or_fixed_settings_and_on_test(apple_run, tmp_path, fixed,
                                                                                 monkeypatch):
    cfg = fake_cfg(apple_run, "apple", *RIG)
    check_pose_supported(cfg)
    with pytest.raises(NotImplementedError, match="pose.fuse=imu with split=test is not supported"):
        check_pose_supported(fake_cfg(apple_run, "apple", *RIG, "split=test", "tuning=false"))
    monkeypatch.setattr(fusion_module, "has_gtsam", lambda: False)
    with pytest.raises(NotImplementedError, match=r"pose.fuse=imu needs gtsam \(the conda-forge build in the environment files\)"):
        check_pose_supported(cfg)
    monkeypatch.setattr(fusion_module, "has_gtsam", lambda: True)
    monkeypatch.setattr(fusion_module, "FUSE_FIXED_PATH", tmp_path / "none" / "fuse_fixed.yaml")
    with pytest.raises(FileNotFoundError, match="fuse_fixed.yaml: no fixed fusion settings; pose.fuse reads them"):
        check_pose_supported(cfg)
    raw = yaml.safe_load((EVAL_CONFIGS / "apple_dev.yaml").read_text(encoding="utf-8"))
    raw.update(runs=[str(apple_run)], output_dir=str(tmp_path / "out"), pose={"rig": True, "rig_ref": 30, "fuse": "imu"})
    path = tmp_path / "fused.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    opened = []
    monkeypatch.setattr(SynthRun, "open", lambda root: opened.append(root))
    r = runner.invoke(eval_app, ["run", "--config", str(path), "--splits", SPLITS])
    assert r.exit_code == 2 and "no fixed fusion settings" in r.output, r.output
    assert opened == [] and not (tmp_path / "out").exists()


def test_fused_truth_repeats_and_takes_its_estimator_and_standstill(apple_run, fixed, drives_seen):
    run = SynthRun.open(apple_run)
    cfg = fake_cfg(apple_run, "apple", *RIG, *D_EXT_VO, *I_EXT, *BELIEF)

    def matrices(cams):
        return np.stack([pose_matrix(c) for key in sorted(cams, key=str) for c in cams[key].values()])

    assert matrices(_cams(run, cfg, 1)).tobytes() == matrices(_cams(run, cfg, 1)).tobytes()
    drives_seen.clear()
    plain = _cams(run, fake_cfg(apple_run, "apple", "pose.rig_ref=30", "pose.fuse=imu"), None)   # no rig, no drift
    assert plain is not None and len(drives_seen) == 4                 # still fused, one drive per pass

    def first_drive(*sets):
        drives_seen.clear()
        _cams(run, fake_cfg(apple_run, "apple", *RIG, *D_EXT_VO, *I_EXT, *BELIEF, *sets), 0)
        return drives_seen[0][0]

    know = first_drive("pose.vio.estimator_imu=generator")
    assert (know.imu.gyro_n, know.imu.acc_n, know.imu.gyro_w, know.imu.acc_w) == (1.16e-3, 4.97e-3, 1.82e-5, 4.75e-4)
    still = first_drive("pose.vio.stop_s=5")
    start = still.imu.stream.t[0]
    assert still.standstill == (start, start + 5.0)
    assert first_drive("pose.vio.stop_s=0").standstill is None              # 0 is no standstill
    with pytest.raises(ValueError, match="pose.vio.stop_s=6 is longer than the standstill of the drive on side A"):
        first_drive("pose.vio.stop_s=6")


LOW = ImuProfile(gyro_n=1e-6, acc_n=1e-5, gyro_b0_dps=1e-3, acc_b0_mg=1e-3)
TIGHT_SIGMA = ("pose.vio.vo_sigma_trans_mm=0.01", "pose.vio.vo_sigma_rot_deg=0.0001")


def _files(run, where, *, profile=LOW, imu_seed=0, seeds=(0,)):
    """The true poses of `run` as a pose cache and an IMU file of `profile` under `where`; the two folders."""
    write_truth_pose_cache(run, where / "pose")
    streams = {}
    for layout in LAYOUTS:
        drives = session_drives(run, layout, AssumedMotion())
        for s in seeds:
            made = imu_session(drives, profile, ImuSampling(), imu_seed=imu_seed, noise_seed=s, scene=run.meta.seed)
            streams.update({d.stream_key(s): m for d, m in zip(drives, made)})
    meta = {"run_id": run.meta.run_id, "fingerprint": run.meta.fingerprint, "imu_seed": imu_seed, "rate_hz": 200.0,
            "profile": "I_low", "noise": dataclasses.asdict(profile), "motion": dataclasses.asdict(AssumedMotion())}
    write_imu_file(imu_file_path(where / "imu", run.meta.run_id), streams, meta)
    return where / "pose", where / "imu"


def _file_cfg(root, pose_dir, imu_dir, *sets):
    files = ("pose.source=file", "pose.cache=vo", f"pose.caches.vo.dir={pose_dir}")
    imu = () if imu_dir is None else ("pose.imu=i", f"pose.imu_files.i.dir={imu_dir}")
    return fake_cfg(root, "apple", "pose.rig=true", "pose.rig_ref=30", *files, *imu, *sets)


def _opened(run, cfg):
    pose_file = PoseFile(pose_cache_path(Path(cfg.pose.caches["vo"].dir), run.meta.run_id))
    imu = cfg.pose.imu
    imu_file = None if imu is None else ImuFile(imu_file_path(Path(cfg.pose.imu_files[imu].dir), run.meta.run_id))
    return pose_file, imu_file


def test_fused_file_source_from_true_caches_is_near_the_truth(apple_run, tmp_path, fixed):
    run = SynthRun.open(apple_run)
    pose_dir, imu_dir = _files(run, tmp_path)
    for fuse, imu, tol in (("imu", imu_dir, (1e-3, 0.01)), ("vo", None, (1e-9, 1e-7))):
        cfg = _file_cfg(apple_run, pose_dir, imu, f"pose.fuse={fuse}", "pose.anchor=first_truth", *TIGHT_SIGMA)
        pose_file, imu_file = _opened(run, cfg)
        passes = _passes(run, cfg)
        cams = _cams(run, cfg, None, pose_file=pose_file, imu_file=imu_file)
        pos = max(np.abs(cams[k][f.idx].C - f.camera.C).max() for k, fs in passes.items() for f in fs)
        rot = max(np.degrees(Rotation.from_matrix(cams[k][f.idx].R_c2w.T @ f.camera.R_c2w).magnitude())
                  for k, fs in passes.items() for f in fs)
        assert pos < tol[0] and rot < tol[1], (fuse, pos, rot)
        if fuse == "imu":            # without a noise seed the file source takes the file's seed-0 streams
            late = _file_cfg(apple_run, pose_dir, imu, "pose.fuse=imu", *TIGHT_SIGMA, "noise.imu_time_offset_ms=50",
                             "pose.vio.stop_s=5")             # over 5 s from 50 ms in true time the body moves
            inputs = fusion_inputs(run, late, None, imu_file=imu_file)
            d = session_drives(run, "rig", AssumedMotion())[0]
            key, stream = (d.side, d.aisle, None), imu_file.stream("rig", 0, d.side, d.aisle)
            assert inputs.imu["stream"] == "noise_seed_0"
            assert inputs.streams[key].acc.tobytes() == stream.acc.tobytes()
            t = inputs.streams[key].t
            np.testing.assert_array_equal(t, stream.t + 50 / 1000)          # the IMU clock is believed late
            assert inputs.standstill[key] == (t[0], t[0] + 5.0)           # held in believed time, checked in true time


def test_file_fusion_pairs_cameras_by_position_and_skips_failed_pairs(apple_run, tmp_path, fixed):
    run = SynthRun.open(apple_run)
    pose_dir, imu_dir = _files(run, tmp_path)
    cache = load_pose_cache(pose_cache_path(pose_dir, run.meta.run_id))
    code = pass_key_code(next(k for k in run.passes() if k.pitch_deg == 10.0 and k.side == "A"))
    jump = rigid(np.zeros(3), [0.5, 0.0, 0.0])

    def rewrite(name, moved):
        passes = {}
        for c, p in cache.passes.items():
            failed, rel = p.failed.copy(), p.rel_c2w.copy()
            if c == code:
                failed[5] = 1
                if moved:
                    rel[5:] = jump @ rel[5:]          # only the pair into row 5 changes
            passes[c] = PassOdometry(idx=p.idx, rel_c2w=rel, inliers=p.inliers, failed=failed, prior=p.prior)
        write_pose_cache(pose_cache_path(tmp_path / name, run.meta.run_id), passes,
                         {k: v for k, v in cache.meta.items() if k != "format"})
        return tmp_path / name

    folders = {"clean": rewrite("clean", False), "garbage": rewrite("garbage", True)}
    out = {}
    for fuse in ("imu", "vo"):
        for name, folder in folders.items():
            cfg = _file_cfg(apple_run, folder, imu_dir if fuse == "imu" else None, f"pose.fuse={fuse}",
                            "pose.anchor=first_truth", *TIGHT_SIGMA)
            pose_file, imu_file = _opened(run, cfg)
            cams = _cams(run, cfg, None, pose_file=pose_file, imu_file=imu_file)
            out[fuse, name] = np.stack([pose_matrix(c) for k in sorted(cams, key=str) for c in cams[k].values()])
    assert out["imu", "clean"].tobytes() == out["imu", "garbage"].tobytes()       # a failed pair's pose is not used
    assert np.isfinite(out["vo", "garbage"]).all() and np.abs(out["vo", "garbage"] - out["vo", "clean"]).max() > 0.1


def test_file_fusion_needs_sigmas_and_a_matching_imu_file(apple_run, ctx_of, tmp_path, fixed):
    run = SynthRun.open(apple_run)
    pose_dir, imu_dir = _files(run, tmp_path / "a")
    cfg = _file_cfg(apple_run, pose_dir, imu_dir, "pose.fuse=imu", "pose.anchor=first_truth")
    pose_file, imu_file = _opened(run, cfg)
    with pytest.raises(ValueError, match="pose.source=file with pose.fuse=imu needs pose.vio.vo_sigma_trans_mm and "
                                         "pose.vio.vo_sigma_rot_deg"):
        _cams(run, cfg, None, pose_file=pose_file, imu_file=imu_file)
    seeded = _file_cfg(apple_run, pose_dir, imu_dir, "pose.fuse=imu", "pose.anchor=first_truth", *TIGHT_SIGMA,
                       "noise.imu_extr_rot_deg=0.1", "noise.seeds=[4]")
    with pytest.raises(ValueError, match=r"no IMU stream for noise seed 4; the file has \[0\]"):
        _cams(run, seeded, 4, pose_file=pose_file, imu_file=imu_file)
    other = _files(run, tmp_path / "b", imu_seed=1)[1]
    with pytest.raises(ValueError, match="IMU file was made with imu_seed 1; the config seed is 0"):
        open_imu_file(ctx_of["apple"], _file_cfg(apple_run, pose_dir, other, "pose.fuse=imu", *TIGHT_SIGMA))
    zero = _files(run, tmp_path / "c", profile=ImuProfile(gyro_n=1e-6, acc_n=1e-5, gyro_b0_dps=1e-3))[1]
    with pytest.raises(ValueError, match="has acc_b0_mg = 0; a noiseless IMU is the truth"):
        open_imu_file(ctx_of["apple"], _file_cfg(apple_run, pose_dir, zero, "pose.fuse=imu", *TIGHT_SIGMA))


def test_file_fusion_ignores_renumbered_positions_and_the_step_value(apple_run, tmp_path, fixed):
    run = SynthRun.open(apple_run)
    pose_dir, imu_dir = _files(run, tmp_path)
    cfg = _file_cfg(apple_run, pose_dir, imu_dir, "pose.fuse=imu", "pose.anchor=first_truth",
                    "pose.vio.vo_sigma_trans_mm=2.1", "pose.vio.vo_sigma_rot_deg=0.039")
    pose_file, imu_file = _opened(run, cfg)
    inputs = fusion_inputs(run, cfg, None, imu_file=imu_file)
    passes = _passes(run, cfg)
    a = model_cameras(run.frames, passes, cfg, None, scene=5, base_step_m=0.15, pose_file=pose_file, fusion=inputs)
    moved = [dataclasses.replace(f, pos_k=f.pos_k + 1) for f in run.frames]
    by_idx = {f.idx: f for f in moved}
    renumbered = {k: [by_idx[f.idx] for f in fs] for k, fs in passes.items()}
    b = model_cameras(moved, renumbered, cfg, None, scene=5, base_step_m=0.30, pose_file=pose_file, fusion=inputs)
    for key, frames in passes.items():
        for f in frames:
            assert pose_matrix(a[key][f.idx]).tobytes() == pose_matrix(b[key][f.idx]).tobytes()


def test_time_offsets_under_fusion(apple_run, shake_root, tmp_path, fixed, drives_seen):
    run = SynthRun.open(shake_root)
    plans = run.plans()
    cfg = fake_cfg(shake_root, "apple", *RIG, "pose.anchor=first_truth", "noise.drift_time_offset_s=0.1",
                   "noise.imu_time_offset_ms=50")
    _cams(run, cfg, 0)
    memory = imu_session(session_drives(run, "rig", AssumedMotion()), ImuProfile.from_noise(cfg.noise), ImuSampling(),
                         imu_seed=0, noise_seed=0, scene=5)
    for (drive, _), side, stream in zip(drives_seen, "AB", memory):
        refs = _refs(run, side)
        t = np.array([plans[f.idx].t for f in refs])
        body = _bodies(refs)
        dt = time_offset((0, 0, STEP_TIME, 5, *pass_key_code(refs[0].pass_key)), 0.1)
        src = np.stack([interpolate_pose(t, body, x + dt) for x in t])
        np.testing.assert_array_equal(drive.times, t)                                  # the frame times stay
        np.testing.assert_array_equal(drive.imu.stream.t, stream.t + 50 / 1000)       # the IMU clock is believed late
        steps = np.stack([np.linalg.inv(a) @ b for a, b in zip(src[:-1], src[1:])])
        np.testing.assert_allclose(body_motions(drive.cameras[0]), steps, rtol=0, atol=1e-9)
        np.testing.assert_allclose(drive.start, src[0], rtol=0, atol=1e-12)
    drives_seen.clear()                     # a standstill is held in the believed IMU time but checked in true time
    _cams(SynthRun.open(apple_run), fake_cfg(apple_run, "apple", *RIG, "noise.imu_time_offset_ms=50", "pose.vio.stop_s=5"), 0)
    late = drives_seen[0][0]
    assert late.imu.stream.t[0] == pytest.approx(0.05, abs=1e-12)
    assert late.standstill == (late.imu.stream.t[0], late.imu.stream.t[0] + 5.0)
    file = fake_cfg(shake_root, "apple", "pose.source=file", "pose.cache=vo", f"pose.caches.vo.dir={tmp_path}",
                    "noise.drift_time_offset_s=0.01")
    with pytest.raises(NotImplementedError, match="the camera-IMU time belief of a file source is noise.imu_time_offset_ms"):
        check_pose_supported(file)


def _fused_config(where, root, pose) -> str:
    where.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load((EVAL_CONFIGS / "apple_dev.yaml").read_text(encoding="utf-8"))
    raw.update(runs=[str(root)], output_dir=str(where / "out"), pose=pose)
    path = where / "fused.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_fused_results_name_the_fusion_and_change_with_the_fixed_settings(small_root, tmp_path, fixed):
    config = _fused_config(tmp_path, small_root, {"rig": True, "rig_ref": 30, "fuse": "imu"})
    r = runner.invoke(eval_app, ["run", "--config", config, "--splits", SPLITS])
    assert r.exit_code == 0, r.output
    folder = Path(r.output.splitlines()[0])
    results = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    sha, version = hashlib.sha256(fixed.read_bytes()).hexdigest(), importlib.metadata.version("gtsam")
    pose = results["runs"][0]["rungs"]["run"]["pose"]
    assert results["schema_version"] == 10 and pose["fuse"] == "imu" and pose["start"] == "nominal_from_first_frame"
    assert pose["imu"] == {"source": "memory", "layout": "rig", "stream": "noiseless", "name": None, "file": None,
                           "digest": None, "profile": None}
    assert pose["fusion"] == {"gtsam": version, "fuse_fixed_sha256": sha, "estimator": "lab", "standstill_s": None}
    record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    assert record["fusion"] == {"gtsam": version, "cpus": os.cpu_count(), "fuse_fixed_sha256": [sha]}
    assert "imu_files" not in record
    report = (folder / "report.md").read_text(encoding="utf-8")
    assert f"> 결합: gtsam {version} · 고정값 {sha[:16]} · 추정기 lab" in report and "결합 imu(모의)" in report
    fixed.write_text(yaml.safe_dump({**FUSE_FIXED, "start_speed_sigma_mps": 0.3}), encoding="utf-8")
    again = runner.invoke(eval_app, ["run", "--config", config, "--splits", SPLITS])
    assert again.exit_code == 0 and Path(again.output.splitlines()[0]) != folder   # the fixed values join the id


def test_file_fusion_results_name_the_imu_file(small_root, tmp_path, fixed):
    run = SynthRun.open(small_root)
    pose_dir, imu_dir = _files(run, tmp_path)
    digest = ImuFile(imu_file_path(imu_dir, run.meta.run_id)).digest
    pose = {"rig": True, "rig_ref": 30, "source": "file", "cache": "vo", "caches": {"vo": {"dir": str(pose_dir)}},
            "fuse": "imu", "imu": "i", "imu_files": {"i": {"dir": str(imu_dir), "digests": {run.meta.run_id: digest}}},
            "anchor": "first_truth", "vio": {"vo_sigma_trans_mm": 0.01, "vo_sigma_rot_deg": 0.0001}}
    r = runner.invoke(eval_app, ["run", "--config", _fused_config(tmp_path / "a", small_root, pose), "--splits", SPLITS])
    assert r.exit_code == 0, r.output
    folder = Path(r.output.splitlines()[0])
    entry = json.loads((folder / "results.json").read_text(encoding="utf-8"))["runs"][0]["rungs"]["run"]["pose"]
    name = f"{run.meta.run_id}.npz"
    assert entry["imu"] == {"source": "file", "layout": "rig", "stream": "noise_seed_0", "name": "i", "file": name,
                            "digest": digest, "profile": "I_low"}
    record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    assert record["imu_files"] == [{"run_id": run.meta.run_id, "name": "i", "file": name, "digest": digest,
                                    "profile": "I_low"}]
    assert record["data"][0]["imu_file_digests"] == [digest]
    assert f"> IMU 파일: i · 프로필 I_low · digest {digest[:16]}" in (folder / "report.md").read_text(encoding="utf-8")
    pose["imu_files"]["i"]["digests"][run.meta.run_id] = "0" * 64
    r = runner.invoke(eval_app, ["run", "--config", _fused_config(tmp_path / "b", small_root, pose), "--splits", SPLITS])
    assert r.exit_code == 2 and "differs from the pinned 0000000000000000" in r.output, r.output


def test_fusion_ignores_shaken_frame_poses_and_trajectory(shake_root, tmp_path, fixed):
    """The same pose cache, IMU file and planned poses give the same fused cameras when the frame poses (but the
    first ones) and both recorded trajectories move; the runner then stops at the IMU file of the changed bundle."""
    shake = shake_root
    run = SynthRun.open(shake)
    pose_dir, imu_dir = _files(run, tmp_path / "files")
    cfg = _file_cfg(shake, pose_dir, imu_dir, "pose.fuse=imu", *TIGHT_SIGMA)
    pose_file, imu_file = _opened(run, cfg)
    want = _cams(run, cfg, None, pose_file=pose_file, imu_file=imu_file)
    root = _copy(shake, tmp_path)
    firsts = _first_frames(shake)
    for p in sorted(root.glob("*_frame.json")):
        if int(p.name[:6]) not in firsts:
            _edit(p, lambda f: f["pose"].update(
                matrix_world=_moved(f["pose"]["matrix_world"], dx=0.002, dy=0.01, yaw_deg=0.3, tilt_deg=0.5)))
    digests = {}
    for side in "AB":
        npz = root / f"trajectory_{side}.npz"
        _edit_npz(npz, lambda a: a["p"].__iadd__(np.outer(a["t"], (0.0, 0.01, 0.0))))   # 1 cm/s sideways
        digests[npz.name] = hashlib.sha256(npz.read_bytes()).hexdigest()
    _edit(root / "trajectory.json", lambda m: m["digests"].update(digests))
    moved = SynthRun.open(root)
    assert moved.meta.shake and moved.meta.run_id != run.meta.run_id
    moved_cfg = _file_cfg(root, pose_dir, imu_dir, "pose.fuse=imu", *TIGHT_SIGMA)
    got = _cams(moved, moved_cfg, None, pose_file=pose_file, imu_file=imu_file)
    for key, cams in want.items():
        for idx, cam in cams.items():
            assert pose_matrix(cam).tobytes() == pose_matrix(got[key][idx]).tobytes(), (key, idx)
    ctx = prepare(moved, fake_cfg(root, "apple"), load_splits())
    imu_file_path(imu_dir, moved.meta.run_id).write_bytes(imu_file.path.read_bytes())
    with pytest.raises(ValueError, match="IMU file was made for bundle fingerprint"):
        open_imu_file(ctx, moved_cfg)
