"""Bundles rendered with body shake: frame times, planned poses and the recorded body trajectory are read on
the evaluation side, join the bundle fingerprint, and are checked against the rig and the frame poses."""

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from counting.data.formats.orchard_synth import FrameInput, PassKey, SynthRun, _fingerprint_paths, run_fingerprint
from counting.evaluation.cli import eval_app
from counting.evaluation.posesource import model_cameras, pose_settings
from counting.evaluation.runner import evaluate, prepare
from counting.evaluation.truth import check_rig, check_trajectory
from counting.geometry.noise import STEP_START, STEP_TIME, pass_key_code, seeded_rigid, side_code, time_offset
from counting.geometry.rig import body_from_reference, interpolate_pose, pitch_extrinsic, pose_matrix
from tests.unit.synthfake import fake_cfg, load_splits, robot_matrix, write_run
from tests.unit.test_drift import D_EXT_VO, ZERO_DRIFT, _passes
from tests.unit.test_leak import ALLOWED
from tests.unit.test_rig import _close

pytestmark = pytest.mark.pose
cli = CliRunner()

TEN_A, TEN_B = PassKey(10.0, "A", 2), PassKey(10.0, "B", 1)            # pitch, side, aisle
THIRTY_A, THIRTY_B = PassKey(30.0, "A", 2), PassKey(30.0, "B", 1)
ROW_Y, DISTANCE, HEIGHT = 3.75, 1.75, 0.9      # the apple fake bundle's target row and robot geometry


@pytest.fixture(scope="module")
def shake_run(tmp_path_factory):
    return write_run(tmp_path_factory.mktemp("shake"), crop="apple", shake=True)


@pytest.fixture(scope="module")
def shake_ctx(shake_run):
    return prepare(SynthRun.open(shake_run), fake_cfg(shake_run, "apple"), load_splits())


def _copy(src: Path, where: Path) -> Path:
    return Path(shutil.copytree(src, where / "copy"))


def _edit(path: Path, change) -> None:
    """Load a JSON file, let `change` edit the dict in place, write it back like write_run."""
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def test_straight_bundles_keep_their_bytes(apple_run):
    run = SynthRun.open(apple_run)
    assert run.meta.run_id == "apple_s5_74edf9bb" and run.meta.shake is False and run.plans() is None
    assert run_fingerprint(apple_run) == run.meta.fingerprint


def test_shake_bundle_opens_as_a_sequence(shake_run):
    run = SynthRun.open(shake_run)
    assert run.meta.shake is True and run.meta.is_sequence and run.meta.base_step_m == pytest.approx(0.15)
    p = run.passes()
    assert list(p) == [TEN_A, TEN_B, THIRTY_A, THIRTY_B] and all(len(v) == 31 for v in p.values())
    assert [f.pos_k for f in p[TEN_A]] == list(range(31)) and [f.pos_k for f in p[THIRTY_B]] == list(range(30, -1, -1))
    plans = run.plans()
    gaps = [np.abs(pose_matrix(f.camera)[:3, 3] - plans[f.idx].plan.C).max() for f in run.frames]
    assert 1e-3 < max(gaps) < 0.06                 # the frame camera is the shaken one, the plan the straight one
    assert all(f.cam_x == pose_matrix(f.camera)[0, 3] for f in run.frames)


def test_plans_carry_time_position_and_planned_camera(shake_run):
    run = SynthRun.open(shake_run)
    plans = run.plans()
    assert set(plans) == {f.idx for f in run.frames}
    for f in run.frames:
        fr = json.loads((shake_run / f"{f.idx:06d}_frame.json").read_text(encoding="utf-8"))
        pl = plans[f.idx]
        assert (pl.idx, pl.t, pl.pos_k) == (f.idx, fr["t"], f.pos_k)
        assert np.array_equal(pl.plan.C, np.array(fr["plan"])[:3, 3]) and np.array_equal(pl.plan.K, f.camera.K)
    times: dict = {}
    for f in run.frames:
        times.setdefault((f.pass_key.side, f.pos_k), set()).add(plans[f.idx].t)
    assert all(len(ts) == 1 for ts in times.values())
    t_a = [plans[f.idx].t for f in run.passes()[TEN_A]]
    assert all(b > a for a, b in zip(t_a, t_a[1:]))


def test_shake_bundle_pitch_checks(shake_run, tmp_path):
    root = _copy(shake_run, tmp_path)
    frame = root / "000000_frame.json"                         # 10 deg, side A, x = -1

    def tilt(deg):
        return lambda f: f["pose"].update(matrix_world=robot_matrix(-1.0, ROW_Y, "A", deg, DISTANCE, HEIGHT))

    _edit(frame, tilt(12.0))
    SynthRun.open(root)                                        # 2 deg from the name: within the shake allowance
    _edit(frame, tilt(14.0))
    with pytest.raises(ValueError, match="more than 3 deg from matrix_world"):
        SynthRun.open(root)

    def bad_plan(f):
        f["pose"]["matrix_world"] = f["plan"]
        f["plan"] = robot_matrix(-1.0, ROW_Y, "A", 10.05, DISTANCE, HEIGHT)

    _edit(frame, bad_plan)
    with pytest.raises(ValueError, match="pitch in pose name does not match plan"):
        SynthRun.open(root)


def _drop_flag(root):
    _edit(root / "manifest.json", lambda m: m["args"].pop("shake"))


def _drop_t(root):
    _edit(root / "000005_frame.json", lambda f: f.pop("t"))


def _shift_pos_k(root):
    _edit(root / "000005_frame.json", lambda f: f.update(pos_k=f["pos_k"] + 1))


def _t_in_straight(root):
    _edit(root / "000000_frame.json", lambda f: f.update(t=1.0))


@pytest.mark.parametrize("change, shaken, match", [
    (_drop_flag, True, "without args.shake"),
    (_t_in_straight, False, "t, pos_k or plan in a bundle without args.shake"),
    (_drop_t, True, "missing field 't'"),
    (_shift_pos_k, True, "pos_k 6 does not match the planned position 5"),
])
def test_half_shake_bundles_are_refused(shake_run, tmp_path, change, shaken, match):
    root = _copy(shake_run, tmp_path) if shaken else write_run(tmp_path / "s", crop="apple", pitches=(30.0,), max_frames=4)
    change(root)
    with pytest.raises(ValueError, match=match):
        SynthRun.open(root)


def test_trajectory_files_join_the_fingerprint(shake_run, tmp_path):
    names = [p.name for p in _fingerprint_paths(shake_run)]
    assert names[-3:] == ["trajectory.json", "trajectory_A.npz", "trajectory_B.npz"] and len(names) == 2 + 124 + 3
    root = _copy(shake_run, tmp_path)
    before = SynthRun.open(root).meta
    meta = root / "trajectory.json"
    meta.write_text(meta.read_text(encoding="utf-8") + " ", encoding="utf-8")
    after = SynthRun.open(root).meta
    assert after.fingerprint != before.fingerprint and after.run_id != before.run_id


def _edit_npz(path: Path, change) -> None:
    with np.load(path) as z:
        arrays = {k: z[k].copy() for k in z.files}
    change(arrays)
    np.savez(path, **arrays)


@pytest.mark.parametrize("case, match", [
    ("edited", "trajectory_B.npz: sha256 does not match trajectory.json"),
    ("missing", "trajectory_B.npz: listed in trajectory.json but missing"),
    ("straight", "trajectory_A.npz: trajectory file in a bundle without args.shake"),
])
def test_trajectory_files_are_checked(shake_run, tmp_path, case, match):
    if case == "straight":
        root = write_run(tmp_path / "s", crop="apple", pitches=(30.0,), max_frames=4)
        shutil.copy(shake_run / "trajectory_A.npz", root)
    else:
        root = _copy(shake_run, tmp_path)
        npz = root / "trajectory_B.npz"
        if case == "edited":
            _edit_npz(npz, lambda a: a["p"].__setitem__((100, 0), a["p"][100, 0] + 0.001))
        else:
            npz.unlink()
    with pytest.raises(ValueError, match=match):
        SynthRun.open(root)


def test_trajectory_gives_the_recorded_body(shake_run, apple_run):
    run = SynthRun.open(shake_run)
    tr = run.trajectory("A")
    assert tr.side == "A" and tr.t.ndim == 1 and tr.p.shape == (len(tr.t), 3) and tr.q_wxyz.shape == (len(tr.t), 4)
    assert np.allclose(np.diff(tr.t), 1e-3, rtol=0, atol=1e-12)
    B = tr.body_poses()
    assert B.shape == (len(tr.t), 4, 4) and np.array_equal(B[:, :3, 3], tr.p)
    assert np.allclose(B[:, :3, :3] @ B[:, :3, :3].transpose(0, 2, 1), np.eye(3), atol=1e-12)
    assert run.trajectory("A") is tr
    with pytest.raises(ValueError, match="not a shake bundle"):
        SynthRun.open(apple_run).trajectory("A")


def test_trajectory_stays_on_the_evaluation_side():
    assert {f.name for f in dataclasses.fields(FrameInput)} == {
        "run_id", "idx", "pass_key", "pos_k", "camera", "cam_x", "rgb_path", "depth_path"}
    assert not {"Trajectory", "FramePlan", "SynthRun", "SceneTruth"} & ALLOWED


def _move_frame_x(root: Path, idx: int, dx: float) -> None:
    def move(f):
        f["pose"]["matrix_world"][0][3] += dx
    _edit(root / f"{idx:06d}_frame.json", move)


def test_rig_and_trajectory_checks_pass(shake_run):
    run = SynthRun.open(shake_run)
    rig, traj = check_rig(run), check_trajectory(run)
    assert rig["passed"] and (rig["positions"], rig["pairs"]) == (62, 62) and rig["max_dt_s"] == 0.0
    assert rig["max_pos_m"] < 1e-9 and rig["max_rot_deg"] < 1e-7
    assert traj["passed"] and traj["frames"] == 124 and traj["max_pos_m"] < 1e-6 and traj["max_rot_deg"] < 1e-4


def test_checks_catch_a_moved_camera_and_an_edited_trajectory(shake_run, tmp_path):
    moved = _copy(shake_run, tmp_path / "moved")
    _move_frame_x(moved, 62, 0.001)                            # 30 deg, side A, pos_k 0
    run = SynthRun.open(moved)
    rig, traj = check_rig(run), check_trajectory(run)
    assert not rig["passed"] and rig["max_pos_m"] == pytest.approx(0.001, abs=1e-6)
    assert not traj["passed"] and traj["max_pos_m"] == pytest.approx(0.001, abs=1e-6)
    edited = _copy(shake_run, tmp_path / "edited")
    npz = edited / "trajectory_A.npz"
    _edit_npz(npz, lambda a: a["p"].__iadd__(np.array([0.0, 0.002, 0.0])))
    _edit(edited / "trajectory.json", lambda m: m["digests"].update(
        {"trajectory_A.npz": hashlib.sha256(npz.read_bytes()).hexdigest()}))
    run = SynthRun.open(edited)
    assert run.meta.run_id != SynthRun.open(shake_run).meta.run_id
    assert check_rig(run)["passed"] and check_trajectory(run)["max_pos_m"] == pytest.approx(0.002, abs=1e-6)
    assert not check_trajectory(run)["passed"]


def test_checks_on_one_pitch_and_partial_bundles(tmp_path, apple_run):
    one = SynthRun.open(write_run(tmp_path / "one", crop="apple", pitches=(30.0,), shake=True))
    rig = check_rig(one)
    assert rig["passed"] and (rig["positions"], rig["pairs"]) == (62, 0) and check_trajectory(one)["passed"]
    part = SynthRun.open(write_run(tmp_path / "part", crop="apple", shake=True, max_frames=10))
    assert not part.meta.is_sequence and all(f.pos_k == -1 for f in part.frames)
    assert check_rig(part)["passed"] and check_trajectory(part)["passed"] and check_trajectory(part)["frames"] == 10
    with pytest.raises(ValueError, match="not a shake bundle"):
        check_rig(SynthRun.open(apple_run))


def test_eval_truth_prints_rig_and_trajectory_lines(shake_run, apple_run, tmp_path):
    ok = cli.invoke(eval_app, ["truth", str(shake_run), "--check-geometry"])
    assert ok.exit_code == 0, ok.output
    for text in ("summary OK", "geometry OK", "rig OK (positions=62, pairs=62,", "trajectory OK (frames=124,"):
        assert text in ok.output
    plain = cli.invoke(eval_app, ["truth", str(apple_run)])
    assert plain.exit_code == 0 and not any(line.startswith(("rig ", "trajectory ")) for line in plain.output.splitlines())
    bad = _copy(shake_run, tmp_path)
    _move_frame_x(bad, 62, 0.001)
    res = cli.invoke(eval_app, ["truth", str(bad)])
    assert res.exit_code == 1 and "rig FAIL (" in res.output and "trajectory FAIL (" in res.output


def test_truth_with_the_rig_matches_l3_on_a_shake_bundle(shake_run, shake_ctx):
    rig = evaluate(shake_ctx, fake_cfg(shake_run, "apple", "pose.rig=true", "pose.rig_ref=30"))
    plain = evaluate(shake_ctx, fake_cfg(shake_run, "apple"))
    for key in ("stages", "lift_depth", "y1", "y6"):
        assert _close(rig[key], plain[key], 1e-9), key


NO_START = tuple(k for k in ZERO_DRIFT if "pass_side" not in k and "pass_pitch" not in k)   # keys a first_truth config may hold


def _body(cam, pitch):
    return body_from_reference(pose_matrix(cam), pitch_extrinsic(pitch))


def test_nominal_start_comes_from_the_plan(shake_run):
    run = SynthRun.open(shake_run)
    plans = run.plans()
    cfg = fake_cfg(shake_run, "apple", "pose.rig_ref=30", *NO_START)
    passes = _passes(run, cfg)
    cams = model_cameras(run.frames, passes, cfg, 0, scene=5, base_step_m=0.15, plans=plans)
    for key, frames in passes.items():
        first, last, q = frames[0], frames[-1], key.pitch_deg
        start = _body(plans[first.idx].plan, q)
        assert np.abs(_body(cams[key][first.idx], q) - start).max() < 1e-12
        want = start @ np.linalg.inv(_body(first.camera, q)) @ _body(last.camera, q)
        assert np.abs(_body(cams[key][last.idx], q) - want).max() < 1e-9
        assert np.abs(pose_matrix(cams[key][first.idx]) - pose_matrix(first.camera)).max() > 1e-4
    truth_cfg = fake_cfg(shake_run, "apple", "pose.rig_ref=30", "pose.anchor=first_truth", *NO_START)
    same = model_cameras(run.frames, passes, truth_cfg, 0, scene=5, base_step_m=0.15, plans=plans)
    plain = model_cameras(run.frames, passes, cfg, 0, scene=5, base_step_m=0.15)
    for k, frames in passes.items():
        assert all(same[k][f.idx] is f.camera and plain[k][f.idx] is f.camera for f in frames)
    # With the rig and with a side start error: a side starts at the planned body of its first frame at the
    # reference pitch (30 deg under the rig, the pass's own pitch without), times the start error, and every
    # frame of the side rides the body chained from there.
    side_error = ("noise.pass_side_trans_m=0.05", "noise.pass_side_rot_deg=0.5")
    for sets in (("pose.rig=true",), side_error, ("pose.rig=true", *side_error)):
        cfg = fake_cfg(shake_run, "apple", "pose.rig_ref=30", *NO_START, *sets)
        cams = model_cameras(run.frames, passes, cfg, 0, scene=5, base_step_m=0.15, plans=plans)
        for key, frames in passes.items():
            q = key.pitch_deg
            r = 30.0 if cfg.pose.rig else q
            refs = passes[PassKey(r, key.side, key.aisle)]
            at = {f.pos_k: f for f in refs}
            error = np.eye(4)
            if side_error[0] in sets:
                error = seeded_rigid((cfg.seed, 0, STEP_START, 5, *side_code(key)), 0.05, 0.5)
            start = _body(plans[refs[0].idx].plan, r) @ error
            for f in frames:
                want = start @ np.linalg.inv(_body(refs[0].camera, r)) @ _body(at[f.pos_k].camera, r)
                assert np.abs(_body(cams[key][f.idx], q) - want).max() < 1e-9, (sets, key, f.pos_k)


def test_pose_settings_name_the_planned_start(shake_run, shake_ctx):
    cfg = fake_cfg(shake_run, "apple", "pose.rig_ref=30", *D_EXT_VO)
    assert pose_settings(cfg, 0, planned=True)["start"] == "nominal_from_plan"
    assert pose_settings(cfg, 0)["start"] == "nominal_from_first_frame"
    first = fake_cfg(shake_run, "apple", "pose.rig_ref=30", "pose.anchor=first_truth", *D_EXT_VO)
    assert pose_settings(first, 0, planned=True)["start"] == "first_frame_truth"
    res = evaluate(shake_ctx, cfg, noise_seed=0)
    assert res["pose"]["start"] == "nominal_from_plan"
    # The cameras of every pass start at the planned pose, away from the shaken first frame.
    assert min(p["start"]["pos_m"] for p in res["pose_quality"]["passes"]) > 1e-3


def test_time_offset_shifts_the_flow_on_shake_bundles(shake_run):
    run = SynthRun.open(shake_run)
    plans = run.plans()
    sets = ("pose.rig=true", "pose.rig_ref=30", "pose.anchor=first_truth")
    cfg = fake_cfg(shake_run, "apple", *sets, "noise.drift_time_offset_s=0.1")
    cams = model_cameras(run.frames, _passes(run, cfg), cfg, 0, scene=5, base_step_m=0.15, plans=plans)
    for side in "AB":
        refs = sorted((f for f in run.frames if f.pass_key.side == side and f.pass_key.pitch_deg == 30.0),
                      key=lambda f: f.idx)
        t = np.array([plans[f.idx].t for f in refs])
        body = np.stack([_body(f.camera, 30.0) for f in refs])
        dt = time_offset((0, 0, STEP_TIME, 5, *pass_key_code(refs[0].pass_key)), 0.1)
        for k, f in enumerate(refs):
            want = interpolate_pose(t, body, t[k] + dt) @ pitch_extrinsic(30.0)
            np.testing.assert_allclose(pose_matrix(cams[f.pass_key][f.idx]), want, rtol=0, atol=1e-9)
    zero = fake_cfg(shake_run, "apple", "pose.rig_ref=30", "pose.anchor=first_truth", "noise.drift_time_offset_s=0",
                    "noise.depth_rel_sigma=0.02")                       # without the rig a flat flow keeps the frames' cameras
    flat = model_cameras(run.frames, _passes(run, zero), zero, 0, scene=5, base_step_m=0.15, plans=plans)
    assert all(flat[key][f.idx] is f.camera for key, frames in _passes(run, zero).items() for f in frames)
