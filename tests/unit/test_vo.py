"""Frame-to-frame RGB-D visual odometry on textured fake bundles: the settings file, the recovered motion,
repeatable output and its cache, failed pairs, the `counting pose vo` command, and odometry that reads
neither frame poses nor planned poses nor the recorded trajectory."""

import dataclasses
import hashlib
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

import counting.odometry.vo as vo_module
from counting.cli import app
from counting.config.hashing import config_hash
from counting.config.loader import load_yaml_model
from counting.config.pose_schema import VoConfig
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.posesource import vo_inputs
from counting.geometry.noise import pass_key_code
from counting.geometry.rig import pose_matrix
from counting.odometry.cache import load_pose_cache, write_pose_cache
from counting.odometry.vo import run_vo, vo_to_cache
from tests.unit.posefake import TEST_VO, TEXTURE_RUN, textured_run
from tests.unit.synthfake import write_run
from tests.unit.test_traj_adapter import _copy, _edit, _edit_npz

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]
VO_YAML = str(ROOT / "configs" / "pose" / "vo.yaml")
SPLITS_POSE = str(ROOT / "configs" / "eval" / "splits_pose.yaml")
VO_DEFAULTS = {"n_features": 6000, "orb_scale": 1.2, "orb_levels": 8, "fast_threshold": 7, "clahe_clip": 3.0,
               "clahe_tile": 8, "match_ratio": 0.8, "ransac_px": 3.0, "final_px": 2.0, "ransac_iters": 300,
               "ransac_confidence": 0.999, "min_inliers": 30, "guided_radius_px": 12.0, "prior_radius_px": 60.0,
               "guided_max_hamming": 64, "guided_ratio": 0.9, "depth_max_m": 10.0, "depth_nb": 2, "depth_k": 0.0036,
               "noise_seed": 0, "seed": 0, "threads": 1, "step_m": None, "output_dir": "runs/pose"}


@pytest.fixture(scope="module")
def straight(tmp_path_factory):
    return textured_run(tmp_path_factory.mktemp("textured") / "straight")


@pytest.fixture(scope="module")
def straight_odo(straight):
    return run_vo(vo_inputs(SynthRun.open(straight).frames), TEST_VO)


def _blank(root: Path, *idxs: int) -> None:
    """Overwrite the RGB image of each frame in `idxs` with flat grey: no features, so the pairs into and
    out of that frame fail and take the motion prior."""
    for idx in idxs:
        Image.fromarray(np.full((240, 320, 3), 128, np.uint8)).save(root / f"{idx:06d}_rgb.png")


@pytest.fixture(scope="module")
def blank(straight, tmp_path_factory):
    """A copy of `straight` with frame 3 (side A) blank."""
    root = Path(shutil.copytree(straight, tmp_path_factory.mktemp("blank") / "blank"))
    _blank(root, 3)
    return root


@pytest.fixture(scope="module")
def blank_odo(blank):
    return run_vo(vo_inputs(SynthRun.open(blank).frames), TEST_VO)


def _same(a: dict, b: dict) -> None:
    """Two odometry results (or a cache's passes and a result) hold the same passes and arrays."""
    assert list(a) == list(b)
    for code in a:
        for name in ("idx", "rel_c2w", "inliers", "failed", "prior"):
            assert np.array_equal(getattr(a[code], name), getattr(b[code], name)), (code, name)


def test_vo_config_file_and_hash():
    cfg = load_yaml_model(VO_YAML, VoConfig)
    assert cfg.model_dump() == VO_DEFAULTS and cfg == VoConfig()
    h = config_hash(cfg)
    assert re.fullmatch(r"[0-9a-f]{16}", h)
    assert config_hash(cfg.model_copy(update={"output_dir": "elsewhere"})) == h
    assert config_hash(cfg.model_copy(update={"depth_k": 0.0})) != h
    for bad in ({"nfeatures": 10}, {"threads": 0}, {"match_ratio": 1.5}, {"step_m": 0}):
        with pytest.raises(ValidationError):
            VoConfig(**bad)


def test_vo_recovers_a_known_motion(straight, straight_odo):
    passes = SynthRun.open(straight).passes()
    assert list(straight_odo) == [pass_key_code(k) for k in passes]
    for key, frames in passes.items():
        odo = straight_odo[pass_key_code(key)]
        assert odo.idx.tolist() == [f.idx for f in frames] and len(frames) == 7
        assert odo.failed.tolist() == [0] * 7 and odo.prior.tolist() == [0] * 7
        assert odo.inliers[0] == 0 and odo.inliers[1:].min() >= 100
        want = np.linalg.inv(pose_matrix(frames[0].camera)) @ pose_matrix(frames[-1].camera)
        assert np.linalg.norm(want[:3, 3]) == pytest.approx(1.8, abs=1e-6)            # six steps of 0.3 m
        err = np.linalg.inv(want) @ odo.rel_c2w[-1]
        assert np.linalg.norm(err[:3, 3]) < 0.05
        assert np.degrees(np.arccos(np.clip((np.trace(err[:3, :3]) - 1) / 2, -1, 1))) < 1.0


def test_vo_repeats_and_writes_one_cache(straight, straight_odo, tmp_path, monkeypatch):
    run = SynthRun.open(straight)
    frames = vo_inputs(run.frames)
    noisy = VoConfig(**{**TEST_VO.model_dump(), "depth_k": 0.0036})
    meta = {"run_id": "r", "fingerprint": "f" * 64}
    summary = vo_to_cache(frames, noisy, tmp_path / "a.npz", meta=meta)
    again = run_vo(frames, noisy)
    assert write_pose_cache(tmp_path / "b.npz", again, meta) == summary["digest"]       # a second run, the same arrays
    code = pass_key_code(frames[0].pass_key)
    assert not np.array_equal(again[code].rel_c2w, straight_odo[code].rel_c2w)         # the depth noise moves the poses
    want = np.linalg.inv(pose_matrix(run.frames[0].camera)) @ pose_matrix(run.frames[6].camera)   # side A, end
    assert np.linalg.norm((np.linalg.inv(want) @ again[code].rel_c2w[-1])[:3, 3]) < 0.1
    assert summary == json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert summary["file"] == "a.npz" and summary["meta"] == meta and summary["n_frames"] == 14
    assert [(p["frames"], p["pairs"], p["failed"], p["prior"]) for p in summary["passes"]] == [(7, 6, 0, 0)] * 2
    assert summary["seconds"] > 0 and "seconds" not in load_pose_cache(tmp_path / "a.npz").meta
    monkeypatch.setattr(vo_module, "run_vo", lambda *a, **k: pytest.fail("the odometry ran"))
    with pytest.raises(FileExistsError, match="pose cache exists"):
        vo_to_cache(frames, noisy, tmp_path / "a.npz", meta=meta)
    (tmp_path / "c.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="pose cache exists"):
        vo_to_cache(frames, noisy, tmp_path / "c.npz", meta=meta)


def test_vo_marks_failed_and_prior_pairs(straight_odo, blank, blank_odo):
    odo = blank_odo
    side_a, side_b = (pass_key_code(k) for k in SynthRun.open(blank).passes())
    a = odo[side_a]
    assert a.failed.tolist() == [0, 0, 0, 1, 1, 0, 0] and a.prior.tolist() == [0, 0, 0, 1, 1, 0, 0]
    assert a.inliers[3] < TEST_VO.min_inliers and a.inliers[4] < TEST_VO.min_inliers
    assert np.isfinite(a.rel_c2w).all() and np.array_equal(a.rel_c2w[0], np.eye(4))
    assert np.array_equal(a.rel_c2w[:3], straight_odo[side_a].rel_c2w[:3])          # the pairs before it are untouched
    for name in ("rel_c2w", "inliers", "failed", "prior"):                            # the other pass is untouched
        assert np.array_equal(getattr(odo[side_b], name), getattr(straight_odo[side_b], name)), name


def test_pose_vo_writes_one_cache_per_bundle(straight, straight_odo, tmp_path, monkeypatch):
    out = tmp_path / "poses"
    args = ["pose", "vo", "--config", VO_YAML, "--run", str(straight), "--splits", SPLITS_POSE,
            "--set", "n_features=1000", "--set", "depth_k=0", "--set", f"output_dir={out}"]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    run = SynthRun.open(straight)
    path = out / config_hash(TEST_VO) / f"{run.meta.run_id}.npz"
    cache = load_pose_cache(path)
    _same(cache.passes, straight_odo)
    m = cache.meta
    assert (m["run_id"], m["fingerprint"], m["vo_config_hash"]) == (run.meta.run_id, run.meta.fingerprint,
                                                                    config_hash(TEST_VO))
    assert m["vo"] == {k: v for k, v in TEST_VO.model_dump(mode="json").items() if k != "output_dir"}
    assert m["opencv"] == cv2.__version__ and m["depth_sha256"] == run.depth_fingerprint()
    assert m["image_size"] == [320, 240] and re.fullmatch(r"[0-9a-f]{64}", m["rgb_sha256"]) and "seconds" not in m
    assert json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["digest"] == cache.digest
    assert str(path) in r.output
    assert f"{run.meta.run_id}: frames 14, pairs 12, failed 0, prior 0, digest {cache.digest[:16]}" in r.output
    monkeypatch.setattr(vo_module, "run_vo", lambda *a, **k: pytest.fail("the odometry ran"))
    again = CliRunner().invoke(app, args)
    assert again.exit_code == 1 and "pose cache exists" in again.output, again.output


def test_pose_vo_refuses_what_it_does_not_cache(straight, tmp_path, monkeypatch):
    monkeypatch.setattr(vo_module, "run_vo", lambda *a, **k: pytest.fail("the odometry ran"))
    out = tmp_path / "o"

    def call(*extra):
        return CliRunner().invoke(app, ["pose", "vo", "--config", VO_YAML, "--splits", SPLITS_POSE,
                                        "--set", f"output_dir={out}", *extra])

    for seed, message in ((1003, "seed 1003 is a test seed"), (999, "seed 999 is in no split")):
        r = call("--run", str(write_run(tmp_path / f"s{seed}", **TEXTURE_RUN, seed=seed)))
        assert r.exit_code == 1 and message in r.output, r.output
    r = call("--run", str(write_run(tmp_path / "part", **TEXTURE_RUN, max_frames=4)))
    assert r.exit_code == 2 and "not a sequence bundle" in r.output, r.output
    r = call("--run", str(straight), "--set", "nfeatures=10")
    assert r.exit_code == 2 and r.output.startswith("error:"), r.output
    assert not out.exists()
    shown = CliRunner().invoke(app, ["pose", "vo", "--help"])                  # `pose` is a group, not the command
    assert shown.exit_code == 0 and "--config" in shown.output


def _moved(matrix, *, dx=0.0, dy=0.0, dz=0.0, yaw_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
    """A Blender camera matrix moved by (dx, dy, dz) and turned by yaw about the world z axis, tilt about
    its own x axis and roll about its own view axis, all through its centre."""

    def turn(axis, deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        i, j = ((1, 2), (2, 0), (0, 1))[axis]
        R = np.eye(3)
        R[i, i] = R[j, j] = c
        R[i, j], R[j, i] = -s, s
        return R

    m = np.asarray(matrix, dtype=float).copy()
    m[:3, :3] = turn(2, yaw_deg) @ m[:3, :3] @ turn(0, tilt_deg) @ turn(2, roll_deg)
    m[:3, 3] += (dx, dy, dz)
    return m.tolist()


def _first_frames(root) -> set[int]:
    return {frames[0].idx for frames in SynthRun.open(root).passes().values()}


def test_vo_ignores_straight_bundle_poses(blank, blank_odo, tmp_path):
    """Frame poses moved sideways, up and about the view axis (x and pitch kept, so the bundle stays a
    sequence) change the bundle id, not the odometry; neither do renumbered positions with another step value.
    The blank frame's pairs take the motion prior, so the prior's path is held to the same outcome as the
    matched pairs."""
    assert [p.prior.tolist() for p in blank_odo.values()] == [[0, 0, 0, 1, 1, 0, 0], [0] * 7]
    root = Path(shutil.copytree(blank, tmp_path / "moved"))
    firsts = _first_frames(blank)
    for p in sorted(root.glob("*_frame.json")):
        if int(p.name[:6]) not in firsts:
            _edit(p, lambda f: f["pose"].update(matrix_world=_moved(f["pose"]["matrix_world"], dy=0.02, dz=-0.01,
                                                                    yaw_deg=0.5, roll_deg=-0.5)))
    run = SynthRun.open(root)
    assert run.meta.is_sequence and run.meta.run_id != SynthRun.open(blank).meta.run_id
    _same(run_vo(vo_inputs(run.frames), TEST_VO), blank_odo)
    renumbered = [dataclasses.replace(f, pos_k=f.pos_k + 3) for f in run.frames]
    _same(run_vo(vo_inputs(renumbered), TEST_VO.model_copy(update={"step_m": 0.6})), blank_odo)


def _move_shake_frame(f: dict) -> None:
    """A shaken frame pose moved along and tilted (within the pitch allowance) and its planned pose moved
    sideways, up and about the view axis (x and pitch kept, so its position and pass hold)."""
    f["pose"].update(matrix_world=_moved(f["pose"]["matrix_world"], dx=0.002, dy=0.01, yaw_deg=0.3, tilt_deg=0.5))
    f["plan"] = _moved(f["plan"], dy=0.01, dz=-0.01, yaw_deg=0.5, roll_deg=-0.5)


def test_vo_ignores_shake_bundle_poses_and_trajectory(tmp_path):
    """On a bundle rendered with body shake, with one blank frame per side so each pass takes the motion
    prior, moved frame poses, moved planned poses and both trajectory files given a sideways drift (with
    their sidecar digests) change the bundle id, not the odometry."""
    shake = textured_run(tmp_path / "shake", shake=True)
    _blank(shake, 3, 10)
    ref = run_vo(vo_inputs(SynthRun.open(shake).frames), TEST_VO)
    assert [p.prior.tolist() for p in ref.values()] == [[0, 0, 0, 1, 1, 0, 0]] * 2
    root = _copy(shake, tmp_path)
    firsts = _first_frames(shake)
    for p in sorted(root.glob("*_frame.json")):
        if int(p.name[:6]) not in firsts:
            _edit(p, _move_shake_frame)
    digests = {}
    for side in "AB":
        npz = root / f"trajectory_{side}.npz"
        _edit_npz(npz, lambda a: a["p"].__iadd__(np.outer(a["t"], (0.0, 0.01, 0.0))))   # 1 cm/s sideways
        digests[npz.name] = hashlib.sha256(npz.read_bytes()).hexdigest()
    _edit(root / "trajectory.json", lambda m: m["digests"].update(digests))
    run = SynthRun.open(root)
    assert run.meta.shake and run.meta.is_sequence and run.meta.run_id != SynthRun.open(shake).meta.run_id
    _same(run_vo(vo_inputs(run.frames), TEST_VO), ref)
