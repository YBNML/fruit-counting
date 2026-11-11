"""Pose caches: the file format, the prediction-side reader, the file pose source and its start, and the
truth check: a pose cache made from the true poses scores like the true poses."""

import json
import re
import shutil

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

import counting.geometry.noise as noise_module
from counting.data.formats.orchard_synth import PassKey, SynthRun
from counting.evaluation.cli import eval_app
from counting.evaluation.posesource import check_pose_supported, drift_active, model_cameras, pose_settings
from counting.evaluation.runner import _patched, evaluate
from counting.geometry.noise import STEP_PITCH, STEP_RIG, STEP_START, pass_key_code, seeded_rigid, side_code
from counting.geometry.posefile import PoseFile
from counting.geometry.rig import body_from_reference, pitch_extrinsic, pose_matrix, relative_extrinsic
from counting.odometry.cache import (
    POSE_ARRAYS, POSE_CACHE_FORMAT, PassOdometry, load_pose_cache, pose_cache_digest, pose_cache_path,
    write_pose_cache,
)
from tests.unit.posefake import TEXTURE_RUN, write_truth_pose_cache
from tests.unit.synthfake import EVAL_CONFIGS, fake_cfg, write_run
from tests.unit.test_drift import _passes
from tests.unit.test_rig import _close

pytestmark = pytest.mark.pose

META = {"run_id": "apple_s5_00000000", "fingerprint": "f" * 64}
SPLITS = str(EVAL_CONFIGS / "splits.yaml")
CODE_A, CODE_B = (10000, 0, 2), (10000, 1, 1)        # 10 deg, side A aisle 2 and side B aisle 1


def _turn(deg: float, dx: float) -> np.ndarray:
    """A pose turned by `deg` about the camera y axis and moved by `dx` along x."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T = np.eye(4)
    T[:3, :3] = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    T[0, 3] = dx
    return T


def _odometry() -> dict:
    """Two hand-made passes: side A frames 0-2 (a failed pair, two prior pairs), side B frames 3-4."""
    a = PassOdometry(idx=np.array([0, 1, 2], np.int32), rel_c2w=np.stack([np.eye(4), _turn(1, 0.15), _turn(2, 0.3)]),
                     inliers=np.array([0, 210, 190], np.int32), failed=np.array([0, 0, 1], np.uint8),
                     prior=np.array([0, 1, 1], np.uint8))
    b = PassOdometry(idx=np.array([3, 4], np.int32), rel_c2w=np.stack([np.eye(4), _turn(-1, -0.15)]),
                     inliers=np.array([0, 300], np.int32), failed=np.zeros(2, np.uint8), prior=np.zeros(2, np.uint8))
    return {CODE_A: a, CODE_B: b}


def test_pose_cache_round_trip_keeps_dtypes_order_and_meta(tmp_path):
    path = pose_cache_path(tmp_path, META["run_id"])
    assert path == tmp_path / "apple_s5_00000000.npz"
    digest = write_pose_cache(path, _odometry(), {**META, "note": "x"})
    cache = load_pose_cache(path)
    assert cache.path == path and cache.digest == digest == pose_cache_digest(path)
    assert cache.meta == {**META, "note": "x", "format": POSE_CACHE_FORMAT}
    assert list(cache.passes) == [CODE_A, CODE_B]
    for code, want in _odometry().items():
        for name in ("idx", "rel_c2w", "inliers", "failed", "prior"):
            got = getattr(cache.passes[code], name)
            assert got.dtype == getattr(want, name).dtype and np.array_equal(got, getattr(want, name)), (code, name)
    with np.load(path) as z:
        assert set(z.files) == {*POSE_ARRAYS, "meta"} and z["frame_idx"].tolist() == [0, 1, 2, 3, 4]
        assert z["pass_code"].tolist() == [list(CODE_A)] * 3 + [list(CODE_B)] * 2


def test_pose_cache_digest_follows_content_not_file_bytes(tmp_path):
    one = tmp_path / "one.npz"
    digest = write_pose_cache(one, _odometry(), META)
    with np.load(one) as z:
        arrays = {k: z[k] for k in z.files}
    two = tmp_path / "two.npz"
    np.savez(two, **arrays)                                   # the same content, uncompressed
    assert one.read_bytes() != two.read_bytes() and pose_cache_digest(two) == digest
    moved = _odometry()
    moved[CODE_A].rel_c2w[2, 0, 3] += 1e-9
    assert write_pose_cache(tmp_path / "three.npz", moved, META) != digest
    assert write_pose_cache(tmp_path / "four.npz", _odometry(), {**META, "note": "x"}) != digest


def test_write_pose_cache_refuses_overwrite_and_missing_identity(tmp_path):
    path = tmp_path / "c.npz"
    write_pose_cache(path, _odometry(), META)
    with pytest.raises(FileExistsError, match="pose cache exists"):
        write_pose_cache(path, _odometry(), META)
    for meta in ({"run_id": "r"}, {"fingerprint": "f"}):
        with pytest.raises(ValueError, match="pose cache meta needs run_id and fingerprint"):
            write_pose_cache(tmp_path / "d.npz", _odometry(), meta)
    assert not (tmp_path / "d.npz").exists() and not list(tmp_path.glob("*.tmp"))


def test_load_pose_cache_refuses_malformed_files(tmp_path):
    good = tmp_path / "good.npz"
    write_pose_cache(good, _odometry(), META)
    with np.load(good) as z:
        arrays = {k: z[k] for k in z.files}
    (tmp_path / "text.npz").write_text("not a zip", encoding="utf-8")
    with open(tmp_path / "one.npz", "wb") as fh:                         # one .npy array under an .npz name
        np.save(fh, np.zeros(3))
    np.savez(tmp_path / "less.npz", **{k: v for k, v in arrays.items() if k != "prior"})
    np.savez(tmp_path / "tag.npz", **{**arrays, "meta": np.array(json.dumps({**META, "format": "x/1"}))})
    np.savez(tmp_path / "pickled.npz", **{**arrays, "meta": np.array([{"a": 1}], dtype=object)})
    (tmp_path / "cut.npz").write_bytes(good.read_bytes()[:-40])
    for name in ("text.npz", "one.npz", "less.npz", "tag.npz", "pickled.npz", "cut.npz"):
        reads = (load_pose_cache,) if name == "tag.npz" else (load_pose_cache, pose_cache_digest)
        for read in reads:
            with pytest.raises(ValueError, match=re.escape(f"{name}: bad pose cache")):
                read(tmp_path / name)
    with pytest.raises(ValueError, match="expected arrays"):
        load_pose_cache(tmp_path / "less.npz")
    with pytest.raises(ValueError, match=re.escape("format is 'x/1', expected 'counting-poses/1'")):
        load_pose_cache(tmp_path / "tag.npz")


BAD = [   # (file name, edit of the saved arrays, reason)
    ("dtype", lambda a: a.__setitem__("inliers", a["inliers"].astype(np.int64)),
     "inliers has dtype int64, expected int32"),
    ("order", lambda a: a["frame_idx"].__setitem__(1, 0), "frame_idx must strictly increase"),
    ("finite", lambda a: a["rel_c2w"].__setitem__((2, 1, 3), np.nan), "poses must be finite"),
    ("last_row", lambda a: a["rel_c2w"].__setitem__((1, 3, 0), 0.5), "the last row of every pose must be 0, 0, 0, 1"),
    ("rotation", lambda a: a["rel_c2w"].__setitem__((1, 0, 0), 1.1), "rotations must be orthonormal"),
    ("flags", lambda a: a["failed"].__setitem__(1, 2), "failed and prior must be 0 or 1"),
    ("first_pose", lambda a: a["rel_c2w"].__setitem__((0, 0, 3), 0.1),
     "the first pose of each pass must be the identity"),
    ("first_pair", lambda a: a["inliers"].__setitem__(3, 5), "the first frame of each pass has no pair"),
]


def test_pose_cache_writer_and_reader_refuse_bad_arrays(tmp_path):
    good = tmp_path / "good.npz"
    write_pose_cache(good, _odometry(), META)
    for name, edit, reason in BAD:
        with np.load(good) as z:
            arrays = {k: z[k].copy() for k in z.files}
        edit(arrays)
        np.savez(tmp_path / f"{name}.npz", **arrays)
        with pytest.raises(ValueError, match=re.escape(f"{name}.npz: bad pose cache: {reason}")):
            load_pose_cache(tmp_path / f"{name}.npz")
    first, flag = _odometry(), _odometry()
    first[CODE_B].rel_c2w[0, 1, 3] = 0.01
    flag[CODE_A].prior[1] = 3
    for odometry, reason in ((first, BAD[6][2]), (flag, BAD[5][2])):
        with pytest.raises(ValueError, match=re.escape(reason)):
            write_pose_cache(tmp_path / "w.npz", odometry, META)
        assert not (tmp_path / "w.npz").exists()


def test_pose_file_refuses_foreign_or_unpinned_caches(apple_run, tmp_path):
    run = SynthRun.open(apple_run)
    path, digest = write_truth_pose_cache(run, tmp_path)
    assert path == pose_cache_path(tmp_path, run.meta.run_id)
    pf = PoseFile(path, fingerprint=run.meta.fingerprint, run_id=run.meta.run_id, digest=digest)
    assert pf.digest == digest and pf.meta["run_id"] == run.meta.run_id and pf.path == path
    with pytest.raises(ValueError, match="pose cache was made for bundle fingerprint"):
        PoseFile(path, fingerprint="0" * 64)
    with pytest.raises(ValueError, match="pose cache is for run"):
        PoseFile(path, run_id="apple_s5_ffffffff")
    pinned = f"pose cache digest {digest[:16]} differs from the pinned {'0' * 16}"
    with pytest.raises(ValueError, match=re.escape(pinned)):
        PoseFile(path, digest="0" * 64)


def test_pose_file_reads_once_and_chains_from_the_start(apple_run, tmp_path, monkeypatch):
    run = SynthRun.open(apple_run)
    path, _ = write_truth_pose_cache(run, tmp_path)
    loads, real = [], np.load
    monkeypatch.setattr(np, "load", lambda *a, **k: loads.append(a[0]) or real(*a, **k))
    pf = PoseFile(path)
    key, frames = next(iter(run.passes().items()))                  # 10 deg, side A
    code = pass_key_code(key)
    assert pf.anchor(code) == frames[0].idx and pf.idxs(code).tolist() == [f.idx for f in frames]
    start = _turn(3, 1.0)
    got = pf.poses(code, start, [frames[0].idx, frames[5].idx])
    first, fifth = pose_matrix(frames[0].camera), pose_matrix(frames[5].camera)
    assert np.array_equal(got[0], start) and np.abs(got[1] - start @ np.linalg.inv(first) @ fifth).max() < 1e-12
    assert len(loads) == 1
    with pytest.raises(ValueError, match=re.escape(f"{path.name}: no pose for frame 999 of pass")):
        pf.poses(code, start, [999])
    with pytest.raises(ValueError, match="no poses for pass"):
        pf.anchor((45000, 0, 2))


def _file_cfg(root, folder, *sets, name="truth"):
    return fake_cfg(root, "apple", "pose.source=file", f"pose.cache={name}", f"pose.caches.{name}.dir={folder}", *sets)


def _cams(run, cfg, pf, passes=None, ns=None, plans=None):
    return model_cameras(run.frames, passes or _passes(run, cfg), cfg, ns, scene=run.meta.seed,
                         base_step_m=run.meta.base_step_m, plans=plans, pose_file=pf)


def _gap(cam, T) -> float:
    return float(np.abs(pose_matrix(cam) - T).max())


def test_truth_pose_file_gives_the_truth_cameras(apple_run, tmp_path):
    run = SynthRun.open(apple_run)
    path, _ = write_truth_pose_cache(run, tmp_path)
    pf = PoseFile(path, fingerprint=run.meta.fingerprint)
    for sets in (("pose.rig=true", "pose.rig_ref=30"), ()):
        cfg = _file_cfg(apple_run, tmp_path, "pose.anchor=first_truth", *sets)
        assert not drift_active(cfg, 0)
        cams = _cams(run, cfg, pf)
        for key, frames in _passes(run, cfg).items():
            assert max(_gap(cams[key][f.idx], pose_matrix(f.camera)) for f in frames) < 1e-12, (sets, key)
    # a cache without the first frame of each pass starts each pass at its first cached frame
    thin, _ = write_truth_pose_cache(run, tmp_path / "thin", drop_first=True)
    cfg = _file_cfg(apple_run, tmp_path / "thin", "pose.anchor=first_truth")
    pf = PoseFile(thin)
    rest = {k: fs[1:] for k, fs in _passes(run, cfg).items()}
    cams = _cams(run, cfg, pf, rest)
    assert max(_gap(cams[k][f.idx], pose_matrix(f.camera)) for k, fs in rest.items() for f in fs) < 1e-12
    with pytest.raises(ValueError, match=re.escape(f"{thin.name}: no pose for frame 0 of pass")):
        _cams(run, cfg, pf)


def test_file_source_start_rig_and_pitch_errors(apple_run, tmp_path, monkeypatch):
    run = SynthRun.open(apple_run)
    path, _ = write_truth_pose_cache(run, tmp_path)
    pf = PoseFile(path)
    start = ("noise.pass_side_trans_m=0.05", "noise.pass_side_rot_deg=0.5")
    cfg = _file_cfg(apple_run, tmp_path, "pose.rig=true", "pose.rig_ref=30", *start,
                    "noise.rig_extr_trans_mm=2", "noise.rig_extr_rot_deg=0.1", "noise.depth_k=0.0036")
    steps, real = [], noise_module.noise_rng
    monkeypatch.setattr(noise_module, "noise_rng", lambda *key: steps.append(key[2]) or real(*key))
    cams = _cams(run, cfg, pf, ns=0)
    assert set(steps) == {STEP_START, STEP_RIG} and not drift_active(cfg, 0)      # no flow draws
    rig = seeded_rigid((0, 0, STEP_RIG, run.meta.seed, 10000), 0.002, 0.1)
    for key, frames in _passes(run, cfg).items():
        refs = sorted((f for f in run.frames if f.pass_key == PassKey(30.0, key.side, key.aisle)), key=lambda f: f.idx)
        at, anchor = {f.pos_k: f for f in refs}, pose_matrix(refs[0].camera)
        side = seeded_rigid((0, 0, STEP_START, run.meta.seed, *side_code(key)), 0.05, 0.5)
        start_cam = body_from_reference(anchor, pitch_extrinsic(30.0)) @ side @ pitch_extrinsic(30.0)
        turn = relative_extrinsic(30.0, key.pitch_deg) @ (rig if key.pitch_deg == 10.0 else np.eye(4))
        for f in frames:
            want = start_cam @ np.linalg.inv(anchor) @ pose_matrix(at[f.pos_k].camera) @ turn
            assert _gap(cams[key][f.idx], want) < 1e-9, (key, f.idx)
    # without the rig each pass is its own flow, and a pitch-to-pitch error moves it about its middle
    plain = _file_cfg(apple_run, tmp_path, *start)
    pitched = _file_cfg(apple_run, tmp_path, *start, "noise.pass_pitch_trans_m=0.02", "noise.pass_pitch_rot_deg=0.25")
    kept, moved = _cams(run, plain, pf, ns=0), _cams(run, pitched, pf, ns=0)
    for key, frames in _passes(run, plain).items():
        draw = (0, 0, STEP_PITCH, run.meta.seed, *side_code(key), round(key.pitch_deg * 1000))
        error = seeded_rigid(draw, 0.02, 0.25)

        def middle(c):
            return (c[frames[0].idx].C + c[frames[-1].idx].C) / 2

        assert np.abs(middle(moved[key]) - middle(kept[key]) - error[:3, 3]).max() < 1e-9, key


def test_file_source_stops_before_running(apple_run, tmp_path):
    base = ("pose.source=file", "pose.cache=vo", f"pose.caches.vo.dir={tmp_path}")
    check_pose_supported(fake_cfg(apple_run, "apple", *base))
    for sets, message in ((("pose.fuse=vo", "split=test", "tuning=false"),
                           "pose.fuse=vo with split=test is not supported"),
                          (("noise.drift_time_offset_s=0.01",),
                           "noise.drift_time_offset_s > 0 with pose.source=file is not supported"),
                          (("split=test", "tuning=false"), "pose.source=file with split=test is not supported")):
        with pytest.raises(NotImplementedError, match=re.escape(message)):
            check_pose_supported(fake_cfg(apple_run, "apple", *base, *sets))
    run, cfg = SynthRun.open(apple_run), fake_cfg(apple_run, "apple", *base)
    with pytest.raises(ValueError, match="pose.source=file needs an opened pose file"):
        model_cameras(run.frames, _passes(run, cfg), cfg, None, scene=5, base_step_m=0.15)


def test_file_source_starts_at_the_plan_on_shake_bundles(tmp_path):
    root = write_run(tmp_path / "shake", **TEXTURE_RUN, shake=True)
    run = SynthRun.open(root)
    plans = run.plans()
    path, _ = write_truth_pose_cache(run, tmp_path / "c")
    pf = PoseFile(path)
    nominal = _file_cfg(root, tmp_path / "c", "pose.rig=true", "pose.rig_ref=10")
    first = _file_cfg(root, tmp_path / "c", "pose.rig=true", "pose.rig_ref=10", "pose.anchor=first_truth")
    cams, truth = _cams(run, nominal, pf, plans=plans), _cams(run, first, pf, plans=plans)
    for key, frames in _passes(run, nominal).items():
        a, last = frames[0], frames[-1]
        plan = pose_matrix(plans[a.idx].plan)
        assert np.abs(plan - pose_matrix(a.camera)).max() > 1e-4                   # the plan is not the shaken pose
        assert _gap(cams[key][a.idx], plan) < 1e-12
        assert _gap(cams[key][last.idx], plan @ np.linalg.inv(pose_matrix(a.camera)) @ pose_matrix(last.camera)) < 1e-9
        assert max(_gap(truth[key][f.idx], pose_matrix(f.camera)) for f in frames) < 1e-12
    assert pose_settings(nominal, None, planned=True)["start"] == "nominal_from_plan"
    assert pose_settings(nominal, None)["start"] == "nominal_from_first_frame"
    settings = pose_settings(first, None, planned=True)
    assert (settings["start"], settings["drift"], settings["cache"]) == ("first_frame_truth", False, None)


def test_truth_pose_file_matches_l3(apple_run, ctx_of, tmp_path, monkeypatch):
    ctx = ctx_of["apple"]
    path, digest = write_truth_pose_cache(ctx.run, tmp_path)
    plain = evaluate(ctx, fake_cfg(apple_run, "apple"))
    loads, real = [], np.load
    monkeypatch.setattr(np, "load", lambda *a, **k: loads.append(a[0]) or real(*a, **k))
    for sets in (("pose.rig=true", "pose.rig_ref=30"), ()):
        res = evaluate(ctx, _file_cfg(apple_run, tmp_path, "pose.anchor=first_truth", *sets))
        for key in ("stages", "lift_depth", "y1", "y6"):
            assert _close(res[key], plain[key], 1e-9), (sets, key)
        pose = res["pose"]
        assert (pose["source"], pose["drift"], pose["start"]) == ("file", False, "first_frame_truth")
        assert pose["cache"] == {"name": "truth", "file": path.name, "digest": digest, "vo_config_hash": None,
                                 "opencv": None}
        assert res["pose_quality"]["summary"]["in_pass"]["pos_m"] <= 1e-9
    assert len(loads) == 1                                                   # one read for the context


def test_pose_cache_is_pinned_and_checked(apple_run, pear_run, ctx_of, tmp_path):
    ctx = ctx_of["apple"]
    run_id = ctx.run.meta.run_id
    cfg = _file_cfg(apple_run, tmp_path, "pose.anchor=first_truth")
    with pytest.raises(FileNotFoundError, match="no pose cache for this bundle"):
        evaluate(ctx, cfg)
    path, digest = write_truth_pose_cache(ctx.run, tmp_path)

    def pinned(digests):
        return _patched(cfg, {"pose": {"caches": {"truth": {"dir": str(tmp_path), "digests": digests}}}})

    with pytest.raises(ValueError, match=re.escape(f"pose.caches.truth.digests pins no cache for run {run_id}")):
        evaluate(ctx, pinned({"other_run": digest}))
    with pytest.raises(ValueError, match=re.escape(f"pose cache digest {digest[:16]} differs from the pinned")):
        evaluate(ctx, pinned({run_id: "0" * 64}))
    assert evaluate(ctx, pinned({run_id: digest}))["pose"]["cache"]["digest"] == digest
    other, _ = write_truth_pose_cache(SynthRun.open(pear_run), tmp_path / "pear")
    (tmp_path / "foreign").mkdir()
    shutil.copyfile(other, tmp_path / "foreign" / path.name)                # another bundle's cache under this run id
    with pytest.raises(ValueError, match="pose cache was made for bundle fingerprint"):
        evaluate(ctx, _file_cfg(apple_run, tmp_path / "foreign", "pose.anchor=first_truth"))


def test_eval_run_names_the_pose_cache(leaf_runs, tmp_path):
    root = leaf_runs[0]
    run = SynthRun.open(root)
    poses = tmp_path / "poses"
    path, digest = write_truth_pose_cache(run, poses)
    raw = yaml.safe_load((EVAL_CONFIGS / "apple_dev.yaml").read_text(encoding="utf-8"))
    raw.update(runs=[str(root)], output_dir=str(tmp_path / "out"),
               pose={"source": "file", "cache": "truth", "caches": {"truth": {"dir": str(poses)}},
                     "anchor": "first_truth"})
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    args = ["run", "--config", str(cfg), "--splits", SPLITS]
    r = CliRunner().invoke(eval_app, args)
    assert r.exit_code == 0, r.output
    (d,) = (tmp_path / "out").glob("*/*")
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    entry = {"name": "truth", "file": path.name, "digest": digest, "vo_config_hash": None, "opencv": None}
    assert res["schema_version"] == 9 and res["runs"][0]["rungs"]["run"]["pose"]["cache"] == entry
    assert rec["pose_caches"] == [{"run_id": run.meta.run_id, **entry}]
    assert rec["data"][0]["pose_cache_digests"] == [digest]
    lines = (d / "report.md").read_text(encoding="utf-8").splitlines()
    assert "자세 file(truth)" in lines[1]
    assert f"> 자세 캐시: truth · VO 설정 - · OpenCV - · digest {digest[:16]}" in lines
    path.unlink()
    _, again = write_truth_pose_cache(run, poses, meta={"note": "same poses, other meta"})
    assert again != digest
    r = CliRunner().invoke(eval_app, args)
    assert r.exit_code == 0, r.output
    assert len(list((tmp_path / "out").glob("*/*"))) == 2          # another cache digest, another experiment
