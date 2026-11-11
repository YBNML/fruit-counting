"""IMU files: the on-disk format of measured IMU streams, the prediction-side reader and its checks, the
interval rule that integrates them, the command that writes one per bundle, and the IMU check of eval truth."""

import hashlib
import json
import re

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from counting.cli import app
from counting.config.loader import load_yaml_model
from counting.config.pose_schema import ImuConfig
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.cli import eval_app
from counting.evaluation.imusim import AssumedMotion, ImuProfile, ImuSampling, imu_session, session_drives
from counting.odometry.imufile import (
    IMU_ARRAYS, IMU_FILE_FORMAT, ImuFile, ImuStream, imu_file_digest, imu_file_path, load_imu_file, sample_spans,
    stream_key, write_imu_file,
)
from tests.unit.synthfake import ROOT, fake_cfg, write_run
from tests.unit.test_traj_adapter import _copy, _edit, _edit_npz

pytestmark = pytest.mark.pose
runner = CliRunner()
NOISE = {"gyro_n": 1.16e-3, "acc_n": 4.97e-3, "gyro_w": 1.82e-5, "acc_w": 4.75e-4, "gyro_b0_dps": 1.0, "acc_b0_mg": 4.0,
         "temp_k": 1.0, "vib_acc_ms2": 0.0, "vib_gyro_dps": 0.0}
META = {"run_id": "apple_s5_00000000", "fingerprint": "f" * 64, "imu_seed": 0, "rate_hz": 200.0, "profile": "I_test",
        "noise": NOISE}


def _streams() -> dict:
    def stream(n, value):
        t = 7.0 + np.arange(n) * 0.005
        return ImuStream(t=t, gyro=np.full((n, 3), value), acc=np.full((n, 3), value + 9.8))
    return {stream_key("rig", 0, "A", 3): stream(5, 0.1), stream_key("rig", 0, "B", 2): stream(4, 0.2),
            stream_key("pass", 1, "A", 3, 10.0): stream(3, 0.3)}


def test_imu_file_round_trip_keeps_dtypes_order_and_meta(tmp_path):
    path = imu_file_path(tmp_path, META["run_id"])
    assert path == tmp_path / "apple_s5_00000000.npz"
    digest = write_imu_file(path, _streams(), META)
    data = load_imu_file(path)
    assert data.digest == digest == imu_file_digest(path) and data.meta == {**META, "format": IMU_FILE_FORMAT}
    assert list(data.streams) == list(_streams()) == [(0, 0, 0, 3, -1), (0, 0, 1, 2, -1), (1, 1, 0, 3, 10000)]
    for key, want in _streams().items():
        for name in ("t", "gyro", "acc"):
            got = getattr(data.streams[key], name)
            assert got.dtype == np.float64 and np.array_equal(got, getattr(want, name)), (key, name)
    with np.load(path) as z:
        assert set(z.files) == {*IMU_ARRAYS, "meta"} and z["stream_of"].tolist() == [0] * 5 + [1] * 4 + [2] * 3
        np.savez(tmp_path / "plain.npz", **{k: z[k] for k in z.files})
    assert imu_file_digest(tmp_path / "plain.npz") == digest          # the content, not the file's bytes
    moved = _streams()
    moved[(0, 0, 1, 2, -1)].acc[1, 2] += 1e-9
    assert write_imu_file(tmp_path / "moved.npz", moved, META) != digest


BAD = [   # (file name, edit of the streams, reason)
    ("order", lambda s: s[(0, 0, 0, 3, -1)].t.__setitem__(2, 7.0), "t must strictly increase within a stream"),
    ("finite", lambda s: s[(0, 0, 1, 2, -1)].gyro.__setitem__((1, 0), np.nan), "values must be finite"),
    ("empty", lambda s: s.__setitem__((0, 1, 0, 3, -1), ImuStream(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)))),
     "a stream has no samples"),
    ("rig_pitch", lambda s: s.__setitem__((0, 1, 0, 3, 10000), s[(0, 0, 0, 3, -1)]),
     "a rig stream has no pitch and a pass stream has one"),
]


def test_imu_file_writer_refuses_overwrite_identity_and_bad_streams(tmp_path):
    path = tmp_path / "a.npz"
    write_imu_file(path, _streams(), META)
    with pytest.raises(FileExistsError, match="IMU file exists; remove it first to make it again"):
        write_imu_file(path, _streams(), META)
    for key in ("run_id", "fingerprint", "imu_seed", "noise", "rate_hz"):
        with pytest.raises(ValueError, match="IMU file meta needs run_id, fingerprint, imu_seed, noise and rate_hz"):
            write_imu_file(tmp_path / "m.npz", _streams(), {k: v for k, v in META.items() if k != key})
    for name, edit, reason in BAD:
        streams = _streams()
        edit(streams)
        with pytest.raises(ValueError, match=re.escape(f"{name}.npz: bad IMU file: {reason}")):
            write_imu_file(tmp_path / f"{name}.npz", streams, META)
        assert not (tmp_path / f"{name}.npz").exists() and not list(tmp_path.glob("*.tmp"))
    (tmp_path / "text.npz").write_text("not a zip", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape("text.npz: bad IMU file")):
        load_imu_file(tmp_path / "text.npz")


def test_imu_file_reader_checks_bundle_seed_and_profile(tmp_path):
    path = tmp_path / "a.npz"
    digest = write_imu_file(path, _streams(), META)
    f = ImuFile(path, fingerprint="f" * 64, run_id=META["run_id"], digest=digest, imu_seed=0)
    assert f.noise_seeds == [0, 1] and f.digest == digest and f.meta["profile"] == "I_test"
    np.testing.assert_array_equal(f.stream("pass", 1, "A", 3, 10.0).gyro, _streams()[(1, 1, 0, 3, 10000)].gyro)
    checks = [({"fingerprint": "e" * 64}, "IMU file was made for bundle fingerprint ffffffffffff, this bundle has eeeeeeeeeeee"),
              ({"run_id": "apple_s5_11111111"}, "IMU file is for run apple_s5_00000000, this bundle is apple_s5_11111111"),
              ({"digest": "0" * 64}, f"IMU file digest {digest[:16]} differs from the pinned 0000000000000000"),
              ({"imu_seed": 3}, "IMU file was made with imu_seed 0; the config seed is 3")]
    for kwargs, message in checks:
        with pytest.raises(ValueError, match=re.escape(f"{path}: {message}")):
            ImuFile(path, **kwargs)
    with pytest.raises(ValueError, match=re.escape(f"{path}: no IMU stream for noise seed 2; the file has [0, 1]")):
        f.stream("rig", 2, "A", 3)
    with pytest.raises(ValueError, match=re.escape(f"{path}: no IMU stream for the pass drive of side B, aisle 2, pitch 10")):
        f.stream("pass", 1, "B", 2, 10.0)
    zero = tmp_path / "zero.npz"
    write_imu_file(zero, _streams(), {**META, "noise": {**NOISE, "acc_b0_mg": 0.0}})
    with pytest.raises(ValueError, match=re.escape("IMU file profile I_test has acc_b0_mg = 0; a noiseless IMU is the truth")):
        ImuFile(zero)


def test_sample_spans_hold_each_sample_over_its_interval():
    t = 7.0 + np.arange(5) * 0.005
    spans = sample_spans(t, 7.0031, 7.0299)          # a partial first interval, the last sample held past its own
    assert [i for i, _ in spans] == [0, 1, 2, 3, 4]
    np.testing.assert_allclose([h for _, h in spans], [0.0019, 0.005, 0.005, 0.005, 0.0099], rtol=0, atol=1e-12)
    assert [i for i, _ in sample_spans(t, 7.0, 7.005)] == [0]
    for ta, tb in ((6.99, 7.01), (7.0, 7.0302)):
        with pytest.raises(ValueError, match=re.escape(f"IMU samples cover 7.000 to 7.020 s, not {ta:.3f} to {tb:.3f} s")):
            sample_spans(t, ta, tb)
    with pytest.raises(ValueError, match="an IMU window must end after it starts"):
        sample_spans(t, 7.01, 7.01)


SPLITS = str(ROOT / "configs" / "eval" / "splits_pose.yaml")
STAGE_SPLITS = str(ROOT / "configs" / "eval" / "splits.yaml")


@pytest.fixture(scope="module")
def shake_root(tmp_path_factory):
    return write_run(tmp_path_factory.mktemp("shake"), crop="apple", size=(64, 36), shake=True)


def _imu_config(where, **over) -> str:
    keys = ("gyro_n", "acc_n", "gyro_w", "acc_w", "gyro_b0_dps", "acc_b0_mg", "temp_k")
    raw = {"profile": "I_test", "noise": {k: NOISE[k] for k in keys}, "noise_seeds": [0, 1],
           "output_dir": str(where / "imu"), **over}
    path = where / "imu.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module")
def made(apple_run, shake_root, tmp_path_factory):
    """One `counting pose imu` run over a straight and a shake bundle: (its folder, config, arguments, result)."""
    where = tmp_path_factory.mktemp("made")
    config = _imu_config(where)
    args = ["pose", "imu", "--config", config, "--splits", SPLITS, "--run", str(apple_run), "--run", str(shake_root)]
    return where, config, args, runner.invoke(app, args)


def test_pose_imu_writes_one_file_per_bundle(apple_run, shake_root, made):
    where, _, args, r = made
    assert r.exit_code == 0, r.output
    (folder,) = (where / "imu").iterdir()
    for root, source in ((apple_run, "straight_assumed"), (shake_root, "recorded")):
        run = SynthRun.open(root)
        path = imu_file_path(folder, run.meta.run_id)
        f = ImuFile(path, fingerprint=run.meta.fingerprint, run_id=run.meta.run_id, imu_seed=0)
        assert f.noise_seeds == [0, 1] and f.meta["traj_source"] == source and f.meta["profile"] == "I_test"
        assert f.meta["motion"] == {"speed_mps": 0.5, "stop_s": 5.0, "accel_s": 2.0}
        assert len(load_imu_file(path).streams) == (2 + 4) * 2
        side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert side["digest"] == f.digest and len(side["streams"]) == 12
        assert f"{run.meta.run_id}: profile I_test, drives rig 2 pass 4, noise seeds 0,1, samples " in r.output
    again = runner.invoke(app, args)
    assert again.exit_code == 1 and "IMU file exists; remove it first to make it again" in again.output
    other = runner.invoke(app, [*args, "--set", f"output_dir={where / 'again'}"])
    assert other.exit_code == 0, other.output
    (twin,) = (where / "again").iterdir()
    assert twin.name == folder.name                                   # output_dir is left out of the config hash
    for p in sorted(folder.glob("*.npz")):
        assert imu_file_digest(twin / p.name) == imu_file_digest(p)


def test_pose_imu_refuses_what_it_does_not_write(heldout_run, tmp_path):
    base = ["pose", "imu", "--config", _imu_config(tmp_path), "--splits", SPLITS]
    held = write_run(tmp_path / "s1003", crop="apple", pitches=(30.0,), seed=1003, size=(64, 36))
    part = write_run(tmp_path / "part", crop="apple", pitches=(30.0,), max_frames=4, size=(64, 36))
    cases = [(["--run", str(held)], 1, "is a test seed; this command makes no IMU file of a test bundle"),
             (["--run", str(heldout_run)], 1, "seed 1000 is in no split"),
             (["--run", str(part)], 2, "not a sequence bundle"),
             (["--run", str(held), "--set", "noise.gyro_n=0"], 2, "gyro_n"),
             (["--run", str(held), "--set", "nope=1"], 2, "error:")]
    for extra, code, text in cases:
        r = runner.invoke(app, [*base, *extra])
        assert r.exit_code == code and text in r.output, (extra, r.output)
    assert not (tmp_path / "imu").exists()


def test_memory_imu_matches_the_file(shake_root, made):
    where, config, _, r = made
    assert r.exit_code == 0, r.output
    run = SynthRun.open(shake_root)
    (folder,) = (where / "imu").iterdir()
    f = ImuFile(imu_file_path(folder, run.meta.run_id))
    keys = [f"noise.imu_{k}={v}" for k, v in NOISE.items() if v]
    cfg = fake_cfg(shake_root, "apple", "pose.rig_ref=10", "pose.fuse=imu", *keys)
    profile = ImuProfile.from_noise(cfg.noise)
    assert cfg.seed == 0 and profile == ImuProfile.from_config(load_yaml_model(config, ImuConfig).noise)
    for layout in ("rig", "pass"):
        drives = session_drives(run, layout, AssumedMotion())
        for seed in (0, 1):
            memory = imu_session(drives, profile, ImuSampling(), imu_seed=cfg.seed, noise_seed=seed, scene=run.meta.seed)
            for d, m in zip(drives, memory):
                s = f.stream(layout, seed, d.side, d.aisle, d.pitch_deg)
                assert all(getattr(m, n).tobytes() == getattr(s, n).tobytes() for n in ("t", "gyro", "acc")), (layout, seed)
    drives = session_drives(run, "rig", AssumedMotion())
    other = imu_session(drives, profile, ImuSampling(), imu_seed=1, noise_seed=0, scene=run.meta.seed)
    assert not np.array_equal(other[0].gyro, f.stream("rig", 0, "A", drives[0].aisle).gyro)


def test_eval_truth_checks_the_imu_of_shake_bundles(shake_root, apple_run, tmp_path):
    r = runner.invoke(eval_app, ["truth", str(shake_root), "--splits", STAGE_SPLITS])
    assert r.exit_code == 0 and re.search(r"^imu OK \(drives=2, max_end_m=", r.output, re.M), r.output
    assert "imu " not in runner.invoke(eval_app, ["truth", str(apple_run), "--splits", STAGE_SPLITS]).output
    # bundles of a few frames: frames of scattered positions, and sides whose frames all share one time
    for k in (10, 3):
        part = write_run(tmp_path / f"part{k}", crop="apple", size=(64, 36), shake=True, max_frames=k)
        r = runner.invoke(eval_app, ["truth", str(part), "--splits", STAGE_SPLITS])
        assert r.exit_code == 0 and re.search(r"^imu OK \(drives=2, ", r.output, re.M), (k, r.output)
    root = _copy(shake_root, tmp_path)
    times = sorted({plan.t for plan in SynthRun.open(root).plans().values()})
    ta, tb = times[len(times) // 2], times[len(times) // 2 + 1]
    npz = root / "trajectory_B.npz"

    def lift(a):    # side B 5 cm up between two frames; every sample within 1 ms of a frame time is kept
        a["p"][(a["t"] > ta + 1e-3) & (a["t"] < tb - 1e-3), 2] += 0.05

    _edit_npz(npz, lift)
    _edit(root / "trajectory.json", lambda m: m["digests"].update({npz.name: hashlib.sha256(npz.read_bytes()).hexdigest()}))
    r = runner.invoke(eval_app, ["truth", str(root), "--splits", STAGE_SPLITS])
    assert r.exit_code == 1, r.output                  # the frames still match the trajectory; only the IMU fails
    assert all(re.search(rf"^{line} \(", r.output, re.M) for line in ("rig OK", "trajectory OK", "imu FAIL")), r.output
