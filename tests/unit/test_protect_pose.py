"""Held-out pose files: the test log row of a pose cache or IMU file, `counting pose vo` and `counting pose imu`
on a test bundle (a frozen, committed config without --set; one test log row per file), the evaluation-side
checks of a held-out request with a pose section (pinned file digests, pinned files and their log rows, shaken
bundles and their trajectories, and what its rungs and grid cells may change), and the protected `eval ladder`
and `eval sweep` on the test bundle that run once those checks pass."""

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest
import yaml
from typer.testing import CliRunner

import counting.evaluation.fusion as fusion_module
import counting.evaluation.protect as protect
import counting.evaluation.record as record
from counting.cli import app
from counting.config.eval_schema import EvalAppConfig
from counting.config.hashing import config_hash
from counting.config.loader import load_yaml_model
from counting.config.pose_schema import ImuConfig
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.cli import eval_app
from counting.evaluation.posesource import check_pose_supported
from counting.evaluation.protect import (
    LOG_HEADER,
    append_log_row,
    check_test_log,
    check_test_pose_caches,
    check_test_pose_rungs,
    check_test_request,
    check_test_trajectories,
    earlier_log_runs,
    pose_log_row,
)
from counting.evaluation.record import git_state
from counting.evaluation.report import SCHEMA_VERSION
from counting.evaluation.runner import ProtectionError, _patched, load_ladder, rung_config
from counting.evaluation.sweep import load_grid, sweep_cells
from counting.odometry.cache import pose_cache_digest, pose_cache_path
from counting.odometry.imufile import imu_file_digest, imu_file_path
from tests.unit.posefake import TEST_VO, TEXTURE_RUN, paint_plane
from tests.unit.synthfake import ROOT, SIDECAR_COMMIT, fake_cfg, write_run
from tests.unit.test_det_cli import _commit_all

pytestmark = pytest.mark.pose
runner = CliRunner()
SPLITS_POSE = ROOT / "configs" / "eval" / "splits_pose.yaml"
POSE_C2 = ROOT / "configs" / "eval" / "apple_dev_pose_c2.yaml"
POSE = ROOT / "configs" / "eval" / "sweeps_pose"
LADDER = load_ladder(POSE / "ladder_pose.yaml")     # the committed pose ladder
# The drift grids that may run on the held-out scenes (78 cells); drift_val and drift_vi run on val only.
TEST_GRIDS = ("drift_rpe", "drift_inpass", "drift_interpass", "drift_interpass_c1x", "drift_imu", "drift_profiles")
# The alignment section of a frozen pose config: no alignment in the base config, and the four knobs it pins.
FROZEN_ALIGN = {"mode": "none", "segment_m": 3.0, "pair_radius_m": 0.5, "min_pairs": 20, "oracle_sigma_m": 0.02}
IMU_NOISE = {"gyro_n": 1.16e-3, "acc_n": 4.97e-3, "gyro_w": 1.82e-5, "acc_w": 4.75e-4, "gyro_b0_dps": 1.0,
             "acc_b0_mg": 4.0}
_ROW_ARGS = {"kind": "pose_vo", "date": "2026-11-13T00:00:00+00:00", "config_rel": "configs/pose/frozen/apple_vo.yaml",
             "config_hash": "h" * 16, "extra": "opencv:5.0.0", "fingerprint": "f" * 64, "digest": "d" * 64,
             "commit": "c" * 40}


@pytest.fixture(autouse=True)
def reopen_closed_seeds(monkeypatch):
    """Reopen the closed held-out seeds for these tests: their fake bundles keep seed 1003, and the rules they
    check apply to the open held-out seeds too. The closed list itself is checked in test_protect_depth.py."""
    monkeypatch.setattr(protect, "CLOSED_TEST_SEEDS", frozenset())


def _bundle(root: Path, seed: int) -> Path:
    """A shaken textured bundle (`TEXTURE_RUN`, images painted by `paint_plane`) with `seed` and the sidecar."""
    root = write_run(root, **TEXTURE_RUN, seed=seed, shake=True)
    paint_plane(root)
    (root / "orchard_synth.commit").write_text(SIDECAR_COMMIT + "\n", encoding="utf-8")
    return root


def _frozen_like() -> EvalAppConfig:
    """`configs/eval/apple_dev_pose_c2.yaml` as a held-out config: split=test, tuning=false, `FROZEN_ALIGN`, a
    detection cache folder and dummy pinned pose files (caches vo and vo_dd, IMU files imu_ext, imu_lab, imu_icm
    and imu_ext_x4), so every rung of the committed ladder builds on it."""
    def files(name):
        return {"dir": f"runs/pose/{name}", "digests": {"apple_s1003_00000000": "0" * 64}}

    cfg = load_yaml_model(POSE_C2, EvalAppConfig)
    imu_names = ("imu_ext", "imu_lab", "imu_icm", "imu_ext_x4")
    return _patched(cfg, {"split": "test", "tuning": False, "align": FROZEN_ALIGN,
                          "detector": {"cache": "runs/detect/0000000000000000"},
                          "pose": {"caches": {n: files(n) for n in ("vo", "vo_dd")},
                                   "imu_files": {n: files(n) for n in imu_names}}})


def _plain_test_cfg(run: Path) -> EvalAppConfig:
    """A held-out config without a pose section: the apple dev config with split=test and tuning=false on `run`."""
    return fake_cfg(run, "apple", "split=test", "tuning=false")


def _small_ladder(moved_dir: str) -> dict:
    """A small pose ladder built from the committed one: rungs L3 (`pose: {}`), L4_drift (the judged rung), LVI
    (with the odometry factor sigmas) and LVI_moved (LVI with the pose cache vo read from `moved_dir`), and
    baselines LVI_lab (LVI's shape, IMU file imu_ext, on val only) and B_drift (the structure rung of the drift
    grids)."""
    rungs = LADDER["rungs"]
    lvi = {**rungs["LVI"], "pose": {**rungs["LVI"]["pose"],
                                    "vio": {"vo_sigma_trans_mm": 2.1, "vo_sigma_rot_deg": 0.039}}}
    moved = {**lvi, "pose": {**lvi["pose"], "caches": {"vo": {"dir": moved_dir}}}}
    lab = {**lvi, "pose": {**lvi["pose"], "imu": "imu_ext"}}
    return {"rungs": {"L3": {"pose": {}}, "L4_drift": rungs["L4_drift"], "LVI": lvi, "LVI_moved": moved},
            "baselines": {"LVI_lab": lab, "B_drift": LADDER["baselines"]["B_drift"]},
            "val_only": ["LVI_lab"]}


def _cells(cfg: EvalAppConfig, grid_name: str, ladder) -> dict:
    """The cells of the committed grid `grid_name` on `cfg`, as `check_test_pose_rungs` takes them:
    "<point> @ <base>" -> (base, cell config)."""
    cells = sweep_cells(cfg, load_grid(POSE / f"{grid_name}.yaml"), ladder)
    return {f"{c.point.name} @ {c.base}": (c.base, c.cfg) for c in cells}


@pytest.fixture(scope="module")
def held(tmp_path_factory):
    """A stand-in counting repository, a test bundle (seed 1003) and a val bundle (seed 200), and the protected
    pose commands run on the test bundle: `pose imu` twice (the first file and its summary moved aside in
    between) and `pose vo` once, in that order (`results`).

    `repo` commits configs/pose/frozen/vo.yaml (`TEST_VO`, caches under `<tmp>/pose`), configs/pose/frozen/imu.yaml
    and configs/pose/loose_imu.yaml (profile I_test, one noise seed, files under `<tmp>/imu`) and a copy of the
    pose splits. `invoke` runs one CLI call with `record.counting_repo` pointing at `repo` for that call only and
    adds `--splits <repo splits>` unless the call gives one. `pose_commit` is the repo commit the pose files
    were made at; `rows(name)` reads `<tmp>/<name>/test_log.csv`; `listing()` is the file names of both output
    folders and the row count of both logs.

    After the pose commands the repo also commits configs/eval/frozen/apple_test_pose.yaml (`frozen_eval`, read as
    `cfg`): the c2 dev pose config on the test bundle at 10 deg, one noise seed, the pose cache vo and the IMU
    file imu_ext with their folders and content digests, and `FROZEN_ALIGN`. `cfg_with(...)` is `cfg` with
    `vo_digests`/`imu_digests` (the digests mapping), `vo_dir`/`imu_dir` (the folder) or `vo_digest` (the test
    bundle's pin) replaced. `copy_of_vo_with_loose_row(tmp)` copies the vo cache folder under `tmp` with its log
    rows' config set to a loose path, `copy_of_imu_with_other_fingerprint(tmp)` the imu_ext folder with its log
    rows' fingerprints set to another bundle's; each returns the copied folder. `splits` is the repo's splits.

    The same commit adds configs/eval/sweeps_pose/ladder_pose.yaml (`ladder`, `_small_ladder`) and the same
    ladder as ladder_moved.yaml (`ladder_moved`), grid_one.yaml (`grid_one`: base B_drift, the common block of
    the committed drift_rpe.yaml, one pose_trans_sigma_m point, so two cells), and copies of the committed
    drift_val.yaml (`drift_val`), configs/eval/criteria_pose.yaml and configs/pose/fuse_fixed.yaml.
    `patch(monkeypatch)` points `record.counting_repo` at `repo` and `fusion.FUSE_FIXED_PATH` at the repo copy
    for one test; `invoke_eval(args, fuse_fixed=None, swap=None)` then runs one eval call on `frozen_eval` with
    the repo's splits and pose criteria (`fuse_fixed`, when given, is `FUSE_FIXED_PATH` for that call only;
    `swap`, when given, maps each path to move aside for that call only to the path moved into its place, or
    None for nothing).
    `eval_rows()` reads `<tmp>/eval/test_log.csv`, `last_results()` is the results.json of its last row's
    experiment, and `eval_listing()` is the experiment folder names and that log's row count.
    """
    tmp = tmp_path_factory.mktemp("held")
    repo = tmp / "repo"
    frozen = repo / "configs" / "pose" / "frozen"
    frozen.mkdir(parents=True)
    (repo / "configs" / "eval").mkdir(parents=True)
    vo = {**TEST_VO.model_dump(mode="json"), "output_dir": str(tmp / "pose")}
    (frozen / "vo.yaml").write_text(yaml.safe_dump(vo), encoding="utf-8")
    imu = {"profile": "I_test", "noise": IMU_NOISE, "noise_seeds": [0], "output_dir": str(tmp / "imu")}
    for path in (frozen / "imu.yaml", repo / "configs" / "pose" / "loose_imu.yaml"):
        path.write_text(yaml.safe_dump(imu), encoding="utf-8")
    shutil.copy(SPLITS_POSE, repo / "configs" / "eval" / "splits_pose.yaml")
    _commit_all(repo)
    loose_splits = tmp / "splits_pose.yaml"         # the same splits, but not a file of the repository
    shutil.copy(SPLITS_POSE, loose_splits)
    test, val = _bundle(tmp / "test_s1003", 1003), _bundle(tmp / "val_s200", 200)

    def invoke(args):
        if "--splits" not in args:
            args = [*args, "--splits", str(repo / "configs" / "eval" / "splits_pose.yaml")]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(record, "counting_repo", lambda: repo)
            return runner.invoke(app, args)

    def rows(name):
        log = tmp / name / "test_log.csv"
        if not log.is_file():
            return []
        with log.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def listing():
        names = sorted(p.relative_to(tmp).as_posix() for d in ("pose", "imu") for p in (tmp / d).rglob("*")
                       if p.is_file())
        return names, len(rows("pose")), len(rows("imu"))

    h = SimpleNamespace(repo=repo, tmp=tmp, test=str(test), val=str(val), frozen_vo=str(frozen / "vo.yaml"),
                        frozen_imu=str(frozen / "imu.yaml"),
                        loose_imu=str(repo / "configs" / "pose" / "loose_imu.yaml"), loose_splits=str(loose_splits),
                        test_fp=SynthRun.open(test).meta.fingerprint, invoke=invoke, rows=rows, listing=listing)
    run_id = SynthRun.open(test).meta.run_id
    h.imu_file = imu_file_path(tmp / "imu" / config_hash(load_yaml_model(h.frozen_imu, ImuConfig)), run_id)
    h.vo_file = pose_cache_path(tmp / "pose" / config_hash(TEST_VO), run_id)
    h.pose_commit = git_state(repo)["commit"]
    imu_args = ["pose", "imu", "--config", h.frozen_imu, "--run", h.test]
    first = invoke(imu_args)
    for p in (h.imu_file, h.imu_file.with_suffix(".json")):
        if p.exists():
            p.rename(p.with_name(f"first_{p.name}"))
    h.results = [first, invoke(imu_args), invoke(["pose", "vo", "--config", h.frozen_vo, "--run", h.test])]

    h.splits = str(repo / "configs" / "eval" / "splits_pose.yaml")
    h.vo_digest, h.imu_digest = pose_cache_digest(h.vo_file), imu_file_digest(h.imu_file)
    frozen_eval = yaml.safe_load(POSE_C2.read_text(encoding="utf-8"))
    frozen_eval.update(run_name="apple_test_pose", split="test", tuning=False, runs=[h.test], align=FROZEN_ALIGN,
                       output_dir=str(tmp / "eval"))
    frozen_eval["passes"]["pitches"] = [10.0]
    frozen_eval["noise"]["seeds"] = [0]
    frozen_eval["pose"] = {"rig_ref": 10, "config_label": "c2",
                           "caches": {"vo": {"dir": str(h.vo_file.parent), "digests": {run_id: h.vo_digest}}},
                           "imu_files": {"imu_ext": {"dir": str(h.imu_file.parent),
                                                     "digests": {run_id: h.imu_digest}}}}
    path = repo / "configs" / "eval" / "frozen" / "apple_test_pose.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(frozen_eval), encoding="utf-8")
    sweeps = repo / "configs" / "eval" / "sweeps_pose"
    sweeps.mkdir()
    small = yaml.safe_dump(_small_ladder(str(tmp / "pose_moved")))
    for name in ("ladder_pose.yaml", "ladder_moved.yaml"):
        (sweeps / name).write_text(small, encoding="utf-8")
    rpe = yaml.safe_load((POSE / "drift_rpe.yaml").read_text(encoding="utf-8"))
    grid_one = {"bases": ["B_drift"], "common": rpe["common"], "factors": {"noise.pose_trans_sigma_m": [0.005]}}
    (sweeps / "grid_one.yaml").write_text(yaml.safe_dump(grid_one), encoding="utf-8")
    shutil.copy(POSE / "drift_val.yaml", sweeps / "drift_val.yaml")
    criteria = repo / "configs" / "eval" / "criteria_pose.yaml"
    shutil.copy(ROOT / "configs" / "eval" / "criteria_pose.yaml", criteria)
    fixed_copy = repo / "configs" / "pose" / "fuse_fixed.yaml"
    shutil.copy(ROOT / "configs" / "pose" / "fuse_fixed.yaml", fixed_copy)
    _commit_all(repo)
    h.frozen_eval, h.cfg = str(path), load_yaml_model(path, EvalAppConfig)
    h.ladder, h.ladder_moved = str(sweeps / "ladder_pose.yaml"), str(sweeps / "ladder_moved.yaml")
    h.grid_one, h.drift_val = str(sweeps / "grid_one.yaml"), str(sweeps / "drift_val.yaml")
    experiments = tmp / "eval" / "apple_test_pose"

    def patch(mp):
        mp.setattr(record, "counting_repo", lambda: repo)
        mp.setattr(fusion_module, "FUSE_FIXED_PATH", fixed_copy)

    def invoke_eval(args, fuse_fixed=None, swap=None):
        args = [*args, "--config", h.frozen_eval, "--splits", h.splits, "--criteria", str(criteria)]
        moved = []                          # (path, what took its place), put back after the call
        try:
            for path, stand_in in (swap or {}).items():
                path = Path(path)
                path.rename(f"{path}.aside")
                moved.append((path, stand_in))
                if stand_in is not None:
                    Path(stand_in).rename(path)
            with pytest.MonkeyPatch.context() as mp:
                if fuse_fixed is not None:
                    mp.setattr(fusion_module, "FUSE_FIXED_PATH", Path(fuse_fixed))
                return runner.invoke(eval_app, args)
        finally:
            for path, stand_in in reversed(moved):
                if stand_in is not None and path.exists():
                    path.rename(stand_in)
                Path(f"{path}.aside").rename(path)

    def eval_listing():
        names = sorted(p.name for p in experiments.iterdir()) if experiments.is_dir() else []
        return names, len(rows("eval"))

    def last_results():
        folder = experiments / rows("eval")[-1]["experiment_id"]
        return json.loads((folder / "results.json").read_text(encoding="utf-8"))

    h.patch, h.invoke_eval, h.eval_listing, h.last_results = patch, invoke_eval, eval_listing, last_results
    h.eval_rows = lambda: rows("eval")

    def cfg_with(**changes):
        data = h.cfg.model_dump()
        for key, value in changes.items():
            which, field = key.split("_", 1)
            files = data["pose"]["caches"]["vo"] if which == "vo" else data["pose"]["imu_files"]["imu_ext"]
            if field == "digest":
                files["digests"] = {run_id: value}
            else:
                files[field] = str(value) if field == "dir" else value
        return EvalAppConfig.model_validate(data)

    def copy_with_rows(src_dir, log_name, dst, change):
        folder = shutil.copytree(src_dir, dst / src_dir.name)
        for row in rows(log_name):
            append_log_row(dst / "test_log.csv", {**row, **change})
        return folder

    h.cfg_with = cfg_with
    h.copy_of_vo_with_loose_row = lambda base: copy_with_rows(h.vo_file.parent, "pose", base / "vo_loose",
                                                              {"config": "configs/pose/vo.yaml"})
    h.copy_of_imu_with_other_fingerprint = lambda base: copy_with_rows(h.imu_file.parent, "imu", base / "imu_other",
                                                                       {"fingerprints": "0" * 64})
    return h


def test_pose_log_rows_fit_the_test_log(tmp_path):
    row = pose_log_row(kind="pose_vo", date="2026-11-13T00:00:00+00:00", config_rel="configs/pose/frozen/apple_vo.yaml",
                       config_hash="h" * 16, extra="opencv:5.0.0", fingerprint="f" * 64, digest="d" * 64,
                       commit="c" * 40)
    assert set(row) == set(LOG_HEADER)
    assert (row["kind"], row["experiment_id"], row["fingerprints"], row["criteria_sha256"], row["verdict"]) == \
        ("pose_vo", "d" * 16, "f" * 64, "", "")
    with pytest.raises(ValueError, match="kind must be pose_vo or pose_imu"):
        pose_log_row(**{**_ROW_ARGS, "kind": "detect"})
    log = tmp_path / "test_log.csv"
    append_log_row(log, row)
    check_test_log(log)
    assert earlier_log_runs(log, config_hash="h" * 16, extra="opencv:5.0.0") == 1


def test_pose_commands_on_test_bundles_log_one_row_per_file(held):
    # imu: two runs, two rows, the second warned; vo: one row
    assert [r.exit_code for r in held.results] == [0, 0, 0]
    imu_rows, vo_rows = held.rows("imu"), held.rows("pose")
    assert [(x["kind"], x["config"], x["extra"]) for x in imu_rows] == \
        [("pose_imu", "configs/pose/frozen/imu.yaml", "imu:I_test")] * 2
    assert [(x["kind"], x["config"], x["extra"]) for x in vo_rows] == \
        [("pose_vo", "configs/pose/frozen/vo.yaml", f"opencv:{cv2.__version__}")]
    assert vo_rows[0]["experiment_id"] == pose_cache_digest(held.vo_file)[:16]
    assert imu_rows[1]["experiment_id"] == imu_file_digest(held.imu_file)[:16]
    assert {x["fingerprints"] for x in imu_rows + vo_rows} == {held.test_fp}
    assert {x["commit"] for x in imu_rows + vo_rows} == {held.pose_commit}
    assert "warning: test_log.csv already lists 1 test pose row(s) of config" in held.results[1].output
    assert "warning:" not in held.results[0].output


def test_pose_commands_refuse_loose_test_requests(held):
    cases = [
        (["pose", "imu", "--config", held.loose_imu, "--run", held.test], 1,
         f"seed 1003 is a test seed; counting pose imu makes an IMU file of a test bundle only from a config file "
         f"under configs/pose/frozen/"),
        (["pose", "imu", "--config", held.frozen_imu, "--run", held.test, "--set", "seed=1"], 1,
         "counting pose imu on a test bundle takes no --set"),
        (["pose", "vo", "--config", held.frozen_vo, "--run", held.test, "--run", held.val], 1,
         "split=test needs a test seed; seed 200 belongs to split val"),
        (["pose", "vo", "--config", held.frozen_vo, "--run", held.test, "--splits", held.loose_splits], 1,
         "counting pose vo on a test bundle needs --splits to be a file tracked in the counting repository"),
    ]
    before = held.listing()
    for args, code, text in cases:
        r = held.invoke(args)               # adds --splits <repo splits> unless the case gives one
        assert r.exit_code == code and text in r.output, (args, r.output)
    assert held.listing() == before         # no file and no log row was written


def test_test_pose_files_need_pins_contents_and_protected_rows(held, tmp_path):
    run = SynthRun.open(held.test)
    out = check_test_pose_caches([run], held.cfg)
    assert [(o["kind"], o["name"], o["run_id"]) for o in out] == \
        [("pose_vo", "vo", run.meta.run_id), ("pose_imu", "imu_ext", run.meta.run_id)]
    cases = [  # (how the config or the files change, the message)
        (held.cfg_with(vo_digests={}), f"pose.caches.vo.digests pins no file for run {run.meta.run_id}"),
        (held.cfg_with(vo_dir=tmp_path / "empty"), "split=test needs the pinned pose cache file"),
        (held.cfg_with(vo_digest="e" * 64), f"differs from the pinned {'e' * 16}"),
        (held.cfg_with(vo_dir=held.copy_of_vo_with_loose_row(tmp_path)),   # same file, row config not frozen
         "no counting pose vo row from a frozen config for file"),
        (held.cfg_with(imu_dir=held.copy_of_imu_with_other_fingerprint(tmp_path)),
         "no counting pose imu row from a frozen config for file"),
    ]
    for cfg, text in cases:
        with pytest.raises(ProtectionError, match=re.escape(text)):
            check_test_pose_caches([run], cfg)


def test_held_out_bundles_must_be_shaken_with_their_trajectories(held, tmp_path):
    (row,) = check_test_trajectories([SynthRun.open(held.test)])
    assert row["trajectory_sha256"] == hashlib.sha256((Path(held.test) / "trajectory.json").read_bytes()).hexdigest()
    straight = write_run(tmp_path / "straight", **TEXTURE_RUN, seed=1003)
    with pytest.raises(ProtectionError, match="split=test with a pose section needs a bundle rendered with body shake"):
        check_test_trajectories([SynthRun.open(straight)])
    copy = shutil.copytree(held.test, tmp_path / "copy")
    opened = SynthRun.open(copy)
    (copy / "trajectory_A.npz").write_bytes((copy / "trajectory_A.npz").read_bytes() + b"x")
    with pytest.raises(ProtectionError, match=re.escape("trajectory_A.npz: sha256 differs from trajectory.json")):
        check_test_trajectories([opened])


def test_held_out_pose_rungs_keep_the_frozen_settings(heldout_run):
    cfg = _frozen_like()     # apple_dev_pose_c2 with split=test, tuning=false, the four knobs and dummy pinned
                             # pose files: caches vo and vo_dd, IMU files imu_ext, imu_lab, imu_icm and imu_ext_x4
                             # (so val rungs build)
    ladder = LADDER          # the committed pose ladder
    names = [*ladder["rungs"], *ladder["baselines"]]
    held_out = [n for n in names if n not in ladder["val_only"]]
    assert len(held_out) == 31
    check_test_pose_rungs(cfg, {n: (n, rung_config(cfg, ladder, n)) for n in held_out}, ladder)
    for grid_name in TEST_GRIDS:                       # the six held-out grids, 78 cells
        cells = sweep_cells(cfg, load_grid(POSE / f"{grid_name}.yaml"), ladder)
        check_test_pose_rungs(cfg, {f"{c.point.name} @ {c.base}": (c.base, c.cfg) for c in cells}, ladder)
    lvi = rung_config(cfg, ladder, "LVI")
    refused = [
        ({"LVI_lab": ("LVI_lab", rung_config(cfg, ladder, "LVI_lab"))}, "LVI_lab runs on val only"),
        (_cells(cfg, "drift_vi", ladder), "baseline @ B_vi runs on val only"),
        (_cells(cfg, "drift_val", ladder),
         "noise.drift_time_offset_s=0.01 @ B_drift sets noise.drift_time_offset_s=0.01; that axis runs on val only"),
        ({"x": ("LVI", _patched(lvi, {"pose": {"caches": {"vo": {"dir": "runs/other"}}}}))},
         "x changes pose.caches or pose.imu_files"),
        ({"x": ("LVI", _patched(lvi, {"pose": {"rig_ref": 10.0}}))}, "x changes pose.rig_ref or pose.config_label"),
        ({"x": ("LVI", _patched(lvi, {"align": {"segment_m": 6.0}}))}, "x changes the alignment knob align.segment_m"),
        ({"x": ("LVI", _patched(lvi, {"tracker": {"predict": "pose_local", "predict_window": 1}}))},
         "x changes the tracker prediction"),
        ({"x": ("LVI", _patched(lvi, {"pose": {"vio": {"vo_sigma_trans_mm": 2.1, "vo_sigma_rot_deg": 0.039,
                                                       "stop_s": 5.0}}}))}, "x sets pose.vio.stop_s"),
        ({"a": ("LVI", _patched(lvi, {"pose": {"vio": {"vo_sigma_trans_mm": 2.1, "vo_sigma_rot_deg": 0.039}}})),
          "b": ("LV", _patched(rung_config(cfg, ladder, "LV"), {"pose": {"vio": {"vo_sigma_trans_mm": 6.3,
                                                                                  "vo_sigma_rot_deg": 0.117}}}))},
         "rungs a and b differ in pose.vio factor sigmas"),
    ]
    for configs, text in refused:
        with pytest.raises(ProtectionError, match=re.escape(f"split=test: {text}")):
            check_test_pose_rungs(cfg, configs, ladder)
    plain = _plain_test_cfg(heldout_run)
    check_test_pose_rungs(plain, {"L3": ("L3", plain)}, None)   # no pose section on either side: no check
    fused = rung_config(plain, ladder, "L3_pass_c1x")           # a pose ladder rung builds on a config without one
    assert fused.pose is not None and fused.pose.fuse == "imu"
    for configs, use in (({"L3": ("L3", plain), "L3_pass_c1x": ("L3_pass_c1x", fused)}, ladder),
                         ({"run": (None, _patched(plain, {"pose": {}}))}, None)):
        name = list(configs)[-1]
        with pytest.raises(ProtectionError, match=re.escape(f"split=test: {name} adds a pose section; a held-out "
                                                            "pose run needs a frozen config with a pose section")):
            check_test_pose_rungs(plain, configs, use)


def test_held_out_request_needs_pinned_pose_files(held):
    ok = dict(config_path=held.frozen_eval, sets=[], runs=[], files={"--splits": held.splits}, grid=None,
              ladder=None, repo=held.repo)
    check_test_request(held.cfg, **ok)
    for cfg, text in ((held.cfg_with(vo_digests={}), "split=test with pose.caches needs pose.caches.vo.digests"),
                      (held.cfg_with(imu_digests={}),
                       "split=test with pose.imu_files needs pose.imu_files.imu_ext.digests")):
        with pytest.raises(ProtectionError, match=re.escape(text)):
            check_test_request(cfg, **ok)


def test_held_out_pose_ladder_runs_after_the_pose_checks(held, monkeypatch):
    pytest.importorskip("gtsam")
    held.patch(monkeypatch)        # record.counting_repo -> repo, fusion.FUSE_FIXED_PATH -> repo copy
    r = held.invoke_eval(["ladder", "--ladder", held.ladder, "--rungs", "L3,LVI"])
    assert r.exit_code == 0, r.output
    res, row = held.last_results(), held.eval_rows()[-1]
    lvi = res["runs"][0]["rungs"]["LVI"]
    assert lvi["pose"]["cache"]["digest"] == held.vo_digest and lvi["pose"]["imu"]["digest"] == held.imu_digest
    assert (row["kind"], row["config"]) == ("ladder", "configs/eval/frozen/apple_test_pose.yaml")
    assert {held.vo_digest, held.imu_digest} <= set(row["fingerprints"].split(";"))
    assert res["split"] == "test" and res["schema_version"] == SCHEMA_VERSION
    s = held.invoke_eval(["sweep", "--ladder", held.ladder, "--grid", held.grid_one])   # fused cells on B_drift
    assert s.exit_code == 0, s.output
    assert held.eval_rows()[-1]["kind"] == "sweep" and held.last_results()["split"] == "test"


def test_held_out_pose_requests_refuse_before_scoring(held, monkeypatch, tmp_path):
    held.patch(monkeypatch)
    untracked = tmp_path / "fuse_fixed.yaml"
    untracked.write_bytes(Path(held.repo, "configs/pose/fuse_fixed.yaml").read_bytes())
    straight = write_run(tmp_path / "straight", **TEXTURE_RUN, seed=1003)    # a test seed rendered without shake
    (straight / "orchard_synth.commit").write_text(SIDECAR_COMMIT + "\n", encoding="utf-8")
    vo_log = held.vo_file.parent.parent / "test_log.csv"                     # the pose vo log with the cache's row
    cases = [
        (["ladder", "--ladder", held.ladder, "--rungs", "L3", "--set", "seed=1"], {}, "split=test takes no --set or --run"),
        (["ladder", "--ladder", held.ladder, "--rungs", "L3,LVI_lab"], {}, "split=test: LVI_lab runs on val only"),
        (["ladder", "--ladder", held.ladder, "--rungs", "LVI_moved"], {}, "split=test: LVI_moved changes pose.caches"),
        (["ladder", "--ladder", held.ladder_moved, "--rungs", "L3"], {},
         "split=test with a pose section needs --ladder configs/eval/sweeps_pose/ladder_pose.yaml"),
        (["sweep", "--ladder", held.ladder, "--grid", held.drift_val], {},
         "split=test: noise.drift_time_offset_s=0.01 @ B_drift sets noise.drift_time_offset_s=0.01; that axis runs on val only"),
        (["ladder", "--ladder", held.ladder, "--rungs", "L3"], {"fuse_fixed": untracked},
         "split=test needs fixed fusion settings to be a file tracked in the counting repository"),
        (["ladder", "--ladder", held.ladder, "--rungs", "L3"], {"swap": {held.test: straight}},   # the frozen run path
         "split=test with a pose section needs a bundle rendered with body shake"),
        (["ladder", "--ladder", held.ladder, "--rungs", "L3"], {"swap": {vo_log: None}},
         "no counting pose vo row from a frozen config for file"),
    ]
    before = held.eval_listing()
    for args, patch, text in cases:
        r = held.invoke_eval(args, **patch)
        assert r.exit_code == 1 and text in r.output, (args, r.output)
    assert held.eval_listing() == before
    cfg = fake_cfg(held.test, "apple", "pose.rig_ref=10", "pose.rig=true", "pose.fuse=imu", "split=test", "tuning=false")
    with pytest.raises(NotImplementedError, match="pose.fuse=imu with split=test is not supported"):
        check_pose_supported(cfg)
    if fusion_module.has_gtsam():
        check_pose_supported(cfg, protected=True)        # the held-out stops are lifted once the request is protected
