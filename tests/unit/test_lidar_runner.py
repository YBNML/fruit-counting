"""LiDAR box depth in evaluate: the camera-only rungs it leaves alone, the camera draws it shares with them, its one
extrinsic draw per scene, what it never reads, the LiDAR result entry, the per-detection depth rows and their file
names, the noisy point cache of a context and the LiDAR file a context opens. Then the truth-label pairing rung, the
error forms of the camera depth and the LiDAR files in the experiment id and the record."""

import copy
import csv
import functools
import hashlib
import io
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import yaml

from counting.config.eval_schema import EvalAppConfig
from counting.data.formats.orchard_synth import SynthRun, Trajectory
from counting.evaluation import lidarsource
from counting.evaluation.lidarsource import (
    DepthRows, depth_rows_csv, depth_rows_file_names, depth_rows_index_csv, lidar_entry, pop_depth_rows,
)
from counting.evaluation.posesource import model_cameras
from counting.evaluation.report import SCHEMA_VERSION, write_experiment
from counting.evaluation.runner import (
    RunContext, _lidar_pass, _merge, describe_run, evaluate, evaluate_seeds, open_lidar_file, prepare,
)
from counting.evaluation.seeds import combine_seeds
from counting.evaluation.sweep import Grid, grid_points, run_sweep, sweep_cells
from counting.multiview.lidardepth import box_lidar_depth, combine_depth, gate_depth, window
from tests.unit.lidarfake import SharedLidar, shared_lidar_bundle, sidecar_sha256, write_lidar_files
from tests.unit.synthfake import EVAL_CONFIGS, fake_cfg, load_splits, write_run

pytestmark = pytest.mark.pose

COMMIT = "1" * 40
LIDAR_KEYS = {"file", "unit", "method", "sidecar_sha256", "sim_commit", "config_sha16", "variant", "mount_m",
              "mount_rpy_deg", "settings", "detections", "coverage", "coverage_all", "fallback_rate", "points",
              "depth", "lidar_only"}
CAMERA_ONLY_KEYS = {"stages", "lift_depth", "y1", "y6", "pose", "pose_quality", "alignment"}
# Noise blocks of the rungs below: the body odometry flow, the side start, the rig calibration, the camera depth, the
# LiDAR points and the depth-unit extrinsic.
VO = {"drift_scale_sigma": 0.0045, "drift_rate_deg_per_m": 0.045, "drift_step_trans_mm": 2.1,
      "drift_step_rot_deg": 0.039}
START = {"pass_side_trans_m": 0.05, "pass_side_rot_deg": 0.5}
RIG = {"rig_extr_trans_mm": 14.0, "rig_extr_rot_deg": 0.4}
DEPTH = {"depth_rel_sigma": 0.02, "depth_dropout": 0.05}
LIDAR_NOISE = {"lidar_range_sigma_m": 0.02, "lidar_range_sigma_near_m": 0.03, "lidar_near_range_m": 1.0,
               "lidar_ang_sigma_deg": 0.15, "lidar_dropout": 0.05, "lidar_range_bias_m": 0.0}
LIDAR_EXTR = {"lidar_extr_trans_mm": 10.0, "lidar_extr_rot_deg": 0.3}
# Rungs on the base config: the drift model on the rig without fusion (camera only, then with the depth-unit LiDAR),
# the true poses with the LiDAR noise and extrinsic alone, the true poses without noise.
DRIFT = {"pose": {"rig": True, "rig_ref": 10}, "noise": {**VO, **START, **RIG, **DEPTH, "seeds": [0, 1]}}
DRIFT_LD = _merge(DRIFT, {"lift": {"depth": "lidar"}, "noise": {**LIDAR_NOISE, **LIDAR_EXTR}})
TRUE_LD = {"pose": {}, "lift": {"depth": "lidar"}, "noise": {**LIDAR_NOISE, **LIDAR_EXTR, "seeds": [0, 1]}}
TRUE = {"pose": {}}
# The truth-label pairing on the true poses, its LiDAR every point taken as it is, and the LiDAR noise and extrinsic
# keys all at 0.
ORACLE = {"pose": {}, "lift": {"depth": "lidar_oracle", "lidar": {"combine": "replace", "min_points": 1}}}
LIDAR_ZERO = {"noise": {**{key: 0.0 for key in LIDAR_NOISE}, "lidar_near_range_m": 1.0,
                        **{key: 0.0 for key in LIDAR_EXTR}}}
FRUIT_PARTS = (3, 4)   # the semantic classes of a fruit and of a bag; 1 is a leaf
FORMS_ZERO = {"noise": {"depth_persist_sigma": 0.0, "depth_persist_len_m": 0.5, "depth_scale_sigma": 0.0,
                        "depth_bias_rel": 0.0}}
PERSIST = {"depth_persist_sigma": 0.02, "depth_persist_len_m": 0.5, "seeds": [0, 1]}


def _plain(obj):
    return obj.tolist() if hasattr(obj, "tolist") else repr(obj)


def _exact(res: dict, keys) -> str:
    """The values of `keys` in `res` as JSON text with every float written in full."""
    return json.dumps({k: res[k] for k in keys}, sort_keys=True, default=_plain)


def _without_rows(res: dict) -> str:
    return _exact(res, [k for k in res if k != "depth_rows"])


@dataclass
class LidarCase:
    """The shared fake shake bundle prepared under a base config that holds both LiDAR files (depth chosen) and the
    LiDAR section with `lift.depth: box_far`; `plain` is that base without the LiDAR keys. `evaluate` keeps each
    result by config and noise seed and hands out copies."""

    shared: SharedLidar
    run: SynthRun
    ctx: RunContext
    raw: dict
    plain: dict
    memo: dict = field(default_factory=dict)

    def config(self, *patches: dict, base: dict | None = None) -> EvalAppConfig:
        out = copy.deepcopy(self.raw if base is None else base)
        for patch in patches:
            out = _merge(out, patch)
        return EvalAppConfig.model_validate(out)

    def evaluate(self, cfg: EvalAppConfig, noise_seed: int | None = None) -> dict:
        key = (json.dumps(cfg.model_dump(mode="json"), sort_keys=True), noise_seed)
        if key not in self.memo:
            self.memo[key] = evaluate(self.ctx, cfg, noise_seed=noise_seed)
        return copy.deepcopy(self.memo[key])

    def combined(self, cfg: EvalAppConfig) -> dict:
        return combine_seeds(cfg, [(s, self.evaluate(cfg, s)) for s in cfg.noise.seeds])


def lidar_files(shared: SharedLidar, run_id: str) -> dict:
    """`lift.lidar_files` of the shared bundle: its depth unit and its navigation unit, each pinned for `run_id`."""
    return {"depth": {"dir": str(shared.depth.parent), "unit": "depth", "sim_config_sha256": "ab" * 32,
                      "sim_commits": [COMMIT], "digests": {run_id: sidecar_sha256(shared.depth)}},
            "nav": {"dir": str(shared.nav.parent), "unit": "nav", "sim_config_sha256": "cd" * 32,
                    "sim_commits": [COMMIT], "digests": {run_id: sidecar_sha256(shared.nav)}}}


@functools.lru_cache(maxsize=None)
def lidar_case() -> LidarCase:
    shared = shared_lidar_bundle()
    run = SynthRun.open(shared.bundle)
    plain = yaml.safe_load((EVAL_CONFIGS / "apple_dev.yaml").read_text(encoding="utf-8"))
    plain["runs"] = [str(shared.bundle)]
    plain["lift"] = {**plain["lift"], "depth": "box_far"}
    raw = _merge(plain, {"lift": {"lidar": {}, "lidar_files": lidar_files(shared, run.meta.run_id),
                                  "lidar_file": "depth"}})
    ctx = prepare(run, EvalAppConfig.model_validate(raw), load_splits())
    return LidarCase(shared=shared, run=run, ctx=ctx, raw=raw, plain=plain)


@functools.lru_cache(maxsize=None)
def low_pitch_case() -> LidarCase:
    """`lidar_case` on the passes of the bundle's lowest pitch alone. The fake LiDAR points are the pixels of those
    frames, so there a fruit's points lie square to the camera, as its truth depth is taken; seen from a steeper
    camera the same flat patch spans about 2 cm of depth."""
    case = lidar_case()
    pitch = min(key.pitch_deg for key in case.ctx.passes)
    raw, plain = (_merge(c, {"passes": {"pitches": [pitch]}}) for c in (case.raw, case.plain))
    ctx = prepare(case.run, EvalAppConfig.model_validate(raw), load_splits())
    return LidarCase(shared=case.shared, run=case.run, ctx=ctx, raw=raw, plain=plain)


@pytest.fixture(scope="module")
def case() -> LidarCase:
    return lidar_case()


def test_lidar_off_is_camera_only(case):
    with_lidar, without = case.config(DRIFT), case.config(DRIFT, base=case.plain)
    assert with_lidar.lift.lidar_files is not None and without.lift.lidar_files is None
    for seed in (0, 1):
        a, b = case.evaluate(with_lidar, seed), case.evaluate(without, seed)
        assert _exact(a, ("y1", "y6", "pose_quality")) == _exact(b, ("y1", "y6", "pose_quality"))
        assert "camera_depth_sha256" in a and "depth_rows" in a and "lidar" not in a
        assert "camera_depth_sha256" not in b and "depth_rows" not in b
        assert set(b) == CAMERA_ONLY_KEYS


def test_skipping_every_box_is_camera_only(case):
    camera = case.evaluate(case.config(DRIFT), 0)
    for combine in ("replace", "ivw"):
        cfg = case.config(DRIFT_LD, {"lift": {"lidar": {"min_points": 10**6, "combine": combine}}})
        res = case.evaluate(cfg, 0)
        assert _exact(res, ("y1", "y6")) == _exact(camera, ("y1", "y6")), combine
        assert res["lidar"]["coverage"] == 0.0 and res["lidar"]["fallback_rate"] == 1.0
        rows = res["depth_rows"]
        assert rows.n_points.max() > 0 and not rows.used_lidar.any()     # points were found, and none was used


def test_camera_draws_are_shared_with_the_camera_only_rung(case):
    ld_cfg, camera_cfg = case.config(DRIFT_LD), case.config(DRIFT)
    for seed in (0, 1):
        ld, camera = case.evaluate(ld_cfg, seed), case.evaluate(camera_cfg, seed)
        sha = camera["camera_depth_sha256"]
        assert re.fullmatch(r"[0-9a-f]{64}", sha) and ld["camera_depth_sha256"] == sha, seed
        assert sha == hashlib.sha256(camera["depth_rows"].z_cam.tobytes()).hexdigest()
        assert _exact(ld, ("pose_quality",)) == _exact(camera, ("pose_quality",))
    sha0 = case.evaluate(ld_cfg, 0)["camera_depth_sha256"]
    assert case.evaluate(case.config(DRIFT_LD, {"noise": {"lidar_range_sigma_m": 0.05}}), 0)[
        "camera_depth_sha256"] == sha0
    assert case.evaluate(case.config(DRIFT_LD, {"noise": {"depth_rel_sigma": 0.03}}), 0)[
        "camera_depth_sha256"] != sha0
    true_ld = case.combined(case.config(TRUE_LD))
    assert [p["camera_depth_sha256"] for p in true_ld["per_seed"]] == ["none", "none"]
    assert "camera_depth_sha256" not in true_ld
    assert case.evaluate(case.config(TRUE))["camera_depth_sha256"] == "none"


def test_lidar_depth_draws_its_extrinsic_once_per_scene(case, monkeypatch):
    draws, pitches = [], []
    real_error, real_transform = lidarsource.lidar_extr_error, lidarsource.camera_from_sensor

    def error(cfg, noise_seed, *, scene):
        out = real_error(cfg, noise_seed, scene=scene)
        draws.append((cfg.seed, noise_seed, scene, out.tobytes()))
        return out

    def transform(cfg, noise_seed, **kw):
        pitches.append(kw["pitch"])
        return real_transform(cfg, noise_seed, **kw)

    monkeypatch.setattr(lidarsource, "lidar_extr_error", error)
    monkeypatch.setattr(lidarsource, "camera_from_sensor", transform)
    bundle_pitches = {key.pitch_deg for key in case.ctx.passes}
    seen = {}
    for name, rung in (("true", TRUE_LD), ("drift", DRIFT_LD)):
        draws.clear()
        pitches.clear()
        evaluate(case.ctx, case.config(rung), noise_seed=0)
        assert len(draws) == len(case.ctx.passes) and set(pitches) == bundle_pitches, name
        assert len({d[3] for d in draws}) == 1 and {d[:3] for d in draws} == {(0, 0, case.run.meta.seed)}, name
        seen[name] = draws[0][3]
    assert seen["true"] == seen["drift"]
    monkeypatch.undo()
    base = case.evaluate(case.config(DRIFT_LD), 0)
    moved = case.evaluate(case.config(DRIFT_LD, {"noise": {"lidar_extr_trans_mm": 60.0, "lidar_extr_rot_deg": 1.0}}), 0)
    assert _exact(moved, ("pose_quality",)) == _exact(base, ("pose_quality",))
    assert _exact(moved["lidar"], ("depth",)) != _exact(base["lidar"], ("depth",))


def _pass_depths(case: LidarCase, cfg: EvalAppConfig, cams) -> list[bytes]:
    """The LiDAR box depth z_L of every frame, from `_lidar_pass` and `box_lidar_depth` on the cameras `cams`."""
    ctx, plans = case.ctx, case.run.plans()
    file = open_lidar_file(ctx, cfg)
    camera_lift = cfg.lift.model_copy(update={"depth": "box_far"})
    ctx._lidar_points.clear()
    out = []
    for key, frames in ctx.passes.items():
        lp = _lidar_pass(ctx, cfg, file, 0, scene=case.run.meta.seed, key=key, frames=frames, cams=cams)
        half = round(lp.settings.window_s * 1e9)
        for f in frames:
            z = box_lidar_depth(ctx.dets[f.idx].boxes, ctx.box_depth(f.idx, camera_lift), cams[key][f.idx],
                                window(lp.cloud, round(plans[f.idx].t * 1e9), half), lp.settings).z
            out.append(z.tobytes())
    return out


def test_trajectory_labels_and_noise_keys_do_not_reach_the_lidar_depth(case, monkeypatch, tmp_path):
    cfg, run, ctx = case.config(DRIFT_LD), case.run, case.ctx
    cams = model_cameras(run.frames, ctx.passes, cfg, 0, scene=run.meta.seed, base_step_m=run.meta.base_step_m,
                         plans=run.plans())
    before = _pass_depths(case, cfg, cams)
    assert np.isfinite(np.concatenate([np.frombuffer(b) for b in before])).sum() > 50
    shaken = {}
    for side in ("A", "B"):
        tr = run.trajectory(side)
        shaken[side] = Trajectory(side=side, t=tr.t, p=tr.p + [0.05, -0.03, 0.02], q_wxyz=tr.q_wxyz[:, [0, 2, 1, 3]])
    monkeypatch.setattr(SynthRun, "trajectory", lambda self, side: shaken[side])
    assert _pass_depths(case, cfg, cams) == before
    monkeypatch.undo()

    copied = tmp_path / "lidar" / ("ab" * 32)[:16] / case.shared.bundle.name   # the label files turned to rubbish
    shutil.copytree(case.shared.depth, copied)
    for side in ("A", "B"):
        (copied / f"lidar_{side}_labels.npz").write_bytes(b"not a label file")
    relabelled = case.config(DRIFT_LD, {"lift": {"lidar_files": {"depth": {"dir": str(copied.parent)}}}})
    a, b = case.evaluate(cfg, 0), evaluate(ctx, relabelled, noise_seed=0)
    assert _without_rows(a) == _without_rows(b)
    assert depth_rows_csv(a["depth_rows"]) == depth_rows_csv(b["depth_rows"])

    other = case.config(DRIFT_LD, {"noise": {"depth_rel_sigma": 0.08, "lidar_range_sigma_m": 0.05,
                                             "lidar_ang_sigma_deg": 0.5, "lidar_dropout": 0.3,
                                             "lidar_range_bias_m": 0.02}})
    file = open_lidar_file(ctx, cfg)
    key, frames = next(iter(ctx.passes.items()))
    common = dict(scene=run.meta.seed, key=key, frames=frames, cams=cams)
    first, second = _lidar_pass(ctx, cfg, file, 0, **common), _lidar_pass(ctx, other, file, 0, **common)
    assert first.settings.combine == "ivw" and first.settings.match == "gate"
    camera_lift = cfg.lift.model_copy(update={"depth": "box_far"})
    members = 0
    for f in frames:
        boxes = ctx.dets[f.idx].boxes
        z_cam = ctx.box_depth(f.idx, camera_lift) * 1.01
        cloud = window(first.cloud, round(run.plans()[f.idx].t * 1e9), round(first.settings.window_s * 1e9))
        a = gate_depth(boxes, z_cam, cams[key][f.idx], cloud, first.settings)
        b = gate_depth(boxes, z_cam, cams[key][f.idx], cloud, second.settings)
        assert len(a.members) == len(b.members)
        assert all(x.tobytes() == y.tobytes() for x, y in zip(a.members, b.members))
        assert a.z.tobytes() == b.z.tobytes() and a.n_points.tobytes() == b.n_points.tobytes()
        za, used_a = combine_depth(z_cam, a, first.settings)
        zb, used_b = combine_depth(z_cam, b, second.settings)
        assert za.tobytes() == zb.tobytes() and used_a.tobytes() == used_b.tobytes()
        members += sum(len(m) for m in a.members)
    assert members > 0


def _hand_rows() -> DepthRows:
    """Six detections: three counted truth-matched ones (LiDAR used, camera fallback, dropout), one off the counted
    row, and two counted ones without a truth match (LiDAR used, fallback)."""
    nan = float("nan")
    return DepthRows(
        run_id=["apple_s5"] * 6, noise_seed=[0] * 6, side=["A"] * 6, pitch=[10.0] * 6, idx=[0, 0, 1, 1, 2, 2],
        det=[0, 1, 0, 1, 0, 1], truth_id=[11, 12, 13, 14, -1, -1], counted=[True, True, True, False, True, True],
        bagged=[False, True, False, False, None, None], depth_med=[2.03, 2.10, 2.00, 1.80, nan, nan],
        z_cam_clean=[2.02, 2.31, 2.01, 1.81, 1.52, 1.70], z_cam=[2.02, 2.30, nan, 1.82, 1.50, 1.71],
        z_lidar=[2.00, nan, nan, 1.80, 1.50, nan], n_points=[9, 1, 0, 4, 5, 0],
        z_used=[2.00, 2.30, nan, 1.80, 1.50, 1.71], used_lidar=[True, False, False, True, True, False])


def test_lidar_entries_rows_and_report_lines(case, tmp_path):
    cfg = case.config(DRIFT_LD)
    single = case.evaluate(cfg, 0)
    entry, rows = single["lidar"], single["depth_rows"]
    assert set(entry) == LIDAR_KEYS
    assert entry["mount_m"] == [0.0, 0.0, 0.1] and entry["mount_rpy_deg"] == [0.0, 0.0, 0.0]
    assert (entry["file"], entry["unit"], entry["method"], entry["variant"]) == ("depth", "depth", "lidar", "judged")
    assert entry["sidecar_sha256"] == sidecar_sha256(case.shared.depth) and entry["config_sha16"] == "ab" * 8
    assert entry["sim_commit"] == COMMIT and entry["settings"] == cfg.lift.lidar.model_dump()

    matched = rows.counted & (rows.truth_id >= 0)       # the counted-row detections matched to a truth record
    e = rows.z_used[matched] - rows.depth_med[matched]
    has = np.isfinite(e)
    dropped = int((~np.isfinite(rows.z_used[matched])).sum())
    assert dropped > 0 and int(has.sum()) + dropped == int(matched.sum())
    depth = entry["depth"]
    assert (depth["n_with_depth"], depth["n_dropout"]) == (int(has.sum()), dropped)
    assert depth["over_10cm_rate"] == float((np.abs(e[has]) > 0.10).sum() / matched.sum())
    assert depth["abs_error_p90_m"] == float(np.percentile(np.abs(e[has]), 90))
    assert entry["coverage"] == float(rows.used_lidar[matched].sum() / matched.sum())

    truth = case.run.truth()
    for i in np.flatnonzero(rows.truth_id >= 0):
        tid, idx = int(rows.truth_id[i]), int(rows.idx[i])
        assert rows.bagged[i] is truth.fruits[tid].bagged
        record = next(r for r in truth.records[idx] if r.id == tid)
        assert rows.depth_med[i] == record.depth_med or (record.depth_med is None and np.isnan(rows.depth_med[i]))
    assert all(b is None for b in rows.bagged[rows.truth_id < 0])
    assert any(rows.bagged[matched]) and not all(rows.bagged[matched])

    hand = _hand_rows()
    file = open_lidar_file(case.ctx, cfg)
    made = lidar_entry(cfg, file=file, mount=lidarsource.sidecar_mount(file.folder), mount_rpy_deg=(0.0, 0.0, 0.0),
                       rows=hand, counts={"frames": 3, "cloud_points": 40, "window_points": 30})
    assert made["coverage"] == 1 / 3 and made["coverage_all"] == 2 / 5 and made["fallback_rate"] == 3 / 6
    assert made["depth"] == {"n_with_depth": 2, "n_dropout": 1, "over_10cm_rate": 1 / 3,
                             "abs_error_p90_m": float(np.percentile([2.03 - 2.00, 2.30 - 2.10], 90))}
    assert made["lidar_only"] == {"n": 1, "over_10cm_rate": 0.0, "abs_error_p90_m": float(abs(2.00 - 2.03))}
    assert made["points"]["p50"] == 1.0 and made["detections"]["n"] == 6

    combined = case.combined(cfg)
    for item, seed in zip(combined["per_seed"], (0, 1)):
        alone = case.evaluate(cfg, seed)
        assert _exact(item, ("lidar", "camera_depth_sha256")) == _exact(alone, ("lidar", "camera_depth_sha256"))
    assert set(combined["lidar"]) == LIDAR_KEYS and len(combined["depth_rows"]) == 2

    one_seed = case.config(DRIFT_LD, {"noise": {"seeds": [0]}})
    grid = Grid(path=Path("combine.yaml"), bases=(), factors={"lift.lidar.combine": ["replace"]}, product={},
                combos={}, common={})
    sweep, _ = run_sweep(case.run, sweep_cells(one_seed, grid, None), load_splits())
    for cell in sweep["cells"]:
        assert set(cell["lidar"]) == LIDAR_KEYS and len(cell["depth_rows"]) == 1
        assert set(cell["per_seed"][0]) >= {"lidar", "camera_depth_sha256"}

    true_cfg = case.config(TRUE_LD)
    ladder = _results(case, "ladder", {"L3_ld": case.combined(true_cfg), "L4_drift_ld": combined})
    ladder_rows = pop_depth_rows(ladder)
    assert list(ladder_rows) == ["L3_ld", "L4_drift_ld"]
    assert len(ladder_rows["L4_drift_ld"]) == 2 * len(rows)
    folder, _ = write_experiment(tmp_path / "ladder", cfg=cfg, results=ladder, record={}, criteria=None,
                                 depth_rows=ladder_rows)
    written = folder / "depth_rows"
    assert {p.name for p in written.iterdir()} == {"L3_ld.csv", "L4_drift_ld.csv", "index.csv"}
    for name, part in ladder_rows.items():
        text = (written / f"{name}.csv").read_bytes().decode("utf-8")
        assert text == depth_rows_csv(part)
        assert text.splitlines()[0] == ",".join(DepthRows.COLUMNS)
    assert "depth_rows" not in (folder / "results.json").read_text(encoding="utf-8")
    report = (folder / "report.md").read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("> LiDAR L4_drift_ld:") for line in report)

    grid_results = _results(case, "sweep", sweep, grid=grid)
    grid_rows = pop_depth_rows(grid_results)
    folder, _ = write_experiment(tmp_path / "grid", cfg=one_seed, results=grid_results, record={}, criteria=None,
                                 depth_rows=grid_rows)
    assert {p.name for p in (folder / "depth_rows").iterdir()} == {
        "baseline@-.csv", "lift.lidar.combine%3Dreplace@-.csv", "index.csv"}
    assert "depth_rows" not in (folder / "results.json").read_text(encoding="utf-8")
    assert any(line.startswith("> LiDAR baseline @ -:")
               for line in (folder / "report.md").read_text(encoding="utf-8").splitlines())


def _results(case: LidarCase, kind: str, body: dict, *, grid: Grid | None = None) -> dict:
    """An experiment result of `kind` on the shared bundle: a ladder of the rungs `body`, or the sweep entry `body`
    of `grid`."""
    if kind == "sweep":
        runs = [body]
    else:
        runs = [{"meta": describe_run(case.ctx), "chain": [], "rungs": body, "interaction_2x2": None}]
    out = {"schema_version": SCHEMA_VERSION, "kind": kind, "run_name": "lidar_rows", "experiment_id": "0123456789ab",
           "split": "dev", "crop": "apple", "runs": runs, "pooled": None, "verdicts": None}
    if grid is not None:
        out["sweep"] = {"grid": grid.path.name, "bases": [],
                        "points": [{"name": p.name, "patch": p.patch, "factor": p.factor, "value": p.value}
                                   for p in grid_points(grid)]}
    return out


def test_depth_rows_file_names_of_grid_cells(case, tmp_path):
    assert depth_rows_file_names(["lift.lidar.match=map@LVI_ld", "k1=a,k2=b@-", "100%", "나 1"]) == {
        "lift.lidar.match=map@LVI_ld": "lift.lidar.match%3Dmap@LVI_ld.csv", "k1=a,k2=b@-": "k1%3Da%2Ck2%3Db@-.csv",
        "100%": "100%25.csv", "나 1": "%EB%82%98%201.csv"}
    odd = ["a b@-", 'lift.center_offset_m={"fruit":0.015}@-', "pose.vio=x:y@L4", "L3_ld+map@-"]
    names = depth_rows_file_names(odd)
    assert list(names) == odd and names["L3_ld+map@-"] == "L3_ld+map@-.csv"
    assert all(re.fullmatch(r"([A-Za-z0-9_.@+-]|%[0-9A-F]{2})+\.csv", n) for n in names.values())
    for clash in (["L3", "l3"], ["index"], ["INDEX"]):
        with pytest.raises(ValueError, match="collides") as e:
            depth_rows_file_names(clash)
        assert all(name in str(e.value) for name in clash)
    assert depth_rows_file_names(["x" * 251]) == {"x" * 251: "x" * 251 + ".csv"}
    with pytest.raises(ValueError, match="255"):
        depth_rows_file_names(["x" * 252])

    base = case.config()                     # LiDAR files, box_far, no noise and no base rungs
    assert not base.is_noisy() and base.lift.depth == "box_far"
    grids = {"factors": (Grid(path=Path("factors.yaml"), bases=(), factors={"lift.far_min_frac": [0.25]},
                                product={}, combos={}, common={}), "lift.far_min_frac%3D0.25@-.csv"),
             "product": (Grid(path=Path("product.yaml"), bases=(), factors={},
                              product={"lift.far_min_frac": [0.25], "lift.far_gap_m": [0.03]}, combos={}, common={}),
                         "lift.far_min_frac%3D0.25%2Clift.far_gap_m%3D0.03@-.csv")}
    written = {}
    for kind, (grid, point_file) in grids.items():
        entry, _ = run_sweep(case.run, sweep_cells(base, grid, None), load_splits())
        results = _results(case, "sweep", entry, grid=grid)
        rows = pop_depth_rows(results)
        folder, _ = write_experiment(tmp_path / kind, cfg=base, results=results, record={}, criteria=None,
                                     depth_rows=rows)
        files = folder / "depth_rows"
        assert {p.name for p in files.iterdir()} == {"baseline@-.csv", point_file, "index.csv"}, kind
        header = depth_rows_csv(next(iter(rows.values()))).splitlines()[0]
        for p in files.iterdir():
            if p.name != "index.csv":
                assert p.read_text(encoding="utf-8").splitlines()[0] == header
        with (files / "index.csv").open(newline="", encoding="utf-8") as f:
            table = list(csv.reader(f))
        assert table[0] == ["cell", "file"]
        assert table[1:] == [[cell, depth_rows_file_names(list(rows))[cell]] for cell in rows]
        assert [cell for cell, _ in table[1:]] == [f"{c['point']}@-" for c in entry["cells"]]
        assert (files / "index.csv").read_bytes().decode("utf-8") == depth_rows_index_csv(
            depth_rows_file_names(list(rows)))
        written[kind] = (results, rows)
    index = tmp_path / "product" / "lidar_rows" / "0123456789ab" / "depth_rows" / "index.csv"
    assert index.read_text(encoding="utf-8").splitlines()[2] == \
        '"lift.far_min_frac=0.25,lift.far_gap_m=0.03@-",lift.far_min_frac%3D0.25%2Clift.far_gap_m%3D0.03@-.csv'

    results, rows = written["factors"]
    part = next(iter(rows.values()))
    with pytest.raises(ValueError, match="collides"):
        write_experiment(tmp_path / "clash", cfg=base, results=results, record={}, criteria=None,
                         depth_rows={"A@-": part, "a@-": part})
    assert not (tmp_path / "clash" / "lidar_rows" / "0123456789ab").exists()


def test_noisy_point_cache_follows_the_file_and_the_seed(case, tmp_path):
    bundle = tmp_path / "apple" / case.shared.bundle.name
    shutil.copytree(case.shared.bundle, bundle)
    sparse = write_lidar_files(bundle, config_sha256="cd" * 32, stride=5)
    run_id, ctx = case.run.meta.run_id, case.ctx
    true_ld = {"pose": {}, "lift": {"depth": "lidar"}}
    dense_cfg = case.config(true_ld)
    sparse_cfg = case.config(true_ld, {"lift": {"lidar_files": {"depth": {
        "dir": str(sparse.parent), "sim_config_sha256": "cd" * 32, "digests": {run_id: sidecar_sha256(sparse)}}}}})
    dense, thin = evaluate(ctx, dense_cfg), evaluate(ctx, sparse_cfg)
    assert dense["depth_rows"].n_points.tobytes() != thin["depth_rows"].n_points.tobytes()

    noise_only = case.config(true_ld, {"noise": {**LIDAR_NOISE, "seeds": [0, 1]}})
    other_seed = case.config(true_ld, {"noise": {**LIDAR_NOISE, "seeds": [0, 1]}, "seed": 1})
    a, b = evaluate(ctx, noise_only, noise_seed=0), evaluate(ctx, other_seed, noise_seed=0)
    assert a["depth_rows"].z_lidar.tobytes() != b["depth_rows"].z_lidar.tobytes()

    first = evaluate(ctx, noise_only, noise_seed=0)
    evaluate(ctx, noise_only, noise_seed=1)
    keys = list(ctx._lidar_points)
    assert len(keys) == 2 and {k[5] for k in keys} == {1} and len({k[4] for k in keys}) == 2
    assert {k[:4] for k in keys} == {(0, str(case.shared.depth), sidecar_sha256(case.shared.depth), "depth")}
    again = evaluate(ctx, noise_only, noise_seed=0)
    assert _without_rows(again) == _without_rows(first) == _without_rows(a)
    assert depth_rows_csv(again["depth_rows"]) == depth_rows_csv(first["depth_rows"])


def test_open_lidar_file_needs_a_pinned_digest_and_frame_times(case, tmp_path):
    run_id = case.run.meta.run_id
    unpinned = copy.deepcopy(case.raw)
    del unpinned["lift"]["lidar_files"]["depth"]["digests"][run_id]
    cfg = case.config(TRUE_LD, base=unpinned)
    with pytest.raises(ValueError, match=re.escape(f"lift.lidar_files.depth: no pinned lidar.json digest for run "
                                                   f"{run_id}")) as e:
        open_lidar_file(case.ctx, cfg)
    assert str(case.shared.depth) in str(e.value)

    straight = write_run(tmp_path / "straight", size=(64, 36))
    ctx = prepare(SynthRun.open(straight), fake_cfg(straight, "apple"), load_splits())
    with pytest.raises(ValueError, match="lift.depth=lidar needs a shaken bundle with frame times"):
        open_lidar_file(ctx, case.config(TRUE_LD))
    assert open_lidar_file(case.ctx, case.config(base=unpinned)) is None     # box_far: the files stay unopened
    assert open_lidar_file(case.ctx, case.config(TRUE, base=unpinned)) is None


def _run_files(case: LidarCase, res: dict) -> tuple[bytes, str]:
    """The `results.json` bytes and the depth rows text of `res` as the one rung `run` of an experiment result."""
    from counting.evaluation.report import results_bytes

    results = _results(case, "run", {"run": res})
    rows = pop_depth_rows(results)
    return results_bytes(results), depth_rows_csv(rows["run"])


def test_true_pose_oracle_converges_and_reruns_match():
    low = low_pitch_case()
    cfg = low.config(ORACLE, LIDAR_ZERO)
    assert not cfg.is_noisy()
    first = evaluate(low.ctx, cfg)
    entry = first["lidar"]
    assert entry["method"] == "lidar_oracle" and entry["depth"]["n_with_depth"] > 50
    assert entry["depth"]["abs_error_p90_m"] <= 0.002 and entry["coverage"] >= 0.9
    again = evaluate(prepare(low.run, cfg, load_splits()), cfg)     # a context of its own
    assert _run_files(low, again) == _run_files(low, first)


def test_oracle_pairing_reads_the_labels_on_the_evaluation_side(case, tmp_path):
    from counting.multiview.lidardepth import PassCloud

    cfg, ctx, plans = case.config(ORACLE, LIDAR_ZERO), case.ctx, case.run.plans()
    file = open_lidar_file(ctx, cfg)
    key, frames = next(iter(ctx.passes.items()))
    lp = _lidar_pass(ctx, cfg, file, None, scene=case.run.meta.seed, key=key, frames=frames, cams=None)
    labels = lidarsource.read_labels(file.folder, key.side, n_points=len(file.points(key.side).t_ns))
    half = round(lp.settings.window_s * 1e9)
    picks = []
    for f in frames:
        near = window(lp.cloud, round(plans[f.idx].t * 1e9), half)
        got = lidarsource.oracle_box_depth(ctx.src[f.idx], f.camera, near, labels, lp.settings)
        picks.append((int(got.n_points.max(initial=0)), f.idx, near, got))
    _, idx, near, base = max(picks, key=lambda p: p[0])   # the frame with the box of the most points
    ids, camera = ctx.src[idx], ctx._frame_of[idx].camera
    i = int(np.argmax(base.n_points))
    assert base.n_points[i] >= 4 and (ids >= 0).all()
    for b, rows in enumerate(base.members):       # a box's points: its fruit or bag, mixed within 0.10 m
        assert len(rows) == base.n_points[b] and (np.diff(rows) > 0).all()
        assert (labels.fruit_id[rows] == ids[b]).all() and np.isin(labels.part[rows], FRUIT_PARTS).all()
        assert (labels.mixed[rows] != 2).all()

    unmatched = ids.copy()
    unmatched[i] = -1
    skipped = lidarsource.oracle_box_depth(unmatched, camera, near, labels, lp.settings)
    assert np.isnan(skipped.z[i]) and skipped.n_points[i] == 0 and len(skipped.members[i]) == 0
    others = np.arange(len(ids)) != i
    assert skipped.z[others].tobytes() == base.z[others].tobytes()

    taken = base.members[i][: len(base.members[i]) // 2 + 1]     # most of the box's points, 0.3 m farther away
    X = near.X_w.copy()
    X[np.isin(near.index, taken)] += 0.3 * camera.R_c2w[:, 2]
    moved = PassCloud(t_ns=near.t_ns, X_w=X, index=near.index)
    assert lidarsource.oracle_box_depth(ids, camera, moved, labels, lp.settings).z[i] > base.z[i] + 0.2
    for name, value in (("mixed", 2), ("part", 1)):              # mixed beyond 0.10 m, then a leaf
        marked = {"side": labels.side, "fruit_id": labels.fruit_id, "part": labels.part, "mixed": labels.mixed}
        marked[name] = marked[name].copy()
        marked[name][taken] = value
        marked = lidarsource.LidarLabels(**marked)
        kept = lidarsource.oracle_box_depth(ids, camera, near, marked, lp.settings)
        shifted = lidarsource.oracle_box_depth(ids, camera, moved, marked, lp.settings)
        assert kept.n_points[i] == base.n_points[i] - len(taken) and not np.isin(taken, kept.members[i]).any(), name
        assert kept.z.tobytes() == shifted.z.tobytes() and kept.n_points.tobytes() == shifted.n_points.tobytes()
        assert all(a.tobytes() == b.tobytes() for a, b in zip(kept.members, shifted.members)), name

    low = low_pitch_case()
    clean = low.evaluate(low.config(ORACLE, LIDAR_ZERO))["lidar"]
    dropped = low.config(ORACLE, LIDAR_ZERO, {"noise": {"lidar_dropout": 0.05, "seeds": [0, 1]}})
    for seed in (0, 1):        # the points left after the dropout keep the labels of their file rows
        entry = low.evaluate(dropped, seed)["lidar"]
        assert entry["points"]["cloud"] < clean["points"]["cloud"], seed
        assert entry["depth"]["abs_error_p90_m"] <= 0.002 and entry["coverage"] >= 0.9, seed

    copied = tmp_path / "lidar" / ("ab" * 32)[:16] / case.shared.bundle.name   # the label files turned to rubbish
    shutil.copytree(case.shared.depth, copied)
    for side in ("A", "B"):
        (copied / f"lidar_{side}_labels.npz").write_bytes(b"not a label file")
    rubbish = case.config(ORACLE, LIDAR_ZERO, {"lift": {"lidar_files": {"depth": {"dir": str(copied.parent)}}}})
    with pytest.raises(ValueError, match="sha256 does not match lidar.json files") as e:
        evaluate(ctx, rubbish)
    assert str(copied) in str(e.value) and "_labels.npz" in str(e.value)


def test_depth_forms_in_evaluate(case):
    from counting.evaluation.report import render_report

    plain_cfg = case.config(DRIFT)
    for seed in (0, 1):        # every form at 0 changes nothing
        plain, zero = case.evaluate(plain_cfg, seed), case.evaluate(case.config(DRIFT, FORMS_ZERO), seed)
        assert _exact(zero, ("y1", "y6", "camera_depth_sha256")) == _exact(plain, ("y1", "y6", "camera_depth_sha256"))
        assert zero["depth_rows"].z_cam.tobytes() == plain["depth_rows"].z_cam.tobytes()
    plain = case.evaluate(plain_cfg, 0)
    rows = plain["depth_rows"]
    assert np.isnan(rows.z_cam).any() and np.isfinite(rows.z_cam).sum() > 50

    bias = {"noise": {"depth_bias_rel": 0.01}}
    biased = case.evaluate(case.config(DRIFT, bias), 0)
    np.testing.assert_allclose(biased["depth_rows"].z_cam, rows.z_cam * 1.01, rtol=1e-12, atol=0)
    sha = biased["camera_depth_sha256"]       # on the camera-only rung, in its digest and shared with its LiDAR pair
    assert re.fullmatch(r"[0-9a-f]{64}", sha) and sha != plain["camera_depth_sha256"]
    assert case.evaluate(case.config(DRIFT_LD, bias), 0)["camera_depth_sha256"] == sha
    alone = case.evaluate(case.config(TRUE, {"noise": {"depth_bias_rel": 0.01, "seeds": [0]}}), 0)
    assert re.fullmatch(r"[0-9a-f]{64}", alone["camera_depth_sha256"])

    scaled = case.evaluate(case.config(DRIFT, {"noise": {"depth_scale_sigma": 0.05}}), 0)["depth_rows"]
    ok = np.isfinite(rows.z_cam)
    ratios: dict[tuple, list[float]] = {}
    for r, side, pitch in zip(scaled.z_cam[ok] / rows.z_cam[ok], rows.side[ok], rows.pitch[ok]):
        ratios.setdefault((side, pitch), []).append(float(r))
    assert len(ratios) == len(case.ctx.passes)
    for values in ratios.values():           # one scale per pass
        np.testing.assert_allclose(values, values[0], rtol=1e-12, atol=0)
    firsts = [values[0] for values in ratios.values()]
    assert len(set(firsts)) == len(firsts) and all(abs(r - 1.0) > 1e-6 for r in firsts)

    true_rows = case.evaluate(case.config(TRUE, {"noise": PERSIST}), 0)["depth_rows"]
    drift = {"pose": {"rig": True, "rig_ref": 10}, "noise": {**VO, **START, **RIG, **PERSIST}}
    drift_rows = case.evaluate(case.config(drift), 0)["depth_rows"]
    assert true_rows.z_cam.tobytes() == drift_rows.z_cam.tobytes()    # the field is read on the true cameras
    field = true_rows.z_cam / true_rows.z_cam_clean - 1.0
    assert np.nanstd(field) > 1e-3 and np.nanmax(np.abs(field)) < 0.2

    report = render_report(_results(case, "ladder", {"plain": case.combined(plain_cfg),
                                                     "bias": case.combined(case.config(DRIFT, bias))}),
                           criteria=None).splitlines()
    assert [line for line in report if line.startswith("- LiDAR·깊이 오차 잡음:")] == \
        ["- LiDAR·깊이 오차 잡음: depth_bias_rel 0.01"]


def test_lidar_files_join_the_fingerprint_and_record(case, tmp_path):
    from typer.testing import CliRunner

    from counting.evaluation.cli import eval_app
    from counting.evaluation.record import experiment_id
    from tests.unit.lidarfake import dump_json

    run_id, shared, cli = case.run.meta.run_id, case.shared, CliRunner()

    def eval_run(name: str, raw: dict) -> tuple[Path, dict]:
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump({**raw, "output_dir": str(tmp_path / "out")}), encoding="utf-8")
        r = cli.invoke(eval_app, ["run", "--config", str(path), "--splits", str(EVAL_CONFIGS / "splits.yaml")])
        assert r.exit_code == 0, r.output
        folder = Path(r.output.splitlines()[0])
        return folder, json.loads((folder / "record.json").read_text(encoding="utf-8"))

    lidar_raw = _merge(case.raw, {"pose": {}, "lift": {"depth": "lidar"}})
    folder, record = eval_run("lidar", lidar_raw)
    sha = sidecar_sha256(shared.depth)
    assert record["lidar_files"] == [{"name": "depth", "run_id": run_id, "sidecar_sha256": sha, "sim_commit": COMMIT}]
    data = record["data"][0]
    fingerprints = [data["fingerprint"], data["depth_fingerprint"], f"lidar:depth:{sha[:16]}"]
    assert experiment_id(config_hash=record["config_hash"], fingerprints=fingerprints, code=record["code"],
                         seed=record["seeds"]["seed"]) == folder.name

    other = tmp_path / "lidar" / ("ab" * 32)[:16] / shared.bundle.name    # the same points, another simulator commit
    shutil.copytree(shared.depth, other)
    sidecar = json.loads((other / "lidar.json").read_bytes())
    sidecar["orchard_synth"]["commit"] = "2" * 40
    (other / "lidar.json").write_bytes(dump_json(sidecar))
    repinned = _merge(lidar_raw, {"lift": {"lidar_files": {"depth": {
        "dir": str(other.parent), "sim_commits": ["2" * 40], "digests": {run_id: sidecar_sha256(other)}}}}})
    folder2, record2 = eval_run("repinned", repinned)
    assert folder2.name != folder.name and sidecar_sha256(other) != sha
    assert record2["lidar_files"] == [{"name": "depth", "run_id": run_id, "sidecar_sha256": sidecar_sha256(other),
                                       "sim_commit": "2" * 40}]

    assert "lidar_files" not in eval_run("unopened", case.raw)[1]    # LiDAR files set, the box depth from the camera
    assert set(eval_run("plain", case.plain)[1]) == {"config_hash", "data", "code", "seeds", "detector", "env",
                                                     "timings", "started", "finished", "extra"}
