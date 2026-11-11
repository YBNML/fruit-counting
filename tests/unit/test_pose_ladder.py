"""The pose ladder, the drift grids and the development pose configs: names, noise points, cells and the
command-line runs of the rungs the runner supports."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import counting.evaluation.fusion as fusion_module
from counting.config.eval_schema import EvalAppConfig
from counting.config.loader import load_yaml_model
from counting.evaluation.cli import eval_app
from counting.evaluation.protect import check_test_grid
from counting.evaluation.runner import _patched, check_ladder_request, load_ladder, rung_config
from counting.evaluation.sweep import load_grid, sweep_cells

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]
POSE = ROOT / "configs" / "eval" / "sweeps_pose"
LADDER = load_ladder(POSE / "ladder_pose.yaml")
SPLITS = str(ROOT / "configs" / "eval" / "splits_pose.yaml")
runner = CliRunner()

# The named noise values of the ladder's noise_blocks.
VO = {"drift_scale_sigma": 0.0045, "drift_rate_deg_per_m": 0.045, "drift_step_trans_mm": 2.1,
      "drift_step_rot_deg": 0.039}
VO_DVO = {"drift_scale_sigma": 0.003, "drift_rate_deg_per_m": 0.03, "drift_step_trans_mm": 0.7,
          "drift_step_rot_deg": 0.013}
START = {"pass_side_trans_m": 0.05, "pass_side_rot_deg": 0.5}
RIG = {"rig_extr_trans_mm": 14.0, "rig_extr_rot_deg": 0.4}
PITCH = {"pass_pitch_trans_m": 0.02, "pass_pitch_rot_deg": 0.25}
DEPTH = {"depth_rel_sigma": 0.02, "depth_dropout": 0.05}
IMU = {"imu_gyro_n": 1.16e-3, "imu_acc_n": 4.97e-3, "imu_gyro_w": 1.82e-5, "imu_acc_w": 4.75e-4,
       "imu_gyro_b0_dps": 1.0, "imu_acc_b0_mg": 4.0, "imu_temp_k": 1.0}
BELIEF = {"imu_time_offset_ms": 0.0, "imu_extr_trans_mm": 10.0, "imu_extr_rot_deg": 0.3}
INDEPENDENT = {"pose_trans_sigma_m": 0.02, "pose_rot_sigma_deg": 0.5, "pass_bias_m": 0.02, "depth_rel_sigma": 0.02,
               "depth_dropout": 0.05}
JUDGED = {**VO, **START, **RIG, **DEPTH, **IMU, **BELIEF}
REF = {**JUDGED, **PITCH}

# (grid file, its one base, number of cells, may run on the held-out scenes)
GRIDS = [
    ("drift_rpe", "B_drift", 10, True),
    ("drift_inpass", "B_drift_fitp", 15, True),
    ("drift_interpass", "B_pass", 16, True),
    ("drift_interpass_c1x", "B_pass_c1x", 8, True),
    ("drift_imu", "B_drift", 18, True),
    ("drift_profiles", "B_drift", 11, True),
    ("drift_val", "B_drift", 14, False),
    ("drift_vi", "B_vi", 11, False),
]

# The tracker, box depth and dedup settings of frozen/{crop}_test_d1.yaml that the development pose configs use.
FROZEN_KNOBS = {
    "apple": {"stages": {"dedup": "hybrid"}, "lift": {"depth": "box_far"},
              "tracker": {"repair_m": 0.08, "score_hi": 0.6, "score_lo": 0.2, "min_obs_cached": 4,
                          "depth_gate": "soft"},
              "dedup": {"radius_m": 0.09, "band_m": 0.05}},
    "pear": {"lift": {"depth": "box_far"},
             "tracker": {"repair_m": 0.18, "score_hi": 0.9, "score_lo": 0.1, "min_obs_cached": 3,
                         "depth_gate": "soft"},
             "dedup": {"radius_m": 0.135, "band_m": 0.10}},
}
# (crop, camera configuration) -> (passes.pitches, pose.rig_ref)
LAYOUTS = {("apple", "c2"): ([10, 50], 50), ("pear", "c2"): ([20, 70], 70),
           ("apple", "c1"): ([30], 30), ("pear", "c1"): ([45], 45)}


def _base(crop: str = "apple", label: str = "c2") -> EvalAppConfig:
    """The development pose config with the cache and IMU file names a val or test config pins."""
    cfg = load_yaml_model(ROOT / "configs" / "eval" / f"{crop}_dev_pose_{label}.yaml", EvalAppConfig)
    files = {"dir": "runs/x", "digests": {}}
    return _patched(cfg, {"detector": {"cache": "runs/detect/x"},
                          "pose": {"caches": {"vo": files, "vo_dd": files},
                                   "imu_files": {n: files for n in ("imu_ext", "imu_lab", "imu_icm", "imu_ext_x4")}}})


def _layout_patch(crop: str, label: str) -> dict:
    """The keys a development pose config sets for its camera configuration: run name, pitches, pose."""
    pitches, ref = LAYOUTS[crop, label]
    return {"run_name": f"{crop}_dev_pose_{label}", "passes": {"pitches": pitches},
            "pose": {"rig_ref": ref, "config_label": label}}


def _pose_cfg_file(tmp_path: Path, run: Path) -> str:
    """apple_dev_pose_c2.yaml on `run`, with the fake bundle's pitches (10 and 30) and the reference camera at 30."""
    raw = yaml.safe_load((ROOT / "configs" / "eval" / "apple_dev_pose_c2.yaml").read_text(encoding="utf-8"))
    raw.update(runs=[str(run)], output_dir=str(tmp_path / "out"))
    raw["passes"]["pitches"] = [10, 30]
    raw["pose"]["rig_ref"] = 30
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(p)


def test_pose_ladder_names_and_counts():
    rungs, baselines, val_only = LADDER["rungs"], LADDER["baselines"], LADDER["val_only"]
    assert list(rungs) == ["L3", "L4_drift", "LVI", "L5", "L5VI"]
    assert len(baselines) == 36 and len(val_only) == 10 and set(val_only) <= set(baselines)
    held_out = [name for name in [*rungs, *baselines] if name not in val_only]
    assert len(held_out) == 31
    assert {"B_drift", "B_drift_fitp", "B_pass", "B_pass_c1x"} <= set(held_out)
    for name, patch in [*rungs.items(), *baselines.items()]:
        assert "pose" in patch, name


def test_every_pose_rung_passes_the_schema():
    names = [*LADDER["rungs"], *LADDER["baselines"]]
    for crop, ref in (("apple", 30.0), ("pear", 45.0)):
        check_ladder_request(_base(crop), LADDER, names)
        c1 = _base(crop, "c1")
        check_ladder_request(c1, LADDER, ["L3", "L4_drift", "LV", "LVI", "L5", "L5VI"])
        drift = rung_config(c1, LADDER, "L4_drift")
        assert drift.pose.rig_ref == ref and drift.passes.pitches == [ref], crop


def test_judged_rungs_carry_the_judged_point():
    rungs, baselines = LADDER["rungs"], LADDER["baselines"]
    assert rungs["L4_drift"]["noise"] == JUDGED
    assert rungs["LVI"]["noise"] == {**START, **RIG, **DEPTH, **BELIEF}
    assert rungs["L5VI"]["noise"] == {**START, **RIG, **DEPTH, **BELIEF}
    assert baselines["L4_drift_noimu"]["noise"] == {**VO, **START, **RIG, **DEPTH}
    assert baselines["L4_drift_c1x"]["noise"] == {**VO, **START, **PITCH, **DEPTH, **IMU, **BELIEF}
    assert baselines["L4_drift_dvo"]["noise"] == {**VO_DVO, **START, **RIG, **DEPTH, **IMU, **BELIEF}
    assert baselines["L4_base"]["noise"] == INDEPENDENT
    drift = rung_config(_base(), LADDER, "L4_drift")
    assert drift.pose.rig is True and drift.pose.fuse == "imu"
    assert drift.align.mode == ["tracks", "row_plane"]


def test_structure_rungs_have_no_noise():
    structure = [name for name in LADDER["baselines"] if name.startswith("B_")]
    assert structure == ["B_drift", "B_drift_fitp", "B_pass", "B_pass_c1x", "B_vi"]
    for name in structure:
        assert "noise" not in LADDER["baselines"][name], name


@pytest.mark.parametrize(("name", "base", "n_cells", "held_out"), GRIDS, ids=[g[0] for g in GRIDS])
def test_pose_grids_build_on_their_one_base(name, base, n_cells, held_out):
    grid = load_grid(POSE / f"{name}.yaml")
    assert grid.bases == (base,)
    assert len(sweep_cells(_base(), grid, LADDER)) == n_cells
    if held_out:
        check_test_grid(grid, LADDER)


def test_grid_commons_repeat_the_judged_point():
    for name, *_ in GRIDS:
        common = load_grid(POSE / f"{name}.yaml").common
        assert list(common) == ["noise"], name
        assert common["noise"] == {k: REF[k] for k in common["noise"]}, name


def test_dev_pose_configs_follow_the_dev_configs_and_frozen_knobs():
    for crop, knobs in FROZEN_KNOBS.items():
        frozen = load_yaml_model(ROOT / "configs" / "eval" / "frozen" / f"{crop}_test_d1.yaml", EvalAppConfig)
        frozen_dump = frozen.model_dump()
        for section, values in knobs.items():
            assert {k: frozen_dump[section][k] for k in values} == values, (crop, section)
        dev = load_yaml_model(ROOT / "configs" / "eval" / f"{crop}_dev.yaml", EvalAppConfig)
        c2 = _patched(dev, {**knobs, "noise": {"seeds": [0, 1, 2]}, "output_dir": "runs/eval_pose",
                            **_layout_patch(crop, "c2")})
        c1 = _patched(c2, _layout_patch(crop, "c1"))
        for label, expected in (("c2", c2), ("c1", c1)):
            cfg = load_yaml_model(ROOT / "configs" / "eval" / f"{crop}_dev_pose_{label}.yaml", EvalAppConfig)
            assert cfg.model_dump() == expected.model_dump(), (crop, label)


def test_eval_ladder_runs_the_supported_pose_rungs(apple_run, tmp_path):
    args = ["ladder", "--config", _pose_cfg_file(tmp_path, apple_run), "--ladder", str(POSE / "ladder_pose.yaml"),
            "--rungs", "L3,L3_al,L4_base", "--splits", SPLITS]
    r = runner.invoke(eval_app, args)
    assert r.exit_code == 0, r.output
    (d,) = (tmp_path / "out" / "apple_dev_pose_c2").iterdir()
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    rungs = res["runs"][0]["rungs"]
    assert res["schema_version"] == 10 and list(rungs) == ["L3", "L3_al", "L4_base"]
    assert rungs["L3_al"]["alignment"]["mode"] == ["tracks", "row_plane"] and rungs["L3"]["alignment"] is None
    for name, rung in rungs.items():
        assert rung["pose"]["config"] == "c2" and rung["pose_quality"] is not None, name
    report = (d / "report.md").read_text(encoding="utf-8")
    assert "자세 출처 칸 있음(칸마다 적음)" in report
    assert report.splitlines().count("**자세 품질**") == 3


def test_eval_refuses_unsupported_pose_rungs(apple_run, tmp_path, monkeypatch):
    base = ["ladder", "--config", _pose_cfg_file(tmp_path, apple_run), "--ladder", str(POSE / "ladder_pose.yaml"),
            "--splits", SPLITS]
    monkeypatch.setattr(fusion_module, "FUSE_FIXED_PATH", tmp_path / "absent" / "fuse_fixed.yaml")
    monkeypatch.setattr(fusion_module, "has_gtsam", lambda: True)
    r = runner.invoke(eval_app, [*base, "--rungs", "L4_drift"])
    assert r.exit_code == 2 and "no fixed fusion settings; pose.fuse reads them" in r.output, r.output
    assert not (tmp_path / "out").exists()
    r = runner.invoke(eval_app, [*base, "--rungs", "LVI"])     # the development config names no pose cache
    assert r.exit_code == 2 and "pose.source=file needs pose.cache, one of pose.caches" in r.output, r.output
    assert not (tmp_path / "out").exists()
