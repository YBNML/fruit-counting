"""The files a held-out run of the depth ladder reads: the criteria of the LiDAR depth rungs (the numbers of the pose
criteria under their names), the depth ladder, its five held-out grids, the val configs of both crops (the c2 pose
val configs with the LiDAR depth knobs, the pinned LiDAR files and the judged configuration label c2l) and the frozen
held-out templates built on them, whose protect section pins the ladder, splits, criteria and grids by sha256.

Every file is read inside the tests. The held-out checks run on the rung and grid cell configs only: nothing here
opens a bundle, a cache or a LiDAR file."""

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from counting.config.eval_schema import EvalAppConfig, LidarLiftConfig, SplitsConfig
from counting.config.loader import load_yaml_model
from counting.evaluation.protect import (
    check_test_frozen_request, check_test_grid, check_test_lidar_rungs, check_test_lift, check_test_pose_rungs,
    check_test_rung_protect, check_test_rungs,
)
from counting.evaluation.runner import check_ladder_request, load_ladder, rung_config
from counting.evaluation.sweep import load_grid, sweep_cells

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "configs" / "eval"
CROPS = ("apple", "pear")
# Repository paths of the files a held-out template pins.
LADDER = "configs/eval/sweeps_depth/ladder_depth.yaml"
SPLITS = "configs/eval/splits_depth.yaml"
CRITERIA = "configs/eval/criteria_depth.yaml"
SPLITS_SHA256 = "139b6645ffa503e4430fbe18300e3328b73f4beb03cdbbf82c1e5fe734f88401"

# The noise values of the LiDAR files' counting contract (range, range within the near range, near range, beam angle,
# dropout and range bias) and the depth unit's own camera-LiDAR extrinsic error.
CONTRACT = {"lidar_range_sigma_m": 0.02, "lidar_range_sigma_near_m": 0.03, "lidar_near_range_m": 1.0,
            "lidar_ang_sigma_deg": 0.15, "lidar_dropout": 0.05, "lidar_range_bias_m": 0.0,
            "lidar_extr_trans_mm": 10.0, "lidar_extr_rot_deg": 0.3}
# The named noise values of the pose ladder: the side start, the rig calibration, the camera depth, the IMU levels
# and the IMU errors the fusion believes. J is the judged camera, pose and IMU point: the body odometry flow and all
# of these, the common block of the drift profile grid.
START = {"pass_side_trans_m": 0.05, "pass_side_rot_deg": 0.5}
RIG = {"rig_extr_trans_mm": 14.0, "rig_extr_rot_deg": 0.4}
DEPTH = {"depth_rel_sigma": 0.02, "depth_dropout": 0.05}
IMU = {"imu_gyro_n": 1.16e-3, "imu_acc_n": 4.97e-3, "imu_gyro_w": 1.82e-5, "imu_acc_w": 4.75e-4,
       "imu_gyro_b0_dps": 1.0, "imu_acc_b0_mg": 4.0, "imu_temp_k": 1.0}
BELIEF = {"imu_time_offset_ms": 0.0, "imu_extr_trans_mm": 10.0, "imu_extr_rot_deg": 0.3}
J = {"drift_scale_sigma": 0.0045, "drift_rate_deg_per_m": 0.045, "drift_step_trans_mm": 2.1,
     "drift_step_rot_deg": 0.039, **START, **RIG, **DEPTH, **IMU, **BELIEF}

# The LiDAR files of each val config (lift.lidar_files dump): the depth unit and the navigation unit, the sha256 of
# the simulation config that made them, the simulator commit, and the sidecar digest of the val scenes 200 and 201.
SIM_COMMIT = "461664412594c5587b35ee8931f940a502ab31af"
DEPTH_SIM = "ab3576dcd4f1dcfde7860df779b33e43ca8af837115428cea401a4cfb4f7eb39"
NAV_SIM = "374e38b82922520925fee8af448065d5b4e83299b852726fc6e5d2478b1ca252"
FILES = {
    "apple": {
        "depth": {"dir": "~/datasets/orchard_synth/apple/lidar/ab3576dcd4f1dcfd", "unit": "depth",
                  "sim_config_sha256": DEPTH_SIM, "sim_commits": [SIM_COMMIT], "digests": {
                      "apple_s200_789edf28": "7df365f2745e6b83935f85a20d4d2fb9a88e5ad484f178527e853ca6fda6f760",
                      "apple_s201_b963f282": "2e2d385c6b5ace43b723ac026ebce5dd2448a9ac7761e1c7c0baff0f4ca62aec"}},
        "nav": {"dir": "~/datasets/orchard_synth/apple/lidar/374e38b829225209", "unit": "nav",
                "sim_config_sha256": NAV_SIM, "sim_commits": [SIM_COMMIT], "digests": {
                    "apple_s200_789edf28": "406099c95ee959b8e402eb982eefa5d2f64a5d20b12858a265683a7b9ddf58a2",
                    "apple_s201_b963f282": "4dd127d7fe8eca759f32c424af7923a0fd61abd1678192b775835683f4431db4"}},
    },
    "pear": {
        "depth": {"dir": "~/datasets/orchard_synth/pear/lidar/ab3576dcd4f1dcfd", "unit": "depth",
                  "sim_config_sha256": DEPTH_SIM, "sim_commits": [SIM_COMMIT], "digests": {
                      "pear_s200_55c0ef00": "c11f874cc501b1685e9a28cb21bd8dc4fad95e43e5530e52b61716633b717efd",
                      "pear_s201_11ab5d03": "906cb6360b77e451f919685c8351b50376a89d6281b6d660e04b3ca4d9fd23b4"}},
        "nav": {"dir": "~/datasets/orchard_synth/pear/lidar/374e38b829225209", "unit": "nav",
                "sim_config_sha256": NAV_SIM, "sim_commits": [SIM_COMMIT], "digests": {
                    "pear_s200_55c0ef00": "4152f61e7b919c07541c6371162f94eae676092d3da0563ee8d8ffd33c801abb",
                    "pear_s201_11ab5d03": "7e60ad6317a01c972e13e51951264171e48f16855ab8c5cdf634257b2b6cb877"}},
    },
}
# The one rung list a held-out ladder request may name: the judgement chain, the judged rungs, their camera-only
# pairs and the report rungs.
DEPTH_LIST = ["L3", "L5", "L3_dd", "L3_ld", "L4_drift_ld", "LVI_ld", "L5VI_ld", "L4_drift", "LVI", "L5VI", "LVI_dd",
              "LVI_ddd", "LVI_ddv", "L4_drift_ld_orc", "L4_drift_ldn", "LVI_ldn"]
# (held-out grid file, its bases, number of cells)
GRIDS = [
    ("depth_cam", ("B_drift",), 11),
    ("depth_lidar", ("B_drift_ld",), 12),
    ("extr_ld", ("B_drift_ld",), 7),
    ("profiles_ld", ("B_drift_ld",), 4),
    ("depth_forms", ("B_drift", "B_drift_ld_noise"), 12),
]


def _grid_path(name: str) -> str:
    return f"configs/eval/sweeps_depth/{name}.yaml"


def _sha256(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def _val(crop: str) -> EvalAppConfig:
    return load_yaml_model(EVAL / f"{crop}_val_depth.yaml", EvalAppConfig)


def _template(crop: str) -> EvalAppConfig:
    return load_yaml_model(EVAL / "frozen" / f"{crop}_test_depth.yaml", EvalAppConfig)


def test_depth_criteria_take_the_pose_numbers():
    pose = yaml.safe_load((EVAL / "criteria_pose.yaml").read_text(encoding="utf-8"))
    depth = yaml.safe_load((ROOT / CRITERIA).read_text(encoding="utf-8"))
    names = {"L4_drift": "L4_drift_ld", "LVI": "LVI_ld", "L5VI": "L5VI_ld"}
    assert depth == {"judged_config": "c2l",
                     **{kind: {names[name]: spec for name, spec in pose[kind].items()} for kind in ("y1", "y6")}}
    assert [list(depth[kind]) for kind in ("y1", "y6")] == [[names[n] for n in pose[kind]] for kind in ("y1", "y6")]
    for crop in CROPS:
        assert sum(len(spec[crop]) for kind in ("y1", "y6") for spec in depth[kind].values()) == 15, crop


def test_depth_ladder_names_and_points():
    ladder = load_ladder(ROOT / LADDER)
    assert list(ladder["rungs"]) == ["L3", "L4_drift_ld", "LVI_ld", "L5", "L5VI_ld"]
    assert list(ladder["baselines"]) == ["L4_drift", "LVI", "L5VI", "L3_dd", "L3_ld", "LVI_dd", "LVI_ddd", "LVI_ddv",
                                         "L4_drift_ld_orc", "L4_drift_ldn", "LVI_ldn", "B_drift", "B_drift_ld",
                                         "B_drift_ld_noise"]
    assert ladder["val_only"] == []
    every = {**ladder["rungs"], **ladder["baselines"]}
    noise = {name: rung.get("noise") for name, rung in every.items()}
    lidar = {k: v for k, v in CONTRACT.items() if not k.startswith("lidar_extr_")}
    file_cam = {**START, **RIG, **DEPTH, **BELIEF}          # the camera side of a rung on a file pose source
    assert noise["L4_drift_ld"] == {**J, **CONTRACT}
    assert noise["LVI_ld"] == noise["L5VI_ld"] == {**file_cam, **CONTRACT}
    assert (noise["L4_drift"], noise["LVI"], noise["L5VI"]) == (J, file_cam, file_cam)
    assert noise["L4_drift_ldn"] == {**J, **lidar} and noise["LVI_ldn"] == {**file_cam, **lidar}
    assert {"imu_extr_trans_mm", "imu_extr_rot_deg"} <= set(noise["L4_drift_ldn"]) & set(noise["LVI_ldn"])
    assert noise["L4_drift_ld_orc"] == noise["L4_drift_ld"]
    ld, ldn = {"depth": "lidar", "lidar_file": "depth"}, {"depth": "lidar", "lidar_file": "nav"}
    assert {name: rung["lift"] for name, rung in every.items() if "lift" in rung} == {
        "L4_drift_ld": ld, "LVI_ld": ld, "L5VI_ld": ld, "L3_dd": {"depth": "det_depth"}, "L3_ld": ld,
        "L4_drift_ld_orc": {"depth": "lidar_oracle", "lidar_file": "depth"}, "L4_drift_ldn": ldn, "LVI_ldn": ldn,
        "B_drift_ld": ld, "B_drift_ld_noise": ld}
    assert not any({"lidar", "lidar_files"} & set(rung.get("lift", {})) for rung in every.values())
    assert (every["LVI_ddv"]["pose"]["cache"], noise["LVI_ddv"]) == ("vo_dd", file_cam)
    assert (every["LVI_ddd"]["pose"]["cache"], noise["LVI_ddd"]) == ("vo", {**START, **RIG, **BELIEF})
    for name in ("B_drift", "B_drift_ld", "B_drift_ld_noise"):
        assert not set(J) & set(noise[name] or {}), name
    assert noise["B_drift_ld_noise"] == CONTRACT


def test_depth_ladder_rungs_build_on_the_val_configs():
    ladder = load_ladder(ROOT / LADDER)
    names = [*ladder["rungs"], *ladder["baselines"]]
    for crop in CROPS:
        cfg = _val(crop)
        check_ladder_request(cfg, ladder, names)
        for name in names:
            lift = rung_config(cfg, ladder, name).lift
            assert (lift.lidar, lift.lidar_files) == (cfg.lift.lidar, cfg.lift.lidar_files), (crop, name)


@pytest.mark.parametrize(("name", "bases", "n_cells"), GRIDS, ids=[g[0] for g in GRIDS])
def test_depth_grids_build_on_their_bases(name, bases, n_cells):
    ladder = load_ladder(ROOT / LADDER)
    grid = load_grid(ROOT / _grid_path(name))
    assert grid.bases == bases
    check_test_grid(grid, ladder)
    for crop in CROPS:
        cells = sweep_cells(_val(crop), grid, ladder)
        assert len(cells) == n_cells, crop
        assert [c.overridden for c in cells] == [()] * n_cells, crop


def test_depth_grid_commons_repeat_the_judged_point():
    profiles = load_grid(EVAL / "sweeps_pose" / "drift_profiles.yaml")
    assert profiles.common == {"noise": J}
    for name in ("depth_cam", "depth_forms"):
        assert load_grid(ROOT / _grid_path(name)).common == {"noise": J}, name
    for name in ("depth_lidar", "extr_ld", "profiles_ld"):
        assert load_grid(ROOT / _grid_path(name)).common == {"noise": {**J, **CONTRACT}}, name
    combos = load_grid(ROOT / _grid_path("profiles_ld")).combos
    assert combos == {name: profiles.combos[name] for name in ("D_lio", "G_end", "D_lio_G_end")}


def test_depth_val_configs_follow_the_pose_val_configs():
    for crop in CROPS:
        pose = load_yaml_model(EVAL / f"{crop}_val_pose_c2.yaml", EvalAppConfig).model_dump()
        cfg = _val(crop)
        assert isinstance(cfg.lift.lidar, LidarLiftConfig)
        dump = cfg.model_dump()
        assert dump["lift"]["lidar_files"] == FILES[crop]
        expected = copy.deepcopy(pose)
        expected["run_name"] = f"{crop}_val_depth"
        expected["lift"].update(lidar=dump["lift"]["lidar"], lidar_files=FILES[crop])
        expected["pose"].update(config_label="c2l", imu_files={"imu_ext": pose["pose"]["imu_files"]["imu_ext"]})
        expected["output_dir"] = "runs/eval_depth_val"
        assert dump == expected, crop


def _unpinned(dump: dict) -> dict:
    """`dump` without the digests and simulator commits of its LiDAR files."""
    out = copy.deepcopy(dump)
    for files in out["lift"]["lidar_files"].values():
        del files["digests"], files["sim_commits"]
    return out


def test_frozen_depth_templates_follow_val():
    test_seeds = load_yaml_model(ROOT / SPLITS, SplitsConfig).test
    for crop in CROPS:
        val, cfg = _val(crop), _template(crop)
        runs = [f"~/datasets/orchard_synth/{crop}/test_s{seed}_shake" for seed in test_seeds]
        expected = _unpinned(val.model_dump())
        expected.update(run_name=f"{crop}_test_depth", split="test", tuning=False, runs=runs,
                        output_dir="runs/eval_depth", protect=cfg.protect.model_dump())
        assert _unpinned(cfg.model_dump()) == expected, crop
        assert (val.output_dir, cfg.output_dir) == ("runs/eval_depth_val", "runs/eval_depth")
        assert cfg.lift.lidar == val.lift.lidar


def test_frozen_depth_protect_pins_the_committed_files():
    for crop in CROPS:
        section = _template(crop).protect
        assert (section.ladder.path, section.splits.path, section.criteria.path) == (LADDER, SPLITS, CRITERIA)
        for pinned in (section.ladder, section.splits, section.criteria):
            assert pinned.sha256 == _sha256(pinned.path), (crop, pinned.path)
        assert section.splits.sha256 == SPLITS_SHA256
        assert section.grids == {_grid_path(name): _sha256(_grid_path(name)) for name, _, _ in GRIDS}
        assert list(section.grids) == [_grid_path(name) for name, _, _ in GRIDS]
        assert section.judged == ["L4_drift_ld", "LVI_ld", "L5VI_ld"]
        assert section.ladders == [DEPTH_LIST]
        assert section.output_dir == "runs/eval_depth"


def _rung_checks(cfg: EvalAppConfig, ladder: dict, configs: dict, baseline_cells: list[str]) -> None:
    """The held-out checks of the rung or grid cell configs `configs` (name -> (its ladder rung, config)), in the
    order the eval commands run them."""
    plain = {name: c for name, (_, c) in configs.items()}
    check_test_rung_protect(cfg, plain)
    check_test_rungs(cfg, plain)
    check_test_pose_rungs(cfg, configs, ladder)
    check_test_lift(cfg, plain)
    check_test_lidar_rungs(cfg, plain, baseline_cells=baseline_cells)


def test_frozen_depth_requests_pass_the_held_out_checks():
    ladder = load_ladder(ROOT / LADDER)
    criteria = yaml.safe_load((ROOT / CRITERIA).read_text(encoding="utf-8"))
    files = {"--ladder": Path(LADDER), "--splits": Path(SPLITS), "--criteria": Path(CRITERIA)}
    pins = {"ladder_sha256": _sha256(LADDER), "criteria_sha256": _sha256(CRITERIA), "splits_sha256": _sha256(SPLITS),
            "with_2x2": False, "subsets": False, "criteria": criteria}
    for crop in CROPS:
        cfg = _template(crop)
        check_test_frozen_request(cfg, kind="ladder", files=files, grid_sha256=None, rungs=DEPTH_LIST, **pins)
        check_ladder_request(cfg, ladder, DEPTH_LIST)
        _rung_checks(cfg, ladder, {name: (name, rung_config(cfg, ladder, name)) for name in DEPTH_LIST}, [])
        n_cells = 0
        for name, _, _ in GRIDS:
            rel = _grid_path(name)
            check_test_frozen_request(cfg, kind="sweep", files={**files, "--grid": Path(rel)}, grid_sha256=_sha256(rel),
                                      rungs=None, **pins)
            cells = sweep_cells(cfg, load_grid(ROOT / rel), ladder)
            _rung_checks(cfg, ladder, {f"{c.point.name} @ {c.base}": (c.base, c.cfg) for c in cells},
                         [f"{c.point.name} @ {c.base}" for c in cells if c.point.name == "baseline"])
            n_cells += len(cells)
        assert n_cells == 46, crop
