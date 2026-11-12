"""The settings fixed on val for the held-out pose files and the pose values chosen on val: each crop's frozen
odometry settings and their noiseless-depth twin (the pose cache folders of the val configs), the frozen copy of
the judged IMU profile, and the odometry drift profile and odometry factor sigmas that the pose ladder and the drift
profile grid carry."""

from pathlib import Path

import pytest

from counting.config.eval_schema import EvalAppConfig
from counting.config.hashing import config_hash
from counting.config.loader import load_yaml_model
from counting.config.pose_schema import ImuConfig, VoConfig
from counting.evaluation.runner import load_ladder, rung_config
from counting.evaluation.sweep import load_grid

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "configs" / "eval"
POSE = EVAL / "sweeps_pose"
POSE_CFG = ROOT / "configs" / "pose"
LADDER_PATH = POSE / "ladder_pose.yaml"
# The odometry knobs chosen on val for each crop; every other setting keeps configs/pose/vo.yaml.
CHOSEN_VO = {"apple": {"depth_nb": 1, "depth_max_m": 3.0}, "pear": {"depth_nb": 2, "depth_max_m": 10.0}}
# The odometry drift profile fitted to the val odometry, one value for both crops.
D_VO = {"drift_scale_sigma": 0.0054, "drift_rate_deg_per_m": 0.036, "drift_step_trans_mm": 0.78,
        "drift_step_rot_deg": 0.019}
# The odometry factor sigmas (mm, deg) chosen on val for every rung that fuses a file pose source.
SIGMA = {"vo_sigma_trans_mm": 0.99, "vo_sigma_rot_deg": 0.021}


def test_frozen_odometry_configs_hold_the_chosen_knobs():
    start = load_yaml_model(POSE_CFG / "vo.yaml", VoConfig).model_dump()
    for crop, knobs in CHOSEN_VO.items():
        vo = load_yaml_model(POSE_CFG / "frozen" / f"{crop}_vo.yaml", VoConfig)
        dd = load_yaml_model(POSE_CFG / "frozen" / f"{crop}_vo_dd.yaml", VoConfig)
        assert vo.model_dump() == {**start, **knobs} and vo.noise_seed == 0 and vo.output_dir == "runs/pose"
        assert dd.model_dump() == {**vo.model_dump(), "depth_k": 0.0}
        for label in ("c2", "c1", "c3", "c1v"):
            val = load_yaml_model(EVAL / f"{crop}_val_pose_{label}.yaml", EvalAppConfig)
            assert (val.pose.caches["vo"].dir, val.pose.caches["vo_dd"].dir) == \
                (f"runs/pose/{config_hash(vo)}", f"runs/pose/{config_hash(dd)}"), (crop, label)


def test_frozen_imu_config_is_the_judged_profile():
    frozen = load_yaml_model(POSE_CFG / "frozen" / "imu_ext.yaml", ImuConfig)
    assert frozen.model_dump() == load_yaml_model(POSE_CFG / "imu_ext.yaml", ImuConfig).model_dump()


def test_ladder_carries_the_values_fitted_and_chosen_on_val():
    ladder = load_ladder(LADDER_PATH)
    assert ladder["noise_blocks"]["vo_dvo"] == D_VO
    assert {k: ladder["baselines"]["L4_drift_dvo"]["noise"][k] for k in D_VO} == D_VO
    combos = load_grid(POSE / "drift_profiles.yaml").combos
    assert combos["D_vo"] == {"noise": D_VO}
    assert combos["D_vo_x3"]["noise"] == pytest.approx({k: 3 * v for k, v in D_VO.items()})
    base = load_yaml_model(EVAL / "apple_val_pose_c2.yaml", EvalAppConfig)
    for name in [*ladder["rungs"], *ladder["baselines"]]:
        pose = rung_config(base, ladder, name).pose
        fused_file = pose.source == "file" and pose.fuse != "none"
        sigma = {} if pose.vio is None else {k: v for k, v in pose.vio.model_dump().items() if k.startswith("vo_sigma")}
        assert sigma == (SIGMA if fused_file else {}), name
    assert rung_config(base, ladder, "LVI_stop5").pose.vio.stop_s == 5.0
