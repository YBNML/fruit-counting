"""The committed pose criteria, the rules for the values chosen on val, the generated IMU profile and the fixed
fusion settings: their rows and values, and the IMU values held equal across the profile and the pose ladder."""

from pathlib import Path

import pytest
import yaml

from counting.config.loader import load_yaml_model
from counting.config.pose_schema import FuseFixedConfig, ImuConfig
from counting.evaluation.runner import load_ladder

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "configs" / "eval"
POSE = ROOT / "configs" / "pose"
VO_KEYS = ("drift_scale_sigma", "drift_rate_deg_per_m", "drift_step_trans_mm", "drift_step_rot_deg")

# crop -> (section, rung, metric) -> upper bound
CRITERIA = {
    "pear": {
        ("y1", "LVI", "e1_median"): 0.02,
        ("y1", "L5VI", "e1_median"): 0.08,
        ("y6", "L4_drift", "row_abs_e_pipe"): 0.05,
        ("y6", "L4_drift", "tree_mae_over_mean"): 0.06,
        ("y6", "L4_drift", "miss_rate"): 0.04,
        ("y6", "L4_drift", "excess_rate"): 0.04,
        ("y6", "LVI", "row_abs_e_pipe"): 0.05,
        ("y6", "LVI", "tree_mae_over_mean"): 0.06,
        ("y6", "LVI", "miss_rate"): 0.04,
        ("y6", "LVI", "excess_rate"): 0.04,
        ("y6", "LVI", "attribution_error"): 0.02,
        ("y6", "L5VI", "row_abs_e_pipe"): 0.08,
        ("y6", "L5VI", "tree_mae_over_mean"): 0.12,
        ("y6", "L5VI", "miss_rate"): 0.08,
        ("y6", "L5VI", "excess_rate"): 0.08,
    },
    "apple": {
        ("y1", "LVI", "e1_median"): 0.04,
        ("y1", "L5VI", "e1_median"): 0.10,
        ("y6", "L4_drift", "row_abs_e_pipe"): 0.10,
        ("y6", "L4_drift", "tree_mae_over_mean"): 0.10,
        ("y6", "L4_drift", "miss_rate"): 0.08,
        ("y6", "L4_drift", "excess_rate"): 0.08,
        ("y6", "LVI", "row_abs_e_pipe"): 0.10,
        ("y6", "LVI", "tree_mae_over_mean"): 0.10,
        ("y6", "LVI", "miss_rate"): 0.08,
        ("y6", "LVI", "excess_rate"): 0.08,
        ("y6", "LVI", "attribution_error"): 0.02,
        ("y6", "L5VI", "row_abs_e_pipe"): 0.12,
        ("y6", "L5VI", "tree_mae_over_mean"): 0.15,
        ("y6", "L5VI", "miss_rate"): 0.12,
        ("y6", "L5VI", "excess_rate"): 0.12,
    },
}


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _ladder():
    return load_ladder(EVAL / "sweeps_pose" / "ladder_pose.yaml")


def test_pose_criteria_rows():
    criteria = _yaml(EVAL / "criteria_pose.yaml")
    assert set(criteria) == {"judged_config", "y1", "y6"}
    assert criteria["judged_config"] == "c2"
    rows: dict[str, dict[tuple[str, str, str], dict]] = {}
    for section in ("y1", "y6"):
        for rung, crops in criteria[section].items():
            for crop, metrics in crops.items():
                for metric, bound in metrics.items():
                    rows.setdefault(crop, {})[(section, rung, metric)] = bound
    assert {crop: len(r) for crop, r in rows.items()} == {"pear": 15, "apple": 15}
    assert rows == {crop: {key: {"le": v} for key, v in want.items()} for crop, want in CRITERIA.items()}
    rungs = {rung for section in ("y1", "y6") for rung in criteria[section]}
    assert rungs == {"L4_drift", "LVI", "L5VI"} and not rungs & {"L3", "L5", "L4_base"}


def test_selection_rules_shape():
    rules = _yaml(EVAL / "selection_pose.yaml")
    assert set(rules) == {"order", "fixed", "items", "vetoes", "detail", "tuning"}
    items = rules["items"]
    assert set(items) == set(rules["order"])
    assert items["pitch_pair"]["start"] == {"apple": [10, 50], "pear": [20, 70]}
    assert items["single_pitch"]["start"] == {"apple": 30, "pear": 45}
    ladder = _ladder()
    assert items["d_vo"]["start"] == dict(zip(VO_KEYS, (0.003, 0.03, 0.7, 0.013)))
    vo = ladder["noise_blocks"]["vo"]
    assert items["fusion_sigma"]["start"] == {"vo_sigma_trans_mm": vo["drift_step_trans_mm"],
                                              "vo_sigma_rot_deg": vo["drift_step_rot_deg"]}
    align = items["align"]
    assert align["start"]["pair_radius_m"] == 0.5 and align["fixed"] == {"min_pairs": 20}
    assert align["candidates"]["segment_m"] == [3.0, 6.0]
    # the first alignment step vetoes on the signed row e_pipe and on a change in either direction
    veto = align["steps"][0]
    assert veto["rung"] == "L3_al" and "either direction" in veto["veto"] and "y6.row.e_pipe" in veto["of"]
    assert veto["limits"] == {"apple": {"row_e_pipe": 0.01, "miss_rate": 0.015, "excess_rate": 0.015},
                              "pear": {"row_e_pipe": 0.005, "miss_rate": 0.005, "excess_rate": 0.005}}


def test_fusion_settings():
    fixed = load_yaml_model(POSE / "fuse_fixed.yaml", FuseFixedConfig)
    assert fixed.model_dump() == {
        "estimator": {"gyro_n": 7.85e-4, "acc_n": 9.81e-3, "gyro_w": 1.82e-5, "acc_w": 4.75e-4},
        "start_speed_mps": 0.5, "start_speed_sigma_mps": 0.2, "start_roll_pitch_sigma_deg": 5.0,
        "shake_share": {"side_m": 0.02, "height_m": 0.01, "along_m": 0.01, "yaw_deg": 0.5},
        "vo_sigma_floor_trans_mm": 0.1, "vo_sigma_floor_rot_deg": 0.001,
    }


def test_imu_profile_matches_the_ladder():
    imu = load_yaml_model(POSE / "imu_ext.yaml", ImuConfig)
    assert imu.profile == "I_ext"
    assert (imu.rate_hz, imu.acc_range_g, imu.noise_seeds) == (200.0, 4.0, [0, 1, 2])
    noise = {f"imu_{k}": v for k, v in imu.noise.model_dump().items() if v}
    assert noise == _ladder()["noise_blocks"]["imu"]
