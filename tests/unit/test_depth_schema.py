"""LiDAR box depth settings, the LiDAR noise and extrinsic keys and the depth error form keys: their start values,
the combinations a config may not hold, what counts as noise, and the hashes of configs that do not use them."""

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from counting.config.eval_schema import (
    DEPTH_FORM_KEYS, DEPTH_NOISE_LEVEL_KEYS, LIDAR_DEPTHS, LIDAR_EXTR_KEYS, LIDAR_NOISE_KEYS, EvalAppConfig,
    LidarFiles, NoiseConfig,
)
from counting.config.hashing import config_hash
from counting.config.loader import load_yaml_model
from counting.config.pose_schema import ImuConfig, VoConfig

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "configs" / "eval"
POSE_FROZEN = ROOT / "configs" / "pose" / "frozen"
SIM_SHA = "ab3576dc" * 8
COMMIT = "4616644125" * 4
FILES = {"depth": {"dir": "runs/lidar/depth", "unit": "depth", "sim_config_sha256": SIM_SHA, "sim_commits": [COMMIT]},
         "nav": {"dir": "runs/lidar/nav", "unit": "nav", "sim_config_sha256": "374e38b8" * 8,
                 "sim_commits": [COMMIT]}}
CAMERA_NOISE = {"pose_trans_sigma_m": 0, "pose_rot_sigma_deg": 0, "pass_bias_m": 0, "depth_rel_sigma": 0,
                "depth_dropout": 0}


def _cfg(lift: dict | None = None, noise: dict | None = None, pose: dict | None = None) -> EvalAppConfig:
    """apple_dev.yaml with `lift` and `noise` merged into its sections and, when given, the pose section `pose`."""
    raw = yaml.safe_load((EVAL / "apple_dev.yaml").read_text(encoding="utf-8"))
    raw["lift"] = {**raw["lift"], **(lift or {})}
    raw["noise"] = {**raw["noise"], **(noise or {})}
    if pose is not None:
        raw["pose"] = pose
    return EvalAppConfig.model_validate(raw)


def _lidar(file: str = "depth", depth: str = "lidar", **lidar) -> dict:
    """A lift section on the LiDAR file `file` of `FILES`, with the LiDAR knobs `lidar`."""
    return {"depth": depth, "lidar": lidar, "lidar_files": FILES, "lidar_file": file}


def _refused(message: str, **kw) -> None:
    with pytest.raises(ValidationError, match=re.escape(message)):
        _cfg(**kw)


def test_lidar_section_takes_its_start_values():
    assert LIDAR_DEPTHS == ("lidar", "lidar_oracle")
    lidar = _cfg(_lidar()).lift.lidar
    assert (lidar.match, lidar.window_s, lidar.combine, lidar.min_points, lidar.gate_k, lidar.box_frac,
            lidar.far_min_frac, lidar.far_gap_m, lidar.belief_cam_rel, lidar.belief_range_m) == \
        ("gate", 0.5, "ivw", 3, 3.0, 0.5, 0.25, 0.03, 0.02, 0.02)
    dump = _cfg().model_dump()
    assert set(dump["lift"]) == {"depth", "box_frac", "far_scale", "far_gap_m", "far_min_frac", "center_offset_m"}
    assert set(dump["noise"]) == {*CAMERA_NOISE, "seeds"}
    assert _cfg(_lidar(match="map", window_s="pass")).lift.lidar.window_s == "pass"
    _refused("lift.lidar.window_s=pass needs lift.lidar.match=map", lift=_lidar(window_s="pass"))
    _refused("greater than or equal to 1", lift=_lidar(min_points=0))
    _refused("greater than 0", lift=_lidar(gate_k=0))


def test_lidar_files_and_the_chosen_file():
    files = LidarFiles(**FILES["depth"], digests={"apple_s5_0a1b2c3d": "0f" * 32})
    assert (files.unit, files.sim_commits, files.digests) == ("depth", [COMMIT], {"apple_s5_0a1b2c3d": "0f" * 32})
    with pytest.raises(ValidationError, match=re.escape("must be a 64-character lowercase hex digest")):
        LidarFiles(**FILES["depth"], digests={"apple_s5_0a1b2c3d": "0" * 63})
    with pytest.raises(ValidationError, match="sim_commits"):
        LidarFiles(**{**FILES["depth"], "sim_commits": ["0" * 39]})
    _refused("lift.depth=lidar needs lift.lidar, lift.lidar_files and lift.lidar_file",
             lift={**_lidar(), "lidar_file": None})
    _refused("lift.lidar_file=x is not one of lift.lidar_files [depth, nav]", lift=_lidar(file="x"))
    _refused("lift.lidar_file needs lift.lidar_files", lift={"lidar_file": "depth"})
    camera = _cfg(_lidar(depth="box_far"))                 # a camera-only rung keeps its base config's LiDAR settings
    assert camera.lift.depth == "box_far" and camera.lift.lidar_unit() == "depth"
    assert _cfg(_lidar(file="nav")).lift.lidar_unit() == "nav" and _cfg().lift.lidar_unit() is None


def test_lidar_noise_keys_need_a_lidar_depth():
    assert LIDAR_NOISE_KEYS == ("lidar_range_sigma_m", "lidar_range_sigma_near_m", "lidar_near_range_m",
                                "lidar_ang_sigma_deg", "lidar_dropout", "lidar_range_bias_m")
    assert LIDAR_EXTR_KEYS == ("lidar_extr_trans_mm", "lidar_extr_rot_deg")
    _refused("noise.lidar_range_sigma_m needs lift.depth=lidar or lidar_oracle",
             lift=_lidar(depth="box_far"), noise={"lidar_range_sigma_m": 0.02})
    _refused("noise.lidar_extr_rot_deg needs a depth-unit LiDAR file (lift.lidar_file)",
             lift=_lidar(file="nav"), noise={"lidar_extr_rot_deg": 0.3})
    _refused("less than or equal to 1", lift=_lidar(), noise={"lidar_dropout": 1.2})
    assert _cfg(_lidar(), {"lidar_range_bias_m": -0.01}).noise.lidar_range_bias_m == -0.01
    assert _cfg(_lidar(depth="lidar_oracle"), {"lidar_extr_rot_deg": 0.3}).noise.lidar_extr_rot_deg == 0.3
    _refused("noise.lidar_range_sigma_near_m and noise.lidar_near_range_m go together",
             lift=_lidar(), noise={"lidar_near_range_m": 1.0})
    _refused("noise.drift_time_offset_s is not supported with lift.depth=lidar",
             lift=_lidar(), noise={"drift_time_offset_s": 0.01}, pose={})


def test_imu_extrinsic_keys_serve_the_navigation_lidar():
    cfg = _cfg(_lidar(file="nav"), {"imu_extr_trans_mm": 10}, pose={})
    assert cfg.pose.fuse == "none" and cfg.noise.imu_extr_trans_mm == 10
    _refused("noise.imu_extr_trans_mm needs pose.fuse=imu or a navigation-unit LiDAR depth",
             lift=_lidar(), noise={"imu_extr_trans_mm": 10}, pose={})
    _refused("noise.imu_extr_trans_mm needs pose.fuse=imu", noise={"imu_extr_trans_mm": 10}, pose={})
    _refused("noise.imu_gyro_n needs pose.fuse=imu", lift=_lidar(file="nav"), noise={"imu_gyro_n": 0.001}, pose={})


def test_depth_form_keys():
    assert DEPTH_FORM_KEYS == ("depth_persist_sigma", "depth_persist_len_m", "depth_scale_sigma", "depth_bias_rel")
    forms = {"depth_persist_sigma": 0.01, "depth_persist_len_m": 0.5, "depth_scale_sigma": 0.02, "depth_bias_rel": 0.01}
    cfg = _cfg(noise=forms)                                 # no pose section and no LiDAR needed
    assert cfg.pose is None and {k: getattr(cfg.noise, k) for k in DEPTH_FORM_KEYS} == forms
    _refused("noise.depth_persist_sigma and noise.depth_persist_len_m go together",
             noise={"depth_persist_sigma": 0.01})
    _refused("greater than 0", noise={"depth_persist_sigma": 0.01, "depth_persist_len_m": 0})
    assert _cfg(noise={"depth_bias_rel": -0.01}).noise.depth_bias_rel == -0.01
    _refused("greater than -1", noise={"depth_bias_rel": -1})
    _refused("greater than or equal to 0", noise={"depth_scale_sigma": -0.1})


def test_new_keys_count_as_noise():
    assert DEPTH_NOISE_LEVEL_KEYS == ("lidar_range_sigma_m", "lidar_range_sigma_near_m", "lidar_ang_sigma_deg",
                                      "lidar_dropout", "lidar_range_bias_m", "lidar_extr_trans_mm",
                                      "lidar_extr_rot_deg", "depth_persist_sigma", "depth_scale_sigma",
                                      "depth_bias_rel")
    assert not _cfg(_lidar()).is_noisy() and _cfg(_lidar(), {"lidar_range_sigma_m": 0.02}).is_noisy()
    shapes = {"lidar_near_range_m": 1.0, "depth_persist_len_m": 0.5}
    assert NoiseConfig(**dict.fromkeys(DEPTH_NOISE_LEVEL_KEYS, 0.0), **shapes).is_zero()
    assert NoiseConfig(**shapes).is_zero() and not NoiseConfig(depth_bias_rel=0.01).is_zero()
    assert not NoiseConfig(lidar_range_bias_m=-0.01).is_zero()
    extr = _cfg(_lidar(), {**CAMERA_NOISE, "lidar_extr_rot_deg": 0.3})
    assert not extr.noise.is_zero() and extr.is_noisy()
    every = {**dict.fromkeys(DEPTH_NOISE_LEVEL_KEYS, 0.0), **shapes}
    assert NoiseConfig(**dict(reversed(every.items()))).depth_keys() == [*LIDAR_NOISE_KEYS, *LIDAR_EXTR_KEYS,
                                                                        *DEPTH_FORM_KEYS]
    assert NoiseConfig(depth_bias_rel=0.0, lidar_dropout=0.0).depth_keys() == ["lidar_dropout", "depth_bias_rel"]
    assert NoiseConfig().depth_keys() == [] and NoiseConfig(lidar_dropout=0.0).pose_keys() == []


# The config hashes of the frozen odometry settings and the frozen IMU profile (the folder names of their files).
FIXED_POSE_HASHES = {"apple_vo": (VoConfig, "b16b37e34b609dd2"), "apple_vo_dd": (VoConfig, "52ad7d4423dea8fb"),
                     "pear_vo": (VoConfig, "a45a4d9dca516ca8"), "pear_vo_dd": (VoConfig, "510f2d4b21947576"),
                     "imu_ext": (ImuConfig, "682fc7ac53717171")}


def test_fixed_pose_settings_keep_their_hashes():
    for name, (model, digest) in FIXED_POSE_HASHES.items():
        assert config_hash(load_yaml_model(POSE_FROZEN / f"{name}.yaml", model)) == digest, name
