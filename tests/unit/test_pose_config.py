"""Pose, alignment and drift/IMU noise settings: defaults, dumps, and the combinations a config may not hold."""

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from counting.config.eval_schema import (
    DRIFT_KEYS, GPS_KEYS, IMU_BELIEF_KEYS, IMU_LEVEL_KEYS, POSE_NOISE_KEYS, AlignConfig, EvalAppConfig, NoiseConfig,
    PoseConfig,
)
from counting.config.hashing import config_hash
from counting.config.loader import load_yaml_model
from counting.config.pose_schema import FuseFixedConfig, ImuConfig
from counting.evaluation.runner import _patched

pytestmark = pytest.mark.pose

EVAL = Path(__file__).resolve().parents[2] / "configs" / "eval"
FILES = {"vo": {"dir": "runs/pose/x"}}
IMUS = {"i": {"dir": "runs/imu/x"}}
VO = {"drift_scale_sigma": 0.0045, "drift_rate_deg_per_m": 0.045, "drift_step_trans_mm": 2.1, "drift_step_rot_deg": 0.039}
START, RIG = {"pass_side_trans_m": 0.05, "pass_side_rot_deg": 0.5}, {"rig_extr_trans_mm": 2.0, "rig_extr_rot_deg": 0.1}
IMU = {"imu_gyro_n": 1.16e-3, "imu_acc_n": 4.97e-3, "imu_gyro_w": 1.82e-5, "imu_acc_w": 4.75e-4,
       "imu_gyro_b0_dps": 1.0, "imu_acc_b0_mg": 20.0, "imu_temp_k": 1.0}
BELIEF = {"imu_time_offset_ms": 0.0, "imu_extr_trans_mm": 2.0, "imu_extr_rot_deg": 0.1}


def _dev():
    return load_yaml_model(EVAL / "apple_dev.yaml", EvalAppConfig)


def test_new_settings_stay_out_of_stage_one_dumps():
    cfg = _dev()
    dump = cfg.model_dump()
    assert "pose" not in dump and "align" not in dump
    assert set(dump["noise"]) == {"pose_trans_sigma_m", "pose_rot_sigma_deg", "pass_bias_m", "depth_rel_sigma",
                                  "depth_dropout", "seeds"}
    assert config_hash(_patched(cfg, {"pose": None, "align": None})) == config_hash(cfg) == "38b74b45e42d9213"
    pose = _patched(cfg, {"pose": {"rig_ref": 10}, "noise": {"drift_step_trans_mm": 0}})
    assert config_hash(pose) != config_hash(cfg)
    assert EvalAppConfig.model_validate(pose.model_dump()) == pose


def test_pose_noise_keys_count_as_noise():
    assert len(POSE_NOISE_KEYS) == len(set(POSE_NOISE_KEYS)) == 26
    assert (len(DRIFT_KEYS), len(IMU_LEVEL_KEYS), len(IMU_BELIEF_KEYS), len(GPS_KEYS)) == (12, 9, 3, 2)
    for key in POSE_NOISE_KEYS:
        assert NoiseConfig(**{key: 0.0}).is_zero() and not NoiseConfig(**{key: 0.5}).is_zero(), key
        assert NoiseConfig().factor(key) == 0.0 and NoiseConfig(**{key: 0.0}).pose_keys() == [key], key
    assert NoiseConfig().pose_keys() == []


def test_pose_and_align_defaults():
    assert PoseConfig().model_dump() == {"source": "truth", "anchor": "nominal", "rig": False, "fuse": "none"}
    assert AlignConfig().model_dump() == {"mode": "none"}


CONTRADICTIONS = [
    ({"noise": {"drift_step_trans_mm": 1}}, "noise.drift_step_trans_mm needs a pose section"),
    ({"align": {"mode": "fit_truth_pass"}}, "align needs a pose section"),
    ({"pose": {"rig": True}}, "need pose.rig_ref"),
    ({"pose": {"rig_ref": 50}, "passes": {"pitches": [10, 30]}}, "pose.rig_ref=50 is not one of passes.pitches"),
    ({"pose": {"rig_ref": 10}, "noise": {"imu_gyro_n": 0.001}}, "noise.imu_gyro_n needs pose.fuse=imu"),
    ({"pose": {"rig_ref": 10, "source": "file", "cache": "vo", "caches": FILES, "fuse": "imu", "imu": "i",
               "imu_files": IMUS}, "noise": {"imu_acc_b0_mg": 20}}, "noise.imu_acc_b0_mg needs pose.source=truth"),
    ({"pose": {"fuse": "vo"}}, "pose.fuse=vo needs pose.source=file"),
    ({"pose": {"rig_ref": 10}, "noise": {"rig_extr_trans_mm": 2}}, "noise.rig_extr_trans_mm needs pose.rig=true"),
    ({"pose": {"rig": True, "rig_ref": 10}, "noise": {"pass_pitch_trans_m": 0.02}},
     "noise.pass_pitch_trans_m needs pose.rig=false"),
    ({"pose": {"anchor": "first_truth"}, "noise": {"pass_side_trans_m": 0.05}},
     "noise.pass_side_trans_m is a start error of pose.anchor=nominal"),
    ({"pose": {"source": "file"}}, "pose.source=file needs pose.cache, one of pose.caches"),
    ({"pose": {"cache": "vo"}}, "pose.cache needs pose.source=file"),
    ({"pose": {"source": "file", "cache": "vo", "caches": FILES, "fuse": "imu"}},
     "pose.source=file with pose.fuse=imu needs pose.imu"),
    ({"pose": {"imu": "i"}}, "pose.imu needs pose.source=file, pose.fuse=imu and a name of pose.imu_files"),
    ({"pose": {"vio": {"stop_s": 5}}}, "pose.vio needs pose.fuse=vo or imu"),
    ({"pose": {"fuse": "imu", "vio": {"vo_sigma_trans_mm": 2.1}}},
     "pose.vio.vo_sigma_trans_mm needs pose.source=file"),
    ({"pose": {"rig_ref": 10}, "align": {"mode": ["tracks"], "stress_row_bias_m": 0.05}},
     "align.stress_row_bias_m needs row_plane in align.mode"),
    ({"pose": {"rig_ref": 10}, "align": {"mode": ["row_plane"], "stress_pair_keep": 0.5}},
     "align.stress_pair_keep needs tracks in align.mode"),
    ({"pose": {"rig_ref": 10}, "align": {"mode": ["tracks", "tracks"]}}, "align.mode repeats a method"),
    ({"pose": {"rig_ref": 10}, "noise": {"gps_end_h_m": 0.02, "gps_end_v_m": 0.03}},
     "noise.gps_end_h_m needs pose.fuse=imu"),
    ({"pose": {"rig_ref": 10, "source": "file", "cache": "vo", "caches": FILES, "fuse": "imu", "imu": "i",
               "imu_files": IMUS}, "noise": {"gps_end_h_m": 0.02, "gps_end_v_m": 0.03}},
     "noise.gps_end_h_m needs pose.source=truth"),
    ({"pose": {"rig_ref": 10, "fuse": "imu", "anchor": "first_truth"},
      "noise": {"gps_end_h_m": 0.02, "gps_end_v_m": 0.03}}, "noise.gps_end_h_m needs pose.anchor=nominal"),
    ({"pose": {"rig_ref": 10, "fuse": "imu"}, "noise": {"gps_end_h_m": 0.02}},
     "noise.gps_end_h_m and noise.gps_end_v_m go together"),
    ({"pose": {"rig_ref": 10, "fuse": "imu"}, "noise": {"gps_end_h_m": 0.0, "gps_end_v_m": 0.03}},
     "noise.gps_end_h_m must be > 0"),
]


@pytest.mark.parametrize("patch, message", CONTRADICTIONS)
def test_pose_settings_reject_contradictions(patch, message):
    with pytest.raises(ValidationError, match=re.escape(message)):
        _patched(_dev(), patch)


SUPPORTED = [
    ({"pose": {"rig": True, "rig_ref": 10, "fuse": "imu"}, "align": {"mode": ["tracks", "row_plane"]},
      "noise": {**VO, **START, **RIG, **IMU, **BELIEF}}, True),
    ({"pose": {"rig_ref": 10, "rig": True, "source": "file", "cache": "vo", "caches": FILES, "fuse": "imu",
               "imu": "i", "imu_files": IMUS}, "noise": {**START, **RIG, **BELIEF}}, True),
    ({"pose": {"rig_ref": 10, "fuse": "imu"}, "noise": {"pass_pitch_trans_m": 0.02, "pass_pitch_rot_deg": 0.25, **IMU}},
     True),
    ({"pose": {"rig_ref": 10, "rig": True, "anchor": "first_truth"}, "noise": {**VO, **RIG}}, True),
    ({"pose": {"rig_ref": 10, "fuse": "imu", "vio": {"stop_s": 5, "estimator_imu": "generator"}}}, False),
    ({"pose": {"rig_ref": 10}, "align": {"mode": ["tracks", "row_plane"], "stress_row_bias_m": 0.05,
                                         "stress_pair_keep": 0.5}}, False),
    ({"pose": {"rig": True, "rig_ref": 10, "fuse": "imu"}, "align": {"mode": ["tracks", "row_plane"]},
      "noise": {**VO, **START, **RIG, **IMU, **BELIEF, "gps_end_h_m": 0.02, "gps_end_v_m": 0.03}}, True),
]


def test_supported_combinations_validate():
    cfg = _dev()
    for patch, noisy in SUPPORTED:
        assert _patched(cfg, patch).is_noisy() is noisy, patch


def test_pose_files_need_hex_digests():
    files = {"vo": {"dir": "x", "digests": {"r": "abc"}}}
    patch = {"pose": {"source": "file", "cache": "vo", "caches": files}}
    with pytest.raises(ValidationError, match=re.escape("digests[r] must be a 64-character lowercase hex digest")):
        _patched(_dev(), patch)
    files["vo"]["digests"]["r"] = "a" * 64
    assert _patched(_dev(), patch).pose.caches["vo"].digests == {"r": "a" * 64}


IMU_NOISE = {"gyro_n": 1.16e-3, "acc_n": 4.97e-3, "gyro_w": 1.82e-5, "acc_w": 4.75e-4, "gyro_b0_dps": 1.0,
             "acc_b0_mg": 20.0, "temp_k": 1.0}
FUSE_FIXED = {"estimator": {"gyro_n": 2.32e-4, "acc_n": 9.94e-4, "gyro_w": 1.82e-6, "acc_w": 4.75e-5},
              "start_speed_mps": 0.5, "start_speed_sigma_mps": 0.2, "start_roll_pitch_sigma_deg": 5.0,
              "shake_share": {"side_m": 0.02, "height_m": 0.01, "along_m": 0.01, "yaw_deg": 0.5},
              "vo_sigma_floor_trans_mm": 0.1, "vo_sigma_floor_rot_deg": 0.001}


def test_imu_and_fusion_schemas():
    imu = ImuConfig(profile="I_ext", noise=IMU_NOISE)
    assert (imu.noise_seeds, imu.rate_hz, imu.seed) == ([0, 1, 2], 200, 0)
    bad = [({"noise": {**IMU_NOISE, "gyro_n": 0}}, "gyro_n"), ({"noise": {**IMU_NOISE, "acc_b0_mg": 0}}, "acc_b0_mg"),
           ({"noise_seeds": [0, 0]}, "noise_seeds repeats a seed"), ({"foo": 1}, "foo")]
    for over, message in bad:
        with pytest.raises(ValidationError, match=re.escape(message)):
            ImuConfig(**{"profile": "I_ext", "noise": IMU_NOISE, **over})
    FuseFixedConfig(**FUSE_FIXED)
    with pytest.raises(ValidationError, match="estimator"):
        FuseFixedConfig(**{k: v for k, v in FUSE_FIXED.items() if k != "estimator"})
