"""The val base configs of the pose ladder and the generated IMU profiles of the report rungs: each val config is a
development config on the shaken val scene with the val pose caches, IMU files, detection caches and alignment
knobs, each passes the ladder request, and the IMU profiles differ from the judged profile in noise levels only."""

from pathlib import Path

import pytest

from counting.config.eval_schema import EvalAppConfig
from counting.config.hashing import config_hash
from counting.config.loader import load_yaml_model
from counting.config.pose_schema import ImuConfig, VoConfig
from counting.evaluation.runner import _patched, check_ladder_request, load_ladder

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "configs" / "eval"
POSE_CFG = ROOT / "configs" / "pose"
LADDER = load_ladder(EVAL / "sweeps_pose" / "ladder_pose.yaml")
# Pose cache and IMU file folders: <output_dir>/<config hash> of each committed settings file; a crop's pose caches
# come from its frozen odometry settings (vo) and their depth_k 0 twin (vo_dd).
VO_DIRS = {crop: {n: f"runs/pose/{config_hash(load_yaml_model(POSE_CFG / 'frozen' / f'{crop}_{n}.yaml', VoConfig))}"
                  for n in ("vo", "vo_dd")} for crop in ("apple", "pear")}
IMU_DIRS = {n: f"runs/imu/{config_hash(load_yaml_model(POSE_CFG / f'{n}.yaml', ImuConfig))}"
            for n in ("imu_ext", "imu_lab", "imu_icm", "imu_ext_x4")}
# The alignment knobs of each crop's val base configs, chosen on val; pear's roll_pitch_sigma_deg stays unset (its
# start value).
KNOBS = {"apple": {"mode": "none", "segment_m": 3.0, "pair_radius_m": 0.5, "roll_pitch_sigma_deg": 1.0, "min_pairs": 20,
                   "oracle_sigma_m": 0.02},
         "pear": {"mode": "none", "segment_m": 6.0, "pair_radius_m": 0.5, "min_pairs": 20, "oracle_sigma_m": 0.02}}
# The tracker prediction chosen on val for each crop.
PREDICT = {"apple": "pose", "pear": "pose"}
# (crop, camera configuration) -> (pass pitches, reference camera pitch)
LAYOUTS = {("apple", "c2"): ([10.0, 50.0], 10.0), ("pear", "c2"): ([20.0, 70.0], 20.0),
           ("apple", "c1"): ([30.0], 30.0), ("pear", "c1"): ([45.0], 45.0),
           ("apple", "c3"): ([10.0, 30.0, 50.0], 10.0), ("pear", "c3"): ([20.0, 45.0, 70.0], 20.0),
           ("apple", "c1v"): ([22.5], 22.5), ("pear", "c1v"): ([45.0], 45.0)}


def test_val_pose_configs_follow_the_development_configs():
    for (crop, label), (pitches, ref) in LAYOUTS.items():
        dev = load_yaml_model(EVAL / f"{crop}_dev_pose_{'c1' if label in ('c1', 'c1v') else 'c2'}.yaml", EvalAppConfig)
        scene = "val_s200_shake_portrait" if label == "c1v" else "val_s200_shake"
        expected = _patched(dev, {
            "run_name": f"{crop}_val_pose_{label}", "split": "val", "runs": [f"~/datasets/orchard_synth/{crop}/{scene}"],
            "passes": {"pitches": pitches}, "detector": {"cache": "runs/detect/88e8c93628216703"},
            "tracker": {"predict": PREDICT[crop]},
            "pose": {"rig_ref": ref, "config_label": label, "caches": {n: {"dir": d} for n, d in VO_DIRS[crop].items()},
                     "imu_files": {n: {"dir": d} for n, d in IMU_DIRS.items()}},
            "align": KNOBS[crop]})
        cfg = load_yaml_model(EVAL / f"{crop}_val_pose_{label}.yaml", EvalAppConfig)
        assert cfg.model_dump() == expected.model_dump(), (crop, label)
        assert cfg.align.roll_pitch_sigma_deg == KNOBS[crop].get("roll_pitch_sigma_deg") and cfg.tuning is True


def test_val_pose_configs_pass_the_ladder_request():
    names = [*LADDER["rungs"], *LADDER["baselines"]]
    six = ["L3", "L4_drift", "LV", "LVI", "L5", "L5VI"]
    for crop in ("apple", "pear"):
        check_ladder_request(load_yaml_model(EVAL / f"{crop}_val_pose_c2.yaml", EvalAppConfig), LADDER, names)
        for label in ("c1", "c3", "c1v"):
            check_ladder_request(load_yaml_model(EVAL / f"{crop}_val_pose_{label}.yaml", EvalAppConfig), LADDER, six)


def test_imu_profiles_change_only_the_noise_levels():
    ext = load_yaml_model(POSE_CFG / "imu_ext.yaml", ImuConfig)
    levels = {"imu_lab": ("I_lab", (7.85e-5, 9.81e-4, 1.82e-6, 4.75e-5)),
              "imu_icm": ("I_icm", (3.93e-4, 4.91e-3, 1.82e-5, 4.75e-4)),
              "imu_ext_x4": ("I_ext_x4", (4.64e-3, 1.988e-2, 7.28e-5, 1.9e-3))}
    for name, (profile, (gn, an, gw, aw)) in levels.items():
        cfg = load_yaml_model(POSE_CFG / f"{name}.yaml", ImuConfig)
        assert cfg.profile == profile
        assert cfg.model_dump(exclude={"profile", "noise"}) == ext.model_dump(exclude={"profile", "noise"})
        assert cfg.noise.model_dump() == {**ext.noise.model_dump(), "gyro_n": gn, "acc_n": an, "gyro_w": gw, "acc_w": aw}
    assert ext.profile == "I_ext" and ext.noise.acc_b0_mg == 4.0
    assert len({config_hash(load_yaml_model(POSE_CFG / f"{n}.yaml", ImuConfig)) for n in IMU_DIRS}) == 4
