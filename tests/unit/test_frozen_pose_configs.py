"""The settings fixed on val for the held-out pose files and the pose values chosen on val: each crop's frozen
odometry settings and their noiseless-depth twin (the pose cache folders of the val configs), the frozen copy of
the judged IMU profile, and the odometry drift profile and odometry factor sigmas that the pose ladder and the drift
profile grid carry. Then the frozen held-out configs of the pose ladder: each is a val config with the held-out
scenes, tuning off and the digests of its detection caches, pose caches and judged IMU files pinned, and its rungs
and the held-out drift grids pass the held-out checks."""

import re
from pathlib import Path

import pytest

from counting.config.eval_schema import EvalAppConfig, SplitsConfig
from counting.config.hashing import config_hash
from counting.config.loader import load_yaml_model
from counting.config.pose_schema import ImuConfig, VoConfig
from counting.evaluation.protect import check_test_pose_rungs, check_test_request
from counting.evaluation.runner import ProtectionError, _patched, load_ladder, rung_config
from counting.evaluation.sweep import load_grid, sweep_cells

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
# The drift grids that may run on the held-out scenes; drift_val and drift_vi run on val only.
TEST_GRIDS = ("drift_rpe", "drift_inpass", "drift_interpass", "drift_interpass_c1x", "drift_imu", "drift_profiles")
# The content digests of the held-out files that the frozen held-out configs pin, per crop and bundle run_id:
# detection caches (det), pose caches (vo, vo_dd) and judged IMU files (imu_ext).
DIGESTS = {
    "apple": {
        "det": {"apple_s1003_7d0fa63d": "6a4a5c003eb39435f0d86a2ab42a9bd857e75a7b0b2de7e8ebf119b2beea76c5",
                "apple_s1004_8c3e380d": "85a0ef592944a6a0e46cebef7c9063c6138ffdb28e127dfa118da564d02b5eb5",
                "apple_s1005_33c3058e": "777bd350314291714375be55662071d2e21ae31b8d8541916941aeb819d87f3e"},
        "vo": {"apple_s1003_7d0fa63d": "a7d120dc6d9c6136fd23bb12f701339601638c2e74ad0f19189b86831608ec23",
               "apple_s1004_8c3e380d": "4225cd8be238d50dd479c02dab58c1d8bc27f5a3e83ed6040edc0410acbb90ee",
               "apple_s1005_33c3058e": "680e7cb2afeddcd4b6c77a9d83609bf26ae2010c57a15f51fe22f6219f620961"},
        "vo_dd": {"apple_s1003_7d0fa63d": "c194cb4c071d8a68459e4c45d45f61d06b05830e92590431f47286af65d95c59",
                  "apple_s1004_8c3e380d": "9c30a34e1c70e84c09e79590d634e4fe01144f89e1f5fd0dd4172e13b1d769d2",
                  "apple_s1005_33c3058e": "2cb6264d92082a5efe9ac3a133a25cf29616e127874f0b32501410997d4c7faa"},
        "imu_ext": {"apple_s1003_7d0fa63d": "c05f253279ef5b2963d5158dd4beeec6e83b105e5f3f28e512af6aa12e13c064",
                    "apple_s1004_8c3e380d": "a9e03276881b4a6f918d9dc6af7de130e3c32c3b1747ad5398ae4add636ab56b",
                    "apple_s1005_33c3058e": "d9662f791f35d34b2057d942f1c3f9b44d2a7b6705203a1ef77db5d20da89124"},
    },
    "pear": {
        "det": {"pear_s1003_2f536246": "046f4e1268570f8a2f5e4c06adad667ba6ddfd61c9ee124e88c20a3e2d64e5d3",
                "pear_s1004_df156985": "1fce9f81c7b23f3b5cb2763e3eead5545f68dc2e283ba40e1060fbed29277605",
                "pear_s1005_0f79160e": "f93291dc2a7682b9e02c92ef4b3b3247170129601292d6b73a0d6edbe5a5d853"},
        "vo": {"pear_s1003_2f536246": "28cebc296ac8b3081239173202c4bf91cf96efb5684c6e4ad8c9fa5bc0f50880",
               "pear_s1004_df156985": "60ec341ac23a07a01a3b7749052b0efd6fe1b39baa0ec1c94ec451437003b558",
               "pear_s1005_0f79160e": "da00bcf566d6bf1301eb9dcc769475b489dbd1648c6b6fee4bb840b03675c1a7"},
        "vo_dd": {"pear_s1003_2f536246": "67bbcc6a52c9348fc2b64cc567d5cf143cd99c89b7b63a652749fadeb1f75b66",
                  "pear_s1004_df156985": "3bb1ab2682033694310c6028563da81792dbb4f00e8cad849f73c977c965fa62",
                  "pear_s1005_0f79160e": "fefae7ad84242f6e24ed4a410a1c484940d9879e10babe79cd32f11b8bc6bb36"},
        "imu_ext": {"pear_s1003_2f536246": "32bd52eff844c680d63aa58a6dff127602c2b1efea12563d8c17e5f380cb4fa9",
                    "pear_s1004_df156985": "451cb3d9f7cc3c267fa3836b80996c62d1fd65c02caf4172c3420b25be18a16f",
                    "pear_s1005_0f79160e": "d934b8f38703bdb6c2418246a5ae773b4ae2bddedd21b8d87fa4a13d123c44d0"},
    },
}


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


def _files() -> dict:
    """The tracked files a held-out pose request reads besides its config, keyed as the request names them."""
    return {"--splits": EVAL / "splits_pose.yaml", "--criteria": EVAL / "criteria_pose.yaml", "--ladder": LADDER_PATH,
            "fixed fusion settings": ROOT / "configs/pose/fuse_fixed.yaml"}


def test_frozen_pose_test_configs_follow_val():
    splits = load_yaml_model(EVAL / "splits_pose.yaml", SplitsConfig)
    for crop in ("apple", "pear"):
        for label, name in (("c2", f"{crop}_test_pose"), ("c1", f"{crop}_test_pose_c1")):
            val = load_yaml_model(EVAL / f"{crop}_val_pose_{label}.yaml", EvalAppConfig)
            cfg = load_yaml_model(EVAL / "frozen" / f"{name}.yaml", EvalAppConfig)
            runs = [f"~/datasets/orchard_synth/{crop}/test_s{s}_shake" for s in splits.test]
            d = DIGESTS[crop]
            expected = _patched(val, {
                "run_name": name, "split": "test", "tuning": False, "runs": runs,
                "detector": {"digests": d["det"]},
                "pose": {"caches": {"vo": {"digests": d["vo"]}, "vo_dd": {"digests": d["vo_dd"]}}}})
            dump = expected.model_dump()
            dump["pose"]["imu_files"] = {"imu_ext": {**dump["pose"]["imu_files"]["imu_ext"], "digests": d["imu_ext"]}}
            assert cfg.model_dump() == dump, name
            for pins in (d["det"], d["vo"], d["vo_dd"], d["imu_ext"]):
                assert len(pins) == 3 and all(k.startswith(f"{crop}_s100") for k in pins)
                assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in pins.values())


def test_frozen_pose_test_requests_pass_the_rung_checks():
    ladder = load_ladder(LADDER_PATH)
    clean = {"commit": "c" * 40, "dirty": False, "diff_sha256": None}
    held_out = [n for n in [*ladder["rungs"], *ladder["baselines"]] if n not in ladder["val_only"]]
    six = ["L3", "L4_drift", "LV", "LVI", "L5", "L5VI"]
    for crop in ("apple", "pear"):
        for name, rungs in ((f"{crop}_test_pose", held_out), (f"{crop}_test_pose_c1", six)):
            path = EVAL / "frozen" / f"{name}.yaml"
            cfg = load_yaml_model(path, EvalAppConfig)
            check_test_request(cfg, config_path=path, sets=[], runs=[], files=_files(), grid=None, ladder=ladder,
                               repo=ROOT, code=clean)
            check_test_pose_rungs(cfg, {n: (n, rung_config(cfg, ladder, n)) for n in rungs}, ladder)
        cfg = load_yaml_model(EVAL / "frozen" / f"{crop}_test_pose.yaml", EvalAppConfig)
        for grid in TEST_GRIDS:
            cells = sweep_cells(cfg, load_grid(POSE / f"{grid}.yaml"), ladder)
            check_test_pose_rungs(cfg, {f"{c.point.name} @ {c.base}": (c.base, c.cfg) for c in cells}, ladder)
        for grid in ("drift_val", "drift_vi"):
            cells = sweep_cells(cfg, load_grid(POSE / f"{grid}.yaml"), ladder)
            with pytest.raises(ProtectionError, match="that axis runs on val only|runs on val only"):
                check_test_pose_rungs(cfg, {f"{c.point.name} @ {c.base}": (c.base, c.cfg) for c in cells}, ladder)
