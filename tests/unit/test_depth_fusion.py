"""LiDAR box depth beside the fused pose source: the navigation unit takes the camera-IMU extrinsic draw of the
fusion, and the depth unit's own extrinsic leaves the fused poses alone and is the same draw with or without them.
On a small fused ladder the LiDAR rungs take the camera depth draws of their camera-only rung, seed by seed."""

import re

import pytest
import yaml

import counting.evaluation.fusion as fusion_module
from counting.evaluation import lidarsource
from counting.evaluation.runner import _merge, evaluate, run_ladder
from tests.unit.test_lidar_runner import DEPTH, LIDAR_EXTR, LIDAR_NOISE, RIG, START, TRUE_LD, VO, _exact, lidar_case

pytestmark = pytest.mark.pose
pytest.importorskip("gtsam")

FUSE_FIXED = {"estimator": {"gyro_n": 7.85e-5, "acc_n": 9.81e-4, "gyro_w": 1.82e-6, "acc_w": 4.75e-5},
              "start_speed_mps": 0.5, "start_speed_sigma_mps": 0.2, "start_roll_pitch_sigma_deg": 5.0,
              "shake_share": {"side_m": 0.02, "height_m": 0.01, "along_m": 0.01, "yaw_deg": 0.5},
              "vo_sigma_floor_trans_mm": 0.1, "vo_sigma_floor_rot_deg": 0.001}
IMU_BELIEF = {"imu_time_offset_ms": 0.0, "imu_extr_trans_mm": 10.0, "imu_extr_rot_deg": 0.3}
IMU = {"imu_gyro_n": 1.16e-3, "imu_acc_n": 4.97e-3, "imu_gyro_w": 1.82e-5, "imu_acc_w": 4.75e-4,
       "imu_gyro_b0_dps": 1.0, "imu_acc_b0_mg": 4.0, "imu_temp_k": 1.0}
FUSED = {"pose": {"rig": True, "rig_ref": 10, "fuse": "imu"}}
# The navigation-unit rung on the fused truth source, and the fused drift rung with the depth unit.
NAV = _merge(FUSED, {"lift": {"depth": "lidar", "lidar_file": "nav"},
                     "noise": {**IMU_BELIEF, **LIDAR_NOISE, "seeds": [0]}})
FUSED_LD = _merge(FUSED, {"lift": {"depth": "lidar"},
                          "noise": {**VO, **START, **RIG, **DEPTH, **IMU_BELIEF, **LIDAR_NOISE, **LIDAR_EXTR,
                                    "seeds": [0]}})


@pytest.fixture
def fixed(tmp_path, monkeypatch):
    path = tmp_path / "fixed" / "fuse_fixed.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(FUSE_FIXED), encoding="utf-8")
    monkeypatch.setattr(fusion_module, "FUSE_FIXED_PATH", path)
    return path


def test_navigation_lidar_shares_the_fusion_extrinsic(fixed, monkeypatch):
    case = lidar_case()
    calls = []
    real_imu, real_lidar = fusion_module.imu_extr_error, lidarsource.lidar_extr_error

    def imu_on(where):
        def record(cfg, noise_seed, *, scene, pitch):
            out = real_imu(cfg, noise_seed, scene=scene, pitch=pitch)
            calls.append((where, (cfg.seed, noise_seed, scene, pitch), out.tobytes()))
            return out
        return record

    def lidar(cfg, noise_seed, *, scene):
        out = real_lidar(cfg, noise_seed, scene=scene)
        calls.append(("lidar", (cfg.seed, noise_seed, scene), out.tobytes()))
        return out

    monkeypatch.setattr(fusion_module, "imu_extr_error", imu_on("fusion"))
    monkeypatch.setattr(lidarsource, "imu_extr_error", imu_on("depth"))
    monkeypatch.setattr(lidarsource, "lidar_extr_error", lidar)

    nav = evaluate(case.ctx, case.config(NAV), noise_seed=0)
    assert nav["pose"]["fusion"] is not None and nav["lidar"]["unit"] == "nav"
    assert nav["lidar"]["mount_m"] == [0.0, 0.0, 0.5]
    fusion = {(args, value) for where, args, value in calls if where == "fusion"}
    depth = {(args, value) for where, args, value in calls if where == "depth"}
    assert fusion and depth == fusion == {((0, 0, case.run.meta.seed, 10.0), next(iter(fusion))[1])}
    assert not any(where == "lidar" for where, _, _ in calls)

    calls.clear()
    base = evaluate(case.ctx, case.config(FUSED_LD), noise_seed=0)
    assert base["pose"]["fusion"] is not None
    fused_draws = {value for where, _, value in calls if where == "lidar"}
    calls.clear()
    evaluate(case.ctx, case.config(TRUE_LD), noise_seed=0)
    true_draws = {value for where, _, value in calls if where == "lidar"}
    assert len(fused_draws) == 1 and true_draws == fused_draws

    moved = evaluate(case.ctx, case.config(FUSED_LD, {"noise": {"lidar_extr_trans_mm": 60.0,
                                                                 "lidar_extr_rot_deg": 1.0}}), noise_seed=0)
    assert _exact(moved, ("pose_quality",)) == _exact(base, ("pose_quality",))
    assert _exact(moved["lidar"], ("depth",)) != _exact(base["lidar"], ("depth",))


def test_fused_ladder_camera_draw_digests_pair_up(fixed):
    case = lidar_case()
    # The fused drift rung (camera only), with the depth unit, with the navigation unit; the true poses with the
    # depth unit and no camera depth noise.
    drift = {"stages": {"track": "landmark", "center": "measured", "assign": "grid"}, **FUSED,
             "align": {"mode": ["tracks", "row_plane"]}, "noise": {**VO, **START, **RIG, **DEPTH, **IMU, **IMU_BELIEF}}
    ladder = {"rungs": {"L4_drift": drift},
              "baselines": {"L4_drift_ld": _merge(drift, {"lift": {"depth": "lidar", "lidar_file": "depth"},
                                                          "noise": {**LIDAR_NOISE, **LIDAR_EXTR}}),
                            "L4_drift_ldn": _merge(drift, {"lift": {"depth": "lidar", "lidar_file": "nav"},
                                                           "noise": LIDAR_NOISE}),
                            "L3_ld": TRUE_LD}}
    out = run_ladder(case.ctx, case.config({"noise": {"seeds": [0, 1]}}), ladder,
                     [*ladder["rungs"], *ladder["baselines"]])
    rungs = out["rungs"]
    assert all(rungs[name]["pose"]["fusion"] is not None for name in ("L4_drift", "L4_drift_ld", "L4_drift_ldn"))
    assert rungs["L4_drift_ld"]["lidar"]["unit"] == "depth" and rungs["L4_drift_ldn"]["lidar"]["unit"] == "nav"

    def digests(name: str) -> list[str]:
        assert [p["noise_seed"] for p in rungs[name]["per_seed"]] == [0, 1], name
        return [p["camera_depth_sha256"] for p in rungs[name]["per_seed"]]

    camera = digests("L4_drift")
    assert all(re.fullmatch(r"[0-9a-f]{64}", sha) for sha in camera) and camera[0] != camera[1]
    assert digests("L4_drift_ld") == camera
    assert digests("L4_drift_ldn") == digests("L4_drift_ld")
    assert digests("L3_ld") == ["none", "none"]
