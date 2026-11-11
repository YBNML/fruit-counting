"""Pydantic schema of the generated IMU settings (`configs/pose/imu_*.yaml`) and of the fixed values of the
fusion graph (`configs/pose/fuse_fixed.yaml`).
The visual odometry settings (`configs/pose/vo.yaml`) are `VoConfig`.

These are files of their own, not sections of the eval config. Noise densities follow the Kalibr form:
white noise in rad/s/sqrt(Hz) and m/s^2/sqrt(Hz), bias random walk in rad/s^2/sqrt(Hz) and m/s^3/sqrt(Hz).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator


class ImuNoise(BaseModel):
    """Noise of a generated IMU. White noise and the turn-on bias must be above 0: a noiseless IMU is the truth."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    gyro_n: float = Field(gt=0)
    acc_n: float = Field(gt=0)
    gyro_w: float = Field(ge=0)
    acc_w: float = Field(ge=0)
    # Turn-on bias sigma, drawn once per session: gyro (deg/s), accelerometer (mg).
    gyro_b0_dps: float = Field(gt=0)
    acc_b0_mg: float = Field(gt=0)
    # Temperature rise over a drive (K).
    temp_k: float = Field(default=0.0, ge=0)
    # Band vibration RMS per axis: accelerometer (m/s^2), gyro (deg/s).
    vib_acc_ms2: float = Field(default=0.0, ge=0)
    vib_gyro_dps: float = Field(default=0.0, ge=0)


class ImuConfig(BaseModel):
    """A generated IMU profile, its sampling, its seeds and the assumed motion of a straight bundle."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    profile: str = Field(pattern=r"^[A-Za-z0-9_]+$")
    noise: ImuNoise
    rate_hz: float = Field(default=200.0, gt=0)
    acc_range_g: float = Field(default=4.0, gt=0)
    seed: NonNegativeInt = 0
    noise_seeds: list[NonNegativeInt] = Field(default_factory=lambda: [0, 1, 2], min_length=1)
    # Assumed motion of a bundle without a recorded trajectory: speed (m/s), standstill before the start (s)
    # and acceleration (s); time at the row end between drives (s).
    speed_mps: float = Field(default=0.5, gt=0)
    stop_s: float = Field(default=5.0, ge=0)
    accel_s: float = Field(default=2.0, gt=0)
    row_end_s: float = Field(default=20.0, ge=0)
    output_dir: str = "runs/imu"

    @field_validator("noise_seeds")
    @classmethod
    def _check_seeds_distinct(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError(f"noise_seeds repeats a seed: {v}")
        return v


class EstimatorImu(BaseModel):
    """IMU noise densities the fusion graph assumes."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    gyro_n: float = Field(gt=0)
    acc_n: float = Field(gt=0)
    gyro_w: float = Field(gt=0)
    acc_w: float = Field(gt=0)


class ShakeShare(BaseModel):
    """Body shake share of the start prior sigma on a bundle with body shake, added in quadrature (m, m, m, deg)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    side_m: float = Field(ge=0)
    height_m: float = Field(ge=0)
    along_m: float = Field(ge=0)
    yaw_deg: float = Field(ge=0)


class FuseFixedConfig(BaseModel):
    """Fixed values of the fusion graph: estimator IMU noise, start priors and the odometry factor sigma floor."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    estimator: EstimatorImu
    start_speed_mps: float = Field(gt=0)
    start_speed_sigma_mps: float = Field(gt=0)
    start_roll_pitch_sigma_deg: float = Field(gt=0)
    shake_share: ShakeShare
    vo_sigma_floor_trans_mm: float = Field(gt=0)
    vo_sigma_floor_rot_deg: float = Field(gt=0)


class VoConfig(BaseModel):
    """Settings of the frame-to-frame RGB-D visual odometry (`counting pose vo`); the config hash names the cache
    folder."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    # ORB keypoints per frame.
    n_features: int = Field(default=6000, gt=0)
    # ORB pyramid scale factor between levels.
    orb_scale: float = Field(default=1.2, gt=1)
    # ORB pyramid levels.
    orb_levels: int = Field(default=8, gt=0)
    # FAST corner threshold of the ORB detector (grey levels).
    fast_threshold: int = Field(default=7, gt=0)
    # CLAHE clip limit of the grey image.
    clahe_clip: float = Field(default=3.0, gt=0)
    # CLAHE tiles per image side (tile x tile grid).
    clahe_tile: int = Field(default=8, gt=0)
    # Ratio test of the global descriptor matching (nearest / second nearest Hamming distance).
    match_ratio: float = Field(default=0.8, gt=0, le=1)
    # Reprojection error of the first EPnP RANSAC fit (px).
    ransac_px: float = Field(default=3.0, gt=0)
    # Reprojection error of the final EPnP RANSAC fit (px).
    final_px: float = Field(default=2.0, gt=0)
    # RANSAC iterations of each fit.
    ransac_iters: int = Field(default=300, gt=0)
    # RANSAC confidence of each fit.
    ransac_confidence: float = Field(default=0.999, gt=0, lt=1)
    # Fewest inliers (and matches) a fit needs for its pose to be taken.
    min_inliers: int = Field(default=30, ge=4)
    # Guided matching radius around points projected with the first fit's pose (px).
    guided_radius_px: float = Field(default=12.0, gt=0)
    # Guided matching radius around points projected with the motion prior (px).
    prior_radius_px: float = Field(default=60.0, gt=0)
    # Largest Hamming distance of a guided match (bits of 256).
    guided_max_hamming: int = Field(default=64, gt=0, le=256)
    # Ratio test of the guided matching (nearest / second nearest Hamming distance).
    guided_ratio: float = Field(default=0.9, gt=0, le=1)
    # Deepest keypoint depth used (m); deeper keypoints get no 3D point.
    depth_max_m: float = Field(default=10.0, gt=0)
    # Keypoint depth window radius r: the nearest depth in (2r+1) x (2r+1) pixels, 0 for the pixel itself.
    depth_nb: int = Field(default=2, ge=0)
    # Keypoint depth noise k: Gaussian sigma k * z^2 (m), 0 for none.
    depth_k: float = Field(default=0.0036, ge=0)
    # Seed of the keypoint depth noise.
    noise_seed: int = Field(default=0, ge=0)
    # Seed of OpenCV's global generator, set per pass with the pass code; solvePnPRansac ignores it (hash only).
    seed: int = Field(default=0, ge=0)
    # OpenCV threads.
    threads: int = Field(default=1, ge=1)
    # Spacing of the frames the odometry runs on (m); None for every frame.
    step_m: float | None = Field(default=None, gt=0)
    # Root folder of the caches (left out of the config hash).
    output_dir: str = "runs/pose"
