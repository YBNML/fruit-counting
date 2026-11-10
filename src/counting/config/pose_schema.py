"""Pydantic schema of the generated IMU settings (`configs/pose/imu_*.yaml`) and of the fixed values of the
fusion graph (`configs/pose/fuse_fixed.yaml`).

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
