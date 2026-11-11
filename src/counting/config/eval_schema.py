"""Pydantic configuration schema for the counting eval tool."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator, model_validator

Crop = Literal["pear", "apple"]
Variant = Literal["count1", "any", "strict"]


class TruthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variant: Variant = "count1"


class PassesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pitches: list[float] | None = None
    sides: list[Literal["A", "B"]] = Field(default=["A", "B"], min_length=1)
    step_m: float | None = Field(default=None, gt=0)


class StagesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    det: Literal["oracle", "oracle_noisy", "cached"] = "oracle"
    track: Literal["oracle", "landmark", "last_box"] = "landmark"
    center: Literal["truth", "measured"] = "measured"
    dedup: Literal["oracle", "self_side", "merge", "hybrid"] = "merge"
    assign: Literal["truth", "grid"] = "grid"


class DetNoise(BaseModel):
    """Detection noise of the noisy oracle; every factor at 0 with score fixed gives the oracle's detections.

    Factors must be finite (an infinite false-positive rate would never finish drawing).
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    miss_p: float = Field(default=0.0, ge=0, le=1)
    fp_lambda: float = Field(default=0.0, ge=0)
    jitter_frac: float = Field(default=0.0, ge=0)
    split_p: float = Field(default=0.0, ge=0, le=1)
    split_visib_max: float = Field(default=0.6, gt=0, le=1)
    score: Literal["fixed", "uniform"] = "fixed"
    score_min: float = Field(default=0.3, ge=0, le=1)


def _unset(value: object) -> bool:
    return value is None


class CachePostConfig(BaseModel):
    """Post-processing of cached detections before matching."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    nms_iou: float = Field(default=0.5, gt=0, le=1)
    edge_snap_px: float = Field(default=2.0, ge=0)
    min_short_px: float = Field(default=0.0, ge=0)


# Cached-detector settings are read under stages.det=cached only; unset, they stay out of the dump and the config hash.
class DetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    positives: Variant = "count1"
    cache: str | None = None
    noise: DetNoise | None = None
    post: CachePostConfig | None = Field(default=None, exclude_if=_unset)
    digests: dict[str, str] | None = Field(default=None, exclude_if=_unset)

    @field_validator("digests")
    @classmethod
    def _check_digest_format(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if v is None:
            return v
        for key, value in v.items():
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"detector.digests[{key}] must be a 64-character lowercase hex digest")
        return v

    def post_settings(self) -> CachePostConfig:
        return self.post if self.post is not None else CachePostConfig()


class LiftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    depth: Literal["det_depth", "box_median", "box_far"] = "box_median"
    # box_median only: side ratio of the centre rectangle of the box.
    box_frac: float = Field(default=0.5, gt=0, le=1)
    # box_far only (far_*): scale of the ellipse inscribed in the box, depth gap (m) that splits
    # clusters, and the least share of the in-ellipse pixels a cluster needs to be picked as the far one.
    far_scale: float = Field(default=1.0, gt=0, le=1)
    far_gap_m: float = Field(default=0.03, gt=0)
    far_min_frac: float = Field(default=0.25, gt=0, le=1)
    center_offset_m: dict[Literal["fruit", "bag"], float] = Field(
        default_factory=lambda: {"fruit": 0.02, "bag": 0.03}
    )

    def offset(self, label: int) -> float:
        key = "bag" if label == 1 else "fruit"
        return self.center_offset_m.get(key, 0.0)


# Pose-model noise keys of `noise`, in the order the reports list them. Drift keys shape the body odometry
# flow, the start, rig and time errors and the depth noise; IMU level keys set the noise of the simulated
# IMU; IMU belief keys are errors the fusion believes; GPS keys set the row-end fixes.
DRIFT_KEYS = ("drift_step_trans_mm", "drift_step_rot_deg", "drift_scale_sigma", "drift_rate_deg_per_m",
              "pass_side_trans_m", "pass_side_rot_deg", "pass_pitch_trans_m", "pass_pitch_rot_deg",
              "rig_extr_trans_mm", "rig_extr_rot_deg", "drift_time_offset_s", "depth_k")
IMU_LEVEL_KEYS = ("imu_gyro_n", "imu_acc_n", "imu_gyro_w", "imu_acc_w", "imu_gyro_b0_dps", "imu_acc_b0_mg",
                  "imu_temp_k", "imu_vib_acc_ms2", "imu_vib_gyro_dps")
IMU_BELIEF_KEYS = ("imu_time_offset_ms", "imu_extr_trans_mm", "imu_extr_rot_deg")
GPS_KEYS = ("gps_end_h_m", "gps_end_v_m")
POSE_NOISE_KEYS = DRIFT_KEYS + IMU_LEVEL_KEYS + IMU_BELIEF_KEYS + GPS_KEYS
# The drift keys of the body odometry flow (walk, scale and heading rate). A file source reads its flow from the
# pose file, so its config does not set them.
FLOW_KEYS = ("drift_step_trans_mm", "drift_step_rot_deg", "drift_scale_sigma", "drift_rate_deg_per_m")


def _factor() -> Any:
    """A pose-model noise factor: unset (None) is no factor and stays out of the dump and the config hash."""
    return Field(default=None, ge=0, exclude_if=_unset)


class NoiseConfig(BaseModel):
    """Pose and depth noise, and the noise seeds; factors must be finite.

    The pose-model keys (`POSE_NOISE_KEYS`) are unset by default. A key set to 0 is present (it is in
    `pose_keys`) but adds no noise (`is_zero` still holds).
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    pose_trans_sigma_m: float = Field(default=0.0, ge=0)
    pose_rot_sigma_deg: float = Field(default=0.0, ge=0)
    pass_bias_m: float = Field(default=0.0, ge=0)
    depth_rel_sigma: float = Field(default=0.0, ge=0)
    depth_dropout: float = Field(default=0.0, ge=0, le=1)
    seeds: list[NonNegativeInt] = Field(default_factory=lambda: [0], min_length=1)
    # Body odometry flow, per base step (0.15 m) per axis: translation (mm) and rotation (deg) noise.
    drift_step_trans_mm: float | None = _factor()
    drift_step_rot_deg: float | None = _factor()
    # Body odometry flow, drawn once per flow: scale error (ratio) and rotation drift rate per axis (deg/m).
    drift_scale_sigma: float | None = _factor()
    drift_rate_deg_per_m: float | None = _factor()
    # Rigid start error of each side's drive (m, deg).
    pass_side_trans_m: float | None = _factor()
    pass_side_rot_deg: float | None = _factor()
    # Extra rigid start error of each pitch pass when one camera passes each pitch (m, deg).
    pass_pitch_trans_m: float | None = _factor()
    pass_pitch_rot_deg: float | None = _factor()
    # Rig extrinsic error of each camera other than the reference one (mm, deg).
    rig_extr_trans_mm: float | None = _factor()
    rig_extr_rot_deg: float | None = _factor()
    # Frame time offset of each pass (s).
    drift_time_offset_s: float | None = _factor()
    # Range-dependent depth noise, sigma = k * z^2 (m).
    depth_k: float | None = _factor()
    # IMU white noise densities: gyro (rad/s/sqrt(Hz)), accelerometer (m/s^2/sqrt(Hz)).
    imu_gyro_n: float | None = _factor()
    imu_acc_n: float | None = _factor()
    # IMU bias random walks: gyro (rad/s^2/sqrt(Hz)), accelerometer (m/s^3/sqrt(Hz)).
    imu_gyro_w: float | None = _factor()
    imu_acc_w: float | None = _factor()
    # IMU turn-on bias sigma, drawn once per session: gyro (deg/s), accelerometer (mg).
    imu_gyro_b0_dps: float | None = _factor()
    imu_acc_b0_mg: float | None = _factor()
    # Temperature rise over a drive (K).
    imu_temp_k: float | None = _factor()
    # Band vibration RMS per axis: accelerometer (m/s^2), gyro (deg/s).
    imu_vib_acc_ms2: float | None = _factor()
    imu_vib_gyro_dps: float | None = _factor()
    # Camera-IMU time offset (ms): a fixed offset the fusion believes; no draw.
    imu_time_offset_ms: float | None = _factor()
    # Camera-IMU extrinsic error (mm, deg).
    imu_extr_trans_mm: float | None = _factor()
    imu_extr_rot_deg: float | None = _factor()
    # Row-end GPS fixes (m): a fix at the first and the last frame of each flow; horizontal sigma per axis.
    gps_end_h_m: float | None = _factor()
    # Row-end GPS fixes (m): a fix at the first and the last frame of each flow; vertical sigma.
    gps_end_v_m: float | None = _factor()

    @field_validator("seeds")
    @classmethod
    def _check_seeds_distinct(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError(f"noise.seeds repeats a seed: {v}")
        return v

    def is_zero(self) -> bool:
        return (
            self.pose_trans_sigma_m == 0
            and self.pose_rot_sigma_deg == 0
            and self.pass_bias_m == 0
            and self.depth_rel_sigma == 0
            and self.depth_dropout == 0
            and all(self.factor(key) == 0 for key in POSE_NOISE_KEYS)
        )

    def factor(self, name: str) -> float:
        """The value of noise factor `name`; 0.0 for an unset pose-model key."""
        value = getattr(self, name)
        return 0.0 if value is None else value

    def pose_keys(self) -> list[str]:
        """The pose-model noise keys that are set (a key set to 0 included), in `POSE_NOISE_KEYS` order."""
        return [key for key in POSE_NOISE_KEYS if getattr(self, key) is not None]


class PoseFiles(BaseModel):
    """A folder of pose or IMU files and the content digest of each bundle's file, keyed by run."""

    model_config = ConfigDict(extra="forbid")
    dir: str
    digests: dict[str, str] = Field(default_factory=dict)

    @field_validator("digests")
    @classmethod
    def _check_digest_format(cls, v: dict[str, str]) -> dict[str, str]:
        for key, value in v.items():
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"digests[{key}] must be a 64-character lowercase hex digest")
        return v


class VioConfig(BaseModel):
    """Settings of the fusion graph; unset keys stay out of the dump and the config hash."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    # Sigma of the odometry factors of a file source (mm, deg).
    vo_sigma_trans_mm: float | None = Field(default=None, gt=0, exclude_if=_unset)
    vo_sigma_rot_deg: float | None = Field(default=None, gt=0, exclude_if=_unset)
    # Standstill before the start (s), held by zero-velocity factors.
    stop_s: float | None = Field(default=None, ge=0, exclude_if=_unset)
    # IMU noise values the estimator assumes: the fixed lab values or the generating profile's.
    estimator_imu: Literal["lab", "generator"] | None = Field(default=None, exclude_if=_unset)


class PoseConfig(BaseModel):
    """Where the camera poses come from and how they are built.

    `source: truth` builds poses from the true poses (with the drift model when drift keys are set);
    `source: file` reads the pose files named by `cache`. With `rig: true` the cameras of one side ride one
    body, and `rig_ref` is the pitch of the reference camera. `fuse` picks the fusion graph.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    source: Literal["truth", "file"] = "truth"
    cache: str | None = Field(default=None, exclude_if=_unset)
    caches: dict[str, PoseFiles] | None = Field(default=None, exclude_if=_unset)
    anchor: Literal["nominal", "first_truth"] = "nominal"
    rig: bool = False
    rig_ref: float | None = Field(default=None, exclude_if=_unset)
    fuse: Literal["none", "vo", "imu"] = "none"
    imu: str | None = Field(default=None, exclude_if=_unset)
    imu_files: dict[str, PoseFiles] | None = Field(default=None, exclude_if=_unset)
    vio: VioConfig | None = Field(default=None, exclude_if=_unset)
    # Name of the camera configuration (for example c2 or c1), shown in results and reports.
    config_label: str | None = Field(default=None, pattern=r"^[a-z0-9_]+$", exclude_if=_unset)

    @model_validator(mode="after")
    def _check_source_and_fusion(self) -> PoseConfig:
        if self.source == "file" and (self.cache is None or self.cache not in (self.caches or {})):
            raise ValueError("pose.source=file needs pose.cache, one of pose.caches")
        if self.cache is not None and self.source == "truth":
            raise ValueError("pose.cache needs pose.source=file")
        if self.source == "file" and self.fuse == "imu" and self.imu is None:
            raise ValueError("pose.source=file with pose.fuse=imu needs pose.imu")
        if self.imu is not None and not (
            self.source == "file" and self.fuse == "imu" and self.imu in (self.imu_files or {})
        ):
            raise ValueError("pose.imu needs pose.source=file, pose.fuse=imu and a name of pose.imu_files")
        if self.source == "truth" and self.fuse == "vo":
            raise ValueError("pose.fuse=vo needs pose.source=file; a truth source has one body odometry flow")
        if self.vio is not None and self.fuse == "none":
            raise ValueError("pose.vio needs pose.fuse=vo or imu")
        if self.vio is not None and self.source == "truth":
            for key in ("vo_sigma_trans_mm", "vo_sigma_rot_deg"):
                if getattr(self.vio, key) is not None:
                    raise ValueError(f"pose.vio.{key} needs pose.source=file; a truth source takes its factor "
                                     "sigmas from its drift keys")
        return self


AlignMethod = Literal["tracks", "row_plane"]


class AlignConfig(BaseModel):
    """Alignment between passes and between aisles.

    `mode` is one method name, or a list of the methods that run in order. The stress keys degrade a listed
    method: `stress_row_bias_m` adds a bias (m, any sign) to the row centre plane `row_plane` estimates, and
    `stress_pair_keep` is the share of the track pairs seen from both sides that `tracks` keeps.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mode: (Literal["none", "fit_truth_pass", "fit_truth_segment", "oracle_struct"]
           | Annotated[list[AlignMethod], Field(min_length=1)]) = "none"
    stress_row_bias_m: float | None = Field(default=None, exclude_if=_unset)
    stress_pair_keep: float | None = Field(default=None, gt=0, le=1, exclude_if=_unset)

    @model_validator(mode="after")
    def _check_methods(self) -> AlignConfig:
        methods = self.mode if isinstance(self.mode, list) else []
        if len(set(methods)) != len(methods):
            raise ValueError("align.mode repeats a method")
        if self.stress_row_bias_m is not None and "row_plane" not in methods:
            raise ValueError("align.stress_row_bias_m needs row_plane in align.mode")
        if self.stress_pair_keep is not None and "tracks" not in methods:
            raise ValueError("align.stress_pair_keep needs tracks in align.mode")
        return self


class TrackerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predict: Literal["pose"] = "pose"
    gate_px_frac: float = Field(default=0.5, gt=0)
    gate_m: float = Field(default=0.05, gt=0)
    depth_gate: Literal["hard", "soft"] = "hard"
    # depth_gate=soft only: extra cost of a pair inside the screen gate that fails the depth gate.
    soft_cost: float = Field(default=0.5, ge=0)
    max_gap: int | None = Field(default=None, ge=1)
    repair_m: float = Field(default=0.04, ge=0)
    min_obs: int = Field(default=1, ge=1)
    min_obs_cached: int | None = Field(default=None, ge=1, exclude_if=_unset)
    score_hi: float = Field(default=0.5, ge=0, le=1)
    score_lo: float = Field(default=0.1, ge=0, le=1)

    @model_validator(mode="after")
    def _check_score_order(self) -> TrackerConfig:
        if self.score_lo > self.score_hi:
            raise ValueError("score_lo must be <= score_hi")
        return self

    def min_obs_for(self, det: str) -> int:
        if det == "cached" and self.min_obs_cached is not None:
            return self.min_obs_cached
        return self.min_obs


class DedupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    radius_m: float = Field(default=0.04, gt=0)
    band_m: float = Field(default=0.10, ge=0)


class EvalAppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_name: str
    crop: Crop
    split: Literal["dev", "val", "test", "exp"]
    tuning: bool = True
    runs: list[str] = Field(min_length=1)
    truth: TruthConfig = Field(default_factory=TruthConfig)
    passes: PassesConfig = Field(default_factory=PassesConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    lift: LiftConfig = Field(default_factory=LiftConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    # Pose source and alignment; without a pose section the config runs on the true camera poses.
    pose: PoseConfig | None = Field(default=None, exclude_if=_unset)
    align: AlignConfig | None = Field(default=None, exclude_if=_unset)
    seed: int = 0
    stop_after: Literal["y1", "y6"] = "y6"
    output_dir: str = "runs/eval"
    # Pass subsets (names from the pass-subset table) whose rung L5 row is judged too; read by eval ladder only.
    verdict_subsets: list[str] | None = Field(default=None, min_length=1, exclude_if=_unset)

    @model_validator(mode="after")
    def _check_stage_combinations(self) -> EvalAppConfig:
        if self.stages.det == "oracle_noisy" and self.detector.noise is None:
            raise ValueError("stages.det=oracle_noisy needs a detector.noise section")
        if self.detector.noise is not None and self.stages.det != "oracle_noisy":
            raise ValueError(f"detector.noise needs stages.det=oracle_noisy, got stages.det={self.stages.det}")
        if self.stages.det == "cached" and self.stages.track == "oracle":
            raise ValueError("cached detector cannot use track=oracle")
        if self.stages.det == "cached" and self.lift.depth == "det_depth":
            raise ValueError("det_depth needs oracle depth; use a box depth method (box_median or box_far) "
                             "with det=cached")
        if self.stages.det == "cached" and self.detector.cache is None:
            raise ValueError("stages.det=cached needs detector.cache (the folder of detection caches)")
        if self.detector.digests is not None and self.detector.cache is None:
            raise ValueError("detector.digests needs detector.cache")
        if self.split == "test" and self.tuning:
            raise ValueError("split=test requires tuning=false")
        return self

    @model_validator(mode="after")
    def _check_pose_settings(self) -> EvalAppConfig:
        """Refuse pose-model keys a config would not use, and pose settings that contradict each other."""
        nz, pose = self.noise, self.pose
        present = nz.pose_keys()

        def first(keys: tuple[str, ...]) -> str | None:
            return next((key for key in present if key in keys), None)

        if pose is None:
            if present:
                raise ValueError(f"noise.{present[0]} needs a pose section")
            if self.align is not None:
                raise ValueError("align needs a pose section")
            return self
        aligned = self.align is not None and self.align.mode != "none"
        if (pose.rig or aligned) and pose.rig_ref is None:
            raise ValueError("pose.rig and an align mode need pose.rig_ref (the reference camera's pitch)")
        pitches = self.passes.pitches
        if pose.rig_ref is not None and pitches is not None and not any(
            abs(pose.rig_ref - p) <= 1e-6 for p in pitches
        ):
            shown = "[" + ", ".join(f"{p:g}" for p in pitches) + "]"
            raise ValueError(f"pose.rig_ref={pose.rig_ref:g} is not one of passes.pitches {shown}")
        if pose.fuse != "imu" and (key := first(IMU_LEVEL_KEYS + IMU_BELIEF_KEYS + GPS_KEYS)):
            raise ValueError(f"noise.{key} needs pose.fuse=imu")
        if pose.source == "file" and (key := first(IMU_LEVEL_KEYS)):
            raise ValueError(f"noise.{key} needs pose.source=truth; a file source takes its IMU noise from the "
                             "IMU file")
        if pose.source == "file" and (key := first(GPS_KEYS)):
            raise ValueError(f"noise.{key} needs pose.source=truth; a file source has no GPS fixes")
        if pose.source == "file" and (key := first(FLOW_KEYS)):
            raise ValueError(f"noise.{key} needs pose.source=truth; a file source takes its odometry from the "
                             "pose file")
        if pose.anchor == "first_truth" and (key := first(GPS_KEYS)):
            raise ValueError(f"noise.{key} needs pose.anchor=nominal")
        gps = [key for key in GPS_KEYS if key in present]
        if len(gps) == 1:
            raise ValueError("noise.gps_end_h_m and noise.gps_end_v_m go together")
        if key := next((key for key in gps if nz.factor(key) == 0), None):
            raise ValueError(f"noise.{key} must be > 0; a zero-noise fix is the truth")
        if not pose.rig and (key := first(("rig_extr_trans_mm", "rig_extr_rot_deg"))):
            raise ValueError(f"noise.{key} needs pose.rig=true")
        if pose.rig and (key := first(("pass_pitch_trans_m", "pass_pitch_rot_deg"))):
            raise ValueError(f"noise.{key} needs pose.rig=false (one camera passing each pitch)")
        start_keys = ("pass_side_trans_m", "pass_side_rot_deg", "pass_pitch_trans_m", "pass_pitch_rot_deg")
        if pose.anchor == "first_truth" and (key := first(start_keys)):
            raise ValueError(f"noise.{key} is a start error of pose.anchor=nominal; pose.anchor=first_truth "
                             "starts at the truth")
        return self

    def is_noisy(self) -> bool:
        """True when the config adds detection, pose or depth noise; noise.seeds alone is not noise."""
        return self.stages.det == "oracle_noisy" or not self.noise.is_zero()


class SplitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dev: list[int]
    val: list[int]
    test: list[int]
    exp: list[int]
    train: list[int]

    @model_validator(mode="after")
    def _check_no_overlap(self) -> SplitsConfig:
        seen: dict[int, str] = {}
        overlap: set[int] = set()
        for name in ("dev", "val", "test", "exp", "train"):
            for seed in getattr(self, name):
                if seed in seen:
                    overlap.add(seed)
                else:
                    seen[seed] = name
        if overlap:
            raise ValueError(f"splits overlap: {sorted(overlap)}")
        return self

    def split_of(self, seed: int) -> str | None:
        for name in ("dev", "val", "test", "exp", "train"):
            if seed in getattr(self, name):
                return name
        return None
