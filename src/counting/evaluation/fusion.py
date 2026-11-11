"""Fusion inputs, on the evaluation side.

Fixed settings. The fusion graph takes its fixed values (the noise the estimator assumes for the IMU,
the start priors and the floor of the odometry factor sigmas; `FuseFixedConfig`) from one file,
`FUSE_FIXED_PATH`: `configs/pose/fuse_fixed.yaml` relative to the working folder. It is not a config
key, and its sha256 goes into the results and the experiment id. A fused config stops before any
bundle is opened when the file is missing (`posesource.check_pose_supported`).

Truth source. `fusion_inputs` gathers what the fused truth source takes besides the poses: the frame
time of every frame (`frame_times` under `AssumedMotion()`), the IMU stream of every drive of the
session, made in memory by `imu_session` over `session_drives` (the function that also makes IMU
files) and keyed by drive (side, aisle, and the pitch without the rig or None under it), and the
standstill of each drive. With a noise seed the streams carry the noise of the `noise.imu_*` level keys
(drawn even when every key is 0); without one they are noiseless.

File source. Under `pose.source=file` the IMU streams come from the bundle's IMU file (`ImuFile`,
opened by `runner.open_imu_file`) under `pose.fuse=imu`: the stream of every drive (`drive_keys`) under
the noise seed, or under the file's noise seed 0 without one, with the frame times of the assumed motion
and the profile the file was made with. Under `pose.fuse=vo` there is no IMU: only the frame times
(under `AssumedMotion()`), and `pose.vio.stop_s` is refused.

IMU clock. The fusion believes the IMU clock is late by `noise.imu_time_offset_ms` (no draw; unset is
0): every stream it is handed, from memory or from the IMU file, has each sample time plus that
offset, while the frame times stay. The fusion does not estimate the offset.

Standstill. With `pose.vio.stop_s` above 0, the standstill of a drive runs from the believed time of
its stream's first sample for `stop_s`. Over the window of the same length from the first sample's
true time, the drive's body must stay within `STANDSTILL_TOL_M` of its position at the window's start,
or the run stops with `ValueError`. Unset or 0 is no standstill.

Extrinsic error. `imu_errors` draws the camera-to-IMU calibration error (noise step 11) the fusion
believes: one for the reference camera under the rig, one per pass pitch without it.

Floors. A start prior sigma is never below `START_SIGMA_FLOOR` and a bias prior sigma never below
`BIAS_SIGMA_FLOOR`: a start error key that is unset or 0, or a noiseless IMU, would otherwise give a
sigma of 0 and a singular graph.

Results. A fused result names the IMU streams it read (`FusionInputs.imu`) and the fusion it ran
(`fusion_entry`: the gtsam version, the fixed settings' sha256, the estimator's IMU noise and the
standstill length).

Importing this module loads no cv2, gtsam or torch; `has_gtsam` only looks the package up.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import hypot
from pathlib import Path

import numpy as np

from counting.config.eval_schema import EvalAppConfig
from counting.config.loader import load_yaml_model
from counting.config.pose_schema import FuseFixedConfig
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.imusim import (
    AssumedMotion, Drive, ImuProfile, ImuSampling, drive_keys, frame_times, imu_session, session_drives,
)
from counting.geometry.noise import STEP_IMU_EXTR, seeded_rigid
from counting.geometry.rig import interpolate_pose
from counting.odometry.imufile import ImuFile, ImuStream

FUSE_FIXED_PATH = Path("configs/pose/fuse_fixed.yaml")
START_SIGMA_FLOOR = 1e-4            # start prior sigma floor (rad for the yaw, m for the position)
BIAS_SIGMA_FLOOR = (1e-4, 1e-5)     # bias prior sigma floor: accelerometer (m/s^2), gyro (rad/s)
STANDSTILL_TOL_M = 1e-6             # how far the body may move in a standstill (m)
_EXTR_KEYS = ("imu_extr_trans_mm", "imu_extr_rot_deg")
_PITCH_TOL = 1e-6

DriveKey = tuple[str, int, float | None]


@dataclass(frozen=True)
class FuseFixed:
    """The fixed fusion settings of `FUSE_FIXED_PATH` and the sha256 of the file's bytes."""

    config: FuseFixedConfig
    sha256: str


def load_fuse_fixed() -> FuseFixed:
    """Read the fixed fusion settings from `FUSE_FIXED_PATH` as it is when called. A missing file raises
    `FileNotFoundError`."""
    path = FUSE_FIXED_PATH
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no fixed fusion settings; pose.fuse reads them")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return FuseFixed(config=load_yaml_model(path, FuseFixedConfig), sha256=sha256)


def has_gtsam() -> bool:
    """True when the gtsam package can be imported; it is not imported."""
    return importlib.util.find_spec("gtsam") is not None


def imu_errors(cfg: EvalAppConfig, noise_seed: int | None, *, scene: int,
               pitches: Iterable[float]) -> dict[float, np.ndarray]:
    """The camera-to-IMU calibration error `dX_imu` of each reference camera, keyed by its pitch `float(q)`.

    Only with `pose.fuse=imu`, `noise_seed` and an `imu_extr_*` key set (0 included); otherwise `{}`. The
    reference camera is the one at `pose.rig_ref` under the rig and each distinct pitch q of `pitches` (the
    caller's pass pitches) without it, and gets `seeded_rigid((cfg.seed, noise_seed, STEP_IMU_EXTR, scene,
    round(q * 1000)), imu_extr_trans_mm / 1000, imu_extr_rot_deg)`, an unset key counting as 0.
    """
    pose, nz = cfg.pose, cfg.noise
    if (pose is None or pose.fuse != "imu" or noise_seed is None
            or all(getattr(nz, key) is None for key in _EXTR_KEYS)):
        return {}
    out: dict[float, np.ndarray] = {}
    for q in [pose.rig_ref] if pose.rig else pitches:
        q = float(q)
        if any(abs(q - p) <= _PITCH_TOL for p in out):
            continue
        out[q] = seeded_rigid((cfg.seed, noise_seed, STEP_IMU_EXTR, scene, round(q * 1000)),
                              nz.factor("imu_extr_trans_mm") / 1000.0, nz.factor("imu_extr_rot_deg"))
    return out


def start_sigma(cfg: EvalAppConfig, fixed: FuseFixed, *, planned: bool, pitch: bool) -> np.ndarray:
    """The sigmas (6,) of the start body pose prior: roll, pitch, yaw (rad), then x (along the travel),
    y (to the side) and z (height) in m.

    Roll and pitch are `start_roll_pitch_sigma_deg` under either anchor. Under `pose.anchor=first_truth`
    the start is the truth, so the yaw and the position are `START_SIGMA_FLOOR`. Under `nominal` the yaw
    joins `pass_side_rot_deg`, with `pitch` (one camera passing each pitch) `pass_pitch_rot_deg` and with
    `planned` (a start at the planned pose of a bundle rendered with body shake) the shake share
    `yaw_deg` by `hypot`; each position axis joins `pass_side_trans_m`, with `pitch` `pass_pitch_trans_m`
    and with `planned` the shake share of its axis (x `along_m`, y `side_m`, z `height_m`). Each is at least
    `START_SIGMA_FLOOR`.
    """
    fx, nz = fixed.config, cfg.noise
    tilt = float(np.radians(fx.start_roll_pitch_sigma_deg))
    if cfg.pose.anchor == "first_truth":
        return np.array([tilt, tilt] + [START_SIGMA_FLOOR] * 4, dtype=np.float64)
    share = fx.shake_share

    def joined(side: float, per_pitch: float, shake: float) -> float:
        return hypot(side, *([per_pitch] if pitch else []), *([shake] if planned else []))

    yaw = joined(nz.factor("pass_side_rot_deg"), nz.factor("pass_pitch_rot_deg"), share.yaw_deg)
    out = [tilt, tilt, max(float(np.radians(yaw)), START_SIGMA_FLOOR)]
    for axis_share in (share.along_m, share.side_m, share.height_m):
        trans = joined(nz.factor("pass_side_trans_m"), nz.factor("pass_pitch_trans_m"), axis_share)
        out.append(max(trans, START_SIGMA_FLOOR))
    return np.array(out, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class FusionInputs:
    """What a fused pose source takes besides the poses: the fixed settings, the drive layout (`rig` or
    `pass`), the time (s) of every frame by idx, the measured stream of every drive by drive key (side,
    aisle, pitch or None) on the believed IMU clock, the generating IMU profile (the bias prior sigmas
    and, under `pose.vio.estimator_imu=generator`, the estimator's noise come from it), the standstill
    (start, end) in s of every drive (None without one) and the `imu` entry of a result."""

    fixed: FuseFixed
    layout: str
    times: dict[int, float]
    streams: dict[DriveKey, ImuStream] | None
    profile: ImuProfile | None
    standstill: dict[DriveKey, tuple[float, float]] | None
    imu: dict | None


def _moved(drive: Drive, start: float, length: float) -> float:
    """The largest distance (m) of the body of `drive` from its position at `start` over
    [start, start + length] (its trajectory samples in the window and both ends)."""
    end = start + length
    first = interpolate_pose(drive.t, drive.body, start)[:3, 3]
    inside = drive.body[(drive.t >= start) & (drive.t <= end), :3, 3]
    points = np.vstack([inside, interpolate_pose(drive.t, drive.body, end)[None, :3, 3]])
    return float(np.linalg.norm(points - first, axis=1).max())


def _believed(streams: Mapping[DriveKey, ImuStream], offset: float) -> dict[DriveKey, ImuStream]:
    """Copies of `streams` on the believed IMU clock: each sample time plus `offset` (s), the samples
    themselves shared."""
    return {key: ImuStream(t=s.t + offset, gyro=s.gyro, acc=s.acc) for key, s in streams.items()}


def _standstill(run: SynthRun, drives: Sequence[Drive], true_streams: Mapping[DriveKey, ImuStream],
                believed: Mapping[DriveKey, ImuStream], stop: float) -> dict[DriveKey, tuple[float, float]]:
    """The standstill (start, end) in s of every drive of `drives` under `pose.vio.stop_s` = `stop` above 0
    (module docstring): from the first sample of the drive's stream in `believed` for `stop`. Raises
    `ValueError` (naming the bundle) when the drive's body moves over the window of the same length from
    the first sample of its stream in `true_streams`."""
    out: dict[DriveKey, tuple[float, float]] = {}
    for d in drives:
        key = (d.side, d.aisle, d.pitch_deg)
        if _moved(d, float(true_streams[key].t[0]), stop) > STANDSTILL_TOL_M:
            raise ValueError(f"{run.root}: pose.vio.stop_s={stop:g} is longer than the standstill of the drive "
                             f"on side {d.side}")
        begin = float(believed[key].t[0])
        out[key] = (begin, begin + stop)
    return out


def fusion_inputs(run: SynthRun, cfg: EvalAppConfig, noise_seed: int | None, *,
                  imu_file: ImuFile | None = None) -> FusionInputs | None:
    """The fusion inputs of `run` under `cfg` (module docstring), or None without fusion (no pose section
    or `pose.fuse=none`).

    Reads the fixed settings (`load_fuse_fixed`). The layout is `rig` under `pose.rig`, else `pass`. Truth
    source: with `noise_seed` the streams carry the noise of `ImuProfile.from_noise(cfg.noise)`, keyed by
    `cfg.seed`, `noise_seed` and the bundle seed; without one they are noiseless; `imu_file` is not read.
    File source under `pose.fuse=imu`: the streams of `imu_file` under `noise_seed` (0 without one), the
    frame times of its assumed motion and its profile; `ValueError` without `imu_file`, when the file has
    no assumed motion, or (`ImuFile.stream`) no stream for the noise seed or a drive. File source under
    `pose.fuse=vo`: the frame times only; `ValueError` when `pose.vio.stop_s` is set. The streams are
    handed on the believed IMU clock, each sample time plus `noise.imu_time_offset_ms` / 1000. Raises
    `ValueError` (naming the bundle) when `pose.vio.stop_s` is longer than a drive's standstill.
    """
    pose = cfg.pose
    if pose is None or pose.fuse == "none":
        return None
    fixed = load_fuse_fixed()
    layout = "rig" if pose.rig else "pass"
    stop = pose.vio.stop_s if pose.vio is not None else None
    offset = cfg.noise.factor("imu_time_offset_ms") / 1000
    if pose.source == "file":
        if pose.fuse == "vo":
            if stop is not None:
                raise ValueError("pose.vio.stop_s needs pose.fuse=imu")
            return FusionInputs(fixed=fixed, layout=layout, times=frame_times(run, AssumedMotion()), streams=None,
                                profile=None, standstill=None, imu=None)
        if imu_file is None:
            raise ValueError("pose.source=file with pose.fuse=imu needs its IMU file")
        meta = imu_file.meta
        if not isinstance(meta.get("motion"), dict):
            raise ValueError(f"{imu_file.path}: IMU file has no assumed motion")
        motion = AssumedMotion(**meta["motion"])
        times = frame_times(run, motion)
        seed = noise_seed if noise_seed is not None else 0
        true_streams = {key: imu_file.stream(layout, seed, *key) for key in drive_keys(run, layout)}
        streams = _believed(true_streams, offset)
        standstill = (_standstill(run, session_drives(run, layout, motion), true_streams, streams, stop) if stop
                      else None)
        imu = {"source": "file", "layout": layout,
               "stream": "per_noise_seed" if noise_seed is not None else "noise_seed_0", "name": pose.imu,
               "file": imu_file.path.name, "digest": imu_file.digest, "profile": meta.get("profile")}
        return FusionInputs(fixed=fixed, layout=layout, times=times, streams=streams,
                            profile=ImuProfile(**meta["noise"]), standstill=standstill, imu=imu)
    motion = AssumedMotion()
    times = frame_times(run, motion)
    drives = session_drives(run, layout, motion)
    profile = ImuProfile.from_noise(cfg.noise)
    made = imu_session(drives, profile if noise_seed is not None else None, ImuSampling(), imu_seed=cfg.seed,
                       noise_seed=noise_seed, scene=run.meta.seed)
    true_streams = dict(zip([(d.side, d.aisle, d.pitch_deg) for d in drives], made))
    streams = _believed(true_streams, offset)
    standstill = _standstill(run, drives, true_streams, streams, stop) if stop else None
    imu = {"source": "memory", "layout": layout, "stream": "per_noise_seed" if noise_seed is not None else "noiseless",
           "name": None, "file": None, "digest": None, "profile": None}
    return FusionInputs(fixed=fixed, layout=layout, times=times, streams=streams, profile=profile,
                        standstill=standstill, imu=imu)


def fusion_entry(cfg: EvalAppConfig, inputs: FusionInputs | None) -> dict | None:
    """The `fusion` entry of a result, or None without fusion (`inputs` None): the gtsam version (read with
    `importlib.metadata`, so gtsam is not imported), the sha256 of the fixed settings `inputs` read, the
    IMU noise the estimator assumed (`estimator`: None under `pose.fuse=vo`, which has no IMU,
    `generator` under `pose.vio.estimator_imu=generator`, else `lab`, the fixed settings' values) and the
    standstill length `pose.vio.stop_s` in s (`standstill_s`: None when unset or 0)."""
    if inputs is None:
        return None
    vio = cfg.pose.vio
    if cfg.pose.fuse == "vo":
        estimator = None
    else:
        estimator = "generator" if vio is not None and vio.estimator_imu == "generator" else "lab"
    return {"gtsam": importlib.metadata.version("gtsam"), "fuse_fixed_sha256": inputs.fixed.sha256,
            "estimator": estimator, "standstill_s": vio.stop_s if vio is not None and vio.stop_s else None}
