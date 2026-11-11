"""Cameras of the pose sources: the rig model and the chained drift model on the true poses, and the
chained poses of a pose cache.

This module is on the evaluation side: it reads the true camera poses of a bundle and hands the
lift and the tracker the cameras they use. Scoring always keeps the true cameras.

Flows. Under `pose.rig` the cameras of one side ride one body, so each side (`side_code`) is one
body flow, taken from the reference camera at `pose.rig_ref`. Without the rig each pass is its own
flow (`pass_key_code`), taken from that pass's camera. A flow's true body poses come from the
reference frames of the whole bundle (every frame of that pitch, side and aisle, by idx), so
thinned frames keep the drift of their position (`pos_k`).

Drift. Under `pose.source=truth`, with a drift key set (`DRIFT_CAMERA_KEYS`, a key at 0 included)
and a noise seed, every flow draws its walk, scale and rate (`flow_draws`), each side its start
error (`pass_side_*`) and, without the rig, each pass its pitch error (`pass_pitch_*`); draws are
made even when the values are 0, so configs with drift keys share their draws. The body chain
starts at the first true body pose times the side start error (`pose.anchor=nominal`; on a bundle
rendered with body shake, at the planned body pose of the first reference frame instead) or at the
first true body pose (`pose.anchor=first_truth`), then chains the noisy increments
(`noisy_increments`). A pitch error moves the chained pass about its middle (`about_middle`). With
zero sigmas and zero start and pitch errors the chain is the truth itself, unless it starts at the
planned body pose.

Time offset. On a bundle with frame times (`plans`), with drift and `drift_time_offset_s` set (a key at
0 included), each flow draws one time offset dt (`time_offset`, noise step 8, keyed by the pass code of
its reference pass). In place of its true body poses the flow then takes the true body at the time
t + dt of each reference frame (`interpolate_pose` between the reference frames, held at the first and
the last one), and its start moves with it by the inverse of the first true body pose times the first
moved one. Such a flow is chained from its start even when its chain would be the truth. Fusion takes
the same flow and keeps the frame times at its nodes. A dt of 0 keeps the true body poses; a bundle
without frame times stops on a value above 0.

Rig. A camera at pitch q is the body pose times the believed extrinsic: the reference camera's
known extrinsic, the turn from the reference camera to q and, for q other than the reference, the
rig calibration error (`rig_errors`, from `rig_extr_*`). With no error this is the true camera.

File source. Under `pose.source=file` the body poses of a flow come from the bundle's pose cache
(`PoseFile`), which holds each pass's camera poses relative to the pass's first cached frame. The
flows, their reference frames and the rig are as above. A flow's anchor is the first cached frame of
its reference pass, and its chain starts at the anchor's body pose times the side start error
(`pass_side_*`, under `pose.anchor=nominal` with a noise seed); that body pose is the planned one
under `pose.anchor=nominal` on a bundle rendered with body shake, and the true one otherwise. The
config refuses the flow keys (`FLOW_KEYS`): the cache is the odometry. The rig calibration error and,
without the rig, the pitch-to-pitch error are drawn as above. The start pose is all a file source
takes from the true or planned poses; a frame whose reference frame has no pose in the cache
raises `ValueError`. Under `pose.fuse` vo or imu the flow is fused instead of chained (below).

Fusion. Under `pose.source=truth` and `pose.fuse=imu`, each flow is one drive of the fusion graph
(`vio.fuse_drive`), with or without the rig and with or without drift. The body flow above (its
noisy increments with drift, the true ones without) is the odometry of the reference camera r: each
increment M becomes the camera motion X^-1 M X with X = `pitch_extrinsic(r)`, and the graph moves
it back into the body through the believed extrinsic X dX_imu (`believed_extrinsic`, with the
camera-to-IMU error of `fusion.imu_errors`), so the body takes the conjugate of the flow. The start
prior is the first body pose (the planned one where the chain above starts there) times the side
start error and, without the rig, the pitch error; its sigmas are `fusion.start_sigma`. The drive's
IMU stream (on the believed IMU clock), frame times and standstill come from `fusion.fusion_inputs`,
and with the GPS keys and a noise seed the first and the last node get a row-end fix of the true body
position (`gps_end_fixes`, noise step 14). A camera of the flow is the fused body pose times its
believed extrinsic: under the rig `believed_extrinsic(q, r, dX_imu, dX_q)`, without it
`pitch_extrinsic(q) @ dX_imu`.

Under `pose.source=file` and `pose.fuse` vo or imu, each flow is one drive too, whose nodes are the
cached frames of its reference pass at their frame times. Every camera of the flow (each pass of the
flow under the rig, the pass itself without it) brings its own odometry: each two neighbouring rows
of its pass's cache are a factor between the nodes of the reference frames at the same positions
(`pos_k` names a position and is never a distance; the base step is not read), with the motion
between the two cached poses, moved into the body through the camera's believed extrinsic (with
dX_imu under imu only). Under imu a pair whose pose fit failed (`failed` in the cache) gets no factor
and the IMU joins its nodes; under vo every pair is used and there is no IMU, start velocity, bias or
standstill. The cache's `prior` and `inliers` are not read. The factor sigmas are `pose.vio`'s
`vo_sigma_trans_mm` and `vo_sigma_rot_deg` (`ValueError` without them). The start prior is the start
of the chain above times, without the rig, the pitch error, with the sigmas of `fusion.start_sigma`;
there is no GPS fix. The IMU stream, the frame times and the standstill come from the IMU file
through `fusion.fusion_inputs`, and the cameras from the fused body as above.

Modes. `check_pose_supported` lets through `pose.source` truth or file, with or without the rig,
fusion (`pose.fuse=imu` on either source, `pose.fuse=vo` on a file source), and every `align.mode`.
In this order it stops fusion on `split=test` and a file source on `split=test` or with
`noise.drift_time_offset_s` above 0 with `NotImplementedError`, then fusion without gtsam
(`NotImplementedError`) or without the fixed fusion settings (`FileNotFoundError`). `pose_settings`
names the settings a result ran with, and `pose_cache_entry` the pose cache a file source read.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from counting.config.eval_schema import DRIFT_KEYS, GPS_KEYS, EvalAppConfig, NoiseConfig
from counting.data.formats.orchard_synth import FrameInput, FramePlan, PassKey
from counting.evaluation import fusion as fusion_mod
from counting.evaluation.fusion import FusionInputs
from counting.evaluation.imusim import gps_end_fixes
from counting.geometry.camera import Camera
from counting.geometry.noise import (
    STEP_GPS, STEP_PITCH, STEP_RIG, STEP_START, STEP_TIME, DriftSigmas, about_middle, chain_poses, flow_draws,
    noisy_increments, pass_key_code, seeded_rigid, side_code, time_offset, vo_factor_sigma,
)
from counting.geometry.posefile import PoseFile
from counting.geometry.rig import (
    believed_extrinsic, body_from_reference, interpolate_pose, pitch_extrinsic, pose_matrix, rig_pose, with_pose,
)
from counting.odometry import vio as vio_mod
from counting.odometry.imufile import GRAVITY, ImuStream
from counting.odometry.vio import GpsFix, VioCamera, VioDrive, VioImu
from counting.odometry.vo import VoFrame

# The drift keys that change the cameras (`depth_k` changes the depths only).
DRIFT_CAMERA_KEYS = tuple(key for key in DRIFT_KEYS if key != "depth_k")
_PITCH_TOL = 1e-6
_SIDE_KEYS = ("pass_side_trans_m", "pass_side_rot_deg")
_PITCH_KEYS = ("pass_pitch_trans_m", "pass_pitch_rot_deg")
_RIG_KEYS = ("rig_extr_trans_mm", "rig_extr_rot_deg")


def check_pose_supported(cfg: EvalAppConfig) -> None:
    """Refuse the pose settings the runner does not run, in this order: fusion (`pose.fuse` vo or imu)
    with `split=test`, `pose.source=file` with `split=test`, then `pose.source=file` with
    `noise.drift_time_offset_s` above 0, each with `NotImplementedError`; then fusion without gtsam
    (`NotImplementedError`, looked up with `importlib.util.find_spec` only) and fusion without the fixed
    fusion settings `fusion.FUSE_FIXED_PATH` (`FileNotFoundError`). A config without `pose` passes."""
    pose = cfg.pose
    fused = pose is not None and pose.fuse != "none"
    if fused and cfg.split == "test":
        raise NotImplementedError(f"pose.fuse={pose.fuse} with split=test is not supported")
    if pose is not None and pose.source == "file" and cfg.split == "test":
        raise NotImplementedError("pose.source=file with split=test is not supported")
    if pose is not None and pose.source == "file" and cfg.noise.factor("drift_time_offset_s") > 0:
        raise NotImplementedError("noise.drift_time_offset_s > 0 with pose.source=file is not supported; the "
                                  "camera-IMU time belief of a file source is noise.imu_time_offset_ms")
    if fused and not fusion_mod.has_gtsam():
        raise NotImplementedError(f"pose.fuse={pose.fuse} needs gtsam (the conda-forge build in the environment files)")
    if fused and not fusion_mod.FUSE_FIXED_PATH.is_file():
        raise FileNotFoundError(f"{fusion_mod.FUSE_FIXED_PATH}: no fixed fusion settings; pose.fuse reads them")


def _present(noise: NoiseConfig, keys: Iterable[str]) -> bool:
    """True when any of `keys` is set (a key set to 0 included)."""
    return any(getattr(noise, key) is not None for key in keys)


def _all_zero(noise: NoiseConfig, keys: Iterable[str]) -> bool:
    """True when every one of `keys` is unset or 0."""
    return all(noise.factor(key) == 0 for key in keys)


def drift_active(cfg: EvalAppConfig, noise_seed: int | None) -> bool:
    """True when `cfg` has a pose section with `pose.source=truth`, `noise_seed` is given and a key of
    `DRIFT_CAMERA_KEYS` is set (a key set to 0 included)."""
    return (cfg.pose is not None and cfg.pose.source == "truth" and noise_seed is not None
            and _present(cfg.noise, DRIFT_CAMERA_KEYS))


def pose_settings(cfg: EvalAppConfig, noise_seed: int | None, *, planned: bool = False,
                  cache: dict | None = None, imu: dict | None = None, fusion: dict | None = None) -> dict:
    """The pose settings of a result (`cfg` has a pose section): the pose source and its camera
    configuration, whether the drift model ran (`drift_active`; never under `pose.source=file`), the
    anchor and, with drift, a file source or fusion (`pose.fuse` vo or imu), the start
    (`pose.anchor=nominal` gives `nominal_from_plan` when `planned`, the start taken from the planned
    pose of a bundle rendered with body shake, and `nominal_from_first_frame` otherwise;
    `pose.anchor=first_truth` gives `first_frame_truth`), the rig, fusion and `align.mode` ("none"
    without an align section, a list for listed methods), the tracker and dedup settings the pose
    errors meet, `cache`: the pose cache a file source read (None for a truth source), `imu`: the IMU
    streams a fused run read (`FusionInputs.imu`; None without them), `fusion`: the fusion it ran
    (`fusion.fusion_entry`; None without fusion) and, last, `flow`: the optical-flow cache the tracker
    prediction read (None without one)."""
    pose, align = cfg.pose, cfg.align
    drift = drift_active(cfg, noise_seed)
    start = None
    if drift or pose.source == "file" or pose.fuse != "none":
        if pose.anchor == "nominal":
            start = "nominal_from_plan" if planned else "nominal_from_first_frame"
        else:
            start = "first_frame_truth"
    if align is None:
        mode: str | list[str] = "none"
    else:
        mode = list(align.mode) if isinstance(align.mode, list) else align.mode
    return {"config": pose.config_label, "source": pose.source, "drift": drift, "anchor": pose.anchor,
            "start": start, "rig": pose.rig, "rig_ref": pose.rig_ref, "fuse": pose.fuse, "align": mode,
            "predict": cfg.tracker.predict, "predict_window": cfg.tracker.predict_window, "dedup": cfg.stages.dedup,
            "dedup_radius_m": cfg.dedup.radius_m, "dedup_band_m": cfg.dedup.band_m,
            "gate_px_frac": cfg.tracker.gate_px_frac, "cache": cache, "imu": imu, "fusion": fusion, "flow": None}


def pose_cache_entry(cfg: EvalAppConfig, pose_file: PoseFile | None) -> dict | None:
    """The `cache` of `pose_settings` for the opened pose cache `pose_file`, or None without one: the cache
    name `pose.cache`, the file name, the content digest and the meta's `vo_config_hash` and `opencv`
    (None when the meta has none)."""
    if pose_file is None:
        return None
    return {"name": cfg.pose.cache, "file": pose_file.path.name, "digest": pose_file.digest,
            "vo_config_hash": pose_file.meta.get("vo_config_hash"), "opencv": pose_file.meta.get("opencv")}


def vo_inputs(frames: Sequence[FrameInput]) -> list[VoFrame]:
    """The odometry input of `frames`, in their order: idx, pass, a float64 copy of the intrinsics, the image
    size and the image paths. No pose goes in."""
    return [VoFrame(idx=f.idx, pass_key=f.pass_key, K=np.array(f.camera.K, dtype=np.float64),
                    width=f.camera.width, height=f.camera.height, rgb_path=f.rgb_path, depth_path=f.depth_path)
            for f in frames]


def drift_sigmas(noise: NoiseConfig) -> DriftSigmas:
    """The flow sigmas of `noise`: walk translation in m (from mm), walk rotation (deg), scale and
    heading rate (deg/m); an unset key is 0."""
    return DriftSigmas(trans_m=noise.factor("drift_step_trans_mm") / 1000.0,
                       rot_deg=noise.factor("drift_step_rot_deg"),
                       scale=noise.factor("drift_scale_sigma"),
                       rate_deg_per_m=noise.factor("drift_rate_deg_per_m"))


def rig_errors(cfg: EvalAppConfig, noise_seed: int | None, *, scene: int,
               pitches: Iterable[float]) -> dict[float, np.ndarray]:
    """The rig calibration error `dX_q` of each camera, keyed by its pitch `float(q)`.

    Only with a pose section, `pose.rig`, a `rig_extr_*` key set (0 included) and `noise_seed`;
    otherwise `{}`. Each distinct pitch q of `pitches` (the caller's pass pitches) more than 1e-6
    from `pose.rig_ref` gets `seeded_rigid((cfg.seed, noise_seed, STEP_RIG, scene, round(q * 1000)),
    rig_extr_trans_mm / 1000, rig_extr_rot_deg)`, an unset key counting as 0. The reference camera
    has no error.
    """
    pose, nz = cfg.pose, cfg.noise
    if pose is None or not pose.rig or noise_seed is None or not _present(nz, _RIG_KEYS):
        return {}
    out: dict[float, np.ndarray] = {}
    for q in pitches:
        q = float(q)
        if abs(q - pose.rig_ref) <= _PITCH_TOL or q in out:
            continue
        out[q] = seeded_rigid((cfg.seed, noise_seed, STEP_RIG, scene, round(q * 1000)),
                              nz.factor("rig_extr_trans_mm") / 1000.0, nz.factor("rig_extr_rot_deg"))
    return out


def _near(errors: Mapping[float, np.ndarray], pitch: float) -> np.ndarray | None:
    """The value of `errors` whose pitch is within 1e-6 of `pitch`, or None."""
    return next((v for q, v in errors.items() if abs(q - pitch) <= _PITCH_TOL), None)


def _flows(passes: Iterable[PassKey], rig: bool) -> dict[tuple[int, ...], list[PassKey]]:
    """The pass keys of each flow, in `passes` order: one flow per side (`side_code`) under the rig,
    one per pass (`pass_key_code`) without it."""
    flows: dict[tuple[int, ...], list[PassKey]] = {}
    for key in passes:
        flows.setdefault(side_code(key) if rig else pass_key_code(key), []).append(key)
    return flows


def _reference_frames(frames: Sequence[FrameInput], lead: PassKey, ref: float, where: Path) -> list[FrameInput]:
    """The frames of the whole bundle at the side and aisle of `lead` and the pitch `ref` (within 1e-6),
    by idx. Raises `ValueError` naming `where` when there are none."""
    refs = sorted((f for f in frames if f.pass_key.side == lead.side and f.pass_key.aisle == lead.aisle
                   and abs(f.pass_key.pitch_deg - ref) <= _PITCH_TOL), key=lambda f: f.idx)
    if not refs:
        raise ValueError(f"{where}: pose.rig_ref={ref:g} has no pass on side {lead.side} (aisle {lead.aisle})")
    return refs


def _flow_cameras(
    keys: Sequence[PassKey], passes: Mapping[PassKey, Sequence[FrameInput]], refs: Sequence[FrameInput],
    body: np.ndarray, row_of: Callable[[int], int], *, ref: float, rig: bool,
    errors: Mapping[float, np.ndarray], where: Path, own: bool = False, imu_error: np.ndarray | None = None,
) -> dict[PassKey, dict[int, Camera]]:
    """The cameras of the passes `keys` of one flow, by pass key and frame idx.

    Each frame takes the body pose `body[row_of(rf.idx)]` of the reference frame rf at its `pos_k`.
    Under the rig its camera is that body pose times the believed extrinsic of its pitch q (with the
    camera-to-IMU error `imu_error` and, for q other than `ref`, the rig calibration error of
    `errors`); without the rig it is the body pose times `pitch_extrinsic(q)` (times `imu_error` when
    given), or, with `own`, the frame's own camera object. Raises `ValueError` naming `where` when no
    reference frame is at a frame's `pos_k`.
    """
    at = {f.pos_k: f for f in refs}
    out: dict[PassKey, dict[int, Camera]] = {}
    for key in keys:
        q = key.pitch_deg
        believed = None
        if rig:
            dX = None if abs(q - ref) <= _PITCH_TOL else _near(errors, q)
            believed = believed_extrinsic(q, ref, imu_error, dX)
        cams: dict[int, Camera] = {}
        for f in passes[key]:
            rf = at.get(f.pos_k)
            if rf is None:
                raise ValueError(f"{where}: pose.rig_ref={ref:g} has no frame at side {key.side}, "
                                 f"pos_k {f.pos_k}; the rig needs the reference camera at every position")
            i = row_of(rf.idx)
            if believed is not None:
                cams[f.idx] = with_pose(f.camera, rig_pose(body[i], believed))
            elif own:
                cams[f.idx] = f.camera
            elif imu_error is not None:
                cams[f.idx] = with_pose(f.camera, body[i] @ pitch_extrinsic(q) @ imu_error)
            else:
                cams[f.idx] = with_pose(f.camera, body[i] @ pitch_extrinsic(q))
        out[key] = cams
    return out


def _rigid_inverse(T: np.ndarray) -> np.ndarray:
    """The inverse of the rigid transform `T` (4, 4) in closed form."""
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -(T[:3, :3].T @ T[:3, 3])
    return out


def _relative_motions(poses: np.ndarray) -> np.ndarray:
    """The relative motions (n - 1, 4, 4) between the n poses `poses`, `poses[k - 1]^-1 @ poses[k]`."""
    R_prev = poses[:-1, :3, :3]
    out = np.repeat(np.eye(4)[None], len(poses) - 1, axis=0)
    out[:, :3, :3] = np.einsum("nji,njk->nik", R_prev, poses[1:, :3, :3])
    out[:, :3, 3] = np.einsum("nji,nj->ni", R_prev, poses[1:, :3, 3] - poses[:-1, :3, 3])
    return out


def _drive_stream(fusion: FusionInputs, lead: PassKey, pitch: float | None,
                  where: Path) -> tuple[tuple[str, int, float | None], ImuStream]:
    """The drive key and the IMU stream of the drive on the side and aisle of `lead` at `pitch` (within
    1e-6; None under the rig). Raises `ValueError` naming `where` when `fusion` has none."""
    for key, stream in (fusion.streams or {}).items():
        side, aisle, q = key
        if side == lead.side and aisle == lead.aisle and (
                q is None if pitch is None else q is not None and abs(q - pitch) <= _PITCH_TOL):
            return key, stream
    shown = "" if pitch is None else f", pitch {pitch:g}"
    raise ValueError(f"{where}: no IMU stream for the drive on side {lead.side} (aisle {lead.aisle}{shown})")


def _vio_imu(cfg: EvalAppConfig, fusion: FusionInputs, stream: ImuStream) -> VioImu:
    """The IMU of a drive: `stream`, the noise the estimator assumes and the bias prior sigmas.

    The estimator takes the fixed settings' `estimator` unless `pose.vio.estimator_imu=generator`, which
    takes the generating profile's white noise and walks (`ValueError` when one is 0). The bias prior
    sigmas are the profile's turn-on bias sigmas, at least `fusion.BIAS_SIGMA_FLOOR`."""
    profile = fusion.profile
    settings = cfg.pose.vio
    if settings is not None and settings.estimator_imu == "generator":
        values = {name: 0.0 if profile is None else float(getattr(profile, name))
                  for name in ("gyro_n", "acc_n", "gyro_w", "acc_w")}
        if zero := next((name for name, v in values.items() if not v > 0), None):
            raise ValueError(f"pose.vio.estimator_imu=generator needs the generating profile's {zero} above 0")
    else:
        est = fusion.fixed.config.estimator
        values = {"gyro_n": est.gyro_n, "acc_n": est.acc_n, "gyro_w": est.gyro_w, "acc_w": est.acc_w}
    acc_floor, gyro_floor = fusion_mod.BIAS_SIGMA_FLOOR
    gyro_b0 = 0.0 if profile is None else float(np.radians(profile.gyro_b0_dps))
    acc_b0 = 0.0 if profile is None else profile.acc_b0_mg * 1e-3 * GRAVITY
    return VioImu(stream=stream, **values, gyro_b0=max(gyro_b0, gyro_floor), acc_b0=max(acc_b0, acc_floor))


def _fused_body(
    cfg: EvalAppConfig, noise_seed: int | None, fusion: FusionInputs, *, scene: int, flow: tuple[int, ...],
    lead: PassKey, ref: float, refs: Sequence[FrameInput], body_true: np.ndarray, increments: np.ndarray,
    start: np.ndarray, planned: bool, base_step_m: float, imu_error: np.ndarray | None, where: Path,
) -> np.ndarray:
    """The fused body poses (n, 4, 4) of one flow of the truth source (module docstring): one drive whose
    nodes are the reference frames `refs`, its body flow `increments` seen by the reference camera at
    `ref` through the believed extrinsic, the start prior at `start`, the drive's IMU, the row-end GPS
    fixes of `body_true` with the GPS keys, and the drive's standstill."""
    pose, nz, fixed = cfg.pose, cfg.noise, fusion.fixed.config
    n = len(refs)
    X = pitch_extrinsic(ref)
    camera = VioCamera(extrinsic=believed_extrinsic(ref, ref, imu_error, None),
                       pairs=np.stack([np.arange(n - 1), np.arange(1, n)], axis=1),
                       motions=_rigid_inverse(X) @ increments @ X, use=np.ones(n - 1, dtype=bool))
    trans, rot = vo_factor_sigma(drift_sigmas(nz), base_step_m, fixed.vo_sigma_floor_trans_mm / 1000.0,
                                 fixed.vo_sigma_floor_rot_deg)
    key, stream = _drive_stream(fusion, lead, None if pose.rig else ref, where)
    gps: tuple[GpsFix, ...] = ()
    if noise_seed is not None and _present(nz, GPS_KEYS):
        h, v = nz.factor("gps_end_h_m"), nz.factor("gps_end_v_m")
        first, last = gps_end_fixes((cfg.seed, noise_seed, STEP_GPS, scene, *flow), body_true[0][:3, 3],
                                    body_true[-1][:3, 3], h, v)
        sigma = np.array([h, h, v], dtype=np.float64)
        gps = (GpsFix(node=0, position=first, sigma=sigma), GpsFix(node=n - 1, position=last, sigma=sigma))
    drive = VioDrive(
        times=np.array([fusion.times[f.idx] for f in refs], dtype=np.float64), cameras=(camera,), start=start,
        start_sigma=fusion_mod.start_sigma(cfg, fusion.fixed, planned=planned, pitch=not pose.rig),
        start_velocity=start[:3, :3] @ np.array([fixed.start_speed_mps, 0.0, 0.0]),
        start_velocity_sigma=fixed.start_speed_sigma_mps, odometry_sigma=(trans, float(np.radians(rot))),
        imu=_vio_imu(cfg, fusion, stream), gps=gps,
        standstill=None if fusion.standstill is None else fusion.standstill[key])
    return vio_mod.fuse_drive(drive)


def model_cameras(
    frames: Sequence[FrameInput], passes: Mapping[PassKey, Sequence[FrameInput]], cfg: EvalAppConfig,
    noise_seed: int | None, *, scene: int, base_step_m: float | None,
    plans: Mapping[int, FramePlan] | None = None, pose_file: PoseFile | None = None,
    fusion: FusionInputs | None = None,
) -> dict[PassKey, dict[int, Camera]] | None:
    """The camera of every frame of `passes`, by pass key and frame idx, under `cfg.pose`.

    Returns None (the true cameras) without a pose section, or under `pose.source=truth` without the
    rig, without drift (`drift_active`) and without fusion. `frames` are all the frames of the bundle:
    the reference frames of a flow come from them. `scene` is the bundle seed of the noise keys and
    `base_step_m` its base step. `plans` are the planned frames by idx of a bundle rendered with body
    shake (`SynthRun.plans`): with drift and `pose.anchor=nominal`, a flow's chain then starts at the
    planned body pose of its first reference frame (times the side start error) and runs even with
    zero sigmas and errors. Under `pose.source=truth` and `pose.fuse=imu` each flow's body is fused with
    the drive's IMU from `fusion` (`fusion.fusion_inputs`; see the module docstring), with or without
    the rig and drift.
    Under `pose.source=file` every pass gets the cameras of `pose_file`, the bundle's opened pose cache
    (see the module docstring); with `plans` and `pose.anchor=nominal` a flow then starts at the planned
    body pose of its anchor. Under `pose.source=file` with `pose.fuse` vo or imu each flow fuses the
    odometry of its cameras in the cache (with the IMU of `fusion` under imu).

    With drift, `plans` and `noise.drift_time_offset_s` set, each flow takes its body poses at the
    believed time t + dt of its reference frames, and its start moves with them (module docstring).

    Raises `ValueError` (naming the bundle folder) when the drift model needs frame times
    (`noise.drift_time_offset_s` > 0 without `plans`) or a sequence bundle (`base_step_m` None), when a
    flow has no reference pass, or when a frame has no reference frame at its `pos_k`. Under `pose.source=truth` and `pose.fuse=imu` it also raises
    `ValueError` without `fusion` and on a bundle that is not a sequence. Under `pose.source=file` it
    also raises `ValueError` without `pose_file`, on a bundle that is not a sequence, and (naming the
    cache path) when the anchor is not a frame of the reference pass or a frame's reference frame has
    no pose in the cache; fused, also without `fusion` or the `pose.vio` factor sigmas, and (naming the
    cache path) when a cached frame has no node. Without the rig, a truth-source flow whose chain is the
    truth gives each frame its own camera object.
    """
    pose, nz = cfg.pose, cfg.noise
    if pose is None:
        return None
    if pose.source == "file":
        return _file_cameras(frames, passes, cfg, noise_seed, scene=scene, base_step_m=base_step_m,
                             plans=plans, pose_file=pose_file, fusion=fusion)
    drift = drift_active(cfg, noise_seed)
    fused = pose.fuse == "imu"
    if fused and fusion is None:
        raise ValueError("pose.fuse=imu needs the fusion inputs (frame times, IMU streams, fixed settings)")
    if not fused and not drift and not pose.rig:
        return None
    where = frames[0].depth_path.parent
    if fused and base_step_m is None:
        raise ValueError(f"{where}: the fused truth source needs a sequence bundle (frame positions)")
    if drift and plans is None and nz.factor("drift_time_offset_s") > 0:
        raise ValueError(f"{where}: noise.drift_time_offset_s needs frame times; this bundle has none")
    if drift and base_step_m is None:
        raise ValueError(f"{where}: the drift model needs a sequence bundle (frame positions)")
    timed = drift and plans is not None and nz.drift_time_offset_s is not None
    sigmas = drift_sigmas(nz)
    planned = drift and plans is not None and pose.anchor == "nominal"
    no_start = pose.anchor == "first_truth" or _all_zero(nz, _SIDE_KEYS)
    flat = not drift or (not planned and sigmas.is_zero() and no_start and _all_zero(nz, _PITCH_KEYS))
    errors = rig_errors(cfg, noise_seed, scene=scene, pitches=[key.pitch_deg for key in passes])
    imu_errors = fusion_mod.imu_errors(cfg, noise_seed, scene=scene,
                                       pitches=[key.pitch_deg for key in passes]) if fused else {}

    out: dict[PassKey, dict[int, Camera]] = {}
    for flow, keys in _flows(passes, pose.rig).items():
        lead = keys[0]
        ref = float(pose.rig_ref) if pose.rig else lead.pitch_deg
        side = side_code(lead)
        refs = _reference_frames(frames, lead, ref, where)
        body_true = np.stack([body_from_reference(pose_matrix(f.camera), pitch_extrinsic(ref)) for f in refs])
        draws = start_error = pitch_error = imu_error = None
        if drift:
            draws = flow_draws(cfg.seed, noise_seed, scene, flow, [f.pos_k for f in refs])
            if _present(nz, _SIDE_KEYS):
                start_error = seeded_rigid((cfg.seed, noise_seed, STEP_START, scene, *side),
                                           nz.factor("pass_side_trans_m"), nz.factor("pass_side_rot_deg"))
            if not pose.rig and _present(nz, _PITCH_KEYS):
                pitch_error = seeded_rigid((cfg.seed, noise_seed, STEP_PITCH, scene, *side, round(ref * 1000)),
                                           nz.factor("pass_pitch_trans_m"), nz.factor("pass_pitch_rot_deg"))
        body_src, shift = body_true, None
        if timed:
            dt = time_offset((cfg.seed, noise_seed, STEP_TIME, scene, *pass_key_code(refs[0].pass_key)),
                             nz.factor("drift_time_offset_s"))
            if dt != 0:
                t = np.array([plans[f.idx].t for f in refs], dtype=np.float64)
                body_src = np.stack([interpolate_pose(t, body_true, x + dt) for x in t])
                shift = _rigid_inverse(body_true[0]) @ body_src[0]
        first = body_true[0]
        if planned:
            first = body_from_reference(pose_matrix(plans[refs[0].idx].plan), pitch_extrinsic(ref))
        if shift is not None:
            first = first @ shift
        if fused:
            start = first
            for error in (start_error, pitch_error):
                if error is not None:
                    start = start @ error
            imu_error = _near(imu_errors, ref)
            increments = (noisy_increments(body_src, draws, sigmas, base_step_m) if drift
                          else _relative_motions(body_src))
            body = _fused_body(cfg, noise_seed, fusion, scene=scene, flow=flow, lead=lead, ref=ref, refs=refs,
                               body_true=body_true, increments=increments, start=start, planned=planned,
                               base_step_m=base_step_m, imu_error=imu_error, where=where)
        elif flat and shift is None:
            body = body_true
        else:
            start = first if no_start else first @ start_error
            body = chain_poses(start, noisy_increments(body_src, draws, sigmas, base_step_m))
            if pitch_error is not None:
                body = about_middle(body, pitch_error)
        rows = {f.idx: i for i, f in enumerate(refs)}
        out.update(_flow_cameras(keys, passes, refs, body, rows.__getitem__, ref=ref, rig=pose.rig,
                                 errors=errors, where=where, own=body is body_true, imu_error=imu_error))
    return {key: out[key] for key in passes}


def _file_odometry(
    cfg: EvalAppConfig, pose_file: PoseFile, keys: Sequence[PassKey], frame_of: Mapping[int, FrameInput],
    reference: tuple[int, int, int], *, ref: float, errors: Mapping[float, np.ndarray],
    imu_error: np.ndarray | None, where: Path,
) -> tuple[VioCamera, ...]:
    """The odometry of every camera of one flow of the file source, for its fusion graph (module docstring).

    The nodes are the cached frames of the reference pass `reference` (its pass code). Each pass of
    `keys` (in their order) is one camera: each two neighbouring rows (k - 1, k) of its cache join the
    nodes of the reference frames at their positions (`pos_k`, the name of a position only), with the
    motion `rel[k - 1]^-1 rel[k]`; under `pose.fuse=vo` every pair is used, under `imu` the pairs whose
    pose fit did not fail. The believed extrinsic is `believed_extrinsic(q, ref, dX_imu, dX_q)` under the
    rig and `pitch_extrinsic(q) @ dX_imu` without it (`imu_error` is dX_imu, None under `vo`). Raises
    `ValueError` naming the cache path when a cached frame is not a frame of the bundle or has no node.
    """
    pose = cfg.pose

    def frame(i: int, code: tuple[int, int, int]) -> FrameInput:
        if i not in frame_of:
            raise ValueError(f"{pose_file.path}: frame {i} is not a frame of pass {code} in {where}")
        return frame_of[i]

    node_of_pos = {frame(i, reference).pos_k: k for k, i in enumerate(pose_file.idxs(reference).tolist())}
    cameras = []
    for key in keys:
        q, code = key.pitch_deg, pass_key_code(key)
        odometry = pose_file.odometry(code)
        nodes = []
        for i in odometry.idx.tolist():
            f = frame(i, code)
            if f.pos_k not in node_of_pos:
                raise ValueError(f"{pose_file.path}: frame {i} of pass {code} has no node; the reference pass has "
                                 f"no cached frame at pos_k {f.pos_k}")
            nodes.append(node_of_pos[f.pos_k])
        node = np.asarray(nodes, dtype=np.int64)
        motions = _relative_motions(odometry.rel_c2w)
        use = np.ones(len(motions), dtype=bool) if pose.fuse == "vo" else odometry.failed[1:] == 0
        if pose.rig:
            dX = None if abs(q - ref) <= _PITCH_TOL else _near(errors, q)
            extrinsic = believed_extrinsic(q, ref, imu_error, dX)
        else:
            extrinsic = pitch_extrinsic(q) if imu_error is None else pitch_extrinsic(q) @ imu_error
        cameras.append(VioCamera(extrinsic=extrinsic, pairs=np.stack([node[:-1], node[1:]], axis=1),
                                 motions=motions, use=use))
    return tuple(cameras)


def _fused_file_body(
    cfg: EvalAppConfig, fusion: FusionInputs, cameras: tuple[VioCamera, ...], *, lead: PassKey, ref: float,
    idxs: np.ndarray, start: np.ndarray, planned: bool, where: Path,
) -> np.ndarray:
    """The fused body poses (len(idxs), 4, 4) of one flow of the file source (module docstring): one drive
    whose nodes are the reference pass's cached frames `idxs` at their frame times, the odometry of
    `cameras` with the `pose.vio` sigmas, the start prior at `start` and, under `pose.fuse=imu`, the
    drive's IMU and standstill. No GPS fix."""
    pose, fixed = cfg.pose, fusion.fixed.config
    trans, rot = pose.vio.vo_sigma_trans_mm / 1000.0, float(np.radians(pose.vio.vo_sigma_rot_deg))
    imu, standstill = None, None
    if pose.fuse == "imu":
        key, stream = _drive_stream(fusion, lead, None if pose.rig else ref, where)
        imu = _vio_imu(cfg, fusion, stream)
        standstill = None if fusion.standstill is None else fusion.standstill[key]
    drive = VioDrive(
        times=np.array([fusion.times[int(i)] for i in idxs], dtype=np.float64), cameras=cameras, start=start,
        start_sigma=fusion_mod.start_sigma(cfg, fusion.fixed, planned=planned, pitch=not pose.rig),
        start_velocity=start[:3, :3] @ np.array([fixed.start_speed_mps, 0.0, 0.0]),
        start_velocity_sigma=fixed.start_speed_sigma_mps,
        odometry_sigma=(trans, rot), imu=imu, standstill=standstill)
    return vio_mod.fuse_drive(drive)


def _file_cameras(
    frames: Sequence[FrameInput], passes: Mapping[PassKey, Sequence[FrameInput]], cfg: EvalAppConfig,
    noise_seed: int | None, *, scene: int, base_step_m: float | None,
    plans: Mapping[int, FramePlan] | None, pose_file: PoseFile | None, fusion: FusionInputs | None = None,
) -> dict[PassKey, dict[int, Camera]]:
    """The cameras of `model_cameras` under `pose.source=file`: each flow chains the poses of
    `pose_file` from the body pose of its anchor, the first cached frame of its reference pass, or under
    `pose.fuse` vo or imu fuses the odometry of every camera of the flow with `fusion` (module
    docstring)."""
    pose, nz = cfg.pose, cfg.noise
    if pose_file is None:
        raise ValueError("pose.source=file needs an opened pose file")
    fused = pose.fuse != "none"
    if fused and fusion is None:
        raise ValueError(f"pose.fuse={pose.fuse} needs the fusion inputs (frame times, IMU streams, fixed settings)")
    if fused and (pose.vio is None or pose.vio.vo_sigma_trans_mm is None or pose.vio.vo_sigma_rot_deg is None):
        raise ValueError(f"pose.source=file with pose.fuse={pose.fuse} needs pose.vio.vo_sigma_trans_mm and "
                         "pose.vio.vo_sigma_rot_deg")
    where = frames[0].depth_path.parent
    if base_step_m is None:
        raise ValueError(f"{where}: pose.source=file needs a sequence bundle (frame positions)")
    nominal = pose.anchor == "nominal"
    errors = rig_errors(cfg, noise_seed, scene=scene, pitches=[key.pitch_deg for key in passes])
    imu_errors = fusion_mod.imu_errors(cfg, noise_seed, scene=scene,
                                       pitches=[key.pitch_deg for key in passes]) if fused else {}
    frame_of = {f.idx: f for f in frames}

    out: dict[PassKey, dict[int, Camera]] = {}
    for keys in _flows(passes, pose.rig).values():
        lead = keys[0]
        ref = float(pose.rig_ref) if pose.rig else lead.pitch_deg
        side, extrinsic = side_code(lead), pitch_extrinsic(ref)
        refs = _reference_frames(frames, lead, ref, where)
        code = pass_key_code(refs[0].pass_key)
        a = pose_file.anchor(code)
        anchor = next((f for f in refs if f.idx == a), None)
        if anchor is None:
            raise ValueError(f"{pose_file.path}: frame {a} is not a frame of pass {code} in {where}")
        anchor_cam = plans[a].plan if nominal and plans is not None else anchor.camera
        start = body_from_reference(pose_matrix(anchor_cam), extrinsic)
        if nominal and noise_seed is not None and _present(nz, _SIDE_KEYS):
            start = start @ seeded_rigid((cfg.seed, noise_seed, STEP_START, scene, *side),
                                         nz.factor("pass_side_trans_m"), nz.factor("pass_side_rot_deg"))
        pitch_error = None
        if not pose.rig and noise_seed is not None and _present(nz, _PITCH_KEYS):
            pitch_error = seeded_rigid((cfg.seed, noise_seed, STEP_PITCH, scene, *side, round(ref * 1000)),
                                       nz.factor("pass_pitch_trans_m"), nz.factor("pass_pitch_rot_deg"))
        idxs = pose_file.idxs(code)
        imu_error = _near(imu_errors, ref)
        if fused:
            if pitch_error is not None:
                start = start @ pitch_error
            cameras = _file_odometry(cfg, pose_file, keys, frame_of, code, ref=ref, errors=errors,
                                     imu_error=imu_error, where=where)
            body = _fused_file_body(cfg, fusion, cameras, lead=lead, ref=ref, idxs=idxs, start=start,
                                    planned=nominal and plans is not None, where=where)
        else:
            chained = pose_file.poses(code, start @ extrinsic, idxs)
            body = np.stack([body_from_reference(T, extrinsic) for T in chained])
            if pitch_error is not None:
                body = about_middle(body, pitch_error)
        rows = {int(i): row for row, i in enumerate(idxs)}

        def row_of(idx: int, rows: dict[int, int] = rows, code: tuple[int, int, int] = code) -> int:
            if idx not in rows:
                raise ValueError(f"{pose_file.path}: no pose for frame {idx} of pass {code}")
            return rows[idx]

        out.update(_flow_cameras(keys, passes, refs, body, row_of, ref=ref, rig=pose.rig, errors=errors,
                                 where=where, imu_error=imu_error))
    return {key: out[key] for key in passes}
