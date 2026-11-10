"""Cameras of the truth pose source: the rig model and the chained drift model on the true poses.

This module is on the evaluation side: it reads the true camera poses of a bundle and hands the
lift and the tracker the cameras they use. Scoring always keeps the true cameras.

Flows. Under `pose.rig` the cameras of one side ride one body, so each side (`side_code`) is one
body flow, taken from the reference camera at `pose.rig_ref`. Without the rig each pass is its own
flow (`pass_key_code`), taken from that pass's camera. A flow's true body poses come from the
reference frames of the whole bundle (every frame of that pitch, side and aisle, by idx), so
thinned frames keep the drift of their position (`pos_k`).

Drift. With a drift key set (`DRIFT_CAMERA_KEYS`, a key at 0 included) and a noise seed, every
flow draws its walk, scale and rate (`flow_draws`), each side its start error (`pass_side_*`) and,
without the rig, each pass its pitch error (`pass_pitch_*`); draws are made even when the values
are 0, so configs with drift keys share their draws. The body chain starts at the first true body
pose times the side start error (`pose.anchor=nominal`) or at the first true body pose
(`pose.anchor=first_truth`), then chains the noisy increments (`noisy_increments`). A pitch error
moves the chained pass about its middle (`about_middle`). With zero sigmas and zero start and
pitch errors the chain is the truth itself.

Rig. A camera at pitch q is the body pose times the believed extrinsic: the reference camera's
known extrinsic, the turn from the reference camera to q and, for q other than the reference, the
rig calibration error (`rig_errors`, from `rig_extr_*`). With no error this is the true camera.

Modes. `check_pose_supported` lets through `pose.source=truth` without fusion (`pose.fuse=none`),
with or without the rig, and `align.mode` none or fit_truth_pass. It stops `pose.source=file`,
`pose.fuse=vo` or `imu`, and the other alignment modes with `NotImplementedError`.
`pose_settings` names the settings a result ran with.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from counting.config.eval_schema import DRIFT_KEYS, EvalAppConfig, NoiseConfig
from counting.data.formats.orchard_synth import FrameInput, PassKey
from counting.geometry.camera import Camera
from counting.geometry.noise import (
    STEP_PITCH, STEP_RIG, STEP_START, DriftSigmas, about_middle, chain_poses, flow_draws, noisy_increments,
    pass_key_code, seeded_rigid, side_code,
)
from counting.geometry.rig import (
    believed_extrinsic, body_from_reference, pitch_extrinsic, pose_matrix, rig_pose, with_pose,
)

# The drift keys that change the cameras (`depth_k` changes the depths only).
DRIFT_CAMERA_KEYS = tuple(key for key in DRIFT_KEYS if key != "depth_k")
SUPPORTED_ALIGN = ("none", "fit_truth_pass")
_PITCH_TOL = 1e-6
_SIDE_KEYS = ("pass_side_trans_m", "pass_side_rot_deg")
_PITCH_KEYS = ("pass_pitch_trans_m", "pass_pitch_rot_deg")
_RIG_KEYS = ("rig_extr_trans_mm", "rig_extr_rot_deg")


def check_pose_supported(cfg: EvalAppConfig) -> None:
    """Refuse the pose settings the runner does not run, in this order: `pose.source=file`,
    `pose.fuse` vo or imu, then an `align.mode` other than none or fit_truth_pass (a list of methods
    shown joined by commas), each with `NotImplementedError`. A config without `pose` and `align`
    passes."""
    pose, align = cfg.pose, cfg.align
    if pose is not None and pose.source == "file":
        raise NotImplementedError("pose.source=file is not supported")
    if pose is not None and pose.fuse != "none":
        raise NotImplementedError(f"pose.fuse={pose.fuse} is not supported")
    if align is not None and (isinstance(align.mode, list) or align.mode not in SUPPORTED_ALIGN):
        mode = ",".join(align.mode) if isinstance(align.mode, list) else align.mode
        raise NotImplementedError(f"align.mode={mode} is not supported")


def _present(noise: NoiseConfig, keys: Iterable[str]) -> bool:
    """True when any of `keys` is set (a key set to 0 included)."""
    return any(getattr(noise, key) is not None for key in keys)


def _all_zero(noise: NoiseConfig, keys: Iterable[str]) -> bool:
    """True when every one of `keys` is unset or 0."""
    return all(noise.factor(key) == 0 for key in keys)


def drift_active(cfg: EvalAppConfig, noise_seed: int | None) -> bool:
    """True when `cfg` has a pose section, `noise_seed` is given and a key of `DRIFT_CAMERA_KEYS`
    is set (a key set to 0 included)."""
    return cfg.pose is not None and noise_seed is not None and _present(cfg.noise, DRIFT_CAMERA_KEYS)


def pose_settings(cfg: EvalAppConfig, noise_seed: int | None) -> dict:
    """The pose settings of a result (`cfg` has a pose section): the pose source and its camera
    configuration, whether the drift model ran (`drift_active`), the anchor and, with drift, the
    start (`nominal_from_first_frame` for `pose.anchor=nominal`, else `first_frame_truth`), the rig,
    fusion and `align.mode` ("none" without an align section, a list for listed methods), and the
    tracker and dedup settings the pose errors meet."""
    pose, align = cfg.pose, cfg.align
    drift = drift_active(cfg, noise_seed)
    start = None
    if drift:
        start = "nominal_from_first_frame" if pose.anchor == "nominal" else "first_frame_truth"
    if align is None:
        mode: str | list[str] = "none"
    else:
        mode = list(align.mode) if isinstance(align.mode, list) else align.mode
    return {"config": pose.config_label, "source": pose.source, "drift": drift, "anchor": pose.anchor,
            "start": start, "rig": pose.rig, "rig_ref": pose.rig_ref, "fuse": pose.fuse, "align": mode,
            "predict": cfg.tracker.predict, "dedup": cfg.stages.dedup, "dedup_radius_m": cfg.dedup.radius_m,
            "dedup_band_m": cfg.dedup.band_m, "gate_px_frac": cfg.tracker.gate_px_frac}


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


def model_cameras(
    frames: Sequence[FrameInput], passes: Mapping[PassKey, Sequence[FrameInput]], cfg: EvalAppConfig,
    noise_seed: int | None, *, scene: int, base_step_m: float | None,
) -> dict[PassKey, dict[int, Camera]] | None:
    """The camera of every frame of `passes`, by pass key and frame idx, under `cfg.pose`.

    Returns None (the true cameras) without a pose section, or without the rig and without drift
    (`drift_active`). `frames` are all the frames of the bundle: the reference frames of a flow come
    from them. `scene` is the bundle seed of the noise keys and `base_step_m` its base step.

    Raises `ValueError` (naming the bundle folder) when the drift model needs frame times
    (`noise.drift_time_offset_s` > 0) or a sequence bundle (`base_step_m` None), when a flow has no
    reference pass, or when a frame has no reference frame at its `pos_k`. Without the rig, a flow
    whose chain is the truth gives each frame its own camera object.
    """
    pose, nz = cfg.pose, cfg.noise
    if pose is None:
        return None
    drift = drift_active(cfg, noise_seed)
    if not drift and not pose.rig:
        return None
    where = frames[0].depth_path.parent
    if drift and nz.factor("drift_time_offset_s") > 0:
        raise ValueError(f"{where}: noise.drift_time_offset_s needs frame times; this bundle has none")
    if drift and base_step_m is None:
        raise ValueError(f"{where}: the drift model needs a sequence bundle (frame positions)")
    sigmas = drift_sigmas(nz)
    no_start = pose.anchor == "first_truth" or _all_zero(nz, _SIDE_KEYS)
    flat = not drift or (sigmas.is_zero() and no_start and _all_zero(nz, _PITCH_KEYS))
    errors = rig_errors(cfg, noise_seed, scene=scene, pitches=[key.pitch_deg for key in passes])

    flows: dict[tuple[int, ...], list[PassKey]] = {}
    for key in passes:
        flows.setdefault(side_code(key) if pose.rig else pass_key_code(key), []).append(key)
    out: dict[PassKey, dict[int, Camera]] = {}
    for flow, keys in flows.items():
        lead = keys[0]
        ref = float(pose.rig_ref) if pose.rig else lead.pitch_deg
        side = side_code(lead)
        refs = sorted((f for f in frames if f.pass_key.side == lead.side and f.pass_key.aisle == lead.aisle
                       and abs(f.pass_key.pitch_deg - ref) <= _PITCH_TOL), key=lambda f: f.idx)
        if not refs:
            raise ValueError(f"{where}: pose.rig_ref={ref:g} has no pass on side {lead.side} (aisle {lead.aisle})")
        body_true = np.stack([body_from_reference(pose_matrix(f.camera), pitch_extrinsic(ref)) for f in refs])
        draws = start_error = pitch_error = None
        if drift:
            draws = flow_draws(cfg.seed, noise_seed, scene, flow, [f.pos_k for f in refs])
            if _present(nz, _SIDE_KEYS):
                start_error = seeded_rigid((cfg.seed, noise_seed, STEP_START, scene, *side),
                                           nz.factor("pass_side_trans_m"), nz.factor("pass_side_rot_deg"))
            if not pose.rig and _present(nz, _PITCH_KEYS):
                pitch_error = seeded_rigid((cfg.seed, noise_seed, STEP_PITCH, scene, *side, round(ref * 1000)),
                                           nz.factor("pass_pitch_trans_m"), nz.factor("pass_pitch_rot_deg"))
        if flat:
            body = body_true
        else:
            start = body_true[0] if no_start else body_true[0] @ start_error
            body = chain_poses(start, noisy_increments(body_true, draws, sigmas, base_step_m))
            if pitch_error is not None:
                body = about_middle(body, pitch_error)
        at = {f.pos_k: i for i, f in enumerate(refs)}
        for key in keys:
            q = key.pitch_deg
            believed = None
            if pose.rig:
                dX = None if abs(q - ref) <= _PITCH_TOL else _near(errors, q)
                believed = believed_extrinsic(q, ref, None, dX)
            cams: dict[int, Camera] = {}
            for f in passes[key]:
                i = at.get(f.pos_k)
                if i is None:
                    raise ValueError(f"{where}: pose.rig_ref={ref:g} has no frame at side {key.side}, "
                                     f"pos_k {f.pos_k}; the rig needs the reference camera at every position")
                if believed is not None:
                    cams[f.idx] = with_pose(f.camera, rig_pose(body[i], believed))
                elif body is body_true:
                    cams[f.idx] = f.camera
                else:
                    cams[f.idx] = with_pose(f.camera, body[i] @ pitch_extrinsic(q))
            out[key] = cams
    return {key: out[key] for key in passes}
