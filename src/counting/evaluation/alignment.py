"""Alignment on the evaluation side: the roll and pitch prior of the fits, the settings an alignment runs
with, the fits of each pass to its true cameras, the oracle alignment to the structures and tree trunks of
the orchard, and the `alignment` entry of a result.

This module reads the true cameras and the structures. The alignment of the tracked passes by the tracks
they share and to the row centre plane is `counting.multiview.align`, which reads no truth; this module
hands it its settings.

Roll and pitch prior. Under `pose.fuse=imu` gravity holds the roll and the pitch of the body, so the rigid
fits take a prior on both: sigma = 2 * hypot(b / g, delta), with b the turn-on accelerometer bias (m/s^2,
at least the accelerometer floor of the fusion's bias prior) and delta the camera-IMU rotation error (rad)
that the estimator believes (`roll_pitch_sigma`). The value is `align.roll_pitch_sigma_deg`, or without it
`ROLL_PITCH_SIGMA_START_DEG`; the IMU noise a config generates is never read. Without fusion nothing holds
the roll and the pitch, and the fits take no prior.

Truth fits (`truth_alignment`). For each pass and segment, the frames whose used camera centre lies in the
window of the segment give the rigid transform that best takes their used cameras to their true cameras
(`fit_truth_rigid`); fewer than two frames give no fit of its own. A segment without one falls back as in
`with_fallback`, the fit of the whole group taking every frame of the pass. The residual is the median
distance of the moved used camera centres from the true ones. On `Segments.whole()` the transform of a pass
is its fit over all its frames.

Oracle alignment (`oracle_alignment`), as if the structures and tree trunks were observed. The points
(`structure_points`) lie on the posts every `STRUCT_STEP_M` from half of it up to their height, on the
corners of the arches, along the purlins every `STRUCT_STEP_M` and on each trunk axis at the same heights as
on a post up to its top, within half a row spacing of the target row. A pass observes a point from the first
frame, nearest the principal point first, whose true camera has it in front and in the image and whose
depth16 does not hide it (a depth more than `OCCLUSION_TOL_M` nearer than the point at its pixel): the
point taken into the used camera of that frame, plus a Gaussian noise of `oracle_sigma_m` on each axis
(noise step 17, drawn for every point of every pass), at the x of that used camera. Each group (a side
under the rig, else a pass) gets per segment the rigid transform that takes the observations in the window
to their true points (pairs known, no prior), when there are at least `ORACLE_MIN_POINTS` of them and the
true points spread at least `ORACLE_MIN_SPREAD_M` (the square root of the second largest eigenvalue of
their covariance: the points on one line give no rigid fit). A segment without one falls back as in
`with_fallback`, the fit of the whole group taking all its observations under the same thresholds.

The entry (`alignment_entry`) holds the mode, the settings the mode used (None for the ones it does not
use), a record per stage and group with one row per segment (centre, count, use, residual, the move in m
and the turn as roll, pitch and yaw in degrees) and a summary: the count of segment rows by use, the median
count and residual of the rows of a fit of their own (not those of the row centre plane) and the median
absolute shift across the row of the row centre plane rows (None where there are none).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
from scipy.spatial.transform import Rotation

from counting.config.eval_schema import EvalAppConfig
from counting.data.formats.orchard_synth import OrchardLayout, PassKey, Structure
from counting.evaluation.fusion import BIAS_SIGMA_FLOOR
from counting.evaluation.posequality import fit_truth_rigid
from counting.geometry.camera import Camera
from counting.geometry.noise import STEP_STRUCT, noise_rng, pass_key_code
from counting.multiview.align import (
    MIN_PAIRS, PAIR_RADIUS_M, SEGMENT_M, USES, AlignResult, AlignSettings, PassAlignment, Segments, StageRecord,
    fit_known, with_fallback,
)
from counting.odometry.imufile import GRAVITY

BELIEF_ACC_B0_MG = 4.0       # believed turn-on accelerometer bias of the judged point (mg)
BELIEF_EXTR_ROT_DEG = 0.3    # believed camera-IMU rotation error of the judged point (deg)
ORACLE_SIGMA_M = 0.02        # noise of an oracle structure observation on each axis (m)
STRUCT_STEP_M = 0.5          # spacing of the structure points along a post, a purlin and a trunk (m)
ORACLE_MIN_POINTS = 6        # fewest observations of an oracle fit
ORACLE_MIN_SPREAD_M = 0.1    # least spread of the true points of an oracle fit (m)
OCCLUSION_TOL_M = 0.15       # a depth16 this much nearer than a point hides it (m)
_SETTING_KEYS = ("segment_m", "pair_radius_m", "roll_pitch_sigma_deg", "min_pairs", "stress_row_bias_m",
                 "stress_pair_keep", "oracle_sigma_m")
_TRUTH_MIN_FRAMES = 2        # fewest frames of a truth fit of a segment's own


def roll_pitch_sigma(acc_b0_mg: float, extr_rot_deg: float) -> float:
    """The roll and pitch prior sigma (deg) at the turn-on accelerometer bias `acc_b0_mg` (mg, at least the
    accelerometer floor of the fusion's bias prior) and the camera-IMU rotation error `extr_rot_deg` (deg)."""
    b = max(acc_b0_mg * 1e-3 * GRAVITY, BIAS_SIGMA_FLOOR[0])
    return math.degrees(2 * math.hypot(b / GRAVITY, math.radians(extr_rot_deg)))


# The start value of the roll and pitch prior: the formula at the believed turn-on accelerometer bias and
# camera-IMU rotation of the judged point, the same for every config whatever IMU noise it generates.
ROLL_PITCH_SIGMA_START_DEG = roll_pitch_sigma(BELIEF_ACC_B0_MG, BELIEF_EXTR_ROT_DEG)


def align_settings(cfg: EvalAppConfig) -> AlignSettings:
    """The settings the alignment of `cfg` (with an align section) runs with: each unset knob at its start
    value; the roll and pitch prior under `pose.fuse=imu` only (`align.roll_pitch_sigma_deg`, else
    `ROLL_PITCH_SIGMA_START_DEG`), None otherwise even when set; the fewest observations of a used track as
    the scoring keeps tracks (`tracker.min_obs_for(stages.det)`) and the stress keys (a row bias of 0 when
    unset)."""
    align = cfg.align
    sigma = None
    if cfg.pose is not None and cfg.pose.fuse == "imu":
        sigma = ROLL_PITCH_SIGMA_START_DEG if align.roll_pitch_sigma_deg is None else align.roll_pitch_sigma_deg
    return AlignSettings(
        segment_m=SEGMENT_M if align.segment_m is None else align.segment_m,
        pair_radius_m=PAIR_RADIUS_M if align.pair_radius_m is None else align.pair_radius_m,
        roll_pitch_sigma_deg=sigma,
        min_pairs=MIN_PAIRS if align.min_pairs is None else align.min_pairs,
        min_obs=cfg.tracker.min_obs_for(cfg.stages.det),
        row_bias_m=0.0 if align.stress_row_bias_m is None else align.stress_row_bias_m,
        pair_keep=align.stress_pair_keep,
    )


def _truth_fit(truth: Sequence[Camera], used: Sequence[Camera],
               sel: np.ndarray) -> tuple[np.ndarray, int, float]:
    """The truth fit of the frames `sel` of a pass, their count and the median distance of the moved used
    camera centres from the true ones."""
    idx = np.flatnonzero(sel)
    T, U = [truth[i] for i in idx], [used[i] for i in idx]
    A = fit_truth_rigid(T, U)
    residual = float(np.median([np.linalg.norm(A[:3, :3] @ u.C + A[:3, 3] - t.C) for t, u in zip(T, U)]))
    return A, len(idx), residual


def truth_alignment(items: Sequence[tuple[PassKey, Sequence[Camera], Sequence[Camera]]],
                    segments: Segments) -> AlignResult:
    """The truth fit of every pass of `items` (pass key, true cameras, used cameras, in frame order) by
    segment, a segment taking the frames whose used camera centre x lies in its window. The stage is
    `truth_pass` on `Segments.whole()` and `truth_segment` otherwise; the group is pitch and side
    (`"10B"`) and the count of a segment its frames."""
    stage = "truth_pass" if np.isinf(segments.length_m) else "truth_segment"
    transforms: dict[PassKey, PassAlignment] = {}
    stages: list[StageRecord] = []
    for key, truth, used in items:
        truth, used = list(truth), list(used)
        if len(truth) != len(used):
            raise ValueError(f"pass {key}: a truth fit needs as many true cameras as used cameras, got {len(truth)} "
                             f"and {len(used)}")
        x = np.array([float(c.C[0]) for c in used])
        own = []
        for j in range(len(segments.centres)):
            sel = segments.window(j, x)
            n = int(sel.sum())
            own.append(_truth_fit(truth, used, sel) if n >= _TRUTH_MIN_FRAMES else (None, n, None))
        every = np.ones(len(x), dtype=bool)
        fits = with_fallback(own, segments, lambda: _truth_fit(truth, used, every) if len(x) else (None, 0, None))
        stages.append(StageRecord(stage, f"{key.pitch_deg:g}{key.side}", segments, fits))
        transforms[key] = PassAlignment(segments, np.stack([f.matrix for f in fits]))
    return AlignResult(transforms, stages)


def _steps(start: float, stop: float) -> np.ndarray:
    """`start`, `start + STRUCT_STEP_M`, ... up to `stop` (within 1e-9)."""
    n = int(np.floor((stop - start) / STRUCT_STEP_M + 1e-9)) + 1 if stop >= start - 1e-9 else 0
    return start + STRUCT_STEP_M * np.arange(n)


def _trunk_points(axis: np.ndarray) -> list[np.ndarray]:
    """The points of a trunk polyline `axis` (k, 3) at z = `STRUCT_STEP_M` / 2, then every `STRUCT_STEP_M`
    up to its highest corner, each on the first segment that passes that height."""
    out = []
    for z in _steps(STRUCT_STEP_M / 2, float(axis[:, 2].max())):
        for a, b in zip(axis[:-1], axis[1:]):
            if min(a[2], b[2]) - 1e-9 <= z <= max(a[2], b[2]) + 1e-9 and a[2] != b[2]:
                out.append(a + (z - a[2]) / (b[2] - a[2]) * (b - a))
                break
    return out


def structure_points(structures: Sequence[Structure], layout: OrchardLayout) -> np.ndarray:
    """The points (m, 3) of `structures` in their order: on a `post` (`x`, `y`, `height`) every
    `STRUCT_STEP_M` from half of it up to the height, the corners of an `arch` (`points`), along a `purlin`
    (`x0`, `x1`, `y`, `z`) every `STRUCT_STEP_M` from x0 up to x1 and on a `trunk` (`points`) as on a post
    up to its highest corner (`_trunk_points`). Other kinds give none, and only the points within half a
    row spacing of the target row across the row stay; (0, 3) without any."""
    pts: list = []
    for s in structures:
        p = s.params
        if s.kind == "post":
            pts += [[float(p["x"]), float(p["y"]), z] for z in _steps(STRUCT_STEP_M / 2, float(p["height"]))]
        elif s.kind == "arch":
            pts += [[float(v) for v in q] for q in p["points"]]
        elif s.kind == "purlin":
            pts += [[x, float(p["y"]), float(p["z"])] for x in _steps(float(p["x0"]), float(p["x1"]))]
        elif s.kind == "trunk":
            pts += _trunk_points(np.asarray(p["points"], dtype=np.float64).reshape(-1, 3))
    P = np.array(pts, dtype=np.float64).reshape(-1, 3)
    return P[np.abs(P[:, 1] - layout.row_y) <= layout.row_spacing / 2 + 1e-9]


def _observe(points: np.ndarray, truth: Sequence[Camera], used: Sequence[Camera], noise: np.ndarray,
             depth: Callable[[int], np.ndarray] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The observations of `points` in one pass, the true points they observe and the x of the used camera
    of each, in point order. The frames whose true camera has a point in front and in the image go by the
    distance of its pixel from the principal point (the earlier frame on a tie), and the point is observed
    from the first of them that does not hide it: with `depth`, frame k hides it when `depth(k)` (m) at its
    pixel (clipped into the image) is finite and more than `OCCLUSION_TOL_M` nearer than the point. Without
    such a frame the point is not observed."""
    m = len(points)
    dist = np.full((len(truth), m), np.inf)
    uv, z = np.zeros((len(truth), m, 2)), np.zeros((len(truth), m))
    for k, cam in enumerate(truth):
        with np.errstate(divide="ignore", invalid="ignore"):
            uv[k], z[k] = cam.project(points)
            d = np.hypot(uv[k, :, 0] - cam.K[0, 2], uv[k, :, 1] - cam.K[1, 2])
        dist[k] = np.where((z[k] > 0) & cam.in_image(uv[k]), d, np.inf)
    order = np.argsort(dist, axis=0, kind="stable")
    pick = np.full(m, -1)
    for i in range(m):
        for k in order[:, i]:
            if not np.isfinite(dist[k, i]):
                break
            if depth is not None:
                img = depth(int(k))
                h, w = img.shape
                u = min(max(int(np.floor(uv[k, i, 0])), 0), w - 1)
                v = min(max(int(np.floor(uv[k, i, 1])), 0), h - 1)
                if np.isfinite(img[v, u]) and img[v, u] < z[k, i] - OCCLUSION_TOL_M:
                    continue
            pick[i] = k
            break
    seen = np.flatnonzero(pick >= 0)
    obs = np.empty((len(seen), 3))
    for k in np.unique(pick[seen]):
        sel = pick[seen] == k
        T, U = truth[k], used[k]
        obs[sel] = ((points[seen[sel]] - T.C) @ T.R_c2w) @ U.R_c2w.T + U.C
    return obs + noise[seen], points[seen], np.array([float(used[k].C[0]) for k in pick[seen]])


def _spread(points: np.ndarray) -> float:
    """The square root of the second largest eigenvalue of the covariance of `points` (n >= 2)."""
    return math.sqrt(max(float(np.linalg.eigvalsh(np.cov(points.T))[-2]), 0.0))


def _known_fit(obs: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray | None, int, float | None]:
    """The rigid transform taking the observations `obs` to their true points `ref`, their count and its
    median residual; no transform below `ORACLE_MIN_POINTS` observations or `ORACLE_MIN_SPREAD_M` spread."""
    n = len(obs)
    if n < ORACLE_MIN_POINTS or _spread(ref) < ORACLE_MIN_SPREAD_M:
        return None, n, None
    A, residual = fit_known(obs, ref, sigma_rp_deg=None)
    return A, n, residual


def oracle_alignment(items: Sequence[tuple[PassKey, Sequence[Camera], Sequence[Camera]]], points: np.ndarray,
                     segments: Segments, *, rig: bool, key: Sequence[int], sigma_m: float,
                     depth: Callable[[PassKey, int], np.ndarray] | None = None) -> AlignResult:
    """The oracle alignment of the passes of `items` (pass key, true cameras, used cameras, in frame order)
    to the structure `points` by segment. `key` = (seed, noise seed or 0, scene) keys the noise of each
    pass with `STEP_STRUCT` and its pass code; `depth(pass key, k)` gives the depth16 (m, NaN for no
    return) of frame k of a pass, and without it no point is hidden. The groups are the sides A then B
    under `rig`, else the passes; the stage is `oracle_struct` and the count of a segment its observations
    in the window."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    groups: dict[str, list] = {}
    for pass_key, truth, used in items:
        truth, used = list(truth), list(used)
        if len(truth) != len(used):
            raise ValueError(f"pass {pass_key}: oracle observations need as many true cameras as used cameras, "
                             f"got {len(truth)} and {len(used)}")
        noise = noise_rng(key[0], key[1], STEP_STRUCT, key[2], *pass_key_code(pass_key)).standard_normal(
            (len(points), 3)) * sigma_m
        observed = _observe(points, truth, used, noise, None if depth is None else partial(depth, pass_key))
        label = pass_key.side if rig else f"{pass_key.pitch_deg:g}{pass_key.side}"
        groups.setdefault(label, []).append((pass_key, observed))
    by_key: dict[PassKey, PassAlignment] = {}
    stages: list[StageRecord] = []
    for label in (sorted(groups) if rig else groups):
        obs, ref, x = (np.concatenate([o[i] for _, o in groups[label]]) for i in range(3))
        own = []
        for j in range(len(segments.centres)):
            sel = segments.window(j, x)
            own.append(_known_fit(obs[sel], ref[sel]))
        fits = with_fallback(own, segments, lambda: _known_fit(obs, ref))
        stages.append(StageRecord("oracle_struct", label, segments, fits))
        step = PassAlignment(segments, np.stack([f.matrix for f in fits]))
        for pass_key, _ in groups[label]:
            by_key[pass_key] = step
    return AlignResult({pass_key: by_key[pass_key] for pass_key, _, _ in items}, stages)


def _float(v: float | None) -> float | None:
    return None if v is None else float(v)


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if len(values) else None


def _stage_entry(record: StageRecord) -> dict:
    """One stage record of the entry: a row per segment, the centre None on a segment of unbounded length."""
    whole = bool(np.isinf(record.segments.length_m))
    rows = []
    for centre, fit in zip(record.segments.centres, record.fits):
        yaw, pitch, roll = Rotation.from_matrix(fit.matrix[:3, :3]).as_euler("ZYX", degrees=True)
        rows.append({"centre_x_m": None if whole else float(centre), "n": int(fit.n), "use": fit.use,
                     "residual_m": _float(fit.residual_m), "move_m": [float(v) for v in fit.matrix[:3, 3]],
                     "turn_deg": [float(roll), float(pitch), float(yaw)]})
    return {"stage": record.stage, "group": record.group, "segments": rows}


def _summary(stages: Sequence[StageRecord]) -> dict:
    fits = [(s.stage, f) for s in stages for f in s.fits]
    counts = {use: sum(f.use == use for _, f in fits) for use in USES}
    own = [f for stage, f in fits if stage != "row_plane" and f.use == "own"]
    return {"segments": len(fits), **counts,
            "pairs_p50": _median([f.n for f in own]),
            "residual_p50_m": _median([f.residual_m for f in own if f.residual_m is not None]),
            "row_shift_p50_m": _median([abs(float(f.matrix[1, 3])) for stage, f in fits if stage == "row_plane"])}


def alignment_entry(cfg: EvalAppConfig, settings: AlignSettings, result: AlignResult) -> dict:
    """The `alignment` entry of a result for the alignment `result` that `cfg.align.mode` ran with `settings`.

    `settings` holds the values the mode used, the others None: listed methods use the segment length and
    the fewest pairs, `tracks` the pair radius, the roll and pitch prior and the pair share kept (when set),
    `row_plane` the row bias (when set); `fit_truth_segment` uses the segment length, `oracle_struct` the
    segment length and the observation noise (`align.oracle_sigma_m`, else `ORACLE_SIGMA_M`) and
    `fit_truth_pass` none."""
    align = cfg.align
    mode = list(align.mode) if isinstance(align.mode, list) else align.mode
    used = dict.fromkeys(_SETTING_KEYS)
    if isinstance(mode, list):
        used.update(segment_m=float(settings.segment_m), min_pairs=int(settings.min_pairs))
        if "tracks" in mode:
            used.update(pair_radius_m=float(settings.pair_radius_m),
                        roll_pitch_sigma_deg=_float(settings.roll_pitch_sigma_deg),
                        stress_pair_keep=_float(align.stress_pair_keep))
        if "row_plane" in mode:
            used["stress_row_bias_m"] = _float(align.stress_row_bias_m)
    elif mode == "fit_truth_segment":
        used["segment_m"] = float(settings.segment_m)
    elif mode == "oracle_struct":
        used.update(segment_m=float(settings.segment_m),
                    oracle_sigma_m=float(ORACLE_SIGMA_M if align.oracle_sigma_m is None else align.oracle_sigma_m))
    return {"mode": mode, "settings": used, "stages": [_stage_entry(s) for s in result.stages],
            "summary": _summary(result.stages)}
