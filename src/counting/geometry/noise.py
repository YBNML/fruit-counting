"""Seeded noise for the prediction path: one generator per integer key (`noise_rng`),
pass bias and depth noise. Every draw is made whatever the factor values, so one key
gives the same draws for every factor value.

The drift model adds steps 5 to 14. Each key but the IMU one starts `(cfg.seed, noise seed, step,
scene)` like the steps above, with side = `side_code` (0 for side A or 1 for side B, aisle) and
flow = the side under a rig or `pass_key_code` otherwise. Walk 5 `(..., *flow, pos_k)` draws t(3)
then r(3); pass start 6 `(..., *side)` t(3) then r(3); rig calibration 7
`(..., round(pitch * 1000))` t(3) then r(3); time offset 8 `(..., *pass_key_code)` one value;
depth k 9 `(..., idx)` one value per depth; camera-to-IMU calibration 11
`(..., round(pitch * 1000))` t(3) then r(3); flow scale and rate 12 `(..., *flow)` s then w(3);
pitch-to-pitch rigid 13 `(..., *side, round(pitch * 1000))` t(3) then r(3); line-end GPS fix 14
`(..., *flow, end)`, end 0 for the first frame and 1 for the last, x then y then z. The IMU 10 is
one generator for the whole IMU stream, keyed `(10, IMU seed, noise seed, scene, layout)`.

Which of these steps a run draws. Under pose.source=truth, with a pose section, a noise seed and a
drift camera key set (a key at 0 included): each flow draws 5 and 12, each flow draws 6 (keyed by
its side) when a `pass_side_*` key is set and, without the rig, each pass draws 13 when a
`pass_pitch_*` key is set. With a pose section, `pose.rig`, a `rig_extr_*` key set and a noise seed,
each pitch other than the reference draws 7. With a noise seed and `noise.depth_k` set, each frame
draws 9, drift camera key or not. Step 8 is not drawn: a time offset above 0 stops the run. Steps 10,
11 and 14 have keys here and no draws. A key set to 0 draws like any other value, so configs that
set the same keys share their draws.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot, sqrt

import numpy as np
from scipy.spatial.transform import Rotation

from counting.data.formats.orchard_synth import PassKey
from counting.geometry.rig import rigid

STEP_DET, STEP_POSE, STEP_BIAS, STEP_DEPTH = 1, 2, 3, 4


def noise_rng(*key: int) -> np.random.Generator:
    """A generator seeded by the integer `key`, one generator per distinct key."""
    if any(k < 0 for k in key):
        raise ValueError(f"noise key {list(key)} has a negative part; seeds and pitches must be >= 0")
    return np.random.default_rng([int(k) for k in key])


def pass_key_code(key: PassKey) -> tuple[int, int, int]:
    """`PassKey` as the integer parts used in a noise key."""
    return (round(key.pitch_deg * 1000), 0 if key.side == "A" else 1, key.aisle)


def pass_bias(rng: np.random.Generator, b: float) -> np.ndarray:
    """A length-`b` vector in a random direction, shape (3,)."""
    v = rng.standard_normal(3)
    return b * v / np.linalg.norm(v)


def perturb_depth(z: np.ndarray, rng: np.random.Generator, rel_sigma: float, dropout: float) -> np.ndarray:
    """`z` scaled by relative Gaussian noise, with a `dropout` fraction set to NaN. Does not mutate `z`."""
    z = np.asarray(z, dtype=np.float64)
    g = rng.standard_normal(z.shape[0])
    u = rng.random(z.shape[0])
    out = z * (1 + rel_sigma * g)
    out[u < dropout] = np.nan
    return out


STEP_WALK, STEP_START, STEP_RIG, STEP_TIME, STEP_DEPTH_K, STEP_IMU, STEP_IMU_EXTR, STEP_FLOW, STEP_PITCH, STEP_GPS = (
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
REF_STEP_M = 0.15  # the base step (m) the walk sigmas are given for


def side_code(key: PassKey) -> tuple[int, int]:
    """The side of `key` as the integer parts used in a noise key: (0 for side A or 1 for B, aisle)."""
    return (0 if key.side == "A" else 1, key.aisle)


@dataclass(frozen=True)
class DriftSigmas:
    """Sigmas of one body flow: walk translation (m) and rotation (deg) per axis for each step of
    `REF_STEP_M`, scale (relative) and heading rate per axis (deg/m). The scale and the rate are
    drawn once per flow."""

    trans_m: float = 0.0
    rot_deg: float = 0.0
    scale: float = 0.0
    rate_deg_per_m: float = 0.0

    def is_zero(self) -> bool:
        """True when all four sigmas are 0."""
        return self.trans_m == 0 and self.rot_deg == 0 and self.scale == 0 and self.rate_deg_per_m == 0


@dataclass(frozen=True, eq=False)
class FlowDraws:
    """Unit normal draws of one flow of n frames: scale `s`, rate `w` (3,), and the walk
    translation `t` and rotation `r` (n - 1, 3), row i - 1 for the step into frame i."""

    s: float
    w: np.ndarray
    t: np.ndarray
    r: np.ndarray


def flow_draws(seed: int, noise_seed: int, scene: int, flow: tuple[int, ...], pos_ks: Sequence[int]) -> FlowDraws:
    """The draws of one flow whose frames sit at the positions `pos_ks`.

    `s` then `w` come from the flow key (step 12). The step into frame i (i >= 1) draws `t[i - 1]`
    then `r[i - 1]` from the key of its position `pos_ks[i]` (step 5), so a position gets the same
    draws wherever it sits in the list.
    """
    g = noise_rng(seed, noise_seed, STEP_FLOW, scene, *flow)
    s = float(g.standard_normal())
    w = g.standard_normal(3)
    n = len(pos_ks)
    t = np.zeros((max(n - 1, 0), 3))
    r = np.zeros((max(n - 1, 0), 3))
    for i in range(1, n):
        step = noise_rng(seed, noise_seed, STEP_WALK, scene, *flow, pos_ks[i])
        t[i - 1] = step.standard_normal(3)
        r[i - 1] = step.standard_normal(3)
    return FlowDraws(s=s, w=w, t=t, r=r)


def noisy_increments(true_poses: np.ndarray, draws: FlowDraws, sigmas: DriftSigmas, base_step_m: float) -> np.ndarray:
    """The noisy relative motions (n - 1, 4, 4) between the n poses `true_poses` (n, 4, 4).

    With the true motion `rel = P[i-1]^-1 @ P[i]`, its length `d = |rel_t|` and
    `k = sqrt(base_step_m / REF_STEP_M)`, the rotation is
    `rel_R @ Exp(radians(rot_deg * k) * r[i-1] + radians(rate_deg_per_m) * w * d)` and the
    translation is `(1 + scale * s) * rel_t + trans_m * k * t[i-1]`. The walk sigmas grow with the
    square root of the step, so a coarser base step drifts the same per metre.
    """
    P = np.asarray(true_poses, dtype=np.float64)
    n = P.shape[0]
    if draws.t.shape != (n - 1, 3) or draws.r.shape != (n - 1, 3):
        raise ValueError(f"draws with t {draws.t.shape} and r {draws.r.shape} do not fit {n} poses")
    if n < 2:
        return np.zeros((0, 4, 4))
    R_prev = P[:-1, :3, :3]
    rel_R = np.einsum("nji,njk->nik", R_prev, P[1:, :3, :3])
    rel_t = np.einsum("nji,nj->ni", R_prev, P[1:, :3, 3] - P[:-1, :3, 3])
    d = np.linalg.norm(rel_t, axis=1)
    k = sqrt(base_step_m / REF_STEP_M)
    turn = np.radians(sigmas.rot_deg * k) * draws.r + np.radians(sigmas.rate_deg_per_m) * draws.w * d[:, None]
    out = np.repeat(np.eye(4)[None], n - 1, axis=0)
    out[:, :3, :3] = rel_R @ Rotation.from_rotvec(turn).as_matrix()
    out[:, :3, 3] = (1.0 + sigmas.scale * draws.s) * rel_t + sigmas.trans_m * k * draws.t
    return out


def chain_poses(start: np.ndarray, increments: np.ndarray) -> np.ndarray:
    """The poses (n, 4, 4) from `start` and the n - 1 `increments`: `poses[0] = start`,
    `poses[i] = poses[i - 1] @ increments[i - 1]`."""
    inc = np.asarray(increments, dtype=np.float64).reshape(-1, 4, 4)
    poses = np.empty((inc.shape[0] + 1, 4, 4))
    poses[0] = np.asarray(start, dtype=np.float64)
    for i in range(1, poses.shape[0]):
        poses[i] = poses[i - 1] @ inc[i - 1]
    return poses


def seeded_rigid(key: Sequence[int], trans_m: float, rot_deg: float) -> np.ndarray:
    """A rigid error from the generator of `key`: t(3) then r(3) unit normals, rotation vector
    `radians(rot_deg) * r` and translation `trans_m * t`. Zero sigmas give exactly the identity."""
    g = noise_rng(*key)
    t = g.standard_normal(3)
    r = g.standard_normal(3)
    return rigid(np.radians(rot_deg) * r, trans_m * t)


def about_middle(poses: np.ndarray, error: np.ndarray) -> np.ndarray:
    """`poses` (n, 4, 4) moved by the rigid `error` about the middle of the pass.

    With c the mean of the first and the last camera centers, each center C becomes
    `c + R_e @ (C - c) + t_e` and each rotation R becomes `R_e @ R`.
    """
    P = np.asarray(poses, dtype=np.float64)
    E = np.asarray(error, dtype=np.float64)
    R_e, t_e = E[:3, :3], E[:3, 3]
    c = (P[0, :3, 3] + P[-1, :3, 3]) / 2.0
    out = P.copy()
    out[:, :3, :3] = R_e @ P[:, :3, :3]
    out[:, :3, 3] = c + (P[:, :3, 3] - c) @ R_e.T + t_e
    return out


def time_offset(key: Sequence[int], sigma_s: float) -> float:
    """A time offset (s): `sigma_s` times one unit normal from the generator of `key`."""
    return float(sigma_s * noise_rng(*key).standard_normal())


def depth_k_noise(key: Sequence[int], z: np.ndarray, k: float) -> np.ndarray:
    """Range-dependent depth noise `k * z**2 * g` (m), g one unit normal per depth from the
    generator of `key`. A NaN depth gives NaN. Does not mutate `z`."""
    z = np.asarray(z, dtype=np.float64)
    g = noise_rng(*key).standard_normal(z.shape[0])
    return k * z**2 * g


def vo_factor_sigma(sigmas: DriftSigmas, base_step_m: float, floor_trans_m: float,
                    floor_rot_deg: float) -> tuple[float, float]:
    """The per-step sigmas (m, deg) of a relative-motion factor for the drift `sigmas` at the base
    step `base_step_m`: the walk at `k = sqrt(base_step_m / REF_STEP_M)` and the scale and heading
    rate over one step, joined by `hypot`, each at least its floor."""
    k = sqrt(base_step_m / REF_STEP_M)
    trans = max(floor_trans_m, hypot(sigmas.trans_m * k, sigmas.scale * base_step_m))
    rot = max(floor_rot_deg, hypot(sigmas.rot_deg * k, sigmas.rate_deg_per_m * base_step_m))
    return trans, rot
