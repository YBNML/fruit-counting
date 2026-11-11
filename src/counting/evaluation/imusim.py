"""The simulated IMU, on the evaluation side: it reads the truth, the recorded body trajectory of a bundle
rendered with body shake and the true frame poses of a straight one.

Samples (`true_samples`). A body trajectory is body-to-world poses at evenly spaced times t_0 ... t_{N-1}
(1 kHz in a bundle), with step d. Central differences give the body angular rate
w_i = rotvec(R_{i-1}^T R_{i+1}) / (2 d) and the specific force f_i = R_i^T ((p_{i+1} - 2 p_i + p_{i-1}) / d^2
+ g z) at t_i, i = 1 ... N-2, where g is `GRAVITY` and z the world up axis. IMU sample k sits at
t_0 + k / rate and holds the values at the middle of its interval, linearly interpolated between the
central differences; a sample whose middle is past t_{N-2} is not kept.

Drives (`session_drives`). A drive is one body motion of the session, on one side and one aisle. Layout
`rig`: one drive per side, A then B (the sides with frames). Layout `pass`: one drive per pass, the
pitches of the bundle in ascending order times the sides, each sharing its side's body. A bundle with
body shake drives its recorded trajectory (`recorded`). A straight bundle has none, so its drive is
assumed (`straight_assumed`, `straight_drive`) from the true bodies of the side's lowest-pitch pass:
standing still for `stop_s`, a cosine speed-up over `accel_s` to `speed_mps`, then constant speed past
the last frame by `STRAIGHT_TAIL_S`, at `TRAJECTORY_RATE_HZ`. Frame j of that pass is at
t_j = stop_s + accel_s + s_j / speed_mps, s_j its distance along the travel from the first frame, and
every frame at the same side and position takes that time (`frame_times`).

Integration (`integrate`). Over the pieces of `sample_spans`, midpoint strapdown steps: a sample holds the
values at the middle of its interval, so its specific force turns with the rotation half way through the
piece, a = R Exp(w h / 2) f - g z, then p <- p + v h + a h^2 / 2, v <- v + a h, R <- R Exp(w h). Turning
it with the start rotation R (the step of GTSAM's preintegration) leaves an error of order h in every
step, which over a whole row of body shake reaches centimetres at 200 Hz. `imu_check` integrates
noiseless samples of each side of a shake bundle from its first frame to its last and reports the largest
end errors.

Noise (`imu_session`). One generator per session, `noise_rng(STEP_IMU, imu_seed, noise_seed, scene,
layout code)`, draws in this order: the turn-on bias (6: gyro axes, then accelerometer axes), the
temperature coefficients (6), then for each drive in session order the bias walk over the row end (6,
every drive but the first) and the draws of its K samples (K, 12: gyro walk, accelerometer walk, gyro
white noise, accelerometer white noise), and last the vibration frequencies in `VIB_BAND_HZ` and phases
(6, `VIB_SINES` each: accelerometer axes, then gyro axes). Every draw is made whatever the profile values.
With sample interval dt, the bias b_k = b_{k-1} + sigma_w sqrt(dt) z_k runs on through the session and
walks sigma_w sqrt(row_end_s) more between drives; the temperature drift rises evenly over each drive,
from 0 at its first sample to `temp_k` times the coefficients at its last; white noise is
sigma_n / sqrt(dt) z; each vibration axis is the sum of `VIB_SINES` sines of amplitude
RMS sqrt(2 / VIB_SINES), and a gyro axis also takes `VIB_G_SENSITIVITY_DPS_PER_G` times the vibration
of the same accelerometer axis in g. A measurement is the truth plus bias, temperature drift, white
noise and vibration, added in that order, with the specific force clipped to the accelerometer range.

GPS (`gps_end_fixes`). A row-end fix is a true body position plus one unit normal per axis from the
generator of its end's key, times the horizontal sigma on x and y and the vertical sigma on z.

The formulas are public ones: central differences here, the Kalibr discretization for noise densities.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np
from scipy.spatial.transform import Rotation

from counting.config.eval_schema import IMU_LEVEL_KEYS, NoiseConfig
from counting.config.pose_schema import ImuConfig, ImuNoise
from counting.data.formats.orchard_synth import FrameInput, SynthRun
from counting.geometry.noise import STEP_IMU, noise_rng
from counting.geometry.rig import body_from_reference, interpolate_pose, pitch_extrinsic, pose_matrix
from counting.odometry import imufile
from counting.odometry.imufile import GRAVITY, LAYOUTS, ImuStream, sample_spans

TRAJECTORY_RATE_HZ = 1000.0   # rate of an assumed straight drive
STRAIGHT_TAIL_S = 0.1         # an assumed straight drive runs on this long past its last frame
TEMP_GYRO_DPS_PER_K = 0.015   # gyro bias temperature coefficient (deg/s per K)
TEMP_ACC_MG_PER_K = 0.2       # accelerometer bias temperature coefficient (mg per K)
VIB_SINES = 5                 # sines per vibration axis
VIB_BAND_HZ = (2.0, 20.0)     # band of the vibration frequencies (Hz)
VIB_G_SENSITIVITY_DPS_PER_G = 0.1   # gyro rate per g of accelerometer vibration (deg/s per g)
_SIDES = ("A", "B")
_PITCH_TOL_DEG = 1e-6
_TURN_TOL_RAD = 1e-6
_LINE_TOL_M = 1e-6
_EVEN_TOL_S = 1e-9
_UP = np.array([0.0, 0.0, GRAVITY])


@dataclass(frozen=True)
class AssumedMotion:
    """The assumed motion of a bundle without a recorded trajectory: speed (m/s), standstill before the
    start (s) and the speed-up from standing to that speed (s)."""

    speed_mps: float = 0.5
    stop_s: float = 5.0
    accel_s: float = 2.0

    @classmethod
    def of(cls, cfg: ImuConfig) -> AssumedMotion:
        """The assumed motion of an IMU config (`speed_mps`, `stop_s`, `accel_s`)."""
        return cls(speed_mps=float(cfg.speed_mps), stop_s=float(cfg.stop_s), accel_s=float(cfg.accel_s))


@dataclass(frozen=True)
class ImuProfile:
    """The noise of a generated IMU, in the units of the `noise.imu_*` keys: white noise densities `gyro_n`
    (rad/s/sqrt(Hz)) and `acc_n` (m/s^2/sqrt(Hz)), bias walks `gyro_w` (rad/s^2/sqrt(Hz)) and `acc_w`
    (m/s^3/sqrt(Hz)), turn-on bias sigmas `gyro_b0_dps` (deg/s) and `acc_b0_mg` (mg), the temperature rise
    over a drive `temp_k` (K) and the band vibration RMS per axis `vib_acc_ms2` (m/s^2) and `vib_gyro_dps`
    (deg/s)."""

    gyro_n: float = 0.0
    acc_n: float = 0.0
    gyro_w: float = 0.0
    acc_w: float = 0.0
    gyro_b0_dps: float = 0.0
    acc_b0_mg: float = 0.0
    temp_k: float = 0.0
    vib_acc_ms2: float = 0.0
    vib_gyro_dps: float = 0.0

    @classmethod
    def from_noise(cls, noise: NoiseConfig) -> ImuProfile:
        """The profile of the IMU noise level keys of `noise` (`IMU_LEVEL_KEYS` without their `imu_`
        prefix), 0 for an unset key. The time offset and extrinsic keys are not read."""
        return cls(**{key.removeprefix("imu_"): float(noise.factor(key)) for key in IMU_LEVEL_KEYS})

    @classmethod
    def from_config(cls, noise: ImuNoise) -> ImuProfile:
        """The profile of the noise of an IMU config."""
        return cls(**{f.name: float(getattr(noise, f.name)) for f in fields(cls)})


@dataclass(frozen=True)
class ImuSampling:
    """The sampling of a generated IMU: rate (Hz), accelerometer range (g) and the time at the row end
    between two drives (s). The defaults are those of `ImuConfig`."""

    rate_hz: float = 200.0
    acc_range_g: float = 4.0
    row_end_s: float = 20.0

    @classmethod
    def of(cls, cfg: ImuConfig) -> ImuSampling:
        """The sampling of an IMU config (`rate_hz`, `acc_range_g`, `row_end_s`)."""
        return cls(rate_hz=float(cfg.rate_hz), acc_range_g=float(cfg.acc_range_g), row_end_s=float(cfg.row_end_s))


@dataclass(frozen=True, eq=False)
class Drive:
    """One drive of a session: its layout (`imufile.LAYOUTS`), side, aisle and pitch (None in the `rig`
    layout), the body times `t` (N,) s and body-to-world poses `body` (N, 4, 4), and `source`, `recorded`
    (the trajectory of a shake bundle) or `straight_assumed`."""

    layout: str
    side: str
    aisle: int
    pitch_deg: float | None
    t: np.ndarray
    body: np.ndarray
    source: str

    def stream_key(self, noise_seed: int) -> tuple[int, int, int, int, int]:
        """The key of this drive's stream under `noise_seed` (`imufile.stream_key`)."""
        return imufile.stream_key(self.layout, noise_seed, self.side, self.aisle, self.pitch_deg)


def straight_drive(bodies: np.ndarray, motion: AssumedMotion) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The assumed straight drive through the true bodies (n, 4, 4) of one pass's frames in travel order.

    Returns the times (N,) s from 0 at `TRAJECTORY_RATE_HZ`, the body-to-world poses (N, 4, 4) and the
    frame times (n,). The body keeps the first frame's rotation and moves along its x axis through the
    first frame's position: at s = -speed * accel / 2 until `stop_s`, then
    s = -speed * accel / 2 + speed / 2 * (tau - accel / pi * sin(pi * tau / accel)) for tau = t - stop_s up
    to `accel_s`, then s = speed * (t - stop_s - accel_s), until `STRAIGHT_TAIL_S` past the last frame.
    The frames must share one rotation (1e-6 rad), sit on the body x axis (1e-6 m) and advance along it.
    """
    bodies = np.asarray(bodies, dtype=np.float64)
    R0, p0 = bodies[0, :3, :3], bodies[0, :3, 3]
    turn = float(Rotation.from_matrix(np.matmul(R0.T, bodies[:, :3, :3])).magnitude().max())
    if turn > _TURN_TOL_RAD:
        raise ValueError(f"a straight drive needs one body rotation (frames turn by {np.degrees(turn):.3g} deg)")
    u = R0[:, 0]
    d = bodies[:, :3, 3] - p0
    along = d @ u
    off = float(np.linalg.norm(d - along[:, None] * u, axis=1).max())
    if off > _LINE_TOL_M:
        raise ValueError(f"a straight drive needs frames on the body x axis (off the line by {off:.3g} m)")
    if not np.all(np.diff(along) > 0):
        raise ValueError("a straight drive needs frames that advance along the travel")
    v0, accel, stop = float(motion.speed_mps), float(motion.accel_s), float(motion.stop_s)
    run_up = v0 * accel / 2   # distance of the speed-up, ending at the first frame
    n = int(np.ceil((stop + accel + along[-1] / v0 + STRAIGHT_TAIL_S) * TRAJECTORY_RATE_HZ)) + 1
    t = np.arange(n) / TRAJECTORY_RATE_HZ
    tau = t - stop
    s = np.where(t <= stop, -run_up,
                 np.where(tau <= accel, -run_up + v0 / 2 * (tau - accel / np.pi * np.sin(np.pi * tau / accel)),
                          v0 * (t - stop - accel)))
    body = np.zeros((n, 4, 4))
    body[:, :3, :3] = R0
    body[:, :3, 3] = p0 + s[:, None] * u
    body[:, 3, 3] = 1.0
    return t, body, stop + accel + along / v0


def _side_aisles(run: SynthRun) -> dict[str, int]:
    """The aisle of each side with frames, A then B; a side with frames in two aisles is refused."""
    out = {}
    for side in _SIDES:
        aisles = sorted({f.pass_key.aisle for f in run.frames if f.pass_key.side == side})
        if len(aisles) > 1:
            raise ValueError(f"{run.root}: side {side} has frames in aisles {aisles}; a drive is one aisle")
        if aisles:
            out[side] = aisles[0]
    return out


def _lowest_pass(run: SynthRun, side: str) -> tuple[float, list[FrameInput]]:
    """The lowest pitch of `side` and that pass's frames in idx order."""
    pitch = min(f.pass_key.pitch_deg for f in run.frames if f.pass_key.side == side)
    frames = sorted((f for f in run.frames if f.pass_key.side == side and f.pass_key.pitch_deg == pitch),
                    key=lambda f: f.idx)
    return pitch, frames


def _straight_side(run: SynthRun, side: str, motion: AssumedMotion):
    """The assumed straight drive of `side` (`straight_drive`) and the lowest-pitch frames it passes."""
    pitch, frames = _lowest_pass(run, side)
    X = pitch_extrinsic(pitch)
    bodies = np.stack([body_from_reference(pose_matrix(f.camera), X) for f in frames])
    return pitch, frames, straight_drive(bodies, motion)


def drive_keys(run: SynthRun, layout: str) -> list[tuple[str, int, float | None]]:
    """The (side, aisle, pitch) of every drive of the session of `run`, in `session_drives` order: per side
    (A, B) with pitch None for `rig`, per pitch of the bundle (ascending) and then side for `pass`. Reads
    no trajectory."""
    if layout not in imufile.LAYOUTS:
        raise ValueError(f"unknown IMU layout {layout!r}")
    aisles = _side_aisles(run)
    if layout == "rig":
        return [(side, aisle, None) for side, aisle in aisles.items()]
    pitches: list[float] = []
    for q in sorted(f.pass_key.pitch_deg for f in run.frames):
        if not pitches or q - pitches[-1] > _PITCH_TOL_DEG:
            pitches.append(q)
    return [(side, aisle, q) for q in pitches for side, aisle in aisles.items()]


def session_drives(run: SynthRun, layout: str, motion: AssumedMotion) -> list[Drive]:
    """The drives of the session of `run` in `layout` order (`drive_keys`). A shake bundle rides its
    recorded trajectories, a straight bundle the assumed drive of each side (`straight_drive` under
    `motion`); the drives of one side share its body."""
    keys = drive_keys(run, layout)
    rides = {}   # side -> (t, body, source)
    for side in dict.fromkeys(side for side, _, _ in keys):
        if run.meta.shake:
            trajectory = run.trajectory(side)
            rides[side] = (trajectory.t, trajectory.body_poses(), "recorded")
        else:
            _, _, (t, body, _) = _straight_side(run, side, motion)
            rides[side] = (t, body, "straight_assumed")
    return [Drive(layout=layout, side=side, aisle=aisle, pitch_deg=q, t=rides[side][0], body=rides[side][1],
                  source=rides[side][2]) for side, aisle, q in keys]


def gps_end_fixes(key: Sequence[int], first: np.ndarray, last: np.ndarray, sigma_h: float,
                  sigma_v: float) -> tuple[np.ndarray, np.ndarray]:
    """The row-end GPS fixes (3,) of the body positions `first` and `last` (world, m): end e (0 for the
    first, 1 for the last) adds one unit normal per axis from `noise_rng(*key, e)` times (`sigma_h`,
    `sigma_h`, `sigma_v`)."""
    scale = np.array([sigma_h, sigma_h, sigma_v], dtype=np.float64)
    fixes = [np.asarray(p, dtype=np.float64) + noise_rng(*key, e).standard_normal(3) * scale
             for e, p in enumerate((first, last))]
    return fixes[0], fixes[1]


def frame_times(run: SynthRun, motion: AssumedMotion) -> dict[int, float]:
    """The time (s) of every frame by idx: the recorded frame time of a shake bundle, else the time of
    the frame's position in the assumed straight drive of its side (`straight_drive` under `motion`),
    which needs a sequence bundle."""
    if run.meta.shake:
        return {idx: plan.t for idx, plan in run.plans().items()}
    if not run.meta.is_sequence or any(f.pos_k < 0 for f in run.frames):
        raise ValueError(f"{run.root}: frame times need a sequence bundle (frame positions)")
    out = {}
    for side in _side_aisles(run):
        pitch, frames, (_, _, tf) = _straight_side(run, side, motion)
        at = {f.pos_k: float(tj) for f, tj in zip(frames, tf)}
        for f in run.frames:
            if f.pass_key.side != side:
                continue
            if f.pos_k not in at:
                raise ValueError(f"{run.root}: no frame of pitch {pitch:g} at side {side}, pos_k {f.pos_k}")
            out[f.idx] = at[f.pos_k]
    return dict(sorted(out.items()))


def true_samples(t: np.ndarray, body: np.ndarray, rate_hz: float) -> ImuStream:
    """Noiseless IMU samples at `rate_hz` of the body-to-world poses `body` (N, 4, 4) at evenly spaced
    times `t` (N,): sample k at t[0] + k / rate_hz holds the central-difference rate and specific force
    at the middle of its interval (module docstring)."""
    t = np.asarray(t, dtype=np.float64)
    body = np.asarray(body, dtype=np.float64)
    n = len(t)
    if n < 3:
        raise ValueError("IMU samples need three trajectory samples or more")
    step = (t[-1] - t[0]) / (n - 1)
    if np.abs(np.diff(t) - step).max() > _EVEN_TOL_S:
        raise ValueError("trajectory samples are not evenly spaced")
    interval = 1.0 / rate_hz
    if interval < 2 * step:
        raise ValueError(f"IMU rate {rate_hz:g} Hz needs at least two trajectory samples per interval")
    R, p = body[:, :3, :3], body[:, :3, 3]
    rate = Rotation.from_matrix(np.matmul(np.transpose(R[:-2], (0, 2, 1)), R[2:])).as_rotvec() / (2 * step)
    force = np.einsum("nji,nj->ni", R[1:-1], (p[2:] - 2 * p[1:-1] + p[:-2]) / step**2 + _UP)
    k = int(np.floor((t[n - 2] - t[0] - interval / 2) / interval + 1e-9)) + 1
    if k < 1:
        raise ValueError(f"a trajectory of {t[-1] - t[0]:.3f} s holds no IMU sample at {rate_hz:g} Hz")
    times = t[0] + np.arange(k) * interval
    mid = times + interval / 2
    inner = t[1:n - 1]
    gyro = np.stack([np.interp(mid, inner, rate[:, a]) for a in range(3)], axis=1)
    acc = np.stack([np.interp(mid, inner, force[:, a]) for a in range(3)], axis=1)
    return ImuStream(t=np.ascontiguousarray(times, dtype=np.float64),
                     gyro=np.ascontiguousarray(gyro, dtype=np.float64),
                     acc=np.ascontiguousarray(acc, dtype=np.float64))


def imu_session(drives: Sequence[Drive], profile: ImuProfile | None, sampling: ImuSampling, *,
                imu_seed: int | None = None, noise_seed: int | None = None,
                scene: int | None = None) -> list[ImuStream]:
    """The measured stream of every drive of one session, in drive order (module docstring).

    Without a profile, the noiseless samples with the specific force clipped to the range; nothing is
    drawn. With one, the noise of one generator keyed by `imu_seed`, `noise_seed`, `scene` and the layout
    the drives share.
    """
    if profile is not None and any(key is None for key in (imu_seed, noise_seed, scene)):
        raise ValueError("an IMU noise profile needs imu_seed, noise_seed and scene")
    if len({d.layout for d in drives}) > 1:
        raise ValueError("the drives of a session share one layout")
    true = [true_samples(d.t, d.body, sampling.rate_hz) for d in drives]
    limit = sampling.acc_range_g * GRAVITY
    if profile is None:
        return [ImuStream(t=s.t, gyro=s.gyro, acc=np.clip(s.acc, -limit, limit)) for s in true]
    if not true:
        return []
    rng = noise_rng(STEP_IMU, imu_seed, noise_seed, scene, LAYOUTS.index(drives[0].layout))
    dt = 1.0 / sampling.rate_hz
    turn_on = np.repeat([np.radians(profile.gyro_b0_dps), profile.acc_b0_mg * 1e-3 * GRAVITY], 3)
    per_k = np.repeat([np.radians(TEMP_GYRO_DPS_PER_K), TEMP_ACC_MG_PER_K * 1e-3 * GRAVITY], 3)
    walk_sigma = np.repeat([profile.gyro_w, profile.acc_w], 3)
    white_sigma = np.repeat([profile.gyro_n, profile.acc_n], 3)
    b = rng.standard_normal(6) * turn_on
    c = rng.standard_normal(6) * per_k
    parts = []   # per drive: bias, temperature drift, white noise, each (K, 6)
    for i, s in enumerate(true):
        k = len(s.t)
        if i > 0:
            b = b + rng.standard_normal(6) * walk_sigma * np.sqrt(sampling.row_end_s)
        z = rng.standard_normal((k, 12))
        bias = b + np.cumsum(z[:, :6] * walk_sigma * np.sqrt(dt), axis=0)
        b = bias[-1]
        ramp = np.arange(k) / (k - 1) * profile.temp_k if k > 1 else np.zeros(k)
        parts.append((bias, ramp[:, None] * c, z[:, 6:12] * white_sigma / np.sqrt(dt)))
    freq = rng.uniform(*VIB_BAND_HZ, (6, VIB_SINES))
    phase = rng.uniform(0.0, 2 * np.pi, (6, VIB_SINES))
    amplitude = np.sqrt(2.0 / VIB_SINES)
    out = []
    for s, (bias, drift, white) in zip(true, parts):
        sines = np.sin(2 * np.pi * freq * s.t[:, None, None] + phase).sum(axis=2)
        vib_acc = profile.vib_acc_ms2 * amplitude * sines[:, :3]
        vib_gyro = (np.radians(profile.vib_gyro_dps) * amplitude * sines[:, 3:]
                    + np.radians(VIB_G_SENSITIVITY_DPS_PER_G) * vib_acc / GRAVITY)
        gyro = s.gyro + bias[:, :3] + drift[:, :3] + white[:, :3] + vib_gyro
        acc = s.acc + bias[:, 3:] + drift[:, 3:] + white[:, 3:] + vib_acc
        out.append(ImuStream(t=s.t, gyro=gyro, acc=np.clip(acc, -limit, limit)))
    return out


def integrate(stream: ImuStream, R: np.ndarray, p: np.ndarray, v: np.ndarray, ta: float,
              tb: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation, position and velocity at `tb` from body rotation `R`, position `p` and velocity `v` (world)
    at `ta`, integrating `stream` over the pieces of `sample_spans` (module docstring)."""
    spans = sample_spans(stream.t, ta, tb)
    idx = np.array([j for j, _ in spans])
    h = np.array([dt for _, dt in spans])
    turns = Rotation.from_rotvec(stream.gyro[idx] * h[:, None]).as_matrix()
    halves = Rotation.from_rotvec(stream.gyro[idx] * (h[:, None] / 2)).as_matrix()
    force = np.einsum("nij,nj->ni", halves, stream.acc[idx])   # in the body axes half way through each piece
    R = np.array(R, dtype=np.float64)
    p = np.array(p, dtype=np.float64)
    v = np.array(v, dtype=np.float64)
    for m in range(len(idx)):
        a = R @ force[m] - _UP
        p = p + v * h[m] + a * (h[m] * h[m] / 2)
        v = v + a * h[m]
        R = R @ turns[m]
    return R, p, v


def imu_check(run: SynthRun, *, rate_hz: float = 200.0) -> dict:
    """Integrate noiseless samples of each side of a shake bundle (layout `rig`) from the first frame time
    to the last, starting from the trajectory pose there and the velocity of its positions one trajectory
    step either side; the largest end position (m) and rotation (deg) errors against the trajectory.

    A side whose frames share one time has nothing to integrate and adds no error."""
    if not run.meta.shake:
        raise ValueError(f"{run.root}: the IMU check needs a recorded trajectory (a shake bundle)")
    motion = AssumedMotion()
    times = frame_times(run, motion)
    drives = session_drives(run, "rig", motion)
    worst_m = worst_deg = 0.0
    for d in drives:
        tf = sorted(times[f.idx] for f in run.frames if f.pass_key.side == d.side)
        ta, tb = tf[0], tf[-1]
        if not ta < tb:
            continue
        step = (d.t[-1] - d.t[0]) / (len(d.t) - 1)
        first, last = interpolate_pose(d.t, d.body, ta), interpolate_pose(d.t, d.body, tb)
        v = (interpolate_pose(d.t, d.body, ta + step)[:3, 3]
             - interpolate_pose(d.t, d.body, ta - step)[:3, 3]) / (2 * step)
        R, p, _ = integrate(true_samples(d.t, d.body, rate_hz), first[:3, :3], first[:3, 3], v, ta, tb)
        worst_m = max(worst_m, float(np.linalg.norm(p - last[:3, 3])))
        worst_deg = max(worst_deg, float(np.degrees(Rotation.from_matrix(R.T @ last[:3, :3]).magnitude())))
    return {"drives": len(drives), "max_end_m": worst_m, "max_end_deg": worst_deg}
