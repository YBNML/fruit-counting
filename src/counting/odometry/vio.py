"""Batch visual-inertial fusion of one drive, on the prediction side.

Nodes. Node k is frame k of the drive, at `times[k]`, and holds the body-to-world pose X(k); with an IMU
stream it also holds the world velocity V(k) and the IMU bias B(k) (GTSAM `ConstantBias`: accelerometer,
then gyro). A standstill adds two nodes n and n + 1 at its start and end, before the first frame node. A
drive without an IMU stream has poses only.

Factors, in this order: a prior on the start pose X(0); with an IMU stream, a prior on the first bias node
(the standstill start when there is one, else B(0)) and either a prior on the start velocity V(0) or the
standstill (zero velocity at its two nodes, one pose held between them, and IMU factors from its start to
its end and from its end to the first frame node); for each camera and each of its pairs in order, the
odometry of a used pair moved into the body through the believed camera-to-body extrinsic, X M X^-1
(`body_motions`); with an IMU stream, a combined IMU factor between each two neighbouring frame nodes,
preintegrating the measured samples over the pieces of `sample_spans`; and GPS fixes of body positions.

Inputs. A drive holds node times, each camera's odometry and believed extrinsic, the measured IMU stream
with the noise densities and bias sigmas the estimator assumes, the start pose prior with its sigmas, the
start velocity prior and the GPS fixes. None of its fields holds a true pose, a frame position number or
the position step.

gtsam is imported only inside `build_graph` and `fuse_drive`, so importing this module loads numpy and the
standard library alone. One Levenberg-Marquardt run with default parameters, from initial values that
follow the cameras, gives the same bytes for the same drive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from counting.odometry.imufile import GRAVITY, ImuStream, sample_spans

STILL_VELOCITY_SIGMA = 1e-3   # zero-velocity prior of a standstill node (m/s)
STILL_POSE_SIGMA = 1e-4       # the pose held between the two standstill nodes (rad, m)
INTEGRATION_SIGMA2 = 1e-8     # integration covariance of the IMU preintegration


@dataclass(frozen=True, eq=False)
class VioCamera:
    """The odometry of one camera. `extrinsic` (4, 4) is the believed camera-to-body transform; `pairs`
    (m, 2) int are node pairs (a, b) with a < b; `motions` (m, 4, 4) is the camera pose of b seen from
    the camera of a (OpenCV axes); `use` (m,) bool says whether a pair's factor joins the graph."""

    extrinsic: np.ndarray
    pairs: np.ndarray
    motions: np.ndarray
    use: np.ndarray


@dataclass(frozen=True, eq=False)
class VioImu:
    """A measured IMU stream (on the believed clock) with the white noise densities `gyro_n`
    (rad/s/sqrt(Hz)) and `acc_n` (m/s^2/sqrt(Hz)), the bias walks `gyro_w` (rad/s^2/sqrt(Hz)) and `acc_w`
    (m/s^3/sqrt(Hz)) and the bias prior sigmas `gyro_b0` (rad/s) and `acc_b0` (m/s^2) the estimator
    assumes."""

    stream: ImuStream
    gyro_n: float
    acc_n: float
    gyro_w: float
    acc_w: float
    gyro_b0: float
    acc_b0: float


@dataclass(frozen=True, eq=False)
class GpsFix:
    """A measured body position `position` (3,) (world, m) at node `node`, with sigma `sigma` (3,) per
    axis."""

    node: int
    position: np.ndarray
    sigma: np.ndarray


@dataclass(frozen=True, eq=False)
class VioDrive:
    """One drive. `times` (n,) are the node (frame) times in s; `start` (4, 4) is the mean of the start
    body pose prior and `start_sigma` (6,) its sigmas (roll, pitch, yaw rad, x, y, z m; body axes, the
    tangent order of GTSAM's Pose3); `start_velocity` (3,) is the mean of the start velocity prior (world,
    m/s) and `start_velocity_sigma` its sigma; `odometry_sigma` is (translation m, rotation rad) of every
    odometry factor; `imu` None fuses the odometry alone; `standstill` is (start, end) in s of a standstill
    before the first node, or None."""

    times: np.ndarray
    cameras: tuple[VioCamera, ...]
    start: np.ndarray
    start_sigma: np.ndarray
    start_velocity: np.ndarray
    start_velocity_sigma: float
    odometry_sigma: tuple[float, float]
    imu: VioImu | None
    gps: tuple[GpsFix, ...] = ()
    standstill: tuple[float, float] | None = None


def _rigid_inverse(T: np.ndarray) -> np.ndarray:
    """The inverse of the rigid transform `T` (4, 4) in closed form, [[R^T, -R^T t], [0, 1]]."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def body_motions(camera: VioCamera) -> np.ndarray:
    """The odometry of `camera` in the body frame (m, 4, 4): X M X^-1, with X the believed extrinsic and M
    each pair's camera motion."""
    X = np.asarray(camera.extrinsic, dtype=np.float64)
    return X @ np.asarray(camera.motions, dtype=np.float64) @ _rigid_inverse(X)


def _windows(drive: VioDrive, times: np.ndarray) -> list[tuple[float, float]]:
    """The IMU windows (s) of the graph in factor order: the standstill's start to end and end to the first
    node when there is one, then each two neighbouring nodes."""
    out = []
    if drive.standstill is not None:
        begin, end = (float(v) for v in drive.standstill)
        out += [(begin, end), (end, float(times[0]))]
    return out + [(float(times[k - 1]), float(times[k])) for k in range(1, len(times))]


def _check(drive: VioDrive) -> list[list[tuple[int, float]]]:
    """Refuse a drive the graph cannot hold (`ValueError`, in the order of the messages below); return the
    `sample_spans` of every IMU window (`_windows`), or an empty list without an IMU stream."""
    times = np.asarray(drive.times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("a drive needs two nodes or more")
    if not (np.all(np.isfinite(times)) and np.all(np.diff(times) > 0)):
        raise ValueError("node times must increase")
    n = len(times)
    for c, cam in enumerate(drive.cameras):
        X, pairs, motions, use = (np.asarray(v) for v in (cam.extrinsic, cam.pairs, cam.motions, cam.use))
        m = len(pairs) if pairs.ndim == 2 else -1
        if (X.shape != (4, 4) or m < 0 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer)
                or motions.shape != (m, 4, 4) or use.shape != (m,) or use.dtype != np.bool_):
            raise ValueError(f"camera {c}: needs an extrinsic (4, 4), pairs (m, 2) int, motions (m, 4, 4) "
                             f"and use (m,) bool")
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(motions))):
            raise ValueError(f"camera {c}: the extrinsic and motions must be finite")
        if np.any(pairs[:, 0] < 0) or np.any(pairs[:, 0] >= pairs[:, 1]) or np.any(pairs[:, 1] > n - 1):
            raise ValueError(f"camera {c}: pairs must join nodes 0 to {n - 1} forward")
    sigma = np.asarray(drive.start_sigma, dtype=np.float64)
    if sigma.shape != (6,) or not (np.all(np.isfinite(sigma)) and np.all(sigma > 0)):
        raise ValueError("start_sigma must be six values above 0")
    for fix in drive.gps:
        if not 0 <= fix.node <= n - 1:
            raise ValueError(f"a GPS fix is on node {fix.node}; the drive has nodes 0 to {n - 1}")
    if drive.standstill is not None:
        if drive.imu is None:
            raise ValueError("a standstill needs an IMU stream")
        begin, end = (float(v) for v in drive.standstill)
        if not end <= times[0]:
            raise ValueError("a standstill must end at or before the first node")
        if not begin < end:
            raise ValueError("a standstill must end after it starts")
    if drive.imu is not None:
        return [sample_spans(drive.imu.stream.t, ta, tb) for ta, tb in _windows(drive, times)]
    root = list(range(n))   # union-find over the used pairs: every node must join node 0

    def find(k: int) -> int:
        while root[k] != k:
            root[k] = root[root[k]]
            k = root[k]
        return k

    for cam in drive.cameras:
        for (a, b), u in zip(np.asarray(cam.pairs).tolist(), np.asarray(cam.use).tolist()):
            if u:
                root[find(a)] = find(b)
    for k in range(1, n):
        if find(k) != find(0):
            raise ValueError(f"node {k} is reached by no odometry or IMU factor")
    return []


def build_graph(drive: VioDrive) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values, dict]:
    """The factor graph, initial values and keys of `drive` (module docstring): checks, then factors, then
    initial values. The keys are {"poses": the keys X(0) ... X(n - 1), "standstill": the standstill node
    numbers (n, n + 1), or None}.

    Initial values: X(0) is `start`; X(k) is X(k - 1) times the body motion of the first camera that uses
    the pair (k - 1, k), else X(k - 1) times the motion before it (X(k - 2)^-1 X(k - 1)), and X(1) is X(0)
    when there is none. With an IMU stream, V(0) is `start_velocity`, V(k) the move from X(k - 1) to X(k)
    over their time, every bias zero, and the standstill nodes take `start` and zero velocity.
    """
    spans = _check(drive)
    import gtsam
    from gtsam.symbol_shorthand import B, V, X

    times = np.asarray(drive.times, dtype=np.float64)
    n = len(times)
    imu = drive.imu
    still = drive.standstill is not None
    start = np.asarray(drive.start, dtype=np.float64)
    start_velocity = np.asarray(drive.start_velocity, dtype=np.float64)
    trans, rot = (float(v) for v in drive.odometry_sigma)
    Diagonal, Isotropic = gtsam.noiseModel.Diagonal, gtsam.noiseModel.Isotropic
    moved = [body_motions(cam) for cam in drive.cameras]

    graph = gtsam.NonlinearFactorGraph()
    graph.add(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(start),
                                     Diagonal.Sigmas(np.asarray(drive.start_sigma, dtype=np.float64))))
    if imu is not None:
        pims = [_preintegrate(gtsam, _imu_params(gtsam, imu), imu.stream, s) for s in spans]
        bias_noise = Diagonal.Sigmas(np.array([imu.acc_b0] * 3 + [imu.gyro_b0] * 3, dtype=np.float64))
        graph.add(gtsam.PriorFactorConstantBias(B(n) if still else B(0), gtsam.imuBias.ConstantBias(), bias_noise))
        if not still:
            graph.add(gtsam.PriorFactorVector(V(0), start_velocity,
                                              Isotropic.Sigma(3, float(drive.start_velocity_sigma))))
        else:
            zero = Isotropic.Sigma(3, STILL_VELOCITY_SIGMA)
            graph.add(gtsam.PriorFactorVector(V(n), np.zeros(3), zero))
            graph.add(gtsam.PriorFactorVector(V(n + 1), np.zeros(3), zero))
            graph.add(gtsam.BetweenFactorPose3(X(n), X(n + 1), gtsam.Pose3(), Isotropic.Sigma(6, STILL_POSE_SIGMA)))
            graph.add(gtsam.CombinedImuFactor(X(n), V(n), X(n + 1), V(n + 1), B(n), B(n + 1), pims[0]))
            graph.add(gtsam.CombinedImuFactor(X(n + 1), V(n + 1), X(0), V(0), B(n + 1), B(0), pims[1]))
    odometry = Diagonal.Sigmas(np.array([rot] * 3 + [trans] * 3, dtype=np.float64))
    for cam, motions in zip(drive.cameras, moved):
        for (a, b), u, m in zip(np.asarray(cam.pairs).tolist(), np.asarray(cam.use).tolist(), motions):
            if u:
                graph.add(gtsam.BetweenFactorPose3(X(a), X(b), gtsam.Pose3(m), odometry))
    if imu is not None:
        first = 2 if still else 0
        for k in range(1, n):
            graph.add(gtsam.CombinedImuFactor(X(k - 1), V(k - 1), X(k), V(k), B(k - 1), B(k), pims[first + k - 1]))
    for fix in drive.gps:
        graph.add(gtsam.GPSFactor(X(fix.node), np.asarray(fix.position, dtype=np.float64),
                                  Diagonal.Sigmas(np.asarray(fix.sigma, dtype=np.float64))))

    step = {}   # node k -> the body motion of the first camera that uses the pair (k - 1, k)
    for cam, motions in zip(drive.cameras, moved):
        for (a, b), u, m in zip(np.asarray(cam.pairs).tolist(), np.asarray(cam.use).tolist(), motions):
            if u and b == a + 1 and b not in step:
                step[b] = m
    poses = [start]
    for k in range(1, n):
        if k in step:
            poses.append(poses[k - 1] @ step[k])
        elif k >= 2:
            poses.append(poses[k - 1] @ (_rigid_inverse(poses[k - 2]) @ poses[k - 1]))
        else:
            poses.append(poses[0])
    values = gtsam.Values()
    for k, pose in enumerate(poses):
        values.insert(X(k), gtsam.Pose3(pose))
    if imu is not None:
        values.insert(V(0), start_velocity)
        for k in range(1, n):
            values.insert(V(k), (poses[k][:3, 3] - poses[k - 1][:3, 3]) / (times[k] - times[k - 1]))
        for k in range(n):
            values.insert(B(k), gtsam.imuBias.ConstantBias())
        if still:
            for k in (n, n + 1):
                values.insert(X(k), gtsam.Pose3(start))
                values.insert(V(k), np.zeros(3))
                values.insert(B(k), gtsam.imuBias.ConstantBias())
    keys = {"poses": [X(k) for k in range(n)], "standstill": (n, n + 1) if still else None}
    return graph, values, keys


def _imu_params(gtsam, imu: VioImu):
    """Combined preintegration parameters: gravity `GRAVITY` along world -z, sensor covariances from the
    estimator's densities and walks, integration covariance `INTEGRATION_SIGMA2`."""
    params = gtsam.PreintegrationCombinedParams.MakeSharedU(GRAVITY)
    eye = np.eye(3)
    params.setGyroscopeCovariance(eye * float(imu.gyro_n) ** 2)
    params.setAccelerometerCovariance(eye * float(imu.acc_n) ** 2)
    params.setIntegrationCovariance(eye * INTEGRATION_SIGMA2)
    params.setBiasOmegaCovariance(eye * float(imu.gyro_w) ** 2)
    params.setBiasAccCovariance(eye * float(imu.acc_w) ** 2)
    return params


def _preintegrate(gtsam, params, stream: ImuStream, spans: list[tuple[int, float]]):
    """The combined preintegration from zero bias of the samples of `stream` over `spans`."""
    pim = gtsam.PreintegratedCombinedMeasurements(params, gtsam.imuBias.ConstantBias())
    for j, h in spans:
        pim.integrateMeasurement(stream.acc[j], stream.gyro[j], h)
    return pim


def fuse_drive(drive: VioDrive) -> np.ndarray:
    """The fused body-to-world poses (n, 4, 4) float64 of the nodes of `drive`: one Levenberg-Marquardt run
    with default parameters over `build_graph`. A gtsam failure raises `ValueError`."""
    graph, values, keys = build_graph(drive)
    import gtsam

    try:
        result = gtsam.LevenbergMarquardtOptimizer(graph, values, gtsam.LevenbergMarquardtParams()).optimize()
    except RuntimeError as e:
        raise ValueError(f"the fusion graph of the drive is singular: {e}") from e
    return np.ascontiguousarray(np.stack([result.atPose3(key).matrix() for key in keys["poses"]]),
                                dtype=np.float64)
