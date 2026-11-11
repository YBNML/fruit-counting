"""The batch fusion of one drive: odometry factors of one or more cameras through their believed extrinsics,
IMU factors from a measured stream, start, speed and bias priors, an optional standstill and row-end GPS fixes."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from counting.evaluation.imusim import AssumedMotion, Drive, ImuProfile, ImuSampling, imu_session, straight_drive
from counting.geometry.rig import interpolate_pose, pitch_extrinsic, rigid
from counting.odometry.imufile import GRAVITY
from counting.odometry.vio import GpsFix, VioCamera, VioDrive, VioImu, body_motions, build_graph, fuse_drive

pytestmark = pytest.mark.pose
gtsam = pytest.importorskip("gtsam")

NODES = np.arange(0.5, 11.5, 0.3)                  # 37 node times (s)
ESTIMATOR = {"gyro_n": 7.85e-5, "acc_n": 9.81e-4, "gyro_w": 1.82e-6, "acc_w": 4.75e-5}
X10, X50 = pitch_extrinsic(10.0), pitch_extrinsic(50.0)
TIGHT = np.full(6, 1e-4)
NOMINAL = np.r_[np.radians([5.0, 5.0, 0.5]), [0.05] * 3]


def _truth():
    """1 kHz body poses of a 0.5 m/s drive along x, swinging 1 deg in yaw (5 s) and 1 cm in height (3 s)."""
    t = np.arange(12001) / 1000
    P = np.repeat(np.eye(4)[None], len(t), axis=0)
    P[:, :3, :3] = Rotation.from_rotvec(np.outer(np.radians(1.0) * np.sin(2 * np.pi * t / 5.0),
                                                 [0.0, 0.0, 1.0])).as_matrix()
    P[:, 0, 3] = 0.5 * t
    P[:, 2, 3] = 0.9 + 0.01 * np.sin(2 * np.pi * t / 3.0)
    return t, P


T_FINE, P_FINE = _truth()
BODIES = np.stack([interpolate_pose(T_FINE, P_FINE, x) for x in NODES])


def _camera(X, *, believed=None, turn=None, use=None, bodies=BODIES):
    """The odometry of a camera with the true extrinsic X between neighbouring nodes; `turn` right-multiplies
    every body increment (a drift), `believed` is the extrinsic the fusion is given (X by default)."""
    inc = np.stack([np.linalg.inv(a) @ b for a, b in zip(bodies[:-1], bodies[1:])])
    if turn is not None:
        inc = inc @ turn
    n = len(inc)
    return VioCamera(extrinsic=X if believed is None else believed,
                     pairs=np.stack([np.arange(n), np.arange(1, n + 1)], axis=1),
                     motions=np.linalg.inv(X) @ inc @ X, use=np.ones(n, bool) if use is None else use)


def _imu(profile=None, seed=0):
    d = Drive(layout="rig", side="A", aisle=1, pitch_deg=None, t=T_FINE, body=P_FINE, source="recorded")
    key = {} if profile is None else {"imu_seed": 0, "noise_seed": seed, "scene": 1}
    (stream,) = imu_session([d], profile, ImuSampling(), **key)
    return VioImu(stream=stream, **ESTIMATOR, gyro_b0=np.radians(1.0), acc_b0=4e-3 * GRAVITY)


def _drive(cameras, imu, *, start=None, start_sigma=TIGHT, sigma=(1e-5, np.radians(1e-4)), gps=(), standstill=None,
           times=NODES):
    start = BODIES[0] if start is None else start
    return VioDrive(times=times, cameras=tuple(cameras), start=start, start_sigma=np.asarray(start_sigma, float),
                    start_velocity=start[:3, :3] @ [0.5, 0.0, 0.0], start_velocity_sigma=0.2, odometry_sigma=sigma,
                    imu=imu, gps=tuple(gps), standstill=standstill)


def _err(out, truth=BODIES):
    pos = np.abs(out[:, :3, 3] - truth[:, :3, 3]).max()
    rot = max(np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()) for a, b in zip(out, truth))
    return pos, rot


def _tilt(a, b):
    w = Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).as_rotvec()
    return np.degrees(np.hypot(w[0], w[1]))


def _factors(graph, kind):
    """The factors of `graph` of the gtsam class (or classes) `kind`, in graph order."""
    return [f for f in (graph.at(i) for i in range(graph.size())) if isinstance(f, kind)]


def _priors(graph):
    """The prior factors of `graph` in graph order, by key name ("x0", "b31", ...): (mean, sigmas)."""
    out = {}
    for f in _factors(graph, (gtsam.PriorFactorPose3, gtsam.PriorFactorVector, gtsam.PriorFactorConstantBias)):
        (key,) = f.keys()
        name, mean = gtsam.Symbol(key).string(), f.prior()
        assert name not in out
        if isinstance(mean, gtsam.Pose3):
            mean = mean.matrix()
        elif isinstance(mean, gtsam.imuBias.ConstantBias):
            mean = mean.vector()
        out[name] = (np.asarray(mean), f.noiseModel().sigmas())
    return out


def test_body_motions_are_the_conjugated_camera_motions():
    believed = X10 @ rigid(np.radians([0.3, -0.2, 0.1]), [0.01, 0.0, -0.005])
    cam = _camera(X10, believed=believed)
    np.testing.assert_allclose(body_motions(cam), believed @ cam.motions @ np.linalg.inv(believed), rtol=0, atol=1e-12)
    truth = np.stack([np.linalg.inv(a) @ b for a, b in zip(BODIES[:-1], BODIES[1:])])
    np.testing.assert_allclose(body_motions(_camera(X10)), truth, rtol=0, atol=1e-12)


def test_truth_odometry_and_noiseless_imu_give_the_truth():
    out = fuse_drive(_drive([_camera(X10)], _imu()))
    assert out.shape == (len(NODES), 4, 4) and out.dtype == np.float64
    pos, rot = _err(out)
    assert pos < 1e-5 and rot < 1e-4


def test_imu_takes_out_the_tilt_drift_of_odometry():
    cams = [_camera(X10, turn=rigid([0.0, np.radians(0.05), 0.0], np.zeros(3)))]   # 0.05 deg of body pitch a step
    sigma = (2e-3, np.radians(0.04))
    alone = fuse_drive(_drive(cams, None, start_sigma=NOMINAL, sigma=sigma))
    fused = fuse_drive(_drive(cams, _imu(), start_sigma=NOMINAL, sigma=sigma))
    assert _tilt(alone[-1], BODIES[-1]) > 1.5
    assert _tilt(fused[-1], BODIES[-1]) < 0.1 * _tilt(alone[-1], BODIES[-1])


def test_two_cameras_and_failed_pairs():
    use = np.ones(len(NODES) - 1, bool)
    use[10] = False
    good, bad = _camera(X50, use=use), _camera(X50, use=use)
    bad.motions[10] = rigid([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])           # the pose of a pair that is not used
    one = fuse_drive(_drive([_camera(X10), good], _imu()))
    two = fuse_drive(_drive([_camera(X10), bad], _imu()))
    assert one.tobytes() == two.tobytes()
    pos, rot = _err(fuse_drive(_drive([_camera(X10, use=use), _camera(X50, use=use)], _imu())))
    assert pos < 1e-3 and rot < 1e-2                                     # the IMU bridges the pair both cameras lost
    with pytest.raises(ValueError, match="node 11 is reached by no odometry or IMU factor"):
        fuse_drive(_drive([_camera(X10, use=use)], None))


def test_odometry_only_graph_follows_its_cameras():
    cam = _camera(X10)
    start = BODIES[0] @ rigid([0.0, 0.0, 0.01], [0.02, 0.0, 0.0])
    out = fuse_drive(_drive([cam], None, start=start, start_sigma=NOMINAL))
    chain = [start]
    for m in body_motions(cam):
        chain.append(chain[-1] @ m)
    np.testing.assert_allclose(out, np.stack(chain), rtol=0, atol=1e-9)
    longer = _camera(X10, turn=rigid(np.zeros(3), [0.001, 0.0, 0.0]))   # every step 1 mm longer
    drive = _drive([cam, longer], None, start_sigma=NOMINAL, sigma=(1e-3, np.radians(0.01)))
    both = fuse_drive(drive)
    x = BODIES[-1][0, 3]
    assert abs(both[-1][0, 3] - (x + 0.018)) < 0.002                     # half way between the two cameras
    graph, values, _ = build_graph(drive)
    odometry = _factors(graph, gtsam.BetweenFactorPose3)                 # cameras in order, then pairs in order
    assert [[gtsam.Symbol(k).index() for k in f.keys()] for f in odometry] == 2 * cam.pairs.tolist()
    np.testing.assert_allclose([f.measured().matrix() for f in odometry],
                               np.concatenate([body_motions(cam), body_motions(longer)]), rtol=0, atol=1e-12)
    for f in odometry:                                   # the Pose3 tangent order: rotation, then translation
        np.testing.assert_allclose(f.noiseModel().sigmas(), [np.radians(0.01)] * 3 + [1e-3] * 3, rtol=1e-12)
    assert list(_priors(graph)) == ["x0"] and {gtsam.Symbol(k).string()[0] for k in values.keys()} == {"x"}


def test_same_drive_twice_gives_the_same_bytes():
    profile = ImuProfile(gyro_n=1.16e-3, acc_n=4.97e-3, gyro_w=1.82e-5, acc_w=4.75e-4, gyro_b0_dps=1.0, acc_b0_mg=4.0)
    drive = _drive([_camera(X10, turn=rigid([0.0, np.radians(0.02), 0.0], [0.0005, 0.0, 0.0])), _camera(X50)],
                   _imu(profile), start_sigma=NOMINAL, sigma=(2e-3, np.radians(0.04)))
    assert fuse_drive(drive).tobytes() == fuse_drive(drive).tobytes()


def test_row_end_gps_fixes_pull_the_world_end_in():
    cams = [_camera(X10, turn=rigid([0.0, 0.0, np.radians(0.03)], np.zeros(3)))]
    start = BODIES[0].copy()
    start[:3, 3] += [0.1, -0.1, 0.0]
    start[:3, :3] = Rotation.from_rotvec([0.0, 0.0, np.radians(0.5)]).as_matrix() @ start[:3, :3]
    sigma = (2e-3, np.radians(0.04))
    fixes = (GpsFix(node=0, position=BODIES[0, :3, 3], sigma=np.array([0.02, 0.02, 0.03])),
             GpsFix(node=len(NODES) - 1, position=BODIES[-1, :3, 3], sigma=np.array([0.02, 0.02, 0.03])))
    without = fuse_drive(_drive(cams, _imu(), start=start, start_sigma=NOMINAL, sigma=sigma))
    fixed = fuse_drive(_drive(cams, _imu(), start=start, start_sigma=NOMINAL, sigma=sigma, gps=fixes))

    def end(out):
        return np.linalg.norm(out[-1, :3, 3] - BODIES[-1, :3, 3])

    assert end(without) > 0.05 and end(fixed) < 0.5 * end(without)
    with pytest.raises(ValueError, match="a GPS fix is on node 37; the drive has nodes 0 to 36"):
        fuse_drive(_drive(cams, _imu(), gps=(GpsFix(node=37, position=np.zeros(3), sigma=np.ones(3)),)))


def test_standstill_window_adds_still_nodes():
    from gtsam.symbol_shorthand import B, V, X

    line = np.repeat(np.eye(4)[None], 31, axis=0)
    line[:, 0, 3] = 0.15 * np.arange(31)
    t, body, tf = straight_drive(line, AssumedMotion())
    d = Drive(layout="rig", side="A", aisle=1, pitch_deg=None, t=t, body=body, source="straight_assumed")
    (stream,) = imu_session([d], None, ImuSampling())
    imu = VioImu(stream=stream, **ESTIMATOR, gyro_b0=np.radians(1.0), acc_b0=4e-3 * GRAVITY)
    cam = _camera(X10, bodies=line)
    plain = _drive([cam], imu, start=line[0], times=tf)
    still = _drive([cam], imu, start=line[0], times=tf, standstill=(stream.t[0], stream.t[0] + 5.0))
    g0, v0, k0 = build_graph(plain)
    g1, v1, k1 = build_graph(still)
    assert k0["standstill"] is None and k1["standstill"] == (31, 32) and len(k1["poses"]) == 31
    assert g1.size() == g0.size() + 4     # two zero-speed priors, a hold and two IMU factors, less the start speed prior
    p0, p1 = _priors(g0), _priors(g1)
    assert list(p0) == ["x0", "b0", "v0"] and list(p1) == ["x0", "b31", "v31", "v32"]
    bias = gtsam.imuBias.ConstantBias(np.full(3, imu.acc_b0), np.full(3, imu.gyro_b0)).vector()   # acc, then gyro
    for mean, sigmas in (p0["b0"], p1["b31"]):
        np.testing.assert_array_equal(mean, np.zeros(6))
        np.testing.assert_allclose(sigmas, bias, rtol=1e-12)
    np.testing.assert_array_equal(p0["v0"][0], plain.start_velocity)
    np.testing.assert_allclose(p0["v0"][1], [0.2] * 3, rtol=1e-12)
    for name in ("v31", "v32"):
        np.testing.assert_array_equal(p1[name][0], np.zeros(3))
        np.testing.assert_allclose(p1[name][1], [1e-3] * 3, rtol=1e-12)
    steps = np.diff(line[:, :3, 3], axis=0) / np.diff(tf)[:, None]      # the camera chain's moves over node times
    for values in (v0, v1):
        np.testing.assert_array_equal(values.atVector(V(0)), plain.start_velocity)   # under a standstill too
        np.testing.assert_allclose([values.atVector(V(k)) for k in range(1, 31)], steps, rtol=0, atol=1e-9)
        assert not np.any([values.atConstantBias(B(k)).vector() for k in range(31)])
    for k in (31, 32):
        np.testing.assert_array_equal(v1.atPose3(X(k)).matrix(), line[0])
        assert not v1.atVector(V(k)).any() and not v1.atConstantBias(B(k)).vector().any()
    pos, rot = _err(fuse_drive(still), line)
    assert pos < 1e-5 and rot < 1e-4
    for window, message in (((stream.t[0], tf[0] + 0.1), "a standstill must end at or before the first node"),
                            ((stream.t[0] - 1.0, stream.t[0] + 5.0), "IMU samples cover")):
        with pytest.raises(ValueError, match=message):
            fuse_drive(_drive([cam], imu, start=line[0], times=tf, standstill=window))


def test_standstill_gyro_turn_on_bias_does_not_tilt_the_start():
    line = np.repeat(np.eye(4)[None], 31, axis=0)
    line[:, 0, 3] = 0.15 * np.arange(31)
    t, body, tf = straight_drive(line, AssumedMotion())
    d = Drive(layout="rig", side="A", aisle=1, pitch_deg=None, t=t, body=body, source="straight_assumed")
    cam = _camera(X10, bodies=line)
    profile = ImuProfile(gyro_b0_dps=1.0)            # the gyro turn-on bias alone, drawn with 1 deg/s per axis
    errors = {}
    for seed in (0, 1, 2):
        (stream,) = imu_session([d], profile, ImuSampling(), imu_seed=0, noise_seed=seed, scene=5)
        imu = VioImu(stream=stream, **ESTIMATOR, gyro_b0=np.radians(1.0), acc_b0=4e-3 * GRAVITY)
        for standstill in (None, (stream.t[0], stream.t[0] + 5.0)):
            out = fuse_drive(_drive([cam], imu, start=line[0], start_sigma=NOMINAL, sigma=(2.1e-3, np.radians(0.039)),
                                    times=tf, standstill=standstill))
            errors[seed, standstill is not None] = (_tilt(out[0], line[0]), abs(out[-1, 2, 3] - line[-1, 2, 3]))
    assert max(tilt for tilt, _ in errors.values()) < 0.05, errors      # deg, start tilt
    assert max(height for _, height in errors.values()) < 0.01, errors  # m, end height


def test_drive_inputs_are_checked():
    cam = _camera(X10)
    far = VioCamera(extrinsic=X10, pairs=np.array([[0, 37]]), motions=np.eye(4)[None], use=np.ones(1, bool))
    cases = [({"cameras": [], "times": NODES[:1]}, "a drive needs two nodes or more"),
             ({"times": NODES[::-1].copy()}, "node times must increase"),
             ({"cameras": [far]}, "camera 0: pairs must join nodes 0 to 36 forward"),
             ({"start_sigma": np.r_[TIGHT[:5], 0.0]}, "start_sigma must be six values above 0"),
             ({"times": NODES + 1.0}, "IMU samples cover")]
    for over, message in cases:
        cameras = over.pop("cameras", [cam])
        with pytest.raises(ValueError, match=message):
            fuse_drive(_drive(cameras, _imu(), **over))
