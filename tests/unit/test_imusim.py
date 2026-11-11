"""The simulated IMU: interval-midpoint samples from a body trajectory, the drives of a session with their
assumed straight motion, the strapdown integration that checks them, and the one seeded noise stream of a
session (turn-on bias, bias walk, white noise, temperature and vibration)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from counting.config.pose_schema import ImuConfig
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.imusim import (
    AssumedMotion, Drive, ImuProfile, ImuSampling, frame_times, imu_check, imu_session, integrate, session_drives,
    straight_drive, true_samples,
)
from counting.geometry.rig import body_from_reference, interpolate_pose, pitch_extrinsic, pose_matrix
from counting.odometry.imufile import GRAVITY, ImuStream, stream_key
from tests.unit.synthfake import write_run

pytestmark = pytest.mark.pose

UP = np.array([0.0, 0.0, GRAVITY])


@pytest.fixture(scope="module")
def shake_root(tmp_path_factory):
    return write_run(tmp_path_factory.mktemp("shake"), crop="apple", size=(64, 36), shake=True)


def _swing(seconds: float = 20.0):
    """1 kHz body poses turning about z at 2 deg/s while the height swings 5 cm with a 4 s period."""
    t = np.arange(round(seconds * 1000) + 1) / 1000
    P = np.repeat(np.eye(4)[None], len(t), axis=0)
    P[:, :3, :3] = Rotation.from_rotvec(np.outer(np.radians(2.0) * t, [0.0, 0.0, 1.0])).as_matrix()
    P[:, 2, 3] = 0.05 * np.sin(2 * np.pi * t / 4.0)
    return t, P


def _still(seconds: float, side: str = "A") -> Drive:
    t = np.arange(round(seconds * 1000) + 1) / 1000
    return Drive(layout="rig", side=side, aisle=1, pitch_deg=None, t=t,
                 body=np.repeat(np.eye(4)[None], len(t), axis=0), source="recorded")


def test_true_samples_take_interval_midpoints():
    t, P = _swing()
    s = true_samples(t, P, 200.0)
    assert isinstance(s, ImuStream) and s.t[0] == 0.0 and s.gyro.dtype == np.float64
    np.testing.assert_allclose(np.diff(s.t), 0.005, rtol=0, atol=1e-12)
    assert s.t[-1] + 0.0025 <= t[-2] < s.t[-1] + 0.0075          # every midpoint inside the central differences
    mid = s.t + 0.0025
    np.testing.assert_allclose(s.gyro, np.tile([0.0, 0.0, np.radians(2.0)], (len(mid), 1)), rtol=0, atol=1e-9)
    w = 2 * np.pi / 4.0
    np.testing.assert_allclose(s.acc[:, 2], GRAVITY - 0.05 * w * w * np.sin(w * mid), rtol=0, atol=1e-6)
    np.testing.assert_allclose(s.acc[:, :2], 0.0, rtol=0, atol=1e-9)
    bent = t.copy()
    bent[10:] += 1e-4
    with pytest.raises(ValueError, match="trajectory samples are not evenly spaced"):
        true_samples(bent, P, 200.0)
    with pytest.raises(ValueError, match="IMU rate 800 Hz needs at least two trajectory samples per interval"):
        true_samples(t, P, 800.0)


def test_noiseless_samples_integrate_back_to_the_recorded_drive(shake_root):
    run = SynthRun.open(shake_root)
    motion = AssumedMotion()
    drives = session_drives(run, "rig", motion)
    assert [(d.layout, d.side, d.pitch_deg, d.source) for d in drives] == [
        ("rig", "A", None, "recorded"), ("rig", "B", None, "recorded")]
    times = frame_times(run, motion)
    assert times == {i: p.t for i, p in run.plans().items()}
    for d in drives:
        np.testing.assert_array_equal(d.t, run.trajectory(d.side).t)
        s = true_samples(d.t, d.body, 200.0)
        tf = sorted(times[f.idx] for f in run.frames if f.pass_key.side == d.side)
        a, b = interpolate_pose(d.t, d.body, tf[0]), interpolate_pose(d.t, d.body, tf[-1])
        v = (interpolate_pose(d.t, d.body, tf[0] + 1e-3)[:3, 3]
             - interpolate_pose(d.t, d.body, tf[0] - 1e-3)[:3, 3]) / 2e-3
        R, p, _ = integrate(s, a[:3, :3], a[:3, 3], v, tf[0], tf[-1])
        # 0.1 mm over these 9 s: a step that turns the force with its start rotation ends 0.9 and 2.3 mm off
        # here, an error that grows with the drive's length and the IMU interval
        assert np.linalg.norm(p - b[:3, 3]) < 1e-4
        assert Rotation.from_matrix(R.T @ b[:3, :3]).magnitude() < np.radians(1e-3)
    res = imu_check(run)
    assert res["drives"] == 2 and res["max_end_m"] < 1e-4 and res["max_end_deg"] < 1e-3


def test_straight_bundles_get_an_assumed_drive_and_frame_times(apple_run):
    run = SynthRun.open(apple_run)
    motion = AssumedMotion(speed_mps=0.5, stop_s=5.0, accel_s=2.0)
    profile = {"gyro_n": 1.0, "acc_n": 1.0, "gyro_w": 0.0, "acc_w": 0.0, "gyro_b0_dps": 1.0, "acc_b0_mg": 1.0}
    assert motion == AssumedMotion() == AssumedMotion.of(ImuConfig(profile="x", noise=profile))
    times = frame_times(run, motion)
    for d in session_drives(run, "rig", motion):
        frames = sorted((f for f in run.frames if f.pass_key.side == d.side and f.pass_key.pitch_deg == 10.0),
                        key=lambda f: f.idx)
        bodies = np.stack([body_from_reference(pose_matrix(f.camera), pitch_extrinsic(10.0)) for f in frames])
        t, body, tf = straight_drive(bodies, motion)
        assert d.source == "straight_assumed"
        np.testing.assert_array_equal(d.t, t)
        np.testing.assert_array_equal(d.body, body)
        along = np.linalg.norm(bodies[:, :3, 3] - bodies[0, :3, 3], axis=1)
        np.testing.assert_allclose(tf, 7.0 + along / 0.5, rtol=0, atol=1e-9)
        assert [times[f.idx] for f in frames] == tf.tolist()
        assert np.ptp(body[t <= 5.0, :3, 3], axis=0).max() == 0.0           # standing still
        for k in range(len(frames)):
            np.testing.assert_allclose(interpolate_pose(t, body, tf[k]), bodies[k], rtol=0, atol=1e-9)
        s = true_samples(t, body, 200.0)
        assert np.abs(s.gyro).max() < 1e-12
        cruise = s.t > 7.01
        np.testing.assert_allclose(s.acc[cruise], np.tile(bodies[0, :3, :3].T @ UP, (cruise.sum(), 1)),
                                   rtol=0, atol=1e-6)
        # from rest at t = 0 the speed-up brings the body to the first frame at the cruise speed
        _, p, v = integrate(s, body[0, :3, :3], body[0, :3, 3], np.zeros(3), 0.0, tf[0])
        assert np.linalg.norm(p - bodies[0, :3, 3]) < 1e-4
        np.testing.assert_allclose(v, motion.speed_mps * bodies[0, :3, 0], rtol=0, atol=1e-4)
    for f in run.frames:          # a frame of another pitch takes the time of its position
        same = next(g for g in run.frames if g.pass_key.side == f.pass_key.side and g.pos_k == f.pos_k
                    and g.pass_key.pitch_deg == 10.0)
        assert times[f.idx] == times[same.idx]
    turned = bodies.copy()
    turned[3, :3, :3] = Rotation.from_rotvec([0.0, 0.0, 0.01]).as_matrix() @ turned[3, :3, :3]
    with pytest.raises(ValueError, match="a straight drive needs one body rotation"):
        straight_drive(turned, motion)
    with pytest.raises(ValueError, match="a straight drive needs frames that advance along the travel"):
        straight_drive(bodies[::-1], motion)


def test_session_drives_follow_the_layout_order(apple_run, shake_root):
    for root in (apple_run, shake_root):
        run = SynthRun.open(root)
        aisle = {f.pass_key.side: f.pass_key.aisle for f in run.frames}
        rig = session_drives(run, "rig", AssumedMotion())
        per_pass = session_drives(run, "pass", AssumedMotion())
        assert [(d.layout, d.side, d.pitch_deg) for d in rig] == [("rig", "A", None), ("rig", "B", None)]
        assert [(d.layout, d.side, d.pitch_deg) for d in per_pass] == [
            ("pass", "A", 10.0), ("pass", "B", 10.0), ("pass", "A", 30.0), ("pass", "B", 30.0)]
        assert [d.stream_key(2) for d in rig] == [(0, 2, 0, aisle["A"], -1), (0, 2, 1, aisle["B"], -1)]
        assert per_pass[2].stream_key(0) == stream_key("pass", 0, "A", aisle["A"], 30.0) == (1, 0, 0, aisle["A"], 30000)
        for d in per_pass:        # a pass drive rides its side's body
            twin = next(r for r in rig if r.side == d.side)
            assert np.array_equal(d.t, twin.t) and np.array_equal(d.body, twin.body) and d.aisle == twin.aisle
    with pytest.raises(ValueError, match="unknown IMU layout 'side'"):
        session_drives(SynthRun.open(apple_run), "side", AssumedMotion())
    with pytest.raises(ValueError, match="a rig drive has no pitch and a pass drive has one"):
        stream_key("rig", 0, "A", 1, 10.0)


I_EXT = ImuProfile(gyro_n=1.16e-3, acc_n=4.97e-3, gyro_w=1.82e-5, acc_w=4.75e-4, gyro_b0_dps=1.0, acc_b0_mg=4.0,
                   temp_k=1.0)
KEY = {"imu_seed": 0, "noise_seed": 0, "scene": 5}


def test_one_seeded_stream_per_session_key(shake_root):
    run = SynthRun.open(shake_root)
    drives = session_drives(run, "rig", AssumedMotion())
    true = imu_session(drives, None, ImuSampling())

    def session(profile, **key):
        return imu_session(drives, profile, ImuSampling(), **{**KEY, **key})

    a, b = session(I_EXT), session(I_EXT)
    assert all(x.gyro.tobytes() == y.gyro.tobytes() and x.acc.tobytes() == y.acc.tobytes() for x, y in zip(a, b))
    assert all(np.array_equal(x.t, y.t) for x, y in zip(a, true))
    for key in ({"imu_seed": 1}, {"noise_seed": 1}, {"scene": 6}):
        assert not np.array_equal(session(I_EXT, **key)[0].gyro, a[0].gyro), key
    per_pass = imu_session(session_drives(run, "pass", AssumedMotion())[:2], I_EXT, ImuSampling(), **KEY)
    assert not np.array_equal(per_pass[0].gyro, a[0].gyro)                  # the layout is part of the key
    # every draw is made whatever the values, so doubling one sigma doubles its part only
    one, two = session(ImuProfile(gyro_n=1e-3)), session(ImuProfile(gyro_n=2e-3))
    for x, y, z in zip(one, two, true):
        np.testing.assert_allclose(y.gyro - z.gyro, 2 * (x.gyro - z.gyro), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(x.acc, z.acc)
    with pytest.raises(ValueError, match="an IMU noise profile needs imu_seed, noise_seed and scene"):
        imu_session(drives, I_EXT, ImuSampling())
    with pytest.raises(ValueError, match="the drives of a session share one layout"):
        imu_session([drives[0], session_drives(run, "pass", AssumedMotion())[0]], I_EXT, ImuSampling(), **KEY)


def test_white_noise_walk_and_turn_on_bias_statistics():
    sampling = ImuSampling()
    long = [_still(100.0)]                      # 20000 samples: the standard error of a std is 0.5%
    white = imu_session(long, ImuProfile(gyro_n=1e-3, acc_n=5e-3), sampling, **KEY)[0]
    assert np.std(white.gyro, axis=0) == pytest.approx([1e-3 * np.sqrt(200.0)] * 3, rel=0.03)
    assert np.std(white.acc - UP, axis=0) == pytest.approx([5e-3 * np.sqrt(200.0)] * 3, rel=0.03)
    walk = imu_session(long, ImuProfile(gyro_w=1e-4, acc_w=2e-3), sampling, **KEY)[0]
    assert np.std(np.diff(walk.gyro, axis=0), axis=0) == pytest.approx([1e-4 * np.sqrt(0.005)] * 3, rel=0.03)
    assert np.std(np.diff(walk.acc, axis=0), axis=0) == pytest.approx([2e-3 * np.sqrt(0.005)] * 3, rel=0.03)
    short = [_still(0.02)]
    first = np.array([np.r_[s.gyro[0], s.acc[0] - UP] for s in (
        imu_session(short, ImuProfile(gyro_b0_dps=1.0, acc_b0_mg=4.0), sampling, imu_seed=0, noise_seed=n,
                    scene=5)[0] for n in range(300))])
    assert np.std(first[:, :3]) == pytest.approx(np.radians(1.0), rel=0.1)
    assert np.std(first[:, 3:]) == pytest.approx(4e-3 * GRAVITY, rel=0.1)


def test_bias_runs_on_across_the_drives_of_a_session():
    sampling = ImuSampling(row_end_s=20.0)
    two = [_still(0.05, "A"), _still(0.05, "B")]
    gaps, joins = [], []
    for n in range(300):
        key = {"imu_seed": 0, "noise_seed": n, "scene": 5}
        a, b = imu_session(two, ImuProfile(gyro_w=1e-3), sampling, **key)
        gaps.append(b.gyro[0] - a.gyro[-1])
        a, b = imu_session(two, ImuProfile(gyro_w=1e-3), ImuSampling(row_end_s=0.0), **key)
        joins.append(b.gyro[0] - a.gyro[-1])
    assert np.std(gaps) == pytest.approx(1e-3 * np.sqrt(20.0 + 0.005), rel=0.1)
    # with no row end, the next drive starts one sample step from the bias the last drive ended on
    assert np.std(joins) == pytest.approx(1e-3 * np.sqrt(0.005), rel=0.1)
    one = imu_session(two[:1], ImuProfile(gyro_b0_dps=1.0), sampling, **KEY)
    both = imu_session(two, ImuProfile(gyro_b0_dps=1.0), sampling, **KEY)
    np.testing.assert_array_equal(both[0].gyro, one[0].gyro)                 # one turn-on bias per session
    np.testing.assert_array_equal(both[1].gyro, np.tile(both[0].gyro[0], (len(both[1].t), 1)))


def test_temperature_vibration_and_range():
    sampling = ImuSampling()
    warm = imu_session([_still(10.0)], ImuProfile(temp_k=1.0), sampling, **KEY)[0]
    assert np.all(warm.gyro[0] == 0.0) and np.abs(warm.gyro[-1]).max() > 0
    np.testing.assert_allclose(np.diff(warm.gyro, 2, axis=0), 0.0, rtol=0, atol=1e-15)   # a straight ramp per drive
    np.testing.assert_allclose(np.diff(warm.acc, 2, axis=0), 0.0, rtol=0, atol=1e-12)
    five = [_still(5.0)]
    acc_power, gyro_power = [], []
    for n in range(10):          # the power of random-phase sines, averaged over sessions
        key = {"imu_seed": 0, "noise_seed": n, "scene": 5}
        both = imu_session(five, ImuProfile(vib_acc_ms2=1.0, vib_gyro_dps=0.5), sampling, **key)[0]
        acc_only = imu_session(five, ImuProfile(vib_acc_ms2=1.0), sampling, **key)[0]
        np.testing.assert_allclose(acc_only.gyro, np.radians(0.1) * (acc_only.acc - UP) / GRAVITY, rtol=0, atol=1e-15)
        acc_power.append(np.mean((both.acc - UP) ** 2, axis=0))
        gyro_power.append(np.mean((both.gyro - acc_only.gyro) ** 2, axis=0))
    assert np.sqrt(np.mean(acc_power, axis=0)) == pytest.approx([1.0] * 3, rel=0.1)
    assert np.sqrt(np.mean(gyro_power, axis=0)) == pytest.approx([np.radians(0.5)] * 3, rel=0.1)
    clipped = imu_session([_still(10.0)], ImuProfile(acc_n=10.0), sampling, **KEY)[0]
    assert np.abs(clipped.acc).max() == 4.0 * GRAVITY and np.abs(clipped.gyro).max() == 0.0
