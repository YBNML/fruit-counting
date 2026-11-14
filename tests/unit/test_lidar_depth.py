"""LiDAR box depth: the poses at the point times, the points of a pass in the world, the time window, the box
matches (far, gate, map), the joined depth and its believed sigmas, the LiDAR point noise and the depth error form
draws, and the extrinsic draws of the two LiDAR units."""

import inspect
import re
from math import asin, degrees, radians

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from counting.config.eval_schema import LiftConfig
from counting.evaluation.fusion import imu_errors, imu_extr_error
from counting.evaluation.lidarsource import LIDAR_UNIT_CODE, camera_from_sensor, lidar_extr_error, noisy_points
from counting.evaluation.posesource import rig_errors
from counting.geometry import noise as noise_module
from counting.geometry.camera import Camera
from counting.geometry.lidarfile import LidarPoints
from counting.geometry.noise import (
    STEP_DEPTH_PERSIST, STEP_DEPTH_SCALE, STEP_LIDAR, STEP_LIDAR_EXTR, depth_persist_field, depth_scale,
    lidar_point_noise, seeded_rigid,
)
from counting.geometry.rig import believed_extrinsic, interpolate_pose, interpolate_poses, pitch_extrinsic
from counting.multiview import lidardepth
from counting.multiview.lidardepth import (
    CLOUD_CHUNK, MAP_VOXEL_M, RANGE_BELIEF_FACTOR, BoxLidar, LidarSettings, PassCloud, belief_sigmas, box_lidar_depth,
    combine_depth, far_depth, gate_depth, map_depth, pass_cloud, window,
)
from counting.multiview.lift import box_far_depth
from tests.unit.test_depth_schema import _cfg, _lidar

pytestmark = pytest.mark.pose

# A small camera at the world origin looking along +z, and the boxes and points of a random scene on it.
SMALL = Camera(K=np.array([[50.0, 0.0, 31.5], [0.0, 50.0, 23.5], [0.0, 0.0, 1.0]]), R_c2w=np.eye(3), C=np.zeros(3),
               width=64, height=48)
# A camera for the fruit, leaf and wall scene.
WIDE = Camera(K=np.array([[200.0, 0.0, 99.5], [0.0, 200.0, 99.5], [0.0, 0.0, 1.0]]), R_c2w=np.eye(3),
              C=np.zeros(3), width=200, height=200)
FRUIT_Z, LEAF_Z, WALL_Z = 1.875, 1.70, 2.50


def _settings(**knobs) -> LidarSettings:
    return LidarSettings.from_lift(LiftConfig(lidar=knobs))


def _matrices(quat: np.ndarray, trans: np.ndarray) -> np.ndarray:
    T = np.repeat(np.eye(4)[None], len(quat), axis=0)
    T[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
    T[:, :3, 3] = trans
    return T


def _translation(xyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = xyz
    return T


def test_interpolate_poses_follows_the_single_pose_rule():
    rng = np.random.default_rng(31)
    times = np.cumsum(rng.uniform(0.05, 0.2, 7))
    poses = np.repeat(np.eye(4)[None], 7, axis=0)
    poses[:, :3, :3] = Rotation.from_rotvec(rng.normal(0.0, 0.6, (7, 3))).as_matrix()
    poses[:, :3, 3] = rng.normal(0.0, 2.0, (7, 3))
    t = np.sort(rng.uniform(times[0], times[-1], 50))
    quat, trans = interpolate_poses(times, poses, t)
    assert quat.shape == (50, 4) and trans.shape == (50, 3) and quat.dtype == trans.dtype == np.float64
    want = np.stack([interpolate_pose(times, poses, float(x)) for x in t])
    assert np.abs(_matrices(quat, trans) - want).max() <= 1e-12
    quat, trans = interpolate_poses(times, poses, times)      # at the frame times, the frame poses
    assert np.abs(_matrices(quat, trans) - poses).max() <= 1e-12
    with pytest.raises(ValueError, match="outside the frame times"):
        interpolate_poses(times, poses, np.array([times[0], times[-1] + 1e-3]))


def _camera_at(t_s: float) -> np.ndarray:
    """The camera at pitch 30 on a body moving along x at 0.5 m/s."""
    return _translation((0.5 * t_s, 0.0, 1.2)) @ pitch_extrinsic(30.0)


def test_pass_cloud_puts_points_in_the_world(monkeypatch):
    frame_t = np.arange(1, 22, dtype=np.int64) * 100_000_000          # 0.1 s to 2.1 s
    frame_poses = np.stack([_camera_at(t / 1e9) for t in frame_t])
    last_ns = 2_050_000_000
    kept = np.round(np.linspace(100_000_000, last_ns, 54)).astype(np.int64)
    t_ns = np.concatenate([[20_000_000, 50_000_000, 99_995_000], kept, [2_070_000_000],
                           [2_100_005_000, 2_150_000_000]]).astype(np.int64)
    world = np.array([[1.0, 2.0, 1.5], [0.3, 1.5, 0.8], [-0.5, 3.0, 2.2]])[np.arange(len(t_ns)) % 3]
    T_cs = np.linalg.inv(pitch_extrinsic(30.0)) @ _translation((0.0, 0.0, 0.1))
    p_sensor = np.stack([(np.linalg.inv(_camera_at(t / 1e9) @ T_cs) @ np.append(x, 1.0))[:3]
                         for t, x in zip(t_ns, world)])
    rows = 7 * np.arange(len(t_ns), dtype=np.int64) + 2
    args = dict(first_ns=0, last_ns=last_ns, frame_t_ns=frame_t, frame_poses=frame_poses, T_cam_sensor=T_cs)
    cloud = pass_cloud(t_ns, p_sensor, rows, **args)
    inside = (t_ns >= frame_t[0]) & (t_ns <= last_ns)
    assert ((t_ns < frame_t[0]).sum(), (t_ns > frame_t[-1]).sum(),
            ((t_ns > last_ns) & (t_ns <= frame_t[-1])).sum()) == (3, 2, 1)
    assert len(cloud.t_ns) == len(t_ns) - 6 and np.array_equal(cloud.t_ns, t_ns[inside])
    assert cloud.index.dtype == np.int64 and np.array_equal(cloud.index, rows[inside])
    assert np.abs(cloud.X_w - world[inside]).max() <= 1e-9
    monkeypatch.setattr(lidardepth, "CLOUD_CHUNK", 7)
    small = pass_cloud(t_ns, p_sensor, rows, **args)
    assert small.X_w.tobytes() == cloud.X_w.tobytes() and small.index.tobytes() == cloud.index.tobytes()


def test_window_is_closed_in_integer_nanoseconds():
    c, half = round(7.9576 * 1e9), round(0.5 * 1e9)
    t_ns = np.array([c - half - 5000, c - half, c - 1, c, c + half, c + half + 5000], dtype=np.int64)
    cloud = PassCloud(t_ns=t_ns, X_w=np.zeros((6, 3)), index=3 * np.arange(6, dtype=np.int64))
    got = window(cloud, c, half)
    assert got.t_ns.tolist() == [c - half, c - 1, c, c + half] and got.index.tolist() == [3, 6, 9, 12]
    assert got.X_w.shape == (4, 3) and window(cloud, c, None) is cloud


def test_tilted_mount_uses_the_whole_transform():
    a = radians(-26.5)
    tilted = _translation((0.0, 0.0, 0.3))
    tilted[:3, :3] = [[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]]
    camera = pitch_extrinsic(30.0)                         # a body at the world origin
    frames = dict(first_ns=0, last_ns=1_000_000_000, frame_t_ns=np.array([0, 1_000_000_000]),
                  frame_poses=np.stack([camera, camera]))
    beam = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])     # the sensor origin and a point along its +z

    def elevation(T_body_sensor: np.ndarray) -> float:
        T_cs = np.linalg.inv(camera) @ T_body_sensor
        X = pass_cloud(np.array([0, 500_000_000]), beam, np.arange(2), T_cam_sensor=T_cs, **frames).X_w
        d = X[1] - X[0]
        return degrees(asin(d[2] / np.linalg.norm(d)))

    assert abs(elevation(tilted) - (90.0 - 26.5)) <= 1e-9
    assert abs(elevation(_translation((0.0, 0.0, 0.3))) - 90.0) <= 1e-9


def _random_scene() -> tuple[PassCloud, np.ndarray]:
    """2000 points on five depth layers in front of `SMALL` (some behind it or off the image) and five boxes."""
    rng = np.random.default_rng(7)
    n = 2000
    uv = np.column_stack([rng.uniform(-6.0, 70.0, n), rng.uniform(-5.0, 53.0, n)])
    z = rng.choice([1.00, 1.02, 1.30, 1.755, 1.80], n) + rng.normal(0.0, 0.004, n)
    z[rng.random(n) < 0.05] *= -1.0
    cloud = PassCloud(t_ns=np.arange(n, dtype=np.int64), X_w=SMALL.backproject(uv, z),
                      index=5 * np.arange(n, dtype=np.int64) + 1)
    x0, y0 = rng.uniform(-5.0, 50.0, 5), rng.uniform(-5.0, 36.0, 5)
    boxes = np.column_stack([x0, y0, x0 + rng.uniform(8.0, 24.0, 5), y0 + rng.uniform(8.0, 18.0, 5)])
    return cloud, boxes


def _sparse_image(cloud: PassCloud, camera: Camera) -> np.ndarray:
    """The whole-image depth of the points: the nearest point on each pixel, NaN on a pixel without one."""
    uv, z = camera.project(cloud.X_w)
    col, row = np.floor(uv[:, 0] + 0.5), np.floor(uv[:, 1] + 0.5)
    on = (z > 0) & (col >= 0) & (col < camera.width) & (row >= 0) & (row < camera.height)
    img = np.full((camera.height, camera.width), np.inf)
    np.minimum.at(img, (row[on].astype(int), col[on].astype(int)), z[on])
    img[np.isinf(img)] = np.nan
    return img


def _ellipse_depths(box: np.ndarray, img: np.ndarray) -> np.ndarray:
    """The finite depths of the pixels whose centre is inside the ellipse inscribed in `box`."""
    x0, y0, x1, y1 = box
    rows, cols = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    au, av = max((x1 - x0) / 2, 0.5), max((y1 - y0) / 2, 0.5)
    inside = ((cols + 0.5 - (x0 + x1) / 2) / au) ** 2 + ((rows + 0.5 - (y0 + y1) / 2) / av) ** 2 <= 1.0
    v = img[inside]
    return v[np.isfinite(v)]


def test_far_match_is_box_far_on_the_sparse_image():
    cloud, boxes = _random_scene()
    s = _settings(match="far")
    img = _sparse_image(cloud, SMALL)
    want = box_far_depth(boxes, img, 1.0, s.far_gap_m, s.far_min_frac)
    got = far_depth(boxes, SMALL, cloud, s)
    assert got.z.tobytes() == want.tobytes() and np.isfinite(want).sum() >= 4
    z_of = dict(zip(cloud.index.tolist(), SMALL.project(cloud.X_w)[1]))
    for i, box in enumerate(boxes):
        v = _ellipse_depths(box, img)
        assert got.n_points[i] == np.count_nonzero(np.abs(v - want[i]) <= s.far_gap_m)
        members = got.members[i]
        assert members.dtype == np.int64 and len(members) == got.n_points[i] and np.all(np.diff(members) > 0)
        assert all(abs(z_of[m] - want[i]) <= s.far_gap_m for m in members.tolist())


def _fruit_scene() -> tuple[PassCloud, np.ndarray, np.ndarray]:
    """Points on the box (80, 80)-(120, 120) of `WIDE`: a leaf on its left 30 %, the front of a fruit on the pixels
    92 to 105 of both axes, a wall elsewhere; one point at each pixel centre and a second one 0.3 px off on each
    fruit pixel. Returns the cloud, the box and the fruit's indices."""
    rows, cols = np.mgrid[80:120, 80:120]
    uv = np.column_stack([cols.ravel(), rows.ravel()]).astype(float)
    leaf = uv[:, 0] < 92
    fruit = ~leaf & np.all((uv >= 92) & (uv < 106), axis=1)
    uv = np.concatenate([uv, uv[fruit] + 0.3])
    extra = int(fruit.sum())
    leaf, fruit = np.concatenate([leaf, np.zeros(extra, bool)]), np.concatenate([fruit, np.ones(extra, bool)])
    z = np.where(leaf, LEAF_Z, np.where(fruit, FRUIT_Z, WALL_Z))
    index = 3 * np.arange(len(z), dtype=np.int64) + 11
    cloud = PassCloud(t_ns=np.zeros(len(z), dtype=np.int64), X_w=WIDE.backproject(uv, z), index=index)
    return cloud, np.array([[80.0, 80.0, 120.0, 120.0]]), np.sort(index[fruit])


def test_gate_and_map_pick_the_fruit_in_front_of_a_leaf():
    cloud, box, fruit = _fruit_scene()
    z_cam = np.array([1.88])
    sigma_cam, _ = belief_sigmas(z_cam, np.array([1]), _settings())
    assert abs(sigma_cam[0] - 0.0376) <= 1e-15 and abs(3.0 * sigma_cam[0] - 0.1128) <= 1e-15
    gate = gate_depth(box, z_cam, WIDE, cloud, _settings(match="gate"))
    assert 1.874 <= gate.z[0] <= 1.876 and gate.n_points[0] == len(fruit) == 2 * 14 * 14
    assert np.array_equal(gate.members[0], fruit)
    mapped = map_depth(box, z_cam, WIDE, cloud, _settings(match="map"))
    fruit_voxels = len(np.unique(np.floor(cloud.X_w[np.isin(cloud.index, fruit)] / 0.01), axis=0))
    assert 1.874 <= mapped.z[0] <= 1.876 and 3 <= mapped.n_points[0] == fruit_voxels < len(fruit)
    assert np.array_equal(mapped.members[0], fruit)
    assert np.isnan(map_depth(box, z_cam, WIDE, cloud, _settings(match="map", min_points=10**6)).z[0])
    nearest = map_depth(box, np.array([np.nan]), WIDE, cloud, _settings(match="map"))
    assert abs(nearest.z[0] - LEAF_Z) <= 1e-9
    assert box_lidar_depth(box, z_cam, WIDE, cloud, _settings(match="gate")).z.tobytes() == gate.z.tobytes()
    assert box_lidar_depth(box, z_cam, WIDE, cloud, _settings(match="map")).z.tobytes() == mapped.z.tobytes()


def test_gate_falls_back_to_far_without_camera_depth():
    cloud, boxes = _random_scene()
    s = _settings(match="gate")
    far = far_depth(boxes, SMALL, cloud, s)
    missing = np.flatnonzero(np.isfinite(far.z))[:2]
    z_cam = np.full(len(boxes), 1.35)       # the layer at 1.30 m is within three believed sigmas, not within one
    z_cam[missing] = np.nan
    gate = gate_depth(boxes, z_cam, SMALL, cloud, s)
    for i in missing:
        assert gate.z[i:i + 1].tobytes() == far.z[i:i + 1].tobytes() and gate.n_points[i] == far.n_points[i]
        assert gate.members[i].tobytes() == far.members[i].tobytes()
    uv, z = SMALL.project(cloud.X_w)                       # the gated boxes: the centre rectangle and the window
    col, row = np.floor(uv[:, 0] + 0.5), np.floor(uv[:, 1] + 0.5)
    for i in np.flatnonzero(np.isfinite(z_cam)):
        x0, y0, x1, y1 = boxes[i]
        c0, r0 = int(np.floor(x0 + 0.25 * (x1 - x0))), int(np.floor(y0 + 0.25 * (y1 - y0)))
        c1, r1 = max(c0 + 1, int(np.ceil(x1 - 0.25 * (x1 - x0)))), max(r0 + 1, int(np.ceil(y1 - 0.25 * (y1 - y0))))
        on = ((z > 0) & (col >= max(c0, 0)) & (col < min(c1, SMALL.width)) & (row >= max(r0, 0))
              & (row < min(r1, SMALL.height)) & (np.abs(z - z_cam[i]) <= 3.0 * 0.02 * z_cam[i]))
        assert gate.n_points[i] == on.sum() > 0 and np.array_equal(gate.members[i], cloud.index[on])
        assert gate.z[i] == np.median(z[on])
    assert box_lidar_depth(boxes, z_cam, SMALL, cloud, _settings(match="far")).z.tobytes() == far.z.tobytes()


def test_combine_replace_ivw_and_skip():
    z_cam = np.array([2.0, 2.0, np.nan, 2.0])
    lidar = BoxLidar(z=np.array([1.9, 1.9, 1.9, np.nan]), n_points=np.array([4, 2, 4, 0]),
                     members=tuple(np.zeros(0, dtype=np.int64) for _ in range(4)))
    z, used = combine_depth(z_cam, lidar, _settings(combine="replace", min_points=3))
    assert z.tolist() == [1.9, 2.0, 1.9, 2.0] and used.tolist() == [True, False, True, False]
    z, used = combine_depth(z_cam, lidar, _settings(combine="ivw", min_points=3))
    assert abs(z[0] - 1.9088967971530248) <= 1e-15 and z[1:].tolist() == [2.0, 1.9, 2.0]
    assert used.tolist() == [True, False, True, False]
    for combine in ("replace", "ivw"):
        z, used = combine_depth(z_cam, lidar, _settings(combine=combine, min_points=10**6))
        assert z.tobytes() == z_cam.tobytes() and not used.any()


def test_belief_sigmas_never_read_the_noise_section():
    assert (MAP_VOXEL_M, RANGE_BELIEF_FACTOR, CLOUD_CHUNK) == (0.01, 1.25, 1 << 20)
    sigma_cam, sigma_lidar = belief_sigmas(np.array([2.0]), np.array([4]), _settings())
    assert sigma_cam.tolist() == [0.04] and sigma_lidar.tolist() == [0.0125]
    one = _cfg(_lidar(), {"depth_rel_sigma": 0.02, "depth_dropout": 0.05, "lidar_range_sigma_m": 0.02,
                          "lidar_ang_sigma_deg": 0.15, "lidar_dropout": 0.05})
    other = _cfg(_lidar(), {"depth_rel_sigma": 0.08, "depth_dropout": 0.2, "lidar_range_sigma_m": 0.05,
                            "lidar_ang_sigma_deg": 0.3, "lidar_dropout": 0.2, "lidar_range_bias_m": 0.02})
    assert LidarSettings.from_lift(one.lift) == LidarSettings.from_lift(other.lift) == _settings()
    assert _settings(match="map", window_s="pass").window_s is None and _settings().window_s == 0.5
    source = inspect.getsource(lidardepth)
    for pattern in (r"\bnoise\.[a-z_]", r"\bNoiseConfig\b", r"_labels", r"\bmount\b", r"counting_contract",
                    r"\btrajectory\b"):
        assert not re.search(pattern, source), pattern


def _square_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors square to each unit row of `d` and to each other."""
    helper = np.where(np.abs(d[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    a = np.cross(d, helper)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a, np.cross(d, a)


def test_lidar_and_depth_form_draws():
    rng = np.random.default_rng(5)
    n = 200_000
    d64 = rng.normal(size=(n, 3))
    d64 /= np.linalg.norm(d64, axis=1, keepdims=True)
    dirs = d64.astype(np.float32)
    ranges = np.where(np.arange(n) < n // 2, 0.5, 2.0).astype(np.float32)
    zero = dict(range_sigma_m=0.0, range_sigma_near_m=0.0, near_range_m=1.0, ang_sigma_deg=0.0, dropout=0.0,
                range_bias_m=0.0)
    sizes = dict(zero, range_sigma_m=0.02, range_sigma_near_m=0.03, ang_sigma_deg=0.15, dropout=0.05)
    key = (0, 1, STEP_LIDAR, 5, 0, 3, 0)
    d1, r1, k1 = lidar_point_noise(key, dirs, ranges, **sizes)
    d2, r2, k2 = lidar_point_noise(key, dirs, ranges, **sizes)
    assert (d1.tobytes(), r1.tobytes(), k1.tobytes()) == (d2.tobytes(), r2.tobytes(), k2.tobytes())
    d3, r3, k3 = lidar_point_noise((0, 1, STEP_LIDAR, 5, 0, 3, 1), dirs, ranges, **sizes)
    assert d3.tobytes() != d1.tobytes() and r3.tobytes() != r1.tobytes() and k3.tobytes() != k1.tobytes()
    dz, rz, kz = lidar_point_noise(key, dirs, ranges, **zero)
    assert dz.tobytes() == dirs.tobytes() and rz.tobytes() == ranges.tobytes() and kz.all()

    err = r1.astype(np.float64) - ranges.astype(np.float64)
    assert abs(err[: n // 2].std() - 0.03) <= 0.0005 and abs(err[n // 2:].std() - 0.02) <= 0.0004
    turned = d1.astype(np.float64)
    along = np.sum(turned * d64, axis=1)
    for axis in _square_basis(d64):
        spread = np.arctan2(np.sum(turned * axis, axis=1), along).std()
        assert abs(spread / radians(0.15) - 1.0) <= 0.03
    assert abs((~k1).mean() - 0.05) <= 0.003
    g = noise_module.noise_rng(*key)                         # range g(N), then angle g(N, 2), then dropout u(N)
    range_draw = g.standard_normal(n)
    g.standard_normal((n, 2))
    u = g.random(n)
    assert np.array_equal(k1, u >= 0.05)
    assert np.abs(err - np.where(ranges < 1.0, 0.03, 0.02) * range_draw).max() <= 1e-6
    _, biased, _ = lidar_point_noise(key, dirs, ranges, **dict(zero, range_bias_m=0.01))
    shift = biased.astype(np.float64) - ranges.astype(np.float64)
    assert np.all(np.abs(shift - 0.01) <= np.spacing(biased).astype(np.float64))
    _, _, still = lidar_point_noise(key, dirs, ranges, **dict(zero, dropout=0.05))
    _, _, more = lidar_point_noise(key, dirs, ranges, **dict(sizes, dropout=0.2))
    assert np.array_equal(still, k1) and not np.any(more & ~k1)        # one u, whatever the sigmas

    cfg = _cfg(_lidar(), {"lidar_range_sigma_m": 0.02, "lidar_range_sigma_near_m": 0.03, "lidar_near_range_m": 1.0,
                          "lidar_ang_sigma_deg": 0.15, "lidar_dropout": 0.05})
    m = 5000
    points = LidarPoints(side="A", t_ns=5000 * np.arange(m, dtype=np.int64), dir=dirs[:m], range_m=ranges[:m],
                         row=np.zeros(m, dtype=np.uint32), first_ns=0, last_ns=5000 * m)
    assert LIDAR_UNIT_CODE == {"depth": 0, "nav": 1}
    got = noisy_points(points, cfg, 1, scene=5, side=(0, 3), unit="depth")
    d, r, keep = lidar_point_noise((cfg.seed, 1, STEP_LIDAR, 5, 0, 3, 0), points.dir, points.range_m, **sizes)
    rows = np.flatnonzero(keep)
    assert got.rows.dtype == np.int64 and np.array_equal(got.rows, rows) and len(rows) < m
    assert np.array_equal(got.t_ns, points.t_ns[rows])
    assert np.array_equal(got.p_sensor, d[rows].astype(np.float64) * r[rows].astype(np.float64)[:, None])
    plain = noisy_points(points, cfg, None, scene=5, side=(0, 3), unit="depth")
    assert np.array_equal(plain.rows, np.arange(m)) and np.array_equal(plain.t_ns, points.t_ns)
    assert np.array_equal(plain.p_sensor, dirs[:m].astype(np.float64) * ranges[:m].astype(np.float64)[:, None])

    pts = np.array([[1.0, 2.0, 0.5], [1.05, 2.0, 0.5], [4.0, 2.0, 0.5]])
    flat = depth_persist_field((0, 1, STEP_DEPTH_PERSIST, 5, 0, 3), 0.0, 0.5)(pts)
    assert flat.tolist() == [0.0, 0.0, 0.0]
    field = depth_persist_field((0, 1, STEP_DEPTH_PERSIST, 5, 0, 3), 0.01, 0.5)
    assert field(pts).tobytes() == depth_persist_field((0, 1, STEP_DEPTH_PERSIST, 5, 0, 3), 0.01, 0.5)(pts).tobytes()
    values = np.array([depth_persist_field((0, k, STEP_DEPTH_PERSIST, 5, 0, 3), 0.01, 0.5)(pts) for k in range(500)])
    assert abs(np.sqrt(np.mean(values[:, 0] ** 2)) / 0.01 - 1.0) <= 0.1
    assert np.corrcoef(values[:, 0], values[:, 1])[0, 1] > 0.9
    assert abs(np.corrcoef(values[:, 0], values[:, 2])[0, 1]) < 0.3
    scale_key = (0, 1, STEP_DEPTH_SCALE, 5, 30000, 0, 3)
    assert depth_scale(scale_key, 0.02) == depth_scale(scale_key, 0.02) != 0.0 and depth_scale(scale_key, 0.0) == 0.0


def test_extrinsic_draws_of_the_two_units():
    pitches = [10.0, 30.0, 50.0]
    imu = {"imu_extr_trans_mm": 10.0, "imu_extr_rot_deg": 0.3}
    rig = {"rig": True, "rig_ref": 10.0}
    fused = _cfg(noise=imu, pose={**rig, "fuse": "imu"})
    errors = imu_errors(fused, 2, scene=5, pitches=pitches)
    assert list(errors) == [10.0]
    assert imu_extr_error(fused, 2, scene=5, pitch=10.0).tobytes() == errors[10.0].tobytes()
    nav = _cfg(_lidar(file="nav"), {**imu, "rig_extr_trans_mm": 2.0, "rig_extr_rot_deg": 0.1}, pose=rig)
    assert nav.pose.fuse == "none" and imu_errors(nav, 2, scene=5, pitches=pitches) == {}
    assert imu_extr_error(nav, 2, scene=5, pitch=10.0).tobytes() == errors[10.0].tobytes()
    assert imu_extr_error(nav, None, scene=5, pitch=10.0) is None
    assert imu_extr_error(_cfg(), 2, scene=5, pitch=10.0) is None

    own = {"lidar_extr_trans_mm": 10.0, "lidar_extr_rot_deg": 0.3}
    plain, rigged = _cfg(_lidar(), own, pose={}), _cfg(_lidar(), own, pose=rig)
    dX = lidar_extr_error(plain, 2, scene=5)
    assert dX.tobytes() == lidar_extr_error(rigged, 2, scene=5).tobytes()
    assert dX.tobytes() == seeded_rigid((plain.seed, 2, 19, 5, 0), 10.0 / 1000, 0.3).tobytes()
    assert not np.allclose(dX, lidar_extr_error(plain, 2, scene=6))
    assert not np.allclose(dX, imu_extr_error(fused, 2, scene=5, pitch=0.0))
    assert lidar_extr_error(plain, None, scene=5) is None and lidar_extr_error(_cfg(_lidar()), 2, scene=5) is None

    def fused_depth(trans_mm: float):
        return _cfg(_lidar(), {**imu, "lidar_extr_trans_mm": trans_mm, "lidar_extr_rot_deg": 0.3},
                    pose={**rig, "fuse": "imu"})

    for trans_mm in (10.0, 60.0):
        moved = imu_errors(fused_depth(trans_mm), 2, scene=5, pitches=pitches)
        assert list(moved) == [10.0] and moved[10.0].tobytes() == errors[10.0].tobytes()

    T_depth, T_nav = _translation((0.0, 0.0, 0.1)), _translation((0.0, 0.0, 0.5))
    still = _cfg(_lidar(), {"lidar_extr_trans_mm": 0.0, "lidar_extr_rot_deg": 0.0})
    for q in pitches:
        known = np.linalg.inv(pitch_extrinsic(q)) @ T_depth
        got = camera_from_sensor(plain, 2, unit="depth", scene=5, pitch=q, T_body_sensor=T_depth)
        assert np.abs(got - known @ dX).max() <= 1e-12
        assert got.tobytes() == camera_from_sensor(rigged, 2, unit="depth", scene=5, pitch=q,
                                                   T_body_sensor=T_depth).tobytes()
        for cfg, seed in ((plain, None), (still, 2)):
            no_error = camera_from_sensor(cfg, seed, unit="depth", scene=5, pitch=q, T_body_sensor=T_depth)
            assert np.abs(no_error - known).max() <= 1e-12
        dX_q = rig_errors(nav, 2, scene=5, pitches=[q]).get(q)
        assert (dX_q is None) == (q == 10.0)
        want = np.linalg.inv(believed_extrinsic(q, 10.0, errors[10.0], dX_q)) @ T_nav
        got = camera_from_sensor(nav, 2, unit="nav", scene=5, pitch=q, T_body_sensor=T_nav)
        assert np.abs(got - want).max() <= 1e-12


def test_draw_steps_of_the_lidar():
    names = ("STEP_LIDAR", "STEP_LIDAR_EXTR", "STEP_DEPTH_PERSIST", "STEP_DEPTH_SCALE")
    steps = (STEP_LIDAR, STEP_LIDAR_EXTR, STEP_DEPTH_PERSIST, STEP_DEPTH_SCALE)
    others = [getattr(noise_module, n) for n in dir(noise_module) if n.startswith("STEP_") and n not in names]
    assert steps == (18, 19, 20, 21) and max(others) == 17
    assert len(set(others)) == len(others) and not set(others) & set(steps)
