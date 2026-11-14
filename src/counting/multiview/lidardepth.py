"""LiDAR box depth on the prediction side: the points of a pass in the world, the points of a frame's time window,
the depth of each detection box from them, and that depth joined with the camera box depth.

Inputs. A pass's points come as the sensor-frame points of one side (`t_ns`, `p_sensor`) with their file row numbers,
the span `[first_ns, last_ns]` of the side's beams, the integer frame times (ns, on the side's clock) and the
camera-to-world poses of the pass's frames from the pose source of the rung, and the believed camera <- sensor
transform. The knobs come from `lift.lidar` (`LidarSettings`). Nothing here reads the noise section, the labels, the
true sensor placement or the per-millisecond body poses of a bundle.

Points of a pass (`pass_cloud`). A point is kept only within both the span of the pass's frame times and
`[first_ns, last_ns]`; nothing is held at the ends. A kept point at time tau moves to the world as `X_w = R(tau) (T p) + t(tau)`,
with T the camera <- sensor transform and (R, t) the camera pose interpolated at tau (`rig.interpolate_poses`:
position linear in time, rotation slerp), in chunks of `CLOUD_CHUNK` points. `PassCloud.index` keeps each point's row
number in the side's file, so a point dropped before (a dropout) leaves the rows of the others as they were.

Window (`window`). A frame at time t (ns) with the half width h (ns) takes the points with `|t_ns - t| <= h`; with no
half width, every point of the pass.

Points on the image. A world point projects with the frame camera (`Camera.project`); it falls on the pixel whose
centre is nearest (column floor(u + 0.5), row floor(v + 0.5)) when it is in front of the camera (z > 0) and that
pixel is inside the image. z is the camera-plane depth.

Box matches (`box_lidar_depth` picks one by `match`):

- `far_depth`: the nearest point of each pixel of the box makes a sparse depth image, and the far-cluster box depth
  of `lift.box_far_depth` runs on it with the ellipse at scale 1 and `far_gap_m`, `far_min_frac`. Only the pixels of
  the box are filled, and the result is the bytes `box_far_depth` gives on the whole sparse image. The point count is
  the number of in-ellipse pixels with a depth within `far_gap_m` of the result.
- `gate_depth`: the points on the centre rectangle of the box (side ratio `box_frac`, the rectangle of
  `lift.box_median_depth`) within `gate_k` believed camera sigmas of the camera box depth; the depth is their median
  and the count their number. A box without a camera box depth takes `far_depth`.
- `map_depth`: the window's points averaged into `MAP_VOXEL_M` voxels of the world (again for every window; the map
  of a whole pass once). The voxels on the centre rectangle, sorted by camera depth, split into clusters where
  consecutive depths differ by more than `far_gap_m`. Of the clusters with at least `min_points` voxels, the nearest
  one whose median is within `gate_k` believed camera sigmas of the camera box depth is picked, or the nearest one
  when the box has no camera box depth. The depth is the cluster's median and the count its voxels.

Every match gives each box the file row numbers of the points it used (`BoxLidar.members`): the pixel points within
`far_gap_m` for `far`, the gated points for `gate` and every point of the picked voxels for `map`. A box with no
depth gets NaN and a count of 0.

Joining (`combine_depth`). A box uses its LiDAR depth when it has one and at least `min_points` points; the others
keep the camera box depth (NaN stays NaN). `replace` takes the LiDAR depth. `ivw` weighs the two depths by their
inverse believed variances, and takes the LiDAR depth alone where the camera has none. The believed sigmas
(`belief_sigmas`) are constants of the config: `belief_cam_rel` times the camera box depth, and `belief_range_m`
times `RANGE_BELIEF_FACTOR` over the square root of the point count.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from counting.config.eval_schema import LiftConfig
from counting.geometry.camera import Camera
from counting.geometry.rig import interpolate_poses

MAP_VOXEL_M = 0.01              # voxel side of the point map (m)
RANGE_BELIEF_FACTOR = 1.25      # believed LiDAR sigma of one point, in units of `belief_range_m`
CLOUD_CHUNK = 1 << 20           # points moved to the world per step
_FAR_SCALE = 1.0                # ellipse scale of the far match
_NO_MEMBERS = np.zeros(0, dtype=np.int64)


@dataclass(frozen=True)
class LidarSettings:
    """The LiDAR box depth knobs of `lift.lidar`; `window_s` None is the whole pass."""

    match: str
    window_s: float | None
    combine: str
    min_points: int
    gate_k: float
    box_frac: float
    far_min_frac: float
    far_gap_m: float
    belief_cam_rel: float
    belief_range_m: float

    @classmethod
    def from_lift(cls, lift: LiftConfig) -> LidarSettings:
        """The settings of `lift.lidar`; `ValueError` without that section."""
        knobs = lift.lidar
        if knobs is None:
            raise ValueError("LiDAR box depth needs a lift.lidar section")
        return cls(match=knobs.match, window_s=None if knobs.window_s == "pass" else float(knobs.window_s),
                   combine=knobs.combine, min_points=int(knobs.min_points), gate_k=float(knobs.gate_k),
                   box_frac=float(knobs.box_frac), far_min_frac=float(knobs.far_min_frac),
                   far_gap_m=float(knobs.far_gap_m), belief_cam_rel=float(knobs.belief_cam_rel),
                   belief_range_m=float(knobs.belief_range_m))


@dataclass(frozen=True, eq=False)
class _Voxels:
    """A point map: voxel centroids (V, 3), and the points of voxel v at `order[starts[v]:starts[v] + counts[v]]`."""

    centroids: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    order: np.ndarray


@dataclass(frozen=True, eq=False)
class PassCloud:
    """Points of a pass in the world: time `t_ns` (M,) int64 ascending, position `X_w` (M, 3) float64 and the row
    number `index` (M,) int64 of each point in its side's file."""

    t_ns: np.ndarray
    X_w: np.ndarray
    index: np.ndarray

    @cached_property
    def _voxels(self) -> _Voxels:
        """The `MAP_VOXEL_M` voxel map of these points, made on first use."""
        return _voxel_map(self.X_w)


@dataclass(frozen=True, eq=False)
class BoxLidar:
    """The LiDAR depth of each box: `z` (n,) float64 (NaN without one), the point count `n_points` (n,) and, per box,
    the file row numbers of the points used (`members`, int64 ascending)."""

    z: np.ndarray
    n_points: np.ndarray
    members: tuple[np.ndarray, ...]


def _apply(T: np.ndarray, p: np.ndarray) -> list[np.ndarray]:
    """`T[:3, :3] @ p + T[:3, 3]` for each row of `p` (n, 3), as three component arrays, element by element."""
    return [T[k, 0] * p[:, 0] + T[k, 1] * p[:, 1] + T[k, 2] * p[:, 2] + T[k, 3] for k in range(3)]


def _rotate(quat: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[np.ndarray]:
    """The vectors (x, y, z) turned by the unit quaternions `quat` (n, 4, scipy order), element by element."""
    qx, qy, qz, qw = quat.T
    return [(1.0 - 2.0 * (qy * qy + qz * qz)) * x + 2.0 * (qx * qy - qz * qw) * y + 2.0 * (qx * qz + qy * qw) * z,
            2.0 * (qx * qy + qz * qw) * x + (1.0 - 2.0 * (qx * qx + qz * qz)) * y + 2.0 * (qy * qz - qx * qw) * z,
            2.0 * (qx * qz - qy * qw) * x + 2.0 * (qy * qz + qx * qw) * y + (1.0 - 2.0 * (qx * qx + qy * qy)) * z]


def pass_cloud(t_ns: np.ndarray, p_sensor: np.ndarray, rows: np.ndarray, *, first_ns: int, last_ns: int,
               frame_t_ns: np.ndarray, frame_poses: np.ndarray, T_cam_sensor: np.ndarray) -> PassCloud:
    """The points of one pass in the world: the sensor-frame points `p_sensor` (N, 3) at the ascending times `t_ns`
    (N, ns) with their file rows `rows` (N,), the span `[first_ns, last_ns]` of the side's beams, the pass's frame
    times `frame_t_ns` (F, ns, ascending) and camera poses `frame_poses` (F, 4, 4), and the camera <- sensor
    transform `T_cam_sensor` (4, 4).

    The points within the frame times and the span, in their order, as `X_w = R(tau) (T_cam_sensor p) + t(tau)`.
    """
    t_ns = np.asarray(t_ns, dtype=np.int64)
    p = np.asarray(p_sensor, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.int64)
    frame_t = np.asarray(frame_t_ns, dtype=np.int64)
    if p.shape != (len(t_ns), 3) or rows.shape != t_ns.shape:
        raise ValueError(f"pass_cloud needs points (N, 3) and rows (N,) for N times; got {p.shape} and "
                         f"{rows.shape} for {len(t_ns)}")
    if len(frame_t) == 0:
        raise ValueError("pass_cloud needs at least one frame")
    if np.any(np.diff(t_ns) < 0):
        raise ValueError("pass_cloud needs point times that never decrease")
    lo, hi = max(int(frame_t[0]), int(first_ns)), min(int(frame_t[-1]), int(last_ns))
    i0 = int(np.searchsorted(t_ns, lo, side="left"))
    i1 = max(i0, int(np.searchsorted(t_ns, hi, side="right")))
    t_kept, p_kept = t_ns[i0:i1], p[i0:i1]
    T = np.asarray(T_cam_sensor, dtype=np.float64)
    times = frame_t.astype(np.float64)
    X = np.empty((len(t_kept), 3))
    for s in range(0, len(t_kept), CLOUD_CHUNK):
        e = s + CLOUD_CHUNK
        quat, trans = interpolate_poses(times, frame_poses, t_kept[s:e].astype(np.float64))
        turned = _rotate(quat, *_apply(T, p_kept[s:e]))
        for k in range(3):
            X[s:e, k] = turned[k] + trans[:, k]
    return PassCloud(t_ns=t_kept.copy(), X_w=X, index=rows[i0:i1].copy())


def window(cloud: PassCloud, t_ns: int, half_ns: int | None) -> PassCloud:
    """The points of `cloud` within `half_ns` of the frame time `t_ns`, both in integer ns (a closed interval);
    `half_ns` None gives `cloud` itself."""
    if half_ns is None:
        return cloud
    i0 = int(np.searchsorted(cloud.t_ns, int(t_ns) - int(half_ns), side="left"))
    i1 = int(np.searchsorted(cloud.t_ns, int(t_ns) + int(half_ns), side="right"))
    return PassCloud(t_ns=cloud.t_ns[i0:i1], X_w=cloud.X_w[i0:i1], index=cloud.index[i0:i1])


@dataclass(frozen=True, eq=False)
class _OnImage:
    """The points on the image, by column: camera depth `z`, pixel `col` and `row`, and position `pos` in the
    projected array."""

    z: np.ndarray
    col: np.ndarray
    row: np.ndarray
    pos: np.ndarray

    def within(self, c0: int, c1: int, r0: int, r1: int) -> np.ndarray:
        """Positions (into these arrays) of the points on columns [c0, c1) and rows [r0, r1)."""
        i0 = int(np.searchsorted(self.col, c0, side="left"))
        i1 = int(np.searchsorted(self.col, c1, side="left"))
        rows = self.row[i0:i1]
        return i0 + np.flatnonzero((rows >= r0) & (rows < r1))


def _on_image(camera: Camera, X: np.ndarray) -> _OnImage:
    with np.errstate(divide="ignore", invalid="ignore"):
        uv, z = camera.project(X)
        col, row = np.floor(uv[:, 0] + 0.5), np.floor(uv[:, 1] + 0.5)
        on = (z > 0) & (col >= 0) & (col < camera.width) & (row >= 0) & (row < camera.height)
    pos = np.flatnonzero(on)
    cols = col[pos].astype(np.int64)
    by_col = np.argsort(cols, kind="stable")
    pos = pos[by_col]
    return _OnImage(z=z[pos], col=cols[by_col], row=row[pos].astype(np.int64), pos=pos)


def _box_pixels(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    """The pixel columns [c0, c1) and rows [r0, r1) of `box` in the image, as `lift.box_far_depth` takes them."""
    x0, y0, x1, y1 = box
    c0 = int(np.floor(x0))
    c1 = max(c0 + 1, int(np.ceil(x1)))
    r0 = int(np.floor(y0))
    r1 = max(r0 + 1, int(np.ceil(y1)))
    c0, c1 = max(c0, 0), min(c1, width)
    r0, r1 = max(r0, 0), min(r1, height)
    return None if c1 <= c0 or r1 <= r0 else (c0, c1, r0, r1)


def _centre_rect(box: np.ndarray, frac: float, width: int, height: int) -> tuple[int, int, int, int] | None:
    """The pixel columns [c0, c1) and rows [r0, r1) of the centre `frac` rectangle of `box`, as
    `lift.box_median_depth` takes them."""
    x0, y0, x1, y1 = box
    a = (1.0 - frac) / 2.0
    w, h = x1 - x0, y1 - y0
    c0 = int(np.floor(x0 + a * w))
    c1 = max(c0 + 1, int(np.ceil(x1 - a * w)))
    r0 = int(np.floor(y0 + a * h))
    r1 = max(r0 + 1, int(np.ceil(y1 - a * h)))
    c0, c1 = max(c0, 0), min(c1, width)
    r0, r1 = max(r0, 0), min(r1, height)
    return None if c1 <= c0 or r1 <= r0 else (c0, c1, r0, r1)


def _members(index: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.sort(index[positions]).astype(np.int64, copy=False)


def _far_box(box: np.ndarray, img: _OnImage, index: np.ndarray, width: int, height: int,
             s: LidarSettings) -> tuple[float, int, np.ndarray]:
    """The far match of one box: depth, point count and members.

    The arithmetic is that of `lift.box_far_depth` (`far_scale` 1) on the box's own pixels, in image coordinates."""
    pixels = _box_pixels(box, width, height)
    if pixels is None:
        return np.nan, 0, _NO_MEMBERS
    c0, c1, r0, r1 = pixels
    sel = img.within(c0, c1, r0, r1)
    pix = (img.row[sel] - r0) * (c1 - c0) + (img.col[sel] - c0)
    by_pixel = np.lexsort((img.pos[sel], img.z[sel], pix))   # per pixel, the nearest point first
    sel, pix = sel[by_pixel], pix[by_pixel]
    first = np.ones(len(pix), dtype=bool)
    first[1:] = pix[1:] != pix[:-1]
    sel, pix = sel[first], pix[first]
    depth = np.full((r1 - r0) * (c1 - c0), np.nan)
    depth[pix] = img.z[sel]
    who = np.full((r1 - r0) * (c1 - c0), -1, dtype=np.int64)
    who[pix] = img.pos[sel]
    depth, who = depth.reshape(r1 - r0, c1 - c0), who.reshape(r1 - r0, c1 - c0)
    x0, y0, x1, y1 = box
    cu, cv = (x0 + x1) / 2, (y0 + y1) / 2
    au = max(_FAR_SCALE * (x1 - x0) / 2, 0.5)
    av = max(_FAR_SCALE * (y1 - y0) / 2, 0.5)
    rows, cols = np.mgrid[r0:r1, c0:c1]
    inside = ((cols + 0.5 - cu) / au) ** 2 + ((rows + 0.5 - cv) / av) ** 2 <= 1.0
    values, points = depth[inside], who[inside]
    v = np.sort(values[np.isfinite(values)])
    if len(v) == 0:
        return np.nan, 0, _NO_MEMBERS
    clusters = np.split(v, np.flatnonzero(np.diff(v) > s.far_gap_m) + 1)
    sizes = np.array([len(c) for c in clusters])
    enough = np.flatnonzero(sizes >= s.far_min_frac * len(v))
    k = int(enough[-1]) if len(enough) else int(np.argmax(sizes))
    z = np.median(clusters[k])
    near = np.isfinite(values) & (np.abs(values - z) <= s.far_gap_m)
    return z, int(near.sum()), _members(index, points[near])


def _result(rows: list[tuple[float, int, np.ndarray]]) -> BoxLidar:
    return BoxLidar(z=np.array([r[0] for r in rows], dtype=np.float64),
                    n_points=np.array([r[1] for r in rows], dtype=np.int64),
                    members=tuple(r[2] for r in rows))


def _boxes(boxes: np.ndarray) -> np.ndarray:
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def far_depth(boxes: np.ndarray, camera: Camera, cloud: PassCloud, s: LidarSettings) -> BoxLidar:
    """The far match of each box (n, 4): `lift.box_far_depth` on the sparse depth image of the points."""
    boxes = _boxes(boxes)
    img = _on_image(camera, cloud.X_w)
    return _result([_far_box(box, img, cloud.index, camera.width, camera.height, s) for box in boxes])


def _cam_sigma(z_cam: np.ndarray, s: LidarSettings) -> np.ndarray:
    return s.belief_cam_rel * np.asarray(z_cam, dtype=np.float64)


def gate_depth(boxes: np.ndarray, z_cam: np.ndarray, camera: Camera, cloud: PassCloud, s: LidarSettings) -> BoxLidar:
    """The gate match of each box (n, 4) around its camera box depth `z_cam` (n,); the far match where `z_cam` is
    not finite."""
    boxes = _boxes(boxes)
    z_cam = np.asarray(z_cam, dtype=np.float64)
    img = _on_image(camera, cloud.X_w)
    half = s.gate_k * _cam_sigma(z_cam, s)
    out = []
    for i, box in enumerate(boxes):
        if not np.isfinite(z_cam[i]):
            out.append(_far_box(box, img, cloud.index, camera.width, camera.height, s))
            continue
        rect = _centre_rect(box, s.box_frac, camera.width, camera.height)
        sel = _NO_MEMBERS if rect is None else img.within(*rect)
        sel = sel[np.abs(img.z[sel] - z_cam[i]) <= half[i]]
        z = np.median(img.z[sel]) if len(sel) else np.nan
        out.append((z, len(sel), _members(cloud.index, img.pos[sel])))
    return _result(out)


def _voxel_map(X: np.ndarray) -> _Voxels:
    """The `MAP_VOXEL_M` voxels of the world points `X` (M, 3), ordered by voxel index, with their centroids."""
    keys = np.floor(X / MAP_VOXEL_M).astype(np.int64)
    if len(keys) == 0:
        return _Voxels(centroids=np.zeros((0, 3)), starts=_NO_MEMBERS, counts=_NO_MEMBERS, order=_NO_MEMBERS)
    low = keys.min(axis=0)
    span = keys.max(axis=0) - low + 1
    if float(span[0]) * float(span[1]) * float(span[2]) < 2.0**62:   # one int64 code, in the order of the keys
        code = ((keys[:, 0] - low[0]) * span[1] + (keys[:, 1] - low[1])) * span[2] + (keys[:, 2] - low[2])
        _, inverse = np.unique(code, return_inverse=True)
    else:
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse)
    centroids = np.column_stack([np.bincount(inverse, weights=X[:, k]) for k in range(3)]) / counts[:, None]
    order = np.argsort(inverse, kind="stable")
    return _Voxels(centroids=centroids, starts=np.cumsum(counts) - counts, counts=counts, order=order)


def map_depth(boxes: np.ndarray, z_cam: np.ndarray, camera: Camera, cloud: PassCloud, s: LidarSettings) -> BoxLidar:
    """The map match of each box (n, 4) around its camera box depth `z_cam` (n,), on the voxel map of `cloud`."""
    boxes = _boxes(boxes)
    z_cam = np.asarray(z_cam, dtype=np.float64)
    voxels = cloud._voxels
    img = _on_image(camera, voxels.centroids)
    half = s.gate_k * _cam_sigma(z_cam, s)
    out = []
    for i, box in enumerate(boxes):
        rect = _centre_rect(box, s.box_frac, camera.width, camera.height)
        sel = _NO_MEMBERS if rect is None else img.within(*rect)
        sel = sel[np.argsort(img.z[sel], kind="stable")]
        z = img.z[sel]
        picked = None
        for cluster in np.split(np.arange(len(sel)), np.flatnonzero(np.diff(z) > s.far_gap_m) + 1):
            if len(cluster) < s.min_points:
                continue
            median = np.median(z[cluster])
            if not np.isfinite(z_cam[i]) or abs(median - z_cam[i]) <= half[i]:
                picked = (median, cluster)
                break
        if picked is None:
            out.append((np.nan, 0, _NO_MEMBERS))
            continue
        median, cluster = picked
        ids = img.pos[sel[cluster]]
        positions = np.concatenate([voxels.order[voxels.starts[v]:voxels.starts[v] + voxels.counts[v]] for v in ids])
        out.append((median, len(cluster), _members(cloud.index, positions)))
    return _result(out)


def box_lidar_depth(boxes: np.ndarray, z_cam: np.ndarray, camera: Camera, cloud: PassCloud,
                    s: LidarSettings) -> BoxLidar:
    """The LiDAR depth of each box by the match `s.match`: `far_depth`, `gate_depth` or `map_depth`."""
    if s.match == "far":
        return far_depth(boxes, camera, cloud, s)
    if s.match == "gate":
        return gate_depth(boxes, z_cam, camera, cloud, s)
    if s.match == "map":
        return map_depth(boxes, z_cam, camera, cloud, s)
    raise ValueError(f"unknown LiDAR match {s.match!r}; matches are far, gate and map")


def belief_sigmas(z_cam: np.ndarray, n_points: np.ndarray, s: LidarSettings) -> tuple[np.ndarray, np.ndarray]:
    """The believed sigmas (m) of the camera box depth, `belief_cam_rel * z_cam`, and of a LiDAR depth from
    `n_points` points, `belief_range_m * RANGE_BELIEF_FACTOR / sqrt(n_points)` (inf for no point)."""
    n = np.asarray(n_points, dtype=np.float64)
    with np.errstate(divide="ignore"):
        lidar = s.belief_range_m * RANGE_BELIEF_FACTOR / np.sqrt(n)
    return _cam_sigma(z_cam, s), lidar


def combine_depth(z_cam: np.ndarray, lidar: BoxLidar, s: LidarSettings) -> tuple[np.ndarray, np.ndarray]:
    """The joined depth of each box and whether it used the LiDAR depth: a finite LiDAR depth from at least
    `min_points` points. `replace` takes it; `ivw` weighs it with a finite camera box depth by the inverse believed
    variances and takes it alone without one. A box that does not use it keeps `z_cam`."""
    z_cam = np.asarray(z_cam, dtype=np.float64)
    z_lidar = np.asarray(lidar.z, dtype=np.float64)
    used = np.isfinite(z_lidar) & (np.asarray(lidar.n_points) >= s.min_points)
    z = np.where(used, z_lidar, z_cam)
    if s.combine == "replace":
        return z, used
    if s.combine != "ivw":
        raise ValueError(f"unknown LiDAR combine {s.combine!r}; combines are replace and ivw")
    both = used & np.isfinite(z_cam)
    sigma_cam, sigma_lidar = belief_sigmas(z_cam[both], np.asarray(lidar.n_points)[both], s)
    w_cam, w_lidar = 1.0 / sigma_cam**2, 1.0 / sigma_lidar**2
    z[both] = (z_cam[both] * w_cam + z_lidar[both] * w_lidar) / (w_cam + w_lidar)
    return z, used
