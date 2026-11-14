"""Scoring-side reading of the LiDAR files beside a bundle: the parts the prediction side must not see.

The prediction side (`counting.geometry.lidarfile`) reads the point files and the sidecar fields that tie them to
the bundle. This module reads the rest of the folder:

- `bundle_inputs`: the sha256 of the bundle's `manifest.json` and `trajectory.json` bytes, the inputs the sidecar
  records, handed to `LidarFile` so it refuses the LiDAR files of another bundle.
- `sidecar_mount`: the true sensor mount on the body, `mount.T_body_sensor` (body <- sensor, 4 x 4), which must be
  rigid; `sidecar_mount_rpy`: the same mount as the simulation config states it for a crop, its (roll, pitch,
  yaw) in degrees, checked to give that matrix.
- `sidecar_contract`: the noise values the simulation config records for counting (`config.values.
  counting_contract`); `contract_noise`: those values under the counting `noise.*` keys of the unit.
- `read_labels`: one side's per-point labels from `lidar_<side>_labels.npz` (`LidarLabels`), rows matching the
  point file's rows.

Every failed check raises `ValueError` naming the file path; a missing file raises `FileNotFoundError`.

Noise and extrinsics. The files hold noiseless ranges; this side adds the noise of the `noise.lidar_*` keys per
noise seed and makes the camera <- sensor transform the prediction side believes:

- `noisy_points`: one side's measured points in the sensor frame (`NoisyPoints`), the noise of `lidar_point_noise`
  (noise step 18, keyed by scene, side, aisle and unit) with the dropped points left out and the file row of each
  kept point, so labels still join by row.
- `lidar_extr_error`: the camera-LiDAR extrinsic error of the depth unit (noise step 19), one per scene, whatever the
  pitch, the rig or the pose section.
- `camera_from_sensor`: the believed camera q <- sensor transform from the true mount `T_BS` (body <- sensor). For
  the depth unit `X_{B->q}^-1 T_BS dX_lidar`; the rig calibration error stays with the camera poses. The
  navigation unit holds the IMU, so it is `Xhat_{B->q}^-1 T_BS`, `Xhat_{B->q}` the extrinsic the fusion believes
  (`believed_extrinsic` with the camera-to-IMU error of `fusion.imu_extr_error` at the reference pitch and the rig
  error of `posesource.rig_errors`), fused or not.

Truth-label pairing (`lift.depth=lidar_oracle`). `oracle_box_depth` pairs each detection with the points the labels
give its truth fruit, through the file row of each point, in place of the prediction-side box match.

Results. A result whose boxes took a LiDAR depth gets `lidar_entry`; a result of a config with LiDAR files gets its
per-detection rows (`DepthRows`), which stay out of `results.json`: `pop_depth_rows` takes them out of a result by
cell, `depth_rows_file_names` names the file of each cell (`%XX` for the characters a file name keeps out),
`depth_rows_csv` writes a cell's rows and `depth_rows_index_csv` the index of cell names and file names.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from counting.config.eval_schema import LIDAR_EXTR_KEYS, EvalAppConfig
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.fusion import imu_extr_error
from counting.evaluation.posesource import _rigid_inverse, rig_errors
from counting.geometry.camera import Camera
from counting.geometry.lidarfile import (
    _FIELD_ERRORS, LIDAR_SIDES, LIDAR_UNITS, LidarFile, LidarPoints, _check_arrays, _check_text, _field_error,
    _npz_arrays, _read_only, _read_sidecar, _sha256,
)
from counting.geometry.noise import STEP_LIDAR, STEP_LIDAR_EXTR, lidar_point_noise, seeded_rigid
from counting.geometry.rig import believed_extrinsic, pitch_extrinsic
from counting.multiview.lidardepth import BoxLidar, LidarSettings, PassCloud

LIDAR_LABELS_FORMAT = "orchard-lidar-labels/1"
_LABEL_ARRAYS = {"fruit_id": (np.uint16, (-1,)), "part": (np.uint8, (-1,)), "mixed": (np.uint8, (-1,))}
_BUNDLE_INPUTS = ("manifest.json", "trajectory.json")
_RIGID_TOL = 1e-9
_RPY_TOL = 1e-9
# counting noise key of each contract value; the extrinsic keys the contract must name, per unit.
_CONTRACT_NOISE = {"range_sigma_m": "lidar_range_sigma_m", "range_sigma_near_m": "lidar_range_sigma_near_m",
                   "near_range_m": "lidar_near_range_m", "ang_sigma_deg": "lidar_ang_sigma_deg",
                   "dropout": "lidar_dropout", "range_bias_m": "lidar_range_bias_m"}
_CONTRACT_EXTRINSIC = {"depth": ("lidar_extr_trans_mm", "lidar_extr_rot_deg"),
                       "nav": ("imu_extr_trans_mm", "imu_extr_rot_deg")}
# Unit part of the LiDAR noise keys.
LIDAR_UNIT_CODE = {"depth": 0, "nav": 1}
# The label parts a truth-label pairing takes (the semantic classes of a fruit and of a bag), and the `mixed` value it
# leaves out (sub-ray ranges joined over more than 0.10 m).
ORACLE_PARTS = (3, 4)
ORACLE_MIXED_OUT = 2


@dataclass(frozen=True, eq=False)
class LidarLabels:
    """One side's labels, row for row with its points (read-only): `fruit_id` uint16 (the fruit id, of the fruit
    inside a bag for a bag point; 0 for anything else), `part` uint8 (the semantic class number, 1 to 8) and
    `mixed` uint8 (how the sub-ray ranges were joined: 0 the mean of the main cluster, 1 the mean of every sub-ray
    hit, spread at most 0.10 m, 2 that mean over a wider spread)."""

    side: str
    fruit_id: np.ndarray
    part: np.ndarray
    mixed: np.ndarray


def bundle_inputs(run: SynthRun) -> dict[str, str]:
    """The sha256 of the bytes of the bundle's `manifest.json` and `trajectory.json`, by file name."""
    out = {}
    for name in _BUNDLE_INPUTS:
        path = run.root / name
        if not path.is_file():
            raise ValueError(f"{path}: missing; LiDAR files are made from a bundle rendered with body shake")
        out[name] = _sha256(path.read_bytes())
    return out


def _mount(path: Path, sidecar: Mapping) -> np.ndarray:
    try:
        raw = sidecar["mount"]["T_body_sensor"]
    except _FIELD_ERRORS as e:
        raise _field_error(path, e) from None
    try:
        T = np.array(raw, dtype=np.float64)
    except (TypeError, ValueError):
        T = np.zeros(0)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError(f"{path}: mount T_body_sensor is not a finite 4 x 4 matrix")
    R = T[:3, :3]
    if (np.abs(R.T @ R - np.eye(3)).max() > _RIGID_TOL or abs(np.linalg.det(R) - 1.0) > _RIGID_TOL
            or not np.array_equal(T[3], [0.0, 0.0, 0.0, 1.0])):
        raise ValueError(f"{path}: mount T_body_sensor is not rigid")
    return T


def sidecar_mount(folder: Path) -> np.ndarray:
    """The true mount of the unit, body <- sensor (4, 4) float64, from the sidecar `mount.T_body_sensor`.

    The rotation must be orthonormal with determinant +1 (to 1e-9) and the last row exactly [0, 0, 0, 1];
    otherwise `ValueError` ("T_body_sensor is not rigid")."""
    path, _, sidecar = _read_sidecar(folder)
    return _mount(path, sidecar)


def sidecar_mount_rpy(folder: Path, crop: str) -> tuple[float, float, float]:
    """The mount direction (roll, pitch, yaw) in degrees that the simulation config states for `crop`.

    The config's `xyz_m` and `rpy_deg` of that crop, as rotation Rz(yaw) Ry(pitch) Rx(roll) and translation
    `xyz_m`, must give `sidecar_mount` to 1e-9; otherwise `ValueError` ("mount rpy_deg does not match
    T_body_sensor")."""
    path, _, sidecar = _read_sidecar(folder)
    T = _mount(path, sidecar)
    try:
        mounts = sidecar["config"]["values"]["mount"]
        if crop not in mounts:
            raise ValueError(f"{path}: config values hold no mount for crop {crop}")
        xyz = np.array(mounts[crop]["xyz_m"], dtype=np.float64)
        rpy = np.array(mounts[crop]["rpy_deg"], dtype=np.float64)
    except _FIELD_ERRORS as e:
        raise _field_error(path, e) from None
    if xyz.shape != (3,) or rpy.shape != (3,):
        raise ValueError(f"{path}: mount {crop} xyz_m and rpy_deg must hold three values each")
    stated = np.eye(4)
    stated[:3, :3] = Rotation.from_euler("ZYX", rpy[::-1], degrees=True).as_matrix()
    stated[:3, 3] = xyz
    if not np.abs(stated - T).max() <= _RPY_TOL:
        raise ValueError(f"{path}: mount rpy_deg does not match T_body_sensor")
    return tuple(float(v) for v in rpy)


def sidecar_contract(folder: Path) -> dict:
    """The noise values the simulation config records for counting: `config.values.counting_contract`."""
    path, _, sidecar = _read_sidecar(folder)
    try:
        contract = sidecar["config"]["values"]["counting_contract"]
    except _FIELD_ERRORS as e:
        raise _field_error(path, e) from None
    if not isinstance(contract, dict):
        raise ValueError(f"{path}: counting_contract is not a JSON object")
    return dict(contract)


def contract_noise(contract: Mapping, unit: str) -> dict[str, float]:
    """The contract values of the `unit` as counting `noise.*` keys (without the `noise.` prefix).

    The range, angle, dropout and bias values go to the `lidar_*` keys; the extrinsic error to the two keys the
    contract names, which must be `lidar_extr_*` for the depth unit and `imu_extr_*` for the navigation unit.
    counting has no LiDAR time offset, so a contract `time_offset_ms` other than 0 raises `ValueError`.
    """
    if unit not in LIDAR_UNITS:
        raise ValueError(f"unknown LiDAR unit {unit!r}; units are {', '.join(LIDAR_UNITS)}")
    try:
        offset = float(contract["time_offset_ms"])
        out = {key: float(contract[name]) for name, key in _CONTRACT_NOISE.items()}
        extrinsic = contract["extrinsic"]
        named = tuple(str(k) for k in extrinsic["keys"])
        trans, rot = float(extrinsic["trans_mm"]), float(extrinsic["rot_deg"])
    except (*_FIELD_ERRORS, ValueError) as e:
        raise ValueError(f"LiDAR contract of the {unit} unit: missing or malformed value ({e})") from None
    if offset != 0.0:
        raise ValueError(f"LiDAR contract time_offset_ms is {offset:g}; counting has no LiDAR time offset")
    keys = _CONTRACT_EXTRINSIC[unit]
    if named != tuple(f"noise.{key}" for key in keys):
        raise ValueError(f"LiDAR contract extrinsic keys {list(named)} are not the {unit} unit's "
                         f"{[f'noise.{key}' for key in keys]}")
    out[keys[0]], out[keys[1]] = trans, rot
    return out


def read_labels(folder: Path, side: str, *, n_points: int) -> LidarLabels:
    """The labels of `side` from `folder/lidar_<side>_labels.npz`, checked against the sidecar.

    In order: the file is there; its bytes match the sha256 the sidecar `files` lists; its array set is `format`,
    `side`, `fruit_id`, `part` and `mixed`, read without pickles; format `LIDAR_LABELS_FORMAT` and the side as 0-d
    `<U` strings; the arrays uint16, uint8 and uint8 (N,) with N equal to `n_points`, the side's point count.
    """
    folder = Path(folder)
    if side not in LIDAR_SIDES:
        raise ValueError(f"{folder}: no LiDAR side {side!r}; sides are {', '.join(LIDAR_SIDES)}")
    path = folder / f"lidar_{side}_labels.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no LiDAR label file")
    sidecar_path, _, sidecar = _read_sidecar(folder)
    try:
        want = sidecar["files"].get(path.name)
    except _FIELD_ERRORS as e:
        raise _field_error(sidecar_path, e) from None
    data = path.read_bytes()
    if want is None or _sha256(data) != want:
        raise ValueError(f"{path}: sha256 does not match lidar.json files")
    arrays = _npz_arrays(path, data, ("format", "side", *_LABEL_ARRAYS))
    _check_text(path, arrays, "format", LIDAR_LABELS_FORMAT)
    _check_text(path, arrays, "side", side)
    n = _check_arrays(path, arrays, _LABEL_ARRAYS)
    if n != n_points:
        raise ValueError(f"{path}: {n} labels, expected {n_points}")
    return LidarLabels(side=side, **{name: _read_only(arrays[name]) for name in _LABEL_ARRAYS})


@dataclass(frozen=True, eq=False)
class NoisyPoints:
    """One side's measured points: time `t_ns` (M,) int64, position in the sensor frame `p_sensor` (M, 3) float64
    and the row `rows` (M,) int64 of each point in the side's file."""

    t_ns: np.ndarray
    p_sensor: np.ndarray
    rows: np.ndarray


def _unit_code(unit: str) -> int:
    if unit not in LIDAR_UNIT_CODE:
        raise ValueError(f"unknown LiDAR unit {unit!r}; units are {', '.join(LIDAR_UNIT_CODE)}")
    return LIDAR_UNIT_CODE[unit]


def noisy_points(points: LidarPoints, cfg: EvalAppConfig, noise_seed: int | None, *, scene: int,
                 side: tuple[int, int], unit: str) -> NoisyPoints:
    """The measured points of one side of the `unit`'s file, `side` its (side, aisle) code (`side_code`).

    Without a noise seed, every noiseless point (rows 0 to N - 1). With one, the noise of `lidar_point_noise` keyed
    `(cfg.seed, noise_seed, STEP_LIDAR, scene, *side, LIDAR_UNIT_CODE[unit])` with the `noise.lidar_*` values (an
    unset key counting as 0), and only the points it keeps, each with its file row. A point is its direction
    times its range.
    """
    code = _unit_code(unit)
    t_ns = np.asarray(points.t_ns, dtype=np.int64)
    if noise_seed is None:
        p = points.dir.astype(np.float64) * points.range_m.astype(np.float64)[:, None]
        return NoisyPoints(t_ns=t_ns, p_sensor=p, rows=np.arange(len(t_ns), dtype=np.int64))
    nz = cfg.noise
    dirs, ranges, keep = lidar_point_noise(
        (cfg.seed, noise_seed, STEP_LIDAR, scene, *side, code), points.dir, points.range_m,
        range_sigma_m=nz.factor("lidar_range_sigma_m"), range_sigma_near_m=nz.factor("lidar_range_sigma_near_m"),
        near_range_m=nz.factor("lidar_near_range_m"), ang_sigma_deg=nz.factor("lidar_ang_sigma_deg"),
        dropout=nz.factor("lidar_dropout"), range_bias_m=nz.factor("lidar_range_bias_m"))
    rows = np.flatnonzero(keep).astype(np.int64, copy=False)
    p = dirs[rows].astype(np.float64) * ranges[rows].astype(np.float64)[:, None]
    return NoisyPoints(t_ns=t_ns[rows], p_sensor=p, rows=rows)


def lidar_extr_error(cfg: EvalAppConfig, noise_seed: int | None, *, scene: int) -> np.ndarray | None:
    """The camera-LiDAR extrinsic error `dX_lidar` of the depth unit in `scene`, one for all its cameras.

    Only with `noise_seed` and a `lidar_extr_*` key set (0 included); otherwise None. It is
    `seeded_rigid((cfg.seed, noise_seed, STEP_LIDAR_EXTR, scene, LIDAR_UNIT_CODE["depth"]), lidar_extr_trans_mm /
    1000, lidar_extr_rot_deg)`, an unset key counting as 0; the pose section, the rig and the pitch do not enter.
    """
    nz = cfg.noise
    if noise_seed is None or all(getattr(nz, key) is None for key in LIDAR_EXTR_KEYS):
        return None
    return seeded_rigid((cfg.seed, noise_seed, STEP_LIDAR_EXTR, scene, LIDAR_UNIT_CODE["depth"]),
                        nz.factor("lidar_extr_trans_mm") / 1000.0, nz.factor("lidar_extr_rot_deg"))


def camera_from_sensor(cfg: EvalAppConfig, noise_seed: int | None, *, unit: str, scene: int, pitch: float,
                       T_body_sensor: np.ndarray) -> np.ndarray:
    """The believed camera <- sensor transform (4, 4) of the camera at `pitch`, from the true mount `T_body_sensor`
    (body <- sensor) of the `unit`.

    Depth unit: `pitch_extrinsic(pitch)^-1 @ T_body_sensor @ dX_lidar`, `dX_lidar` the scene's `lidar_extr_error`
    (identity without one). Navigation unit: `believed_extrinsic(pitch, r, dX_imu, dX_q)^-1 @ T_body_sensor`, with r
    `pose.rig_ref` under the rig and `pitch` otherwise, `dX_imu = fusion.imu_extr_error(..., pitch=r)` and `dX_q`
    the `posesource.rig_errors` value of `pitch` (None for none).
    """
    code = _unit_code(unit)
    T_bs = np.asarray(T_body_sensor, dtype=np.float64)
    if code == LIDAR_UNIT_CODE["depth"]:
        T = _rigid_inverse(pitch_extrinsic(pitch)) @ T_bs
        error = lidar_extr_error(cfg, noise_seed, scene=scene)
        return T if error is None else T @ error
    pose = cfg.pose
    ref = float(pose.rig_ref) if pose is not None and pose.rig else float(pitch)
    imu_error = imu_extr_error(cfg, noise_seed, scene=scene, pitch=ref)
    rig_error = rig_errors(cfg, noise_seed, scene=scene, pitches=[pitch]).get(float(pitch))
    return _rigid_inverse(believed_extrinsic(pitch, ref, imu_error, rig_error)) @ T_bs


def oracle_box_depth(ids: np.ndarray, camera: Camera, cloud: PassCloud, labels: LidarLabels,
                     s: LidarSettings) -> BoxLidar:
    """The truth-label pairing of each detection: the LiDAR depth of the points of its truth fruit.

    `ids` (n,) are the truth fruit ids of the detections (-1 for none). The points of a detection are the points of
    `cloud` in front of `camera` whose labels, read at the point's file row (`cloud.index`, so a dropped point
    leaves the others' labels in place), give its fruit id, a fruit or bag part (`ORACLE_PARTS`) and a `mixed`
    other than `ORACLE_MIXED_OUT`. Its depth is the median of their camera-plane depths, its count their number
    and its members their file rows, ascending. A detection without a truth fruit, or without such a point, gets
    NaN, 0 and no member. The pairing takes the place of the box match of `s.match` and reads none of the knobs of
    `s`; `combine_depth` then joins its depths as it joins those of a box match.
    """
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    rows = np.asarray(cloud.index, dtype=np.int64)
    fruit = labels.fruit_id[rows].astype(np.int64)
    pos = np.flatnonzero(np.isin(labels.part[rows], ORACLE_PARTS) & (labels.mixed[rows] != ORACLE_MIXED_OUT)
                         & np.isin(fruit, ids[ids > 0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        _, z = camera.project(cloud.X_w[pos])
    front = z > 0
    pos, z, fruit = pos[front], z[front], fruit[pos[front]]
    depth = np.full(len(ids), np.nan)
    count = np.zeros(len(ids), dtype=np.int64)
    members = []
    for d, fid in enumerate(ids.tolist()):
        sel = np.flatnonzero(fruit == fid) if fid > 0 else np.zeros(0, dtype=np.int64)
        if len(sel):
            depth[d], count[d] = np.median(z[sel]), len(sel)
        members.append(np.sort(rows[pos[sel]]).astype(np.int64, copy=False))
    return BoxLidar(z=depth, n_points=count, members=tuple(members))


# --- per-detection depth rows and the LiDAR result entry ---------------------------------------------------------

_ROW_COLUMNS = {"run_id": object, "noise_seed": object, "side": object, "pitch": np.float64, "idx": np.int64,
                "det": np.int64, "truth_id": np.int64, "counted": bool, "bagged": object, "depth_med": np.float64,
                "z_cam_clean": np.float64, "z_cam": np.float64, "z_lidar": np.float64, "n_points": np.int64,
                "z_used": np.float64, "used_lidar": bool}
OVER_M = 0.10   # an absolute depth error above this (m) counts as over
DEPTH_ROWS_INDEX = "index"
_FILE_NAME_BYTES = 255
_FILE_SAFE = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.@+-")


@dataclass(frozen=True, eq=False)
class DepthRows:
    """One row per detection of every frame a result scored, frame by frame, the detections of a frame in order.

    Columns (`COLUMNS`, each an array of one length): `run_id`, `noise_seed` (None without one), the pass `side` and
    `pitch`, the frame `idx` and the detection `det` in it, `truth_id` (the truth fruit the detection matched, -1
    for none), `counted` (in the counted row: the matched fruit belongs to a target-row tree; a detection without a
    match, when its lifted point lies in the target row), `bagged` (whether the matched fruit is bagged; None without
    a match), `depth_med` (the truth record's depth of the matched fruit in that frame, NaN without one),
    `z_cam_clean` (the camera box depth before noise), `z_cam` (after the camera depth noise, before joining),
    `z_lidar` and `n_points` (the LiDAR box depth and its point count; NaN and 0 without one), `z_used` (the depth
    the lift receives) and `used_lidar` (whether that depth took the LiDAR depth).
    """

    COLUMNS = tuple(_ROW_COLUMNS)

    run_id: np.ndarray
    noise_seed: np.ndarray
    side: np.ndarray
    pitch: np.ndarray
    idx: np.ndarray
    det: np.ndarray
    truth_id: np.ndarray
    counted: np.ndarray
    bagged: np.ndarray
    depth_med: np.ndarray
    z_cam_clean: np.ndarray
    z_cam: np.ndarray
    z_lidar: np.ndarray
    n_points: np.ndarray
    z_used: np.ndarray
    used_lidar: np.ndarray

    def __post_init__(self) -> None:
        n = None
        for name, dtype in _ROW_COLUMNS.items():
            value = getattr(self, name)
            if dtype is object:
                arr = np.empty(len(value), dtype=object)
                arr[:] = list(value)
            else:
                arr = np.asarray(value, dtype=dtype)
            if arr.ndim != 1 or (n is not None and len(arr) != n):
                raise ValueError(f"depth rows column {name} has shape {arr.shape}; every column needs ({n},)")
            n = len(arr)
            object.__setattr__(self, name, arr)

    def __len__(self) -> int:
        return len(self.idx)

    @classmethod
    def concat(cls, parts) -> DepthRows:
        """The rows of `parts` one after the other (no rows for no part)."""
        parts = list(parts)
        return cls(**{name: np.concatenate([getattr(p, name) for p in parts]) if parts else []
                      for name in _ROW_COLUMNS})


def _cell_text(value) -> str:
    """A depth row value as CSV text: None empty, a bool or a number as its `repr`, anything else as text."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return repr(bool(value))
    if isinstance(value, (int, np.integer)):
        return repr(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def depth_rows_csv(rows: DepthRows) -> str:
    """The CSV text of `rows`: the head line `DepthRows.COLUMNS`, then one line per row (`_cell_text` values)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(DepthRows.COLUMNS)
    columns = [getattr(rows, name) for name in DepthRows.COLUMNS]
    writer.writerows([_cell_text(column[i]) for column in columns] for i in range(len(rows)))
    return buf.getvalue()


def pop_depth_rows(results: dict) -> dict[str, DepthRows]:
    """Take the depth rows out of `results` (a run, ladder or sweep result) and return them by cell, in cell order.

    A cell is a rung of a run or ladder (the one rung of `eval run` is `run`) or a sweep cell `<point>@<base>` (`-`
    without a base). Its `depth_rows` (one `DepthRows`, or a list of them, one per noise seed) leaves the result;
    the rows of a cell, over the bundles in order and then its seeds, are joined into one `DepthRows`. Cell names are
    kept as they are; `depth_rows_file_names` makes the file names.
    """
    found: dict[str, list[DepthRows]] = {}
    for run in results["runs"]:
        if results["kind"] == "sweep":
            items = [(f"{cell['point']}@{cell['base'] or '-'}", cell) for cell in run["cells"]]
        else:
            items = list(run["rungs"].items())
        for name, item in items:
            rows = item.pop("depth_rows", None)
            if rows is None:
                continue
            found.setdefault(name, []).extend([rows] if isinstance(rows, DepthRows) else rows)
    return {name: DepthRows.concat(parts) for name, parts in found.items()}


def _file_stem(cell: str) -> str:
    """`cell` with every character outside `[A-Za-z0-9_.@+-]` written as `%XX` per UTF-8 byte (upper-case hex)."""
    return "".join(ch if ch in _FILE_SAFE else "".join(f"%{b:02X}" for b in ch.encode("utf-8")) for ch in cell)


def depth_rows_file_names(cells) -> dict[str, str]:
    """The depth rows file name of each cell name of `cells`, in order: `_file_stem(cell) + ".csv"`.

    `ValueError` when a name is longer than 255 bytes, or when two names are the same but for letter case or a name
    is that of the index file `DEPTH_ROWS_INDEX` (`collides`, naming the cells).
    """
    out: dict[str, str] = {}
    taken: dict[str, str | None] = {DEPTH_ROWS_INDEX.casefold(): None}
    for cell in cells:
        stem = _file_stem(cell)
        name = f"{stem}.csv"
        size = len(name.encode("utf-8"))
        if size > _FILE_NAME_BYTES:
            raise ValueError(f"depth rows file of cell {cell!r} would be {size} bytes long; the limit is "
                             f"{_FILE_NAME_BYTES}")
        key = stem.casefold()
        if key in taken:
            other = taken[key]
            if other is None:
                raise ValueError(f"depth rows file {name} of cell {cell!r} collides with the index file "
                                 f"{DEPTH_ROWS_INDEX}.csv")
            raise ValueError(f"depth rows file {name} of cell {cell!r} collides with the file of cell {other!r} "
                             "(names are compared without letter case)")
        taken[key] = cell
        out[cell] = name
    return out


def depth_rows_index_csv(names) -> str:
    """The index of the depth rows files: the head line `cell,file`, then one line per (cell name, file name)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["cell", "file"])
    writer.writerows([cell, name] for cell, name in names.items())
    return buf.getvalue()


def _sidecar_variant(folder: Path) -> str:
    path, _, sidecar = _read_sidecar(folder)
    try:
        return str(sidecar["variant"])
    except _FIELD_ERRORS as e:
        raise _field_error(path, e) from None


def _rate(num, den) -> float | None:
    return None if den == 0 else float(num / den)


def _p(values: np.ndarray, q: float) -> float | None:
    return float(np.percentile(values, q)) if len(values) else None


def lidar_entry(cfg: EvalAppConfig, *, file: LidarFile, mount: np.ndarray, mount_rpy_deg, rows: DepthRows,
                counts: Mapping) -> dict:
    """The `lidar` entry of a result whose boxes took the LiDAR depth of `file`.

    The file: `file` (the name in `lift.lidar_files`), `unit`, `method` (`lift.depth`), `sidecar_sha256`,
    `sim_commit`, `config_sha16` (the simulation config sha256, 16 characters) and `variant` (the sidecar's);
    the true mount `mount_m` (the translation of `mount`, body <- sensor) and `mount_rpy_deg`; `settings`, the
    `lift.lidar` section.

    From `rows`, with the matched detections the counted-row ones matched to a truth record: `detections` (`n`
    rows, `counted`, `matched`, `used` the LiDAR depth); `coverage`, the share of the matched detections that used
    the LiDAR depth, and `coverage_all`, that share of every counted-row detection; `fallback_rate`, the share of
    every detection that kept the camera depth; `points`, the p10, p50 and p90 of the point counts of the matched
    detections, with `cloud` (the points of the pass clouds, `counts["cloud_points"]`) and `per_window` (the mean
    points of a frame window, `counts["window_points"] / counts["frames"]`).
    `depth`, over the matched detections with e = `z_used - depth_med`: `n_with_depth` (finite e), `n_dropout`
    (without a depth), `over_10cm_rate` (|e| above `OVER_M` over every matched detection, those without a depth
    included and not counted as over) and `abs_error_p90_m` (the p90 of the finite |e|). `lidar_only`, the same with
    e = `z_lidar - depth_med` over the matched detections that used the LiDAR depth: `n`, `over_10cm_rate` and
    `abs_error_p90_m`. A ratio or percentile of nothing is None.
    """
    matched = rows.counted & (rows.truth_id >= 0)
    used = rows.used_lidar
    n_matched = int(matched.sum())
    pts = rows.n_points[matched].astype(np.float64)
    e = rows.z_used[matched] - rows.depth_med[matched]
    has = np.isfinite(e)
    covered = matched & used
    e_lidar = rows.z_lidar[covered] - rows.depth_med[covered]
    has_lidar = np.isfinite(e_lidar)
    frames = int(counts["frames"])
    return {
        "file": cfg.lift.lidar_file, "unit": file.device, "method": cfg.lift.depth, "sidecar_sha256": file.digest,
        "sim_commit": file.sim_commit, "config_sha16": file.config_sha256[:16],
        "variant": _sidecar_variant(file.folder),
        "mount_m": [float(v) for v in np.asarray(mount, dtype=np.float64)[:3, 3]],
        "mount_rpy_deg": [float(v) for v in mount_rpy_deg],
        "settings": cfg.lift.lidar.model_dump(),
        "detections": {"n": len(rows), "counted": int(rows.counted.sum()), "matched": n_matched,
                       "used": int(used.sum())},
        "coverage": _rate(used[matched].sum(), n_matched),
        "coverage_all": _rate(used[rows.counted].sum(), rows.counted.sum()),
        "fallback_rate": _rate((~used).sum(), len(rows)),
        "points": {"p10": _p(pts, 10), "p50": _p(pts, 50), "p90": _p(pts, 90), "cloud": int(counts["cloud_points"]),
                   "per_window": _rate(counts["window_points"], frames)},
        "depth": {"n_with_depth": int(has.sum()), "n_dropout": int((~np.isfinite(rows.z_used[matched])).sum()),
                  "over_10cm_rate": _rate((np.abs(e[has]) > OVER_M).sum(), n_matched),
                  "abs_error_p90_m": _p(np.abs(e[has]), 90)},
        "lidar_only": {"n": int(covered.sum()),
                       "over_10cm_rate": _rate((np.abs(e_lidar[has_lidar]) > OVER_M).sum(), int(covered.sum())),
                       "abs_error_p90_m": _p(np.abs(e_lidar[has_lidar]), 90)},
    }
