"""Fake LiDAR files beside a fake shake bundle, shaped like the files the orchard simulator writes.

`write_lidar_files` turns the pixels of a `write_run(..., shake=True, images=True)` bundle into the five LiDAR files
of one unit: per side a point file `lidar_<side>.npz` and a label file `lidar_<side>_labels.npz`, and the sidecar
`lidar.json`. `shared_lidar_bundle` writes one such bundle with a depth unit and a navigation unit once per process.

Like `synthfake`, this module builds raw numbers: the sidecar, the npz keys and the mount matrix are written out
here, not taken from the modules under test, which it does not import.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from math import cos, radians, sin, tan
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from tests.unit.synthfake import SIDES, _camera_on_body, write_run

POINT_RATE_HZ = 200_000
SCAN_RATE_HZ = 10
NS_PER_POINT = 5_000
PATTERN_ROWS = 800_000
RANGE_M = (0.1, 40.0)   # ranges the fake unit returns
_ROW0 = {"A": 4_000, "B": 8_000}   # first scan pattern row per side
_UNIT_CODE = {"depth": 0, "nav": 1}
_CROP_CODE = {"apple": 0, "pear": 1}
_EXTRINSIC = {"depth": (["noise.lidar_extr_trans_mm", "noise.lidar_extr_rot_deg"], "own"),
              "nav": (["noise.imu_extr_trans_mm", "noise.imu_extr_rot_deg"], "imu")}


@dataclass(frozen=True)
class SharedLidar:
    """The shared fake bundle and the LiDAR file folders of its depth unit and its navigation unit."""

    bundle: Path
    depth: Path
    nav: Path


def lidar_contract(unit: str) -> dict:
    """The noise values the simulation config records for counting, for the `depth` or `nav` unit."""
    keys, draw = _EXTRINSIC[unit]
    return {"range_sigma_m": 0.02, "range_sigma_near_m": 0.03, "near_range_m": 1.0, "ang_sigma_deg": 0.15,
            "dropout": 0.05, "range_bias_m": 0.0,
            "extrinsic": {"keys": list(keys), "trans_mm": 10.0, "rot_deg": 0.3, "draw": draw},
            "time_offset_ms": 0.0}


def mount_matrix(xyz_m, rpy_deg) -> np.ndarray:
    """The body <- sensor transform (4, 4): rotation Rz(yaw) Ry(pitch) Rx(roll) of `rpy_deg` = (roll, pitch,
    yaw), translation `xyz_m`."""
    r, p, y = (radians(float(a)) for a in rpy_deg)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cos(r), -sin(r)], [0.0, sin(r), cos(r)]])
    ry = np.array([[cos(p), 0.0, sin(p)], [0.0, 1.0, 0.0], [-sin(p), 0.0, cos(p)]])
    rz = np.array([[cos(y), -sin(y), 0.0], [sin(y), cos(y), 0.0], [0.0, 0.0, 1.0]])
    out = np.eye(4)
    out[:3, :3] = rz @ ry @ rx
    out[:3, 3] = [float(v) for v in xyz_m]
    return out


def dump_json(obj) -> bytes:
    """Deterministic JSON bytes: sorted keys, no spaces, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sidecar_sha256(folder: Path) -> str:
    """The sha256 of the bytes of `folder/lidar.json`."""
    return _sha256(Path(folder) / "lidar.json")


def _pitch(frame: Mapping) -> float:
    """The camera pitch of a fake frame, from its pose name `<preset>_<pitch>`."""
    return float(frame["pose"]["name"].rsplit("_", 1)[1])


def _png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im)


def _side_points(bundle: Path, frames: list[dict], stride: int, mount: np.ndarray, side: str):
    """The sensor-frame points and labels of the frames `frames` (one side, time order).

    Every `stride`-th pixel with a depth return goes back through the true camera: the depth16 plane depth along
    the pixel-centre ray gives the point in the camera frame, the camera-on-body turn of the frame's pitch gives it
    in the body frame, and the inverse of `mount` gives it in the sensor frame.
    """
    rot, trans = mount[:3, :3], mount[:3, 3]
    parts = {name: [] for name in ("dir", "range_m", "t_ns", "row", "fruit_id", "part", "mixed")}
    for f in frames:
        idx, pose = f["idx"], f["pose"]
        width, height = pose["width"], pose["height"]
        fx = (width / 2) / tan(radians(pose["hfov_deg"] / 2))
        depth = _png(bundle / f"{idx:06d}_depth16.png")[::stride, ::stride]
        inst = _png(bundle / f"{idx:06d}_instances.png")[::stride, ::stride]
        sem = _png(bundle / f"{idx:06d}_semantic.png")[::stride, ::stride]
        ys, xs = np.mgrid[0:height:stride, 0:width:stride]
        d = depth.astype(np.float64) / 1000.0
        cam = np.stack([(xs + 0.5 - width / 2) / fx * d, -(ys + 0.5 - height / 2) / fx * d, -d], axis=-1)
        body = cam @ _camera_on_body(_pitch(f))[:3, :3].T
        sensor = (body - trans) @ rot                      # rows are R^T (p - t)
        rng = np.linalg.norm(sensor, axis=-1)
        keep = (depth > 0) & (rng >= RANGE_M[0]) & (rng <= RANGE_M[1])
        n = int(keep.sum())
        t_ns = round(float(f["t"]) * 1e9)
        parts["dir"].append((sensor[keep] / rng[keep][:, None]).astype(np.float32))
        parts["range_m"].append(rng[keep].astype(np.float32))
        parts["t_ns"].append(np.full(n, t_ns, dtype=np.int64))
        parts["row"].append(np.full(n, (_ROW0[side] + t_ns // NS_PER_POINT) % PATTERN_ROWS, dtype=np.uint32))
        parts["fruit_id"].append(inst[keep].astype(np.uint16))
        parts["part"].append(sem[keep].astype(np.uint8))
        parts["mixed"].append(np.zeros(n, dtype=np.uint8))
    return {name: np.concatenate(arrs) for name, arrs in parts.items()}


def write_lidar_files(bundle: Path, *, unit: str = "depth", mount_xyz=(0.0, 0.0, 0.1), mount_rpy_deg=(0.0, 0.0, 0.0),
                      config_sha256: str = "ab" * 32, commit: str = "1" * 40, dirty: bool = False, stride: int = 3,
                      contract: Mapping | None = None) -> Path:
    """Write the five LiDAR files of `unit` for the shake bundle `bundle` (rendered with images) and return their
    folder `<bundle.parent>/lidar/<config_sha256[:16]>/<bundle.name>/`.

    Each side's points are the frames of the bundle's lowest pitch, in time order: every `stride`-th pixel with a
    depth return seen through the true camera and placed in the sensor frame of the unit mounted at `mount_xyz`
    (m) and `mount_rpy_deg` (roll, pitch, yaw; Rz Ry Rx) on the body. A point's time `t_ns` is its frame's
    `round(t * 1e9)`; its labels are the frame's instance id, semantic class and 0 (not mixed). The sidecar holds
    the seventeen top keys, with `mount.T_body_sensor` and `config.values.mount[<crop>]` stating the same mount,
    and `contract` (default `lidar_contract(unit)`) as `config.values.counting_contract`.
    """
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    if manifest["args"].get("shake") is not True:
        raise ValueError(f"{bundle}: fake LiDAR files need a bundle rendered with body shake")
    crop, seed = manifest["args"]["crop"], manifest["args"]["seed"]
    mount = mount_matrix(mount_xyz, mount_rpy_deg)
    frames = [json.loads(p.read_bytes()) for p in sorted(bundle.glob("*_frame.json"))]
    base = min(_pitch(f) for f in frames)
    folder = bundle.parent / "lidar" / config_sha256[:16] / bundle.name
    folder.mkdir(parents=True, exist_ok=True)

    spans = {}
    for side in SIDES:
        chosen = sorted((f for f in frames if f["side"] == side and _pitch(f) == base), key=lambda f: f["t"])
        a = _side_points(bundle, chosen, stride, mount, side)
        np.savez_compressed(folder / f"lidar_{side}.npz", format=np.array("orchard-lidar/1"), device=np.array(unit),
                            side=np.array(side), **{k: a[k] for k in ("dir", "range_m", "t_ns", "row")})
        np.savez_compressed(folder / f"lidar_{side}_labels.npz", format=np.array("orchard-lidar-labels/1"),
                            side=np.array(side), **{k: a[k] for k in ("fruit_id", "part", "mixed")})
        last_ns = int(a["t_ns"].max())
        n_scans = last_ns // (1_000_000_000 // SCAN_RATE_HZ) + 1
        spans[side] = {"first_ns": 0, "last_ns": last_ns, "n_scans": n_scans,
                       "n_beams": n_scans * (POINT_RATE_HZ // SCAN_RATE_HZ), "n_points": int(len(a["t_ns"]))}

    commit_file = bundle / "orchard_synth.commit"
    render_commit = commit_file.read_text(encoding="utf-8").splitlines()[0].strip() if commit_file.is_file() else None
    name = f"mid360_{unit}"
    sidecar = {
        "format": "orchard-lidar-sidecar/1",
        "device": unit,
        "variant": "judged",
        "crop": crop,
        "bundle": bundle.name,
        "seed": seed,
        "inputs": {"manifest.json": _sha256(bundle / "manifest.json"),
                   "trajectory.json": _sha256(bundle / "trajectory.json"), "render_commit": render_commit},
        "files": {f"lidar_{s}{tail}.npz": _sha256(folder / f"lidar_{s}{tail}.npz")
                  for s in SIDES for tail in ("", "_labels")},
        "config": {"name": name, "path": f"configs/lidar/{name}.yaml", "sha256": config_sha256,
                   "values": {"device": unit, "variant": "judged",
                              "mount": {crop: {"xyz_m": [float(v) for v in mount_xyz],
                                               "rpy_deg": [float(v) for v in mount_rpy_deg]}},
                              "counting_contract": dict(lidar_contract(unit) if contract is None else contract)}},
        "pattern": {"name": "mid360", "source": "fake scan pattern", "sha256": "0" * 64, "rows": PATTERN_ROWS,
                    "license": "fake"},
        "orchard_synth": {"commit": commit, "dirty": dirty},
        "rng": {"key": [seed, 0, _CROP_CODE[crop], _UNIT_CODE[unit]], "row0": dict(_ROW0)},
        "mount": {"T_body_sensor": mount.tolist()},
        "time": {"clock": "body clock of the side, t=0 at the start of the stop", "unit": "ns",
                 "point_rate_hz": POINT_RATE_HZ, "scan_rate_hz": SCAN_RATE_HZ, "sides": spans},
        "env": {"python": "3.11.0", "numpy": "1.26.4", "open3d": "0.19.0"},
        "frames": {"sensor": "scan pattern frame", "body": "x travel, y left, z up, origin at the camera centre",
                   "dir": "float32 unit vector in the sensor frame", "range_m": "float32 noise-free range (m)"},
        "reflectivity": "constant, not stored",
    }
    assert len(sidecar) == 17
    (folder / "lidar.json").write_bytes(dump_json(sidecar))
    return folder


@functools.lru_cache(maxsize=None)
def shared_lidar_bundle() -> SharedLidar:
    """A fake apple shake bundle (160 x 90, three trees per row) with the LiDAR files of a depth unit (0, 0, 0.1) m
    and of a navigation unit (0, 0, 0.5) m beside it, written once per process under a temporary folder removed
    at exit. Read it only; a test that edits files copies them to its `tmp_path` first."""
    root = Path(tempfile.mkdtemp(prefix="lidarfake-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    bundle = write_run(root / "apple" / "dev_s5_shake", size=(160, 90), shake=True, images=True, trees_per_row=3)
    depth = write_lidar_files(bundle)
    nav = write_lidar_files(bundle, unit="nav", mount_xyz=(0.0, 0.0, 0.5), config_sha256="cd" * 32)
    return SharedLidar(bundle=bundle, depth=depth, nav=nav)
