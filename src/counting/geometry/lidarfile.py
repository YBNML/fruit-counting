"""Prediction-side reading of the LiDAR files beside a bundle: one simulated unit's points, per side.

A folder of LiDAR files sits outside the bundle, at `<crop folder>/lidar/<simulation config sha256[:16]>/<bundle
name>/` (`lidar_folder` joins a configured folder and the bundle name). It holds one point file per side,
`lidar_<side>.npz`, the sidecar `lidar.json` and files that only scoring reads. This module opens the point files
by their exact names and no other file of the folder.

Checks. `LidarFile` reads the sidecar once and checks, in this order: that it is there; the sha256 of its bytes
against the pinned digest; its format tag (`LIDAR_SIDECAR_FORMAT`); that its unit (`device`) is the one asked for;
its crop, bundle name and seed against this bundle's; each input file sha256 the caller gives against the one the
sidecar records; the simulation config sha256, and that the folder above is named after its first 16 characters;
that the simulator commit is one of the expected commits. The digest, the config sha256 and the commits are
checked only when given. `LidarFile.points(side)` opens that side's point file on first use and checks: the sha256
of its bytes against the sidecar `files`; its array set (`LIDAR_TEXT_KEYS` and `LIDAR_ARRAYS`); the three text
values, each a 0-d `<U` string (format `LIDAR_FILE_FORMAT`, the sidecar unit, the side of the file name); each
array's dtype and shape (`LIDAR_ARRAYS`) and one point count N; N against the side's `n_points` in the sidecar, and
N above 0; that `t_ns` never decreases and stays within the side's `[first_ns, last_ns]`. A failed check raises
`ValueError` naming the file path; a missing file raises `FileNotFoundError`.

Points. `dir` float32 (N, 3) holds unit vectors in the sensor frame, `range_m` float32 (N,) the noiseless range
along them (m), `t_ns` int64 (N,) the time of each point in ns on the side's clock (the clock of the frame times;
the two sides have separate clocks) and `row` uint32 (N,) the scan pattern row. `LidarPoints` holds the four arrays
read-only, with the side and the span `[first_ns, last_ns]` of the side's beams. The sensor placement on the body,
the simulation settings and every per-point truth stay with the scoring side; this module reads none of them.

This module imports only numpy and the standard library.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LIDAR_FILE_FORMAT = "orchard-lidar/1"
LIDAR_SIDECAR_FORMAT = "orchard-lidar-sidecar/1"
LIDAR_SIDECAR = "lidar.json"
LIDAR_TEXT_KEYS = ("format", "device", "side")
LIDAR_ARRAYS = {"dir": (np.float32, (-1, 3)), "range_m": (np.float32, (-1,)), "t_ns": (np.int64, (-1,)),
                "row": (np.uint32, (-1,))}
LIDAR_SIDES = ("A", "B")
LIDAR_UNITS = ("depth", "nav")
# Errors from indexing a JSON tree that lacks a field or has the wrong shape.
_FIELD_ERRORS = (KeyError, IndexError, TypeError, AttributeError)


def lidar_folder(dir: str | Path, bundle_name: str) -> Path:
    """The LiDAR file folder of the bundle named `bundle_name` under the configured folder `dir`."""
    return Path(dir).expanduser() / bundle_name


@dataclass(frozen=True, eq=False)
class LidarPoints:
    """One side's points (read-only arrays, `LIDAR_ARRAYS`) and the span `[first_ns, last_ns]` of its beams."""

    side: str
    t_ns: np.ndarray
    dir: np.ndarray
    range_m: np.ndarray
    row: np.ndarray
    first_ns: int
    last_ns: int


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _field_error(path: Path, e: Exception) -> ValueError:
    """A `ValueError` naming `path` for a `_FIELD_ERRORS` exception raised while reading its JSON."""
    if isinstance(e, KeyError):
        return ValueError(f"{path}: missing field {e.args[0]!r}")
    return ValueError(f"{path}: malformed field ({type(e).__name__}: {e})")


def _read_sidecar(folder: Path) -> tuple[Path, bytes, dict]:
    """The sidecar path, bytes and JSON object of the LiDAR file folder `folder`."""
    path = Path(folder) / LIDAR_SIDECAR
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no LiDAR sidecar")
    data = path.read_bytes()
    try:
        obj = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"{path}: not a JSON file ({e})") from None
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: not a JSON object")
    return path, data, obj


def _npz_arrays(path: Path, data: bytes, keys: Sequence[str]) -> dict[str, np.ndarray]:
    """The arrays of the npz bytes `data` read from `path`, without pickles; the array set must be `keys`."""
    try:
        obj = np.load(io.BytesIO(data), allow_pickle=False)
        if not isinstance(obj, np.lib.npyio.NpzFile):   # a `.npy` comes back as one array
            raise ValueError("a single array, not an archive")
        with obj as npz:
            arrays = {name: npz[name] for name in npz.files}
    except (zipfile.BadZipFile, EOFError, OSError, zlib.error, ValueError) as e:
        raise ValueError(f"{path}: not a readable npz ({e})") from None
    if set(arrays) != set(keys):
        raise ValueError(f"{path}: arrays {sorted(arrays)}, expected {sorted(keys)}")
    return arrays


def _check_text(path: Path, arrays: Mapping[str, np.ndarray], name: str, want: str) -> None:
    """The npz value `name` must be a 0-d `<U` string equal to `want`."""
    arr = arrays[name]
    if arr.ndim != 0 or not arr.dtype.str.startswith("<U"):
        raise ValueError(f"{path}: {name} is {arr.dtype.str} of shape {arr.shape}, not a 0-d <U string")
    value = str(arr.item())
    if value != want:
        raise ValueError(f"{path}: {name} {value}, expected {want}")


def _check_arrays(path: Path, arrays: Mapping[str, np.ndarray], specs: Mapping[str, tuple]) -> int:
    """Each array of `specs` (name -> (dtype, shape with -1 for N)) must have that dtype and shape, with one N;
    returns N."""
    n = None
    for name, (dtype, shape) in specs.items():
        arr = arrays[name]
        if arr.dtype != dtype:
            raise ValueError(f"{path}: {name} has dtype {arr.dtype}, expected {np.dtype(dtype)}")
        if arr.ndim != len(shape) or arr.shape[1:] != shape[1:]:
            want = "(N,)" if len(shape) == 1 else "(N, " + ", ".join(str(d) for d in shape[1:]) + ")"
            raise ValueError(f"{path}: {name} has shape {arr.shape}, expected {want}")
        if n is None:
            n = len(arr)
        elif len(arr) != n:
            raise ValueError(f"{path}: {name} has {len(arr)} rows, the arrays before it {n}")
    return int(n or 0)


def _spans(path: Path, sides: Mapping) -> dict[str, tuple[int, int, int]]:
    """(first_ns, last_ns, n_points) of each side the sidecar `time.sides` holds; each must be a whole number."""
    out = {}
    for side in LIDAR_SIDES:
        if side not in sides:
            continue
        values = tuple(sides[side][key] for key in ("first_ns", "last_ns", "n_points"))
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            raise ValueError(f"{path}: time.sides.{side} first_ns, last_ns and n_points must be whole numbers")
        out[side] = values
    return out


def _read_only(arr: np.ndarray) -> np.ndarray:
    arr.flags.writeable = False
    return arr


class LidarFile:
    """The LiDAR files of one bundle and one unit, the sidecar read and checked once, the points on demand.

    `unit` is the unit asked for (`LIDAR_UNITS`); `bundle`, `crop` and `seed` are the bundle's folder name, crop
    and seed; `inputs` maps bundle file names to the sha256 of their bytes. `digest` (the sha256 of the sidecar
    bytes), `config_sha256` (the simulation config) and `sim_commits` (the simulator commits allowed) are checked
    only when given. Attributes: `folder`, `digest` (the sidecar sha256), `device` (the unit), `config_sha256` and
    `sim_commit` (as the sidecar records them) and `dirty` (whether the simulator tree had changes).
    """

    def __init__(self, folder: Path, *, unit: str, bundle: str, crop: str, seed: int, inputs: Mapping[str, str],
                 digest: str | None = None, config_sha256: str | None = None,
                 sim_commits: Sequence[str] | None = None) -> None:
        if unit not in LIDAR_UNITS:
            raise ValueError(f"unknown LiDAR unit {unit!r}; units are {', '.join(LIDAR_UNITS)}")
        self.folder = Path(folder)
        path, data, sidecar = _read_sidecar(self.folder)
        self.digest = _sha256(data)
        if digest is not None and self.digest != digest:
            raise ValueError(f"{path}: lidar.json sha256 does not match the pinned digest "
                             f"({self.digest[:16]}, pinned {digest[:16]})")
        try:
            fmt, device = sidecar["format"], sidecar["device"]
            if fmt != LIDAR_SIDECAR_FORMAT:
                raise ValueError(f"{path}: format {fmt}, expected {LIDAR_SIDECAR_FORMAT}")
            if device != unit:
                raise ValueError(f"{path}: device {device}, expected {unit}")
            for name, want in (("crop", crop), ("bundle", bundle), ("seed", seed)):
                if sidecar[name] != want:
                    raise ValueError(f"{path}: {name} {sidecar[name]}, expected {want}")
            recorded = sidecar["inputs"]
            for name, sha in inputs.items():
                if recorded.get(name) != sha:
                    raise ValueError(f"{path}: {name} sha256 differs from the bundle")
            sim_config = str(sidecar["config"]["sha256"])
            if config_sha256 is not None and sim_config != config_sha256:
                raise ValueError(f"{path}: config sha256 {sim_config[:16]} differs from the expected "
                                 f"{config_sha256[:16]}")
            if config_sha256 is not None and self.folder.parent.name != config_sha256[:16]:
                raise ValueError(f"{path}: folder {self.folder.parent.name} is not named after config sha256 "
                                 f"{config_sha256[:16]}")
            commit, dirty = sidecar["orchard_synth"]["commit"], sidecar["orchard_synth"]["dirty"]
            if sim_commits is not None and commit not in tuple(sim_commits):
                raise ValueError(f"{path}: simulator commit {str(commit)[:12]} is not one of the expected commits")
            if not isinstance(dirty, bool):
                raise ValueError(f"{path}: orchard_synth dirty is {dirty!r}, not true or false")
            files = sidecar["files"]
            self._files = {name: files[name] for name in (f"lidar_{s}.npz" for s in LIDAR_SIDES) if name in files}
            self._spans = _spans(path, sidecar["time"]["sides"])
        except _FIELD_ERRORS as e:
            raise _field_error(path, e) from None
        self.device = str(device)
        self.config_sha256 = sim_config
        self.sim_commit = str(commit)
        self.dirty = dirty
        self._points: dict[str, LidarPoints] = {}

    def points(self, side: str) -> LidarPoints:
        """The points of `side` from `lidar_<side>.npz`, read and checked on the first call, then kept."""
        if side in self._points:
            return self._points[side]
        if side not in LIDAR_SIDES:
            raise ValueError(f"{self.folder}: no LiDAR side {side!r}; sides are {', '.join(LIDAR_SIDES)}")
        path = self.folder / f"lidar_{side}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"{path}: no LiDAR point file")
        if path.name not in self._files or side not in self._spans:
            raise ValueError(f"{path}: lidar.json lists no {path.name} or no time span for side {side}")
        data = path.read_bytes()
        if _sha256(data) != self._files[path.name]:
            raise ValueError(f"{path}: sha256 does not match lidar.json files")
        arrays = _npz_arrays(path, data, (*LIDAR_TEXT_KEYS, *LIDAR_ARRAYS))
        for name, want in zip(LIDAR_TEXT_KEYS, (LIDAR_FILE_FORMAT, self.device, side)):
            _check_text(path, arrays, name, want)
        n = _check_arrays(path, arrays, LIDAR_ARRAYS)
        first_ns, last_ns, n_points = self._spans[side]
        if n != n_points:
            raise ValueError(f"{path}: {n} points, lidar.json time.sides.{side}.n_points is {n_points}")
        if n == 0:
            raise ValueError(f"{path}: no points")
        t_ns = arrays["t_ns"]
        if np.any(np.diff(t_ns) < 0):
            raise ValueError(f"{path}: t_ns decreases")
        if t_ns[0] < first_ns or t_ns[-1] > last_ns:
            raise ValueError(f"{path}: t_ns runs outside [first_ns, last_ns] = [{first_ns}, {last_ns}] of "
                             f"lidar.json time.sides.{side} ({int(t_ns[0])} to {int(t_ns[-1])})")
        points = LidarPoints(side=side, **{name: _read_only(arrays[name]) for name in LIDAR_ARRAYS},
                             first_ns=first_ns, last_ns=last_ns)
        self._points[side] = points
        return points
