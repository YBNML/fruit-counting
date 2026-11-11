"""The on-disk pose cache: one bundle's relative camera poses, pass by pass.

A pose cache is one `.npz` file per run id holding six arrays, one row per frame in frame idx
order (`POSE_ARRAYS`), plus a `meta` JSON string. Each pass's poses are camera-to-world poses
relative to the pass's first cached frame, and each row also carries the values of the pair that
ends at that frame: the inlier count and two flags, `failed` and `prior`. `write_pose_cache` and
`load_pose_cache` run the same array checks, so a cache the reader would refuse is never
written. `pose_cache_digest` hashes the content (the meta JSON and the arrays), not the file's
bytes, so a rewrite of the same arrays and meta digests the same.

This module imports only numpy and the standard library.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POSE_CACHE_FORMAT = "counting-poses/1"
POSE_ARRAYS = ("frame_idx", "pass_code", "rel_c2w", "inliers", "failed", "prior")
_ARRAY_SPECS = (   # (name, dtype, shape after the row axis)
    ("frame_idx", np.int32, ()),
    ("pass_code", np.int32, (3,)),
    ("rel_c2w", np.float64, (4, 4)),
    ("inliers", np.int32, ()),
    ("failed", np.uint8, ()),
    ("prior", np.uint8, ()),
)
_ROTATION_TOL = 1e-9


@dataclass(frozen=True, eq=False)
class PassOdometry:
    """One pass of a pose cache, rows in ascending frame idx.

    Row k holds frame `idx[k]`: `rel_c2w[k]` is its camera-to-world pose (OpenCV camera axes)
    relative to the pass's first frame, and `inliers[k]`, `failed[k]`, `prior[k]` are the values
    of the pair from the frame before it to frame `idx[k]`. `failed` is 1 when the final pose fit
    of that pair failed, `prior` is 1 when its global matching failed and the motion prior stood
    in. The first row is the identity with 0, 0, 0. Dtypes: `idx` int32 (n,), `rel_c2w` float64
    (n, 4, 4), `inliers` int32 (n,), `failed` and `prior` uint8 (n,).
    """

    idx: np.ndarray
    rel_c2w: np.ndarray
    inliers: np.ndarray
    failed: np.ndarray
    prior: np.ndarray


@dataclass(frozen=True, eq=False)
class PoseCache:
    """A read pose cache: its path, meta (with `format`), passes keyed by pass code in the order of
    their first frame, and the content digest."""

    path: Path
    meta: dict
    passes: dict[tuple[int, int, int], PassOdometry]
    digest: str


def pose_cache_path(cache_dir: Path, run_id: str) -> Path:
    return Path(cache_dir) / f"{run_id}.npz"


def _bad(path: Path, reason: str) -> ValueError:
    return ValueError(f"{path}: bad pose cache: {reason}")


def write_pose_cache(path: Path, passes: Mapping[tuple[int, int, int], PassOdometry], meta: Mapping) -> str:
    """Write `passes` (one `PassOdometry` per pass code) and `meta` to `path`; return `pose_cache_digest(path)`.

    `path` must not exist, and `meta` must have `run_id` and `fingerprint` (the identity a reader
    checks against). The passes are joined into one set of rows sorted by frame idx (a frame idx in
    two passes is refused by the order check), and the rows pass the reader's `_check_arrays` before
    anything is written. The write is atomic: a sibling `<file name>.tmp` is written, fsynced and
    `os.replace`d onto `path`. The parent folder is made when missing.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path}: pose cache exists; remove it first to make it again")
    if "run_id" not in meta or "fingerprint" not in meta:
        raise ValueError("pose cache meta needs run_id and fingerprint")

    arrays = _join(path, passes)
    _check_arrays(path, arrays)
    meta_json = json.dumps({**meta, "format": POSE_CACHE_FORMAT}, sort_keys=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays, meta=np.array(meta_json))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return pose_cache_digest(path)


def _join(path: Path, passes: Mapping[tuple[int, int, int], PassOdometry]) -> dict[str, np.ndarray]:
    """The file arrays of `passes`: every pass's rows joined, then sorted by frame idx.

    Dtypes are kept as given (`_check_arrays` refuses wrong ones). A pass with no frames, or arrays
    that do not join, raise `ValueError` naming `path`.
    """
    if not passes:
        raise _bad(path, "no passes")
    for code, p in passes.items():
        if np.size(p.idx) == 0:
            raise _bad(path, f"pass {tuple(code)} has no frames")
    try:
        arrays = {
            "frame_idx": np.concatenate([p.idx for p in passes.values()]),
            "pass_code": np.concatenate([np.tile(np.asarray(code, dtype=np.int32), (np.size(p.idx), 1))
                                         for code, p in passes.items()]),
            "rel_c2w": np.concatenate([p.rel_c2w for p in passes.values()]),
            "inliers": np.concatenate([p.inliers for p in passes.values()]),
            "failed": np.concatenate([p.failed for p in passes.values()]),
            "prior": np.concatenate([p.prior for p in passes.values()]),
        }
    except ValueError as e:
        raise _bad(path, f"the passes do not join into arrays ({e})") from None
    frame_idx = arrays["frame_idx"]
    if frame_idx.ndim == 1 and all(len(a) == len(frame_idx) for a in arrays.values()):
        order = np.argsort(frame_idx, kind="stable")
        arrays = {name: arr[order] for name, arr in arrays.items()}
    return arrays


def _check_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Refuse pose cache arrays a reader cannot trust; raises `ValueError` naming `path`.

    In order: every array's dtype and shape (n >= 1 rows), `frame_idx` strictly increasing, finite
    poses whose last row is exactly 0, 0, 0, 1, rotations orthonormal to 1e-9 with a positive
    determinant, `failed` and `prior` 0 or 1, `inliers` not negative, and in every pass (the rows
    with one `pass_code`) a first row that is exactly the identity with 0 inliers, failed and prior.
    """
    for name, dtype, _ in _ARRAY_SPECS:
        if arrays[name].dtype != dtype:
            raise _bad(path, f"{name} has dtype {arrays[name].dtype}, expected {np.dtype(dtype)}")
    frame_idx = arrays["frame_idx"]
    if frame_idx.ndim != 1:
        raise _bad(path, f"frame_idx has shape {frame_idx.shape}, expected (n,)")
    n = len(frame_idx)
    if n == 0:
        raise _bad(path, "no frames")
    for name, _, tail in _ARRAY_SPECS:
        if arrays[name].shape != (n, *tail):
            raise _bad(path, f"{name} has shape {arrays[name].shape}, expected {(n, *tail)}")

    pass_code, rel_c2w = arrays["pass_code"], arrays["rel_c2w"]
    inliers, failed, prior = arrays["inliers"], arrays["failed"], arrays["prior"]
    if np.any(np.diff(frame_idx) <= 0):
        raise _bad(path, "frame_idx must strictly increase")
    if not np.all(np.isfinite(rel_c2w)):
        raise _bad(path, "poses must be finite")
    if not np.all(rel_c2w[:, 3, :] == np.array([0.0, 0.0, 0.0, 1.0])):
        raise _bad(path, "the last row of every pose must be 0, 0, 0, 1")
    rot = rel_c2w[:, :3, :3]
    gram = np.swapaxes(rot, 1, 2) @ rot
    if np.max(np.abs(gram - np.eye(3))) > _ROTATION_TOL or np.any(np.linalg.det(rot) <= 0):
        raise _bad(path, "rotations must be orthonormal")
    if not (np.all(np.isin(failed, (0, 1))) and np.all(np.isin(prior, (0, 1)))):
        raise _bad(path, "failed and prior must be 0 or 1")
    if np.any(inliers < 0):
        raise _bad(path, "inliers must not be negative")

    _, first = np.unique(pass_code, axis=0, return_index=True)
    if not np.all(rel_c2w[first] == np.eye(4)):
        raise _bad(path, "the first pose of each pass must be the identity")
    if np.any(inliers[first] != 0) or np.any(failed[first] != 0) or np.any(prior[first] != 0):
        raise _bad(path, "the first frame of each pass has no pair: inliers, failed and prior must be 0")


def _open_npz(path: Path) -> dict[str, np.ndarray]:
    """The arrays of the npz at `path`, as a plain dict, after every check on the file itself.

    A file `np.load` cannot read (not a zip, a corrupt compressed member, pickled data), a single
    `.npy` array under an `.npz` name, or an archive whose array set is not `POSE_ARRAYS` plus
    `meta` raises `ValueError` naming `path`. `load_pose_cache` and `pose_cache_digest` both read
    through here, so they refuse the same files.
    """
    try:
        obj = np.load(path, allow_pickle=False)
        if isinstance(obj, np.lib.npyio.NpzFile):      # a `.npy` comes back as one array, not a context manager
            with obj as data:
                arrays = {k: data[k] for k in data.files}
    except (zipfile.BadZipFile, EOFError, OSError, zlib.error) as e:
        raise _bad(path, f"not a readable npz ({e})") from None
    except ValueError as e:
        raise _bad(path, str(e)) from None
    if not isinstance(obj, np.lib.npyio.NpzFile):
        raise _bad(path, "not a readable npz (a single array, not an archive)")
    if set(arrays) != {*POSE_ARRAYS, "meta"}:
        raise _bad(path, f"expected arrays {{{', '.join((*POSE_ARRAYS, 'meta'))}}}, got {sorted(arrays)}")
    return arrays


def _digest(arrays: Mapping[str, np.ndarray]) -> str:
    """sha256 over the meta JSON, then each `POSE_ARRAYS` array's (name, dtype, shape, bytes)."""
    h = hashlib.sha256()
    h.update(str(arrays["meta"].item()).encode("utf-8"))
    for name in POSE_ARRAYS:
        arr = arrays[name]
        h.update(f"\n{name}:{arr.dtype.str}:{arr.shape}:".encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def load_pose_cache(path: Path) -> PoseCache:
    """Read and check a pose cache in one pass over the file; return its meta, passes and digest.

    Checks the file and its array set (`_open_npz`), that `meta` is a JSON object with the format
    tag `POSE_CACHE_FORMAT`, and the arrays (`_check_arrays`, shared with `write_pose_cache`). Any
    violation raises `ValueError` naming `path`. Passes are keyed by their pass code as a tuple of
    ints, in the order of their first frame; their arrays are copies.
    """
    path = Path(path)
    arrays = _open_npz(path)
    try:
        meta = json.loads(arrays["meta"].item())
    except (ValueError, TypeError) as e:
        raise _bad(path, f"meta is not a JSON string ({e})") from None
    if not isinstance(meta, dict):
        raise _bad(path, "meta is not a JSON object")
    if meta.get("format") != POSE_CACHE_FORMAT:
        raise _bad(path, f"format is {meta.get('format')!r}, expected {POSE_CACHE_FORMAT!r}")
    _check_arrays(path, arrays)

    pass_code = arrays["pass_code"]
    _, first = np.unique(pass_code, axis=0, return_index=True)
    passes: dict[tuple[int, int, int], PassOdometry] = {}
    for row in sorted(first.tolist()):
        code = pass_code[row]
        rows = np.flatnonzero(np.all(pass_code == code, axis=1))
        passes[tuple(int(v) for v in code)] = PassOdometry(
            idx=arrays["frame_idx"][rows].copy(), rel_c2w=arrays["rel_c2w"][rows].copy(),
            inliers=arrays["inliers"][rows].copy(), failed=arrays["failed"][rows].copy(),
            prior=arrays["prior"][rows].copy(),
        )
    return PoseCache(path=path, meta=meta, passes=passes, digest=_digest(arrays))


def pose_cache_digest(path: Path) -> str:
    """sha256 of the pose cache's content: the meta JSON, then each `POSE_ARRAYS` array's (name,
    dtype, shape, bytes). Independent of the file's own bytes (the npz/zip framing). It refuses
    what `load_pose_cache` refuses about the file and its array set (`ValueError` naming `path`),
    and does not read the format tag."""
    path = Path(path)
    return _digest(_open_npz(path))
