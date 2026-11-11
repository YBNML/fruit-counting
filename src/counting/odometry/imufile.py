"""IMU files and the measured streams they hold, on the prediction side.

This module holds standard gravity (`GRAVITY`), the drive layouts (`LAYOUTS`), the stream type
(`ImuStream`), the key of one stream (`stream_key`), the rule that splits an integration window into
per-sample spans (`sample_spans`), the on-disk IMU file (`write_imu_file`, `load_imu_file`,
`imu_file_digest`) and its prediction-side reader (`ImuFile`).

Layouts. `rig`: one drive per side, the cameras of every pitch riding one body. `pass`: one drive per
pass (pitch and side), each riding its side's body.

Samples. Sample k of a stream holds the body angular rate (rad/s, body axes) and the specific force
(m/s^2, body axes) over [t[k], t[k+1]). `sample_spans` holds the last sample past its own interval, up
to two sample intervals after its time.

File format (`IMU_FILE_FORMAT`). One `.npz` file per run id holds five arrays (`IMU_ARRAYS`) and a
`meta` JSON string. `stream` int32 (S, 5) has one row per stream, its key (layout code, noise seed,
side code, aisle, pitch in millidegrees or -1; see `stream_key`), in the order the writer was given.
`stream_of` int32 (N,) gives each sample's stream row, one block per stream in row order. `t` float64
(N,), `gyro` and `acc` float64 (N, 3) are the samples, in time order within each stream. The meta holds
at least `run_id`, `fingerprint` (the bundle's), `imu_seed`, `noise` (the profile the measurements were
made with) and `rate_hz`, plus the format tag. `write_imu_file` and `load_imu_file` run the same array
checks, so a file the reader would refuse is never written. The digest hashes the content (the meta
JSON and the arrays), not the file's bytes, so a rewrite of the same streams and meta digests the same.

Reading. `ImuFile` reads a file once and checks, each only when given and in this order, the bundle
fingerprint, the run id, the content digest against a pinned digest and the `imu_seed` against the
config seed; then always that the profile's white noise and turn-on bias are above 0, since an IMU
without them hands the true motion to the estimator. Every failed check raises `ValueError` naming the
file path. A file holds measured streams only; this module reads no truth, trajectory, frame pose or
planned pose.

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

GRAVITY = 9.80665   # standard gravity (m/s^2)
LAYOUTS = ("rig", "pass")
_SPAN_TOL_S = 1e-9

IMU_FILE_FORMAT = "counting-imu/1"
IMU_ARRAYS = ("stream", "stream_of", "t", "gyro", "acc")
_SAMPLE_ARRAYS = ("t", "gyro", "acc")
_ARRAY_DTYPES = {"stream": np.int32, "stream_of": np.int32, "t": np.float64, "gyro": np.float64,
                 "acc": np.float64}
_META_KEYS = ("run_id", "fingerprint", "imu_seed", "noise", "rate_hz")
_NOISELESS_KEYS = ("gyro_n", "acc_n", "gyro_b0_dps", "acc_b0_mg")   # white noise and turn-on bias


@dataclass(frozen=True, eq=False)
class ImuStream:
    """One measured stream: sample times `t` (K,) s, body angular rate `gyro` (K, 3) rad/s and specific
    force `acc` (K, 3) m/s^2, both in body axes, all float64. Sample k holds over [t[k], t[k+1])."""

    t: np.ndarray
    gyro: np.ndarray
    acc: np.ndarray


def stream_key(layout: str, noise_seed: int, side: str, aisle: int,
               pitch_deg: float | None = None) -> tuple[int, int, int, int, int]:
    """The key of one stream: (layout code, noise seed, side code, aisle, pitch in millidegrees).

    The layout code is the index in `LAYOUTS`, the side code 0 for side A and 1 otherwise, and the pitch
    -1 for a `rig` drive, which has none; a `pass` drive has one.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown IMU layout {layout!r}")
    if (layout == "rig") != (pitch_deg is None):
        raise ValueError("a rig drive has no pitch and a pass drive has one")
    return (LAYOUTS.index(layout), int(noise_seed), 0 if side == "A" else 1, int(aisle),
            -1 if pitch_deg is None else round(float(pitch_deg) * 1000))


def sample_spans(t: np.ndarray, ta: float, tb: float) -> list[tuple[int, float]]:
    """The (sample index, seconds) pieces that integrate the window [ta, tb] over the samples at times `t`.

    Each piece runs from the window start or a sample time to the next sample time or the window end, and
    pieces of no length are left out; the last sample holds up to the window end. The seconds add up to
    tb - ta. The window must start at or after the first sample and end at most two sample intervals
    (t[1] - t[0]) after the last one.
    """
    t = np.asarray(t, dtype=np.float64)
    n = len(t)
    if n < 2:
        raise ValueError("IMU spans need two samples or more")
    ta, tb = float(ta), float(tb)
    if not ta < tb:
        raise ValueError(f"an IMU window must end after it starts ({ta:.3f} to {tb:.3f} s)")
    interval = float(t[1] - t[0])
    if ta < t[0] - _SPAN_TOL_S or tb > t[-1] + 2 * interval + _SPAN_TOL_S:
        raise ValueError(f"IMU samples cover {t[0]:.3f} to {t[-1]:.3f} s, not {ta:.3f} to {tb:.3f} s")
    spans: list[tuple[int, float]] = []
    j = max(int(np.searchsorted(t, ta, side="right")) - 1, 0)
    start = ta
    while start < tb:
        end = min(float(t[j + 1]), tb) if j + 1 < n else tb
        if end > start:
            spans.append((j, end - start))
            start = end
        j += 1
    return spans


@dataclass(frozen=True, eq=False)
class ImuFileData:
    """A read IMU file: its path, meta (with `format`), streams keyed by `stream_key` in file row order,
    and the content digest."""

    path: Path
    meta: dict
    streams: dict[tuple[int, int, int, int, int], ImuStream]
    digest: str


def imu_file_path(dir: Path, run_id: str) -> Path:
    return Path(dir) / f"{run_id}.npz"


def _bad(path: Path, reason: str) -> ValueError:
    return ValueError(f"{path}: bad IMU file: {reason}")


def write_imu_file(path: Path, streams: Mapping[tuple[int, int, int, int, int], ImuStream], meta: Mapping) -> str:
    """Write `streams` (one `ImuStream` per `stream_key`) and `meta` to `path`; return `imu_file_digest(path)`.

    `path` must not exist, and `meta` must have `run_id`, `fingerprint`, `imu_seed`, `noise` and
    `rate_hz`; the format tag is added. The streams' rows are joined in the order of `streams` (not
    sorted), and the arrays pass the reader's `_check_arrays` before anything is written; errors name
    `path`. The noise profile is not checked here (`ImuFile` refuses a noiseless one). The write is
    atomic: a sibling `<file name>.tmp` is written, fsynced and `os.replace`d onto `path`. The parent
    folder is made when missing.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path}: IMU file exists; remove it first to make it again")
    if any(key not in meta for key in _META_KEYS):
        raise ValueError("IMU file meta needs run_id, fingerprint, imu_seed, noise and rate_hz")

    arrays = _join(path, streams)
    _check_arrays(path, arrays)
    meta_json = json.dumps({**meta, "format": IMU_FILE_FORMAT}, sort_keys=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays, meta=np.array(meta_json))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return imu_file_digest(path)


def _join(path: Path, streams: Mapping[tuple[int, int, int, int, int], ImuStream]) -> dict[str, np.ndarray]:
    """The file arrays of `streams`: one `stream` row per key and every stream's samples joined, in the
    order of `streams`.

    Sample dtypes are kept as given; a stream whose samples are not float64, or whose `t`, `gyro` and
    `acc` do not have the shapes (n,), (n, 3) and (n, 3), raises `ValueError` naming `path`, so no
    sample lands in another stream's rows.
    """
    if not streams:
        raise _bad(path, "no streams")
    for key, s in streams.items():
        parts = {name: np.asarray(getattr(s, name)) for name in _SAMPLE_ARRAYS}
        for name, arr in parts.items():
            if arr.dtype != np.float64:
                raise _bad(path, f"stream {tuple(key)} has {name} dtype {arr.dtype}, expected float64")
        t, gyro, acc = parts["t"], parts["gyro"], parts["acc"]
        if t.ndim != 1 or gyro.shape != (len(t), 3) or acc.shape != (len(t), 3):
            raise _bad(path, f"stream {tuple(key)} has t {t.shape}, gyro {gyro.shape} and acc {acc.shape}; "
                             f"expected (n,), (n, 3) and (n, 3)")
    try:
        stream = np.asarray([tuple(key) for key in streams], dtype=np.int32)
    except (TypeError, ValueError, OverflowError) as e:
        raise _bad(path, f"the stream keys do not make int32 rows ({e})") from None
    counts = [len(np.asarray(s.t)) for s in streams.values()]
    return {
        "stream": stream,
        "stream_of": np.repeat(np.arange(len(counts), dtype=np.int32), counts),
        **{name: np.concatenate([np.asarray(getattr(s, name)) for s in streams.values()])
           for name in _SAMPLE_ARRAYS},
    }


def _check_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Refuse IMU file arrays a reader cannot trust; raises `ValueError` naming `path`.

    In order: every array's dtype and shape (`stream` (S, 5) with S >= 1, `stream_of` and `t` (N,),
    `gyro` and `acc` (N, 3)); `stream` rows distinct; each row a layout code 0 or 1, a noise seed not
    negative and a side code 0 or 1; pitch -1 in a `rig` row and not negative in a `pass` row;
    `stream_of` running 0 to S - 1 in that order, one block per stream, with no stream left empty; `t`
    strictly increasing within each stream; every sample value finite.
    """
    for name in IMU_ARRAYS:
        if arrays[name].dtype != _ARRAY_DTYPES[name]:
            raise _bad(path, f"{name} has dtype {arrays[name].dtype}, expected {np.dtype(_ARRAY_DTYPES[name])}")
    stream, stream_of = arrays["stream"], arrays["stream_of"]
    if stream.ndim != 2 or stream.shape[1] != 5:
        raise _bad(path, f"stream has shape {stream.shape}, expected (S, 5)")
    n_streams = len(stream)
    if n_streams == 0:
        raise _bad(path, "no streams")
    if stream_of.ndim != 1:
        raise _bad(path, f"stream_of has shape {stream_of.shape}, expected (N,)")
    n = len(stream_of)
    for name, shape in (("t", (n,)), ("gyro", (n, 3)), ("acc", (n, 3))):
        if arrays[name].shape != shape:
            raise _bad(path, f"{name} has shape {arrays[name].shape}, expected {shape}")

    if len(np.unique(stream, axis=0)) != n_streams:
        raise _bad(path, "stream rows must be distinct")
    layout, seed, side, pitch = stream[:, 0], stream[:, 1], stream[:, 2], stream[:, 4]
    if not (np.all(np.isin(layout, (0, 1))) and np.all(seed >= 0) and np.all(np.isin(side, (0, 1)))):
        raise _bad(path, "stream rows must be (layout, noise seed, side, aisle, pitch)")
    if np.any(pitch[layout == 0] != -1) or np.any(pitch[layout == 1] < 0):
        raise _bad(path, "a rig stream has no pitch and a pass stream has one")
    if np.any(stream_of < 0) or np.any(stream_of >= n_streams) or np.any(np.diff(stream_of) < 0):
        raise _bad(path, "stream_of must run 0 to S - 1 in order, one block per stream")
    if np.any(np.bincount(stream_of, minlength=n_streams) == 0):
        raise _bad(path, "a stream has no samples")
    within = np.diff(stream_of) == 0
    if np.any(np.diff(arrays["t"])[within] <= 0):
        raise _bad(path, "t must strictly increase within a stream")
    if not all(np.all(np.isfinite(arrays[name])) for name in _SAMPLE_ARRAYS):
        raise _bad(path, "values must be finite")


def _open_npz(path: Path) -> dict[str, np.ndarray]:
    """The arrays of the npz at `path`, as a plain dict, after every check on the file itself.

    A file `np.load` cannot read (not a zip, a corrupt compressed member, pickled data), a single
    `.npy` array under an `.npz` name, or an archive whose array set is not `IMU_ARRAYS` plus `meta`
    raises `ValueError` naming `path`. `load_imu_file` and `imu_file_digest` both read through here, so
    they refuse the same files.
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
    if set(arrays) != {*IMU_ARRAYS, "meta"}:
        raise _bad(path, f"expected arrays {{{', '.join((*IMU_ARRAYS, 'meta'))}}}, got {sorted(arrays)}")
    return arrays


def _digest(arrays: Mapping[str, np.ndarray]) -> str:
    """sha256 over the meta JSON, then each `IMU_ARRAYS` array's (name, dtype, shape, bytes)."""
    h = hashlib.sha256()
    h.update(str(arrays["meta"].item()).encode("utf-8"))
    for name in IMU_ARRAYS:
        arr = arrays[name]
        h.update(f"\n{name}:{arr.dtype.str}:{arr.shape}:".encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def load_imu_file(path: Path) -> ImuFileData:
    """Read and check an IMU file in one pass over the file; return its meta, streams and digest.

    Checks the file and its array set (`_open_npz`), that `meta` is a JSON object with the format tag
    `IMU_FILE_FORMAT`, and the arrays (`_check_arrays`, shared with `write_imu_file`). Any violation
    raises `ValueError` naming `path`. Streams are keyed by their `stream` row as a tuple of ints, in
    row order; their arrays are copies.
    """
    path = Path(path)
    arrays = _open_npz(path)
    try:
        meta = json.loads(arrays["meta"].item())
    except (ValueError, TypeError) as e:
        raise _bad(path, f"meta is not a JSON string ({e})") from None
    if not isinstance(meta, dict):
        raise _bad(path, "meta is not a JSON object")
    if meta.get("format") != IMU_FILE_FORMAT:
        raise _bad(path, f"format is {meta.get('format')!r}, expected {IMU_FILE_FORMAT!r}")
    _check_arrays(path, arrays)

    counts = np.bincount(arrays["stream_of"], minlength=len(arrays["stream"]))
    ends = np.cumsum(counts)
    starts = ends - counts
    streams: dict[tuple[int, int, int, int, int], ImuStream] = {}
    for row, a, b in zip(arrays["stream"], starts.tolist(), ends.tolist()):
        streams[tuple(int(v) for v in row)] = ImuStream(
            t=arrays["t"][a:b].copy(), gyro=arrays["gyro"][a:b].copy(), acc=arrays["acc"][a:b].copy())
    return ImuFileData(path=path, meta=meta, streams=streams, digest=_digest(arrays))


def imu_file_digest(path: Path) -> str:
    """sha256 of the IMU file's content: the meta JSON, then each `IMU_ARRAYS` array's (name, dtype,
    shape, bytes). Independent of the file's own bytes (the npz/zip framing). It refuses what
    `load_imu_file` refuses about the file and its array set (`ValueError` naming `path`), and does not
    read the format tag."""
    path = Path(path)
    return _digest(_open_npz(path))


class ImuFile:
    """One bundle's IMU file, read and checked once, on the prediction side.

    `fingerprint`, `run_id`, `digest` and `imu_seed`, when given, are checked in that order against the
    meta `fingerprint`, the meta `run_id`, the content digest and the meta `imu_seed`. Then the meta
    `noise` profile must have white noise (`gyro_n`, `acc_n`) and turn-on bias (`gyro_b0_dps`,
    `acc_b0_mg`) above 0; a missing one counts as 0. Every failure raises `ValueError` naming the path.
    Attributes: `path`, `meta` (with its `format`), `digest` (the content digest) and `noise_seeds` (the
    noise seeds of the file's streams, ascending).
    """

    def __init__(self, path: Path, *, fingerprint: str | None = None, run_id: str | None = None,
                 digest: str | None = None, imu_seed: int | None = None) -> None:
        data = load_imu_file(path)
        self.path = data.path
        self.meta = data.meta
        self.digest = data.digest
        if fingerprint is not None and self.meta.get("fingerprint") != fingerprint:
            raise ValueError(f"{self.path}: IMU file was made for bundle fingerprint "
                             f"{str(self.meta.get('fingerprint'))[:12]}, this bundle has {fingerprint[:12]}")
        if run_id is not None and self.meta.get("run_id") != run_id:
            raise ValueError(f"{self.path}: IMU file is for run {self.meta.get('run_id')}, this bundle is {run_id}")
        if digest is not None and self.digest != digest:
            raise ValueError(f"{self.path}: IMU file digest {self.digest[:16]} differs from the pinned {digest[:16]}")
        if imu_seed is not None and self.meta.get("imu_seed") != imu_seed:
            raise ValueError(f"{self.path}: IMU file was made with imu_seed {self.meta.get('imu_seed')}; "
                             f"the config seed is {imu_seed}")
        noise = self.meta.get("noise")
        noise = noise if isinstance(noise, dict) else {}
        for key in _NOISELESS_KEYS:
            value = noise.get(key, 0.0)
            if not (isinstance(value, (int, float)) and value > 0):
                raise ValueError(f"{self.path}: IMU file profile {self.meta.get('profile')} has {key} = 0; "
                                 f"a noiseless IMU is the truth")
        self._streams = data.streams
        self.noise_seeds = sorted({key[1] for key in data.streams})

    def stream(self, layout: str, noise_seed: int, side: str, aisle: int,
               pitch_deg: float | None = None) -> ImuStream:
        """The measured stream of one drive under `noise_seed` (the arguments of `stream_key`).

        A noise seed the file has no stream for, or a drive it has no stream for, raises `ValueError`
        naming the path.
        """
        key = stream_key(layout, noise_seed, side, aisle, pitch_deg)
        if key in self._streams:
            return self._streams[key]
        if key[1] not in self.noise_seeds:
            raise ValueError(f"{self.path}: no IMU stream for noise seed {noise_seed}; the file has {self.noise_seeds}")
        pitch = "" if pitch_deg is None else f", pitch {pitch_deg:g}"
        raise ValueError(f"{self.path}: no IMU stream for the {layout} drive of side {side}, aisle {aisle}{pitch}")
