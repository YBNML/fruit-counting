"""orchard_synth render bundles: typed prediction inputs, kept apart from ground truth.

`SynthRun.open` reads only the bundle's JSON and PNG outputs. What the
prediction path may see is in `FrameInput` (pose, intrinsics, image paths) and
`OrchardLayout` (row and tree positions); records, fruit positions and tree
totals stay in `SceneTruth`, which only scoring and oracle code should read.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from counting.geometry.camera import Camera, pitch_from_blender

_TOL = 1e-6
_PITCH_TOL_DEG = 0.01
_PLANAR_DEPTH = "평면 깊이"   # marker orchard_synth writes into conventions.depth16 ("plane depth")
_U16_MODES = ("I;16", "I;16B", "I")
_COMMIT_SIDECAR = "orchard_synth.commit"
# Errors from indexing a JSON tree that lacks a field or has the wrong shape.
_FIELD_ERRORS = (KeyError, IndexError, TypeError, AttributeError)


@dataclass(frozen=True)
class PassKey:
    pitch_deg: float
    side: str
    aisle: int


@dataclass(frozen=True, eq=False)
class FrameInput:
    """Everything the prediction path receives for one frame. `pos_k` is -1 outside sequences."""

    run_id: str
    idx: int
    pass_key: PassKey
    pos_k: int
    camera: Camera
    cam_x: float
    rgb_path: Path
    depth_path: Path


@dataclass(frozen=True)
class OrchardLayout:
    """Row and tree ledger. `tree_ids`/`tree_x` cover the target row in ascending x."""

    crop: str
    row_ys: tuple[float, ...]
    target_row: int
    row_spacing: float
    tree_ids: tuple[int, ...]
    tree_x: tuple[float, ...]

    @property
    def row_y(self) -> float:
        return self.row_ys[self.target_row]


@dataclass(frozen=True)
class GtRecord:
    id: int
    tree: int
    vis_px: int
    full_px: int
    visib: float
    bbox: tuple[int, int, int, int] | None
    trunc: bool
    depth_med: float | None


@dataclass(frozen=True, eq=False)
class FruitGt:
    """One fruit. `center` is the bag center when the fruit is bagged (that is what gets counted)."""

    id: int
    tree: int
    center: np.ndarray
    diameter: float
    bagged: bool


@dataclass(frozen=True)
class SceneTruth:
    """Ground truth of a bundle: per-frame records (every frame idx has an entry), fruits, totals."""

    records: dict[int, tuple[GtRecord, ...]]
    fruits: dict[int, FruitGt]
    totals: dict[int, int]
    count1: tuple[float, int]
    summary: dict[int, dict] | None


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    crop: str
    preset: str
    seed: int
    samples: int
    time: str
    base_step_m: float | None
    is_sequence: bool
    synth_commit: str | None
    fingerprint: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _combine(named_digests: list[tuple[str, str]]) -> str:
    return _sha256(json.dumps(named_digests, separators=(",", ":")).encode("utf-8"))


def _fingerprint_paths(root: Path) -> list[Path]:
    paths = [root / "manifest.json"]
    if (root / "summary.json").is_file():
        paths.append(root / "summary.json")
    return paths + sorted(root.glob("*_frame.json"))


def run_fingerprint(root: Path) -> str:
    """sha256 over (name, sha256) of manifest.json, summary.json (if any) and every *_frame.json.

    RGB and depth images are left out (RGB bytes are not reproducible); see
    `SynthRun.depth_fingerprint` for the depth images.
    """
    return _combine([(p.name, _sha256(p.read_bytes())) for p in _fingerprint_paths(Path(root))])


def _field_error(path: Path, e: Exception) -> ValueError:
    """A `ValueError` naming `path` for a `_FIELD_ERRORS` exception raised while reading its JSON."""
    if isinstance(e, KeyError):
        return ValueError(f"{path}: missing field {e.args[0]!r}")
    return ValueError(f"{path}: malformed field ({type(e).__name__}: {e})")


def read_png_u16(path: Path) -> np.ndarray:
    """Read a 16-bit grayscale PNG as uint16 without any 8-bit conversion.

    A missing, corrupt or truncated file raises `ValueError` naming `path`.
    """
    try:
        with Image.open(path) as im:
            mode = im.mode
            if mode in _U16_MODES:
                return np.array(im, dtype=np.uint16)
    except (OSError, SyntaxError) as e:
        raise ValueError(f"{path}: unreadable PNG ({e})") from None
    raise ValueError(f"{path}: expected a 16-bit grayscale PNG, got mode {mode}")


def read_depth_m(path: Path) -> np.ndarray:
    """Read a depth16 PNG (mm, plane depth) as float32 metres; 0 (no return) becomes NaN."""
    raw = read_png_u16(path)
    z = raw.astype(np.float32) / np.float32(1000.0)
    z[raw == 0] = np.nan
    return z


def _parse_fruits(spec_trees: list[dict], mpath: Path) -> dict[int, FruitGt]:
    bag_centers: dict[int, np.ndarray] = {}
    for t in spec_trees:
        for b in t["bags"]:
            if len(b) != 6:
                raise ValueError(f"{mpath}: bag tuple has length {len(b)}, expected 6")
            bag_centers[int(b[0])] = np.asarray(b[1], dtype=float)
    fruits: dict[int, FruitGt] = {}
    for t in spec_trees:
        for f in t["fruits"]:
            if len(f) != 7:
                raise ValueError(f"{mpath}: fruit tuple has length {len(f)}, expected 7")
            fid, bagged = int(f[0]), bool(f[6])
            if bagged and fid not in bag_centers:
                raise ValueError(f"{mpath}: bagged fruit {fid} has no bag tuple")
            center = bag_centers[fid] if bagged else np.asarray(f[2], dtype=float)
            center.flags.writeable = False
            fruits[fid] = FruitGt(id=fid, tree=int(f[1]), center=center, diameter=float(f[3]), bagged=bagged)
    return fruits


def _parse_cam_param(items, mpath: Path) -> dict:
    params = {}
    for item in items or []:
        name, _, value = item.partition("=")
        try:
            params[name] = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError(f"{mpath}: cam_param item {item!r} is not NAME=JSON") from None
    return params


def _record(r: dict) -> GtRecord:
    bbox, dmed = r["bbox"], r["depth_med"]
    return GtRecord(id=int(r["id"]), tree=int(r["tree"]), vis_px=int(r["vis_px"]), full_px=int(r["full_px"]),
                    visib=float(r["visib"]), bbox=None if bbox is None else tuple(int(v) for v in bbox),
                    trunc=bool(r["trunc"]), depth_med=None if dmed is None else float(dmed))


def _parse_layout(manifest: dict, mpath: Path) -> OrchardLayout:
    """The row and tree layout. `spec.row_ys` must hold at least 2 rows, ascending and evenly spaced."""
    args = manifest["args"]
    row_ys = tuple(float(y) for y in manifest["spec"]["row_ys"])
    if len(row_ys) < 2:
        raise ValueError(f"{mpath}: spec.row_ys needs at least 2 rows, got {len(row_ys)}")
    gaps = np.diff(row_ys)
    if not np.all(gaps > 0):
        raise ValueError(f"{mpath}: spec.row_ys is not strictly ascending: {list(row_ys)}")
    if np.max(np.abs(gaps - gaps[0])) > _TOL * gaps[0]:
        raise ValueError(f"{mpath}: spec.row_ys is not evenly spaced: {list(row_ys)}")
    base_x = {int(t["id"]): float(t["base"][0]) for t in manifest["trees"]}
    target = sorted((int(i) for i in manifest["target_row_trees"]), key=lambda i: (base_x[i], i))
    return OrchardLayout(crop=str(args["crop"]), row_ys=row_ys, target_row=int(args["target_row"]),
                         row_spacing=row_ys[1] - row_ys[0], tree_ids=tuple(target),
                         tree_x=tuple(base_x[i] for i in target))


def _parse_frame(p: Path, f: dict, fruits: dict[int, FruitGt], layout: OrchardLayout):
    """(idx, pass key, camera, camera x, records) of one *_frame.json, after its format checks."""
    idx = int(f["idx"])
    if p.name != f"{idx:06d}_frame.json":
        raise ValueError(f"{p}: frame file name does not match idx")
    pose, matrix = f["pose"], f["pose"]["matrix_world"]
    try:
        pitch = float(pose["name"].rpartition("_")[2])
    except ValueError:
        raise ValueError(f"{p}: pose name {pose['name']!r} has no pitch suffix") from None
    if abs(pitch - pitch_from_blender(matrix)) > _PITCH_TOL_DEG:
        raise ValueError(f"{p}: pitch in pose name does not match matrix_world")
    camera = Camera.from_blender(matrix, int(pose["width"]), int(pose["height"]), float(pose["hfov_deg"]))
    s = layout.row_spacing
    aisle = round((float(matrix[1][3]) - layout.row_ys[0] + s / 2) / s)
    recs = tuple(_record(r) for r in f["records"])
    for r in recs:
        if r.id not in fruits or fruits[r.id].tree != r.tree:
            raise ValueError(f"{p}: record tree != fruit tree for id {r.id}")
    return idx, PassKey(pitch, str(f["side"]), aisle), camera, float(matrix[0][3]), recs


def _parse_summary(data: dict, spath: Path) -> dict[int, dict]:
    """summary.json's per-tree counts (`summary.trees`), keyed by tree id."""
    summary = data.get("summary") if isinstance(data, dict) else None
    trees = summary.get("trees") if isinstance(summary, dict) else None
    if not isinstance(trees, dict):
        raise ValueError(f"{spath}: no 'summary.trees' mapping")
    try:
        return {int(k): v for k, v in trees.items()}
    except ValueError:
        raise ValueError(f"{spath}: summary.trees has a non-integer tree id") from None


def _measured_step(xs_by_pass: dict[PassKey, list[float]]) -> float | None:
    """The common camera step of all passes, or None when the frames are not an even sequence.

    Side A must move toward +x and side B toward -x, with every consecutive
    |dx| equal within 1e-6, and all passes must share the same step.
    """
    steps = []
    for key, xs in xs_by_pass.items():
        direction = {"A": 1.0, "B": -1.0}.get(key.side)
        if direction is None or len(xs) < 2:
            return None
        dx = np.diff(np.asarray(xs, dtype=float)) * direction
        if not np.all(dx > 0) or dx.max() - dx.min() > _TOL:
            return None
        steps.append(float(dx.mean()))
    if max(steps) - min(steps) > _TOL:
        return None
    return float(np.mean(steps))


class SynthRun:
    """One orchard_synth render bundle (a directory of manifest, summary and per-frame files)."""

    def __init__(self, root: Path, meta: RunMeta, layout: OrchardLayout, frames: tuple[FrameInput, ...],
                 *, records: dict[int, tuple[GtRecord, ...]], fruits: dict[int, FruitGt],
                 totals: dict[int, int], count1: tuple[float, int], summary: dict[int, dict] | None) -> None:
        self.root = root
        self.meta = meta
        self.layout = layout
        self.frames = frames
        self._records = records
        self._fruits = fruits
        self._totals = totals
        self._count1 = count1
        self._summary = summary
        self._truth: SceneTruth | None = None

    @classmethod
    def open(cls, path: str | Path) -> SynthRun:
        """Read and validate a bundle. Format errors raise ValueError naming the file."""
        root = Path(path).expanduser()
        mpath = root / "manifest.json"
        if not mpath.is_file():
            raise ValueError(f"{root}: no manifest.json")
        paths = _fingerprint_paths(root)
        frame_paths = [p for p in paths if p.name.endswith("_frame.json")]
        if not frame_paths:
            raise ValueError(f"{root}: no *_frame.json files")

        # Read every fingerprinted file once: hash its bytes and parse its JSON.
        digests, loaded = [], {}
        for p in paths:
            data = p.read_bytes()
            digests.append((p.name, _sha256(data)))
            try:
                loaded[p] = json.loads(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}: invalid JSON ({e})") from None
        fingerprint = _combine(digests)

        spath = root / "summary.json"
        summary = _parse_summary(loaded.pop(spath), spath) if spath in loaded else None
        manifest = loaded.pop(mpath)
        try:
            args = manifest["args"]
            conventions = manifest.get("conventions") or {}
            if _PLANAR_DEPTH not in str(conventions.get("depth16", "")):
                raise ValueError(f"{mpath}: conventions.depth16 is not planar depth")
            if "count1" not in conventions:
                raise ValueError(f"{mpath}: conventions.count1 is missing")
            count1 = (float(conventions["count1"]["visib_min"]), int(conventions["count1"]["px_min"]))
            fruits = _parse_fruits(manifest["spec"]["trees"], mpath)
            cam_param = _parse_cam_param(args.get("cam_param"), mpath)
            layout = _parse_layout(manifest, mpath)
            totals = {int(t["id"]): int(t["total"]) for t in manifest["trees"]}
            crop, seed = str(args["crop"]), int(args["seed"])
            preset, samples, time_of_day = str(args["preset"]), int(args["samples"]), str(args["time"])
            max_frames = args.get("max_frames")
        except _FIELD_ERRORS as e:
            raise _field_error(mpath, e) from None
        del manifest   # drops the leaves and everything else not kept above

        parsed = []
        for p in frame_paths:
            try:
                parsed.append(_parse_frame(p, loaded[p], fruits, layout))
            except _FIELD_ERRORS as e:
                raise _field_error(p, e) from None
        parsed.sort(key=lambda e: e[0])

        xs_by_pass: dict[PassKey, list[float]] = {}
        for _, key, _, cam_x, _ in parsed:
            xs_by_pass.setdefault(key, []).append(cam_x)
        step = None if max_frames is not None else _measured_step(xs_by_pass)
        if step is not None and "step" in cam_param and abs(step - float(cam_param["step"])) > _TOL:
            raise ValueError(f"{mpath}: measured step != cam_param step ({step:.9g} vs {cam_param['step']})")
        x_min = {key: min(xs) for key, xs in xs_by_pass.items()}

        sidecar = root / _COMMIT_SIDECAR
        lines = sidecar.read_text(encoding="utf-8").splitlines() if sidecar.is_file() else []
        commit = lines[0].strip() if lines and lines[0].strip() else None
        meta = RunMeta(run_id=f"{crop}_s{seed}_{fingerprint[:8]}", crop=crop, preset=preset,
                       seed=seed, samples=samples, time=time_of_day, base_step_m=step,
                       is_sequence=step is not None, synth_commit=commit, fingerprint=fingerprint)

        frames = tuple(
            FrameInput(run_id=meta.run_id, idx=idx, pass_key=key,
                       pos_k=-1 if step is None else round((cam_x - x_min[key]) / step),
                       camera=camera, cam_x=cam_x, rgb_path=root / f"{idx:06d}_rgb.png",
                       depth_path=root / f"{idx:06d}_depth16.png")
            for idx, key, camera, cam_x, _ in parsed)
        records = {idx: recs for idx, _, _, _, recs in parsed}
        return cls(root, meta, layout, frames, records=records, fruits=fruits, totals=totals, count1=count1,
                   summary=summary)

    def passes(self) -> dict[PassKey, list[FrameInput]]:
        """Frames grouped by pass, passes in order of first appearance, frames in idx order."""
        out: dict[PassKey, list[FrameInput]] = {}
        for f in self.frames:
            out.setdefault(f.pass_key, []).append(f)
        return out

    def truth(self) -> SceneTruth:
        """Ground truth of the bundle (built once, then cached)."""
        if self._truth is None:
            self._truth = SceneTruth(records=self._records, fruits=self._fruits, totals=self._totals,
                                     count1=self._count1, summary=self._summary)
        return self._truth

    def semantic_path(self, idx: int) -> Path:
        return self.root / f"{idx:06d}_semantic.png"

    def depth_fingerprint(self) -> str:
        """sha256 over (name, sha256) of every *_depth16.png, computed like `run_fingerprint`."""
        return _combine([(p.name, _sha256(p.read_bytes())) for p in sorted(self.root.glob("*_depth16.png"))])


def select_frames(
    frames: Sequence[FrameInput],
    *,
    base_step_m: float | None,
    pitches: Sequence[float] | None = None,
    sides: Sequence[str] | None = None,
    step_m: float | None = None,
) -> list[FrameInput]:
    """Keep frames of the given pitches and sides, thinned to every `step_m / base_step_m`-th position."""
    frames = list(frames)
    where = f"{frames[0].depth_path.parent}: " if frames else ""
    have_p = sorted({f.pass_key.pitch_deg for f in frames})
    have_s = sorted({f.pass_key.side for f in frames})
    available = f"available pitches: {have_p}, sides: {have_s}"
    for p in pitches or ():
        if not any(abs(p - q) < _TOL for q in have_p):
            raise ValueError(f"{where}pitch {p:g} is not in the bundle; {available}")
    for s in sides or ():
        if s not in have_s:
            raise ValueError(f"{where}side {s!r} is not in the bundle; {available}")
    kept = [f for f in frames
            if (pitches is None or any(abs(f.pass_key.pitch_deg - p) < _TOL for p in pitches))
            and (sides is None or f.pass_key.side in sides)]
    if step_m is None:
        return kept
    if base_step_m is None:
        raise ValueError(f"{where}step_m={step_m:g} needs frame positions, but this is not a sequence bundle")
    m = step_m / base_step_m
    k = round(m)
    if k < 1 or abs(m - k) > _TOL:
        raise ValueError(f"{where}step_m is not a multiple of the base step ({step_m:g} / {base_step_m:g})")
    return [f for f in kept if f.pos_k % k == 0]
