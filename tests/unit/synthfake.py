"""Synthetic fixtures for unit tests: fake Blender poses and orchard_synth-shaped data.

This module never imports `counting.geometry` (or any eval/detection code): it
builds raw numbers that the modules under test then consume. `write_run` writes
a small render bundle; `fake_cfg` and `load_splits` read the repo's eval
configs; `python_c` runs a snippet in a fresh interpreter, for checks on import
side effects.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from math import ceil, cos, floor, radians, sin, tan
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from counting.config.eval_schema import EvalAppConfig, SplitsConfig
from counting.config.loader import load_yaml_model

ROOT = Path(__file__).resolve().parents[2]
EVAL_CONFIGS = ROOT / "configs" / "eval"


def look_matrix(eye, forward, up=(0, 0, 1)) -> list[list[float]]:
    """Build a Blender camera-to-world matrix that looks along `forward`.

    The camera's local -Z axis points along the normalized `forward`
    direction, local +Y is `up`-ward, and local +X completes a right-handed
    basis.
    """
    f = np.asarray(forward, dtype=float)
    f = f / np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, dtype=float))
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[:3, 0] = r
    m[:3, 1] = u
    m[:3, 2] = -f
    m[:3, 3] = eye
    return m.tolist()


def robot_matrix(
    x: float, row_y: float, side: str, pitch_deg: float, distance: float, height: float
) -> list[list[float]]:
    """Build a Blender camera-to-world matrix for the robot's pass geometry.

    The robot drives along a row at world y = `row_y`, offset by `distance`
    to one `side` ("A" or "B") of it, tilted up by `pitch_deg`. Side "A" sits
    on the +y side and looks toward -y; side "B" sits on the -y side and
    looks toward +y. Both tilt up by `pitch_deg`.
    """
    s = 1.0 if side == "A" else -1.0
    p = radians(pitch_deg)
    eye = (x, row_y + s * distance, height)
    forward = (0.0, -s * cos(p), sin(p))
    return look_matrix(eye, forward)


def python_c(code: str) -> str:
    """Run `python -c code` in a fresh interpreter with the repo's `src` on PYTHONPATH; return stdout."""
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    if out.returncode != 0:
        raise AssertionError(f"python -c exited with {out.returncode}:\n{out.stderr}")
    return out.stdout


def fake_cfg(run_dir: Path, crop: str, *sets: str) -> EvalAppConfig:
    """`configs/eval/{crop}_dev.yaml` with the `key.path=value` overrides `sets`, scoring only `run_dir`."""
    cfg = load_yaml_model(EVAL_CONFIGS / f"{crop}_dev.yaml", EvalAppConfig, overrides=sets)
    return cfg.model_copy(update={"runs": [str(run_dir)]})


def load_splits() -> SplitsConfig:
    """The repo's `configs/eval/splits.yaml`."""
    return load_yaml_model(EVAL_CONFIGS / "splits.yaml", SplitsConfig)


# --- fake orchard_synth render bundle ------------------------------------------------------

HFOV_DEG = 69.0
COUNT1 = {"visib_min": 0.25, "px_min": 20}
SIDES = ("A", "B")
_NEAR_M = 0.05
_MAX_DEPTH_M = 65.535
_LEAF_FRAC = 0.6
_LEAF_DZ_M = 0.30
_SEM = {"background": 0, "leaf": 1, "branch": 2, "fruit": 3, "bag": 4, "structure": 5, "net": 6,
        "ground": 7, "drip": 8}


@dataclass(frozen=True)
class _Scene:
    """Per-crop constants of the fake orchard: spacing, robot pose, fruit layout."""

    tree_dx: float
    row_spacing: float
    distance: float
    height: float
    preset: str
    pitches: tuple[float, ...]
    block: int
    form: str
    offsets: tuple[tuple[float, float, float], ...]   # (dx, dy, z) from the trunk base
    diameter: float
    bagged_j: frozenset[int]
    bag_w: float
    bag_h: float
    bag_dz: float


_SCENES = {
    "apple": _Scene(
        tree_dx=1.25, row_spacing=3.75, distance=1.75, height=0.9, preset="apple_robot_side",
        pitches=(10.0, 30.0), block=200, form="spindle",
        offsets=((-0.30, 0.20, 1.35), (0.30, 0.25, 1.55), (-0.15, -0.20, 1.45), (0.20, -0.30, 1.60)),
        diameter=0.07, bagged_j=frozenset({0}), bag_w=0.10, bag_h=0.12, bag_dz=0.02),
    "pear": _Scene(
        tree_dx=3.0, row_spacing=4.25, distance=2.125, height=0.5, preset="pear_robot_up",
        pitches=(45.0, 70.0), block=400, form="y_trellis",
        offsets=((-0.9, 1.25, 1.80), (0.0, 1.35, 1.80), (0.9, 1.25, 1.85), (-0.6, -1.30, 1.80),
                 (0.6, -1.35, 1.85)),
        diameter=0.053, bagged_j=frozenset(range(5)), bag_w=0.19, bag_h=0.22, bag_dz=-0.03),
}


def _r6(xs) -> list[float]:
    return [round(float(x), 6) for x in xs]


def _build_trees(sc: _Scene, n_rows: int, trees_per_row: int):
    """Spec trees plus the counted objects (the bag if bagged, else the fruit), in id order."""
    spec_trees, objects = [], []
    for r in range(n_rows):
        for k in range(trees_per_row):
            tid = r * trees_per_row + k
            base = _r6((k * sc.tree_dx, r * sc.row_spacing, 0.0))
            fruits, bags = [], []
            for j, (ox, oy, oz) in enumerate(sc.offsets):
                fid = sc.block * (tid + 1) + j
                center = _r6((base[0] + ox, base[1] + oy, oz))
                bagged = j in sc.bagged_j
                fruits.append([fid, tid, center, sc.diameter, [0.0, 0.0, 0.0], center, bagged])
                if bagged:
                    bag_center = _r6((center[0], center[1], center[2] + sc.bag_dz))
                    bags.append([fid, bag_center, sc.bag_w, sc.bag_h, [0.0, 0.0, 0.0], 0])
                    objects.append((fid, tid, bag_center, sc.bag_w, True))
                else:
                    objects.append((fid, tid, center, sc.diameter, False))
            spec_trees.append({"id": tid, "form": sc.form, "base": base, "fruits": fruits, "bags": bags,
                               "segments": [], "leaves": {}})
    return spec_trees, objects


def _ground(matrix, width: int, height: int, fx: float) -> tuple[np.ndarray, np.ndarray]:
    """Plane depth (mm) and semantic image of the z = 0 ground seen through each pixel-center ray."""
    m = np.asarray(matrix, dtype=float)
    xb = (np.arange(width) + 0.5 - width / 2) / fx
    yb = -(np.arange(height) + 0.5 - height / 2) / fx
    # World z of the ray direction R @ (xb, yb, -1); the plane depth equals the ray parameter t.
    dz = m[2, 0] * xb[None, :] + m[2, 1] * yb[:, None] - m[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -m[2, 3] / dz
    hit = (dz < 0) & (t <= _MAX_DEPTH_M)
    depth = np.zeros((height, width), dtype=np.uint16)
    depth[hit] = np.rint(t[hit] * 1000.0)
    sem = np.zeros((height, width), dtype=np.uint8)
    sem[hit] = _SEM["ground"]
    return depth, sem


def _records(matrix, obj, width: int, height: int, fx: float, dim: Mapping[int, float]):
    """Records for one camera, plus the boxes to paint into depth16/semantic.

    Follows the real rule: an object is recorded whenever its square reaches the
    screen, even when its projected center is off-screen.
    """
    ids, trees, centers, sizes, is_bag = obj
    m = np.asarray(matrix, dtype=float)
    pb = (centers - m[:3, 3]) @ m[:3, :3]                   # rows are R^T (p - t), Blender camera frame
    depth = -pb[:, 2]
    near = np.flatnonzero(depth > _NEAR_M)
    d = depth[near]
    u = width / 2 + fx * pb[near, 0] / d                    # pixel-corner coordinates
    v = height / 2 - fx * pb[near, 1] / d
    rad = fx * (sizes[near] / 2) / d
    box = np.stack([np.floor(u - rad), np.floor(v - rad), np.ceil(u + rad), np.ceil(v + rad)], axis=1)
    clip = np.clip(box, 0, [width, height, width, height])
    area = np.clip(clip[:, 2] - clip[:, 0], 0, None) * np.clip(clip[:, 3] - clip[:, 1], 0, None)
    trunc = (box[:, 0] < 0) | (box[:, 1] < 0) | (box[:, 2] > width) | (box[:, 3] > height)
    records, paints = [], []
    for i in np.flatnonzero(area > 0):
        o = int(near[i])
        oid, full = int(ids[o]), int(area[i])
        vis = int(round(full * dim.get(oid, 1.0)))
        bbox = [int(c) for c in clip[i]]
        records.append({"id": oid, "tree": int(trees[o]), "vis_px": vis, "full_px": full, "visib": vis / full,
                        "bbox": bbox if vis > 0 else None, "trunc": bool(trunc[i]),
                        "depth_med": float(d[i]) if vis > 0 else None})
        paints.append((float(d[i]), oid, bbox, bool(is_bag[o])))
    return records, paints


def _summary(frames: list[tuple[str, list[dict]]], spec_trees: list[dict]) -> dict:
    """Per-tree, per-side 'any' (vis_px > 0) and count1 counts plus their union, like summary.json."""
    seen = {s: {} for s in SIDES}
    good = {s: {} for s in SIDES}
    for side, records in frames:
        for r in records:
            if r["vis_px"] > 0:
                seen[side].setdefault(r["tree"], set()).add(r["id"])
            if r["visib"] >= COUNT1["visib_min"] and r["vis_px"] >= COUNT1["px_min"]:
                good[side].setdefault(r["tree"], set()).add(r["id"])

    def counts(by_side, tid):
        sets = {s: by_side[s].get(tid, set()) for s in SIDES}
        return {**{s: len(v) for s, v in sets.items()}, "union": len(set().union(*sets.values()))}

    trees = {str(t["id"]): {"total": len(t["fruits"]), "bagged": sum(bool(f[6]) for f in t["fruits"]),
                            "any": counts(seen, t["id"]), "count1": counts(good, t["id"])} for t in spec_trees}
    return {"summary": {"trees": trees}, "realism": {}}


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write_run(
    root: Path,
    *,
    crop: str = "apple",
    pitches: Sequence[float] | None = None,
    step: float = 0.15,
    size: tuple[int, int] = (640, 360),
    seed: int = 5,
    n_rows: int = 3,
    trees_per_row: int = 3,
    max_frames: int | None = None,
    dim: Mapping[int, float] | None = None,
    step_param: bool = True,
    occlude: Mapping[int, Iterable[int]] | None = None,
) -> Path:
    """Write a small orchard_synth-shaped render bundle under `root` (created if needed) and return it.

    Each counted object (the bag if bagged, else the fruit) is a square of side
    `fx * size / depth` around its projected center, where size is the fruit
    diameter or the bag width. `full_px` is the
    on-screen area of that square and `dim[id]` scales `vis_px` to fake
    occlusion. depth16/semantic hold the ground plane and the squares painted far
    to near; RGB and instance images are not written.

    `occlude` paints a nearer patch into depth16 only, so geometry checks that read depth at
    fruit centres fail on such a bundle.
    """
    sc = _SCENES[crop]
    pitches = tuple(float(p) for p in (sc.pitches if pitches is None else pitches))
    dim = dict(dim or {})
    occlude = {int(oid): frozenset(int(i) for i in idxs) for oid, idxs in (occlude or {}).items()}
    width, height = size
    fx = (width / 2) / tan(radians(HFOV_DEG / 2))
    target_row = n_rows // 2
    row_ys = _r6(r * sc.row_spacing for r in range(n_rows))
    spec_trees, objects = _build_trees(sc, n_rows, trees_per_row)
    obj = (np.array([o[0] for o in objects]), np.array([o[1] for o in objects]),
           np.array([o[2] for o in objects], dtype=float), np.array([o[3] for o in objects], dtype=float),
           np.array([o[4] for o in objects]))

    x0, x1 = -1.0, (trees_per_row - 1) * sc.tree_dx + 1.0
    xs = [x0 + i * step for i in range(floor((x1 - x0) / step + 1e-9) + 1)]
    poses = [(side, pitch, robot_matrix(x, row_ys[target_row], side, pitch, sc.distance, sc.height))
             for pitch in pitches for side, order in (("A", xs), ("B", xs[::-1])) for x in order]
    chosen = range(len(poses)) if max_frames is None else \
        np.unique(np.round(np.linspace(0, len(poses) - 1, max_frames))).astype(int).tolist()

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    grounds: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    written = []
    for idx, frame_no in enumerate(chosen):
        side, pitch, matrix = poses[frame_no]
        records, paints = _records(matrix, obj, width, height, fx, dim)
        if (side, pitch) not in grounds:              # R and camera height are fixed within a pass
            grounds[(side, pitch)] = _ground(matrix, width, height, fx)
        depth, sem = (a.copy() for a in grounds[(side, pitch)])
        for d, _, (bx0, by0, bx1, by1), is_bag in sorted(paints, key=lambda p: (-p[0], p[1])):
            depth[by0:by1, bx0:bx1] = round(d * 1000.0)
            sem[by0:by1, bx0:bx1] = _SEM["bag"] if is_bag else _SEM["fruit"]
        for r in records:
            if r["bbox"] is None or idx not in occlude.get(r["id"], ()):
                continue
            bx0, by0, bx1, by1 = r["bbox"]
            w, h = bx1 - bx0, by1 - by0
            a = (1 - _LEAF_FRAC) / 2
            px0, px1 = floor(bx0 + a * w), ceil(bx1 - a * w)
            py0, py1 = floor(by0 + a * h), ceil(by1 - a * h)
            leaf = round((r["depth_med"] - _LEAF_DZ_M) * 1000)
            region = depth[py0:py1, px0:px1]
            region[(region == 0) | (region > leaf)] = leaf
        Image.fromarray(depth).save(root / f"{idx:06d}_depth16.png", compress_level=1)
        Image.fromarray(sem).save(root / f"{idx:06d}_semantic.png", compress_level=1)
        pose = {"name": f"{sc.preset}_{pitch:g}", "matrix_world": matrix, "width": width, "height": height,
                "hfov_deg": HFOV_DEG, "side": side, "frame": frame_no}
        _write_json(root / f"{idx:06d}_frame.json",
                    {"idx": idx, "frame": frame_no, "side": side, "pose": pose, "records": records})
        written.append((side, records))

    manifest = {
        "args": {"crop": crop, "preset": sc.preset, "seed": seed, "samples": 4, "rows": n_rows,
                 "trees_per_row": trees_per_row, "target_row": target_row, "time": "day", "size": [width, height],
                 "max_frames": max_frames, "cam_param": [f"step={step}"] if step_param else [], "tree_param": []},
        "spec": {"seed": seed, "crop": crop, "form": sc.form, "row_axis": [1, 0], "row_ys": row_ys,
                 "trees": spec_trees, "structures": []},
        "camera": {"K": [[fx, 0.0, width / 2], [0.0, fx, height / 2], [0.0, 0.0, 1.0]], "width": width,
                   "height": height, "hfov_deg": HFOV_DEG, "preset": sc.preset},
        "trees": [{"id": t["id"], "base": t["base"], "total": len(t["fruits"]),
                   "bagged": sum(bool(f[6]) for f in t["fruits"])} for t in spec_trees],
        "target_row_trees": [t["id"] for t in spec_trees if t["id"] // trees_per_row == target_row],
        "conventions": {"count1": dict(COUNT1), "depth16": "mm, 0 = 반환 없음, 평면 깊이",
                        "instances": "uint16 과실 번호(봉지는 안에 든 과실 번호), 0 = 그 밖", "semantic": dict(_SEM)},
        "unverified": [],
    }
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "summary.json", _summary(written, spec_trees))
    return root
