"""Ground-truth sets, the summary.json self-check, the pass subsets and their table, the geometry check,
and the rig, trajectory and IMU checks of bundles rendered with body shake.

`pass_coverage` compares the G6 of the chosen passes with the G6 of every pitch's passes on the same sides
and step.

Everything here reads `SceneTruth`/`FruitGt`/`GtRecord` from an orchard_synth
bundle, so only evaluation and oracle code should import this module (the
prediction path never does; see `counting.data.formats.orchard_synth`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from counting.data.formats.orchard_synth import (
    FrameInput, FramePlan, GtRecord, OrchardLayout, PassKey, SceneTruth, SynthRun, read_depth_m,
)
from counting.evaluation.imusim import imu_check
from counting.geometry.rig import interpolate_pose, pitch_extrinsic, pose_matrix, relative_extrinsic

_GROUND_SEMANTIC = 7
_GROUND_SAMPLE_MAX = 2000
RIG_TOL_M = 1e-6      # rig and trajectory checks: position tolerance, metres
RIG_TOL_DEG = 1e-4    # rig and trajectory checks: rotation tolerance, degrees
IMU_TOL_M = 0.02      # IMU check: end position tolerance, metres


def is_positive(r: GtRecord, variant: str, count1: tuple[float, int]) -> bool:
    """Whether record `r` counts as seen, under one of the truth variants.

    `count1` (the default): `visib >= vmin and vis_px >= pmin`, read from the
    bundle's `conventions.count1`. `any`: `vis_px > 0`. `strict`: `count1`
    with `trunc` false (drops screen-edge fragments).
    """
    vmin, pmin = count1
    count1_ok = r.visib >= vmin and r.vis_px >= pmin
    if variant == "count1":
        return count1_ok
    if variant == "any":
        return r.vis_px > 0
    if variant == "strict":
        return count1_ok and not r.trunc
    raise ValueError(f"unknown truth variant: {variant!r}")


@dataclass(frozen=True)
class TruthTables:
    """Per-pass and union ground-truth fruit sets for the target-row trees, under one variant."""

    variant: str
    tree_ids: tuple[int, ...]
    tree_of: Mapping[int, int]
    g1: Mapping[PassKey, frozenset[int]]
    g6: frozenset[int]
    g6_total: Mapping[int, int]

    @property
    def target_ids(self) -> frozenset[int]:
        """Fruit ids that belong to a target-row tree (the keys of `tree_of`)."""
        return frozenset(self.tree_of)

    def per_tree(self, ids: Iterable[int]) -> dict[int, int]:
        """Count of `ids` per target-row tree, in `tree_ids` order, 0 included."""
        counts = Counter(self.tree_of[i] for i in ids if i in self.tree_of)
        return {t: counts.get(t, 0) for t in self.tree_ids}


def truth_sets(
    truth: SceneTruth,
    layout: OrchardLayout,
    passes: Mapping[PassKey, Sequence[FrameInput]],
    variant: str = "count1",
) -> TruthTables:
    """Build `TruthTables` from the frames actually received in `passes` (the thinning rule).

    `g1[p]` is the set of target-row fruit ids positive (under `variant`) in
    any frame of pass `p`; `g6` is their union over all of `passes`.
    """
    tree_of = {fid: fr.tree for fid, fr in truth.fruits.items() if fr.tree in layout.tree_ids}
    g1: dict[PassKey, frozenset[int]] = {}
    for key, frames in passes.items():
        ids = {
            r.id
            for f in frames
            for r in truth.records.get(f.idx, ())
            if r.id in tree_of and is_positive(r, variant, truth.count1)
        }
        g1[key] = frozenset(ids)
    g6 = frozenset().union(*g1.values())
    g6_total = {t: truth.totals.get(t, 0) for t in layout.tree_ids}
    return TruthTables(variant=variant, tree_ids=layout.tree_ids, tree_of=tree_of,
                       g1=g1, g6=g6, g6_total=g6_total)


def pass_coverage(
    scene: SceneTruth,
    layout: OrchardLayout,
    chosen: Mapping[PassKey, Sequence[FrameInput]],
    every: Mapping[PassKey, Sequence[FrameInput]],
    *,
    variant: str,
) -> dict:
    """How much of the G6 of `every` (every pitch's passes on the same sides and step) the passes in `chosen` see.

    `{"pitches", "g6", "g6_all", "ratio"}`: the pitches of `chosen` (ascending floats), the G6 sizes of
    `chosen` and of `every` (`truth_sets` under `variant`), and `g6 / g6_all`, None when `g6_all` is 0.
    """
    g6 = len(truth_sets(scene, layout, chosen, variant).g6)
    g6_all = len(truth_sets(scene, layout, every, variant).g6)
    return {"pitches": sorted({float(k.pitch_deg) for k in chosen}), "g6": g6, "g6_all": g6_all,
            "ratio": g6 / g6_all if g6_all else None}


class SummaryMismatch(RuntimeError):
    """The recount from `records[]` disagrees with `summary.json` (an adapter bug)."""


def check_summary(
    truth: SceneTruth, layout: OrchardLayout, passes: Mapping[PassKey, Sequence[FrameInput]]
) -> None:
    """Recount per-tree `count1`/`any` totals from `passes` and compare against `summary.json`.

    Call this only when `passes` is the whole bundle (every pass, every
    frame): the recount must then match `summary[tree][variant][side|union]`
    exactly. A mismatch means the adapter mis-parsed the bundle.
    """
    if truth.summary is None:
        raise ValueError("summary.json is missing (needed for the self-check)")
    for variant in ("count1", "any"):
        tt = truth_sets(truth, layout, passes, variant)
        for side in ("A", "B"):
            side_ids = frozenset().union(*(ids for key, ids in tt.g1.items() if key.side == side))
            side_counts = tt.per_tree(side_ids)
            for t in layout.tree_ids:
                a, b = side_counts[t], truth.summary[t][variant][side]
                if a != b:
                    raise SummaryMismatch(f"tree {t} {variant} {side}: recount {a} != summary {b}")
        union_counts = tt.per_tree(tt.g6)
        for t in layout.tree_ids:
            a, b = union_counts[t], truth.summary[t][variant]["union"]
            if a != b:
                raise SummaryMismatch(f"tree {t} {variant} union: recount {a} != summary {b}")


def pass_subsets(keys: Sequence[PassKey]) -> list[tuple[str, list[PassKey]]]:
    """Named pass subsets of `keys`: `all`, `A`, `B`, then `only {p:g}` and `drop {p:g}` per pitch, ascending.

    Each list keeps the order of `keys`; a subset may be empty (`drop 30` when 30 is the only pitch).
    """
    keys = list(keys)
    out = [("all", keys), ("A", [k for k in keys if k.side == "A"]), ("B", [k for k in keys if k.side == "B"])]
    for p in sorted({k.pitch_deg for k in keys}):
        out.append((f"only {p:g}", [k for k in keys if k.pitch_deg == p]))
        out.append((f"drop {p:g}", [k for k in keys if k.pitch_deg != p]))
    return out


def subset_table(
    truth: SceneTruth, layout: OrchardLayout, passes: Mapping[PassKey, Sequence[FrameInput]]
) -> list[dict]:
    """G6 (count1) for each subset of `pass_subsets`, against `all` and the grand total."""
    tt_all = truth_sets(truth, layout, passes, "count1")
    g6_all = len(tt_all.g6)
    total = sum(tt_all.g6_total.values())

    def row(name: str, subset: Mapping[PassKey, Sequence[FrameInput]]) -> dict:
        g6 = len(truth_sets(truth, layout, subset, "count1").g6)
        return {"S": name, "n_passes": len(subset), "G6": g6,
                "ratio_all": (g6 / g6_all) if g6_all else None,
                "ratio_total": (g6 / total) if total else None}

    return [row(name, {k: passes[k] for k in keys}) for name, keys in pass_subsets(list(passes))]


def check_geometry(run: SynthRun, truth: SceneTruth, *, stride: int = 10) -> dict:
    """Sanity-check the adapter's camera and depth reading on `run.frames[::stride]`.

    1. Unoccluded records (`visib >= 0.95`, not `trunc`, with a `bbox`): their
       ground-truth 3D center, projected, must fall inside `bbox`.
    2. Ground pixels (`semantic == 7`) with valid depth, back-projected: the
       recovered world `|z|` must be near 0 (the ground plane). `n_ground_px`
       counts the back-projected pixels.

    When no sampled frame shows a ground pixel, whatever its depth, the ground check
    is "n/a" and only the centre check decides. Ground in view with no valid depth
    fails the ground check.
    """
    n_unoccluded = n_in_bbox = n_ground_px = n_ground_seen = 0
    ground_abs_z: list[float] = []
    for f in run.frames[::stride]:
        for r in truth.records.get(f.idx, ()):
            if r.visib >= 0.95 and not r.trunc and r.bbox is not None:
                n_unoccluded += 1
                center = truth.fruits[r.id].center.reshape(1, 3)
                uv, _ = f.camera.project(center)
                u, v = float(uv[0, 0]), float(uv[0, 1])
                x0, y0, x1, y1 = r.bbox
                if x0 - 0.5 <= u <= x1 - 0.5 and y0 - 0.5 <= v <= y1 - 0.5:
                    n_in_bbox += 1

        with Image.open(run.semantic_path(f.idx)) as im:
            sem = np.array(im)
        depth_m = read_depth_m(f.depth_path)
        ground_px = sem == _GROUND_SEMANTIC
        n_ground_seen += int(np.count_nonzero(ground_px))
        rows, cols = np.nonzero(ground_px & ~np.isnan(depth_m))
        n = len(rows)
        if n == 0:
            continue
        k = max(1, -(-n // _GROUND_SAMPLE_MAX))
        rows, cols = rows[::k], cols[::k]
        uv = np.stack([cols, rows], axis=1).astype(float)
        Xw = f.camera.backproject(uv, depth_m[rows, cols].astype(float))
        ground_abs_z.extend(np.abs(Xw[:, 2]).tolist())
        n_ground_px += len(rows)

    center_in_bbox = (n_in_bbox / n_unoccluded) if n_unoccluded else None
    median = float(np.median(ground_abs_z)) if ground_abs_z else None
    if n_ground_seen == 0:
        ground = "n/a"
    elif median is not None and median <= 0.001:
        ground = "ok"
    else:
        ground = "fail"
    passed = bool(center_in_bbox == 1.0 and n_unoccluded > 0 and ground != "fail")
    return {"center_in_bbox": center_in_bbox, "n_unoccluded": n_unoccluded,
            "ground_abs_z_median_m": median, "n_ground_px": n_ground_px, "ground": ground, "passed": passed}


def _shake_plans(run: SynthRun) -> dict[int, FramePlan]:
    """The frame plans of `run`; ValueError when `run` is not rendered with body shake."""
    plans = run.plans()
    if plans is None:
        raise ValueError(f"{run.root}: not a shake bundle")
    return plans


def _pose_gap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(position gap in metres, rotation gap in degrees) between the 4x4 poses `a` and `b`."""
    pos = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rot = float(np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()))
    return pos, rot


def check_rig(run: SynthRun, *, tol_m: float = RIG_TOL_M, tol_deg: float = RIG_TOL_DEG) -> dict:
    """Check that the cameras at one position of a bundle rendered with body shake share their body and time.

    Frames are grouped by position: (side, aisle, `FramePlan.pos_k`), the render position, also on a
    bundle that is not a sequence. At each position the frame with the smallest pitch r is the
    reference; every other frame, at pitch q, is a pair. Its pose must match
    `pose_matrix(cam_r) @ relative_extrinsic(r, q)` within `tol_m` and `tol_deg`, and its time
    `FramePlan.t` must equal the reference time exactly. With no pairs the maxima are 0.0.
    Raises ValueError when `run` is not rendered with body shake.
    """
    plans = _shake_plans(run)
    at: dict[tuple[str, int, int], list[FrameInput]] = {}
    for f in run.frames:
        at.setdefault((f.pass_key.side, f.pass_key.aisle, plans[f.idx].pos_k), []).append(f)
    pairs, max_pos, max_rot, max_dt = 0, 0.0, 0.0, 0.0
    for frames in at.values():
        ref = min(frames, key=lambda f: f.pass_key.pitch_deg)
        r, ref_pose, ref_t = ref.pass_key.pitch_deg, pose_matrix(ref.camera), plans[ref.idx].t
        for f in frames:
            if f is ref:
                continue
            pos, rot = _pose_gap(pose_matrix(f.camera), ref_pose @ relative_extrinsic(r, f.pass_key.pitch_deg))
            dt = abs(plans[f.idx].t - ref_t)
            pairs += 1
            max_pos, max_rot, max_dt = max(max_pos, pos), max(max_rot, rot), max(max_dt, dt)
    passed = max_pos <= tol_m and max_rot <= tol_deg and max_dt == 0.0
    return {"passed": passed, "positions": len(at), "pairs": pairs,
            "max_pos_m": max_pos, "max_rot_deg": max_rot, "max_dt_s": max_dt}


def check_trajectory(run: SynthRun, *, tol_m: float = RIG_TOL_M, tol_deg: float = RIG_TOL_DEG) -> dict:
    """Check that the recorded body trajectory of a bundle rendered with body shake gives each frame pose.

    For every frame, at pitch q on one side, the body pose at the frame time `FramePlan.t`
    (`interpolate_pose` over that side's `Trajectory.body_poses()`) times `pitch_extrinsic(q)` must
    match the frame pose within `tol_m` and `tol_deg`.
    Raises ValueError when `run` is not rendered with body shake.
    """
    plans = _shake_plans(run)
    bodies: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    max_pos, max_rot = 0.0, 0.0
    for f in run.frames:
        side = f.pass_key.side
        if side not in bodies:
            tr = run.trajectory(side)
            bodies[side] = (tr.t, tr.body_poses())
        body = interpolate_pose(*bodies[side], plans[f.idx].t)
        pos, rot = _pose_gap(pose_matrix(f.camera), body @ pitch_extrinsic(f.pass_key.pitch_deg))
        max_pos, max_rot = max(max_pos, pos), max(max_rot, rot)
    passed = max_pos <= tol_m and max_rot <= tol_deg
    return {"passed": passed, "frames": len(run.frames), "max_pos_m": max_pos, "max_rot_deg": max_rot}


def check_imu(run: SynthRun, *, tol_m: float = IMU_TOL_M) -> dict:
    """Check that noiseless IMU samples of the recorded body trajectory integrate to its end pose.

    Each side of a bundle rendered with body shake is integrated from its first frame time to its last
    (`imusim.imu_check`); the check passes when the largest end position error is within `tol_m`.
    Raises ValueError when `run` is not rendered with body shake.
    """
    res = imu_check(run)
    return {**res, "passed": res["max_end_m"] <= tol_m}
