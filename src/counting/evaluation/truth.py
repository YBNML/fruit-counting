"""Ground-truth sets, the summary.json self-check, the subset table and the geometry check.

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

from counting.data.formats.orchard_synth import (
    FrameInput, GtRecord, OrchardLayout, PassKey, SceneTruth, SynthRun, read_depth_m,
)

_GROUND_SEMANTIC = 7
_GROUND_SAMPLE_MAX = 2000


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


def subset_table(
    truth: SceneTruth, layout: OrchardLayout, passes: Mapping[PassKey, Sequence[FrameInput]]
) -> list[dict]:
    """G6 (count1) for `all`, each side, and each pitch dropped/kept, against `all` and the grand total."""
    tt_all = truth_sets(truth, layout, passes, "count1")
    g6_all = len(tt_all.g6)
    total = sum(tt_all.g6_total.values())
    pitches = sorted({key.pitch_deg for key in passes})

    def row(name: str, subset: Mapping[PassKey, Sequence[FrameInput]]) -> dict:
        g6 = len(truth_sets(truth, layout, subset, "count1").g6)
        return {"S": name, "n_passes": len(subset), "G6": g6,
                "ratio_all": (g6 / g6_all) if g6_all else None,
                "ratio_total": (g6 / total) if total else None}

    rows = [
        row("all", passes),
        row("A", {k: v for k, v in passes.items() if k.side == "A"}),
        row("B", {k: v for k, v in passes.items() if k.side == "B"}),
    ]
    for p in pitches:
        rows.append(row(f"only {p:g}", {k: v for k, v in passes.items() if k.pitch_deg == p}))
        rows.append(row(f"drop {p:g}", {k: v for k, v in passes.items() if k.pitch_deg != p}))
    return rows


def check_geometry(run: SynthRun, truth: SceneTruth, *, stride: int = 10) -> dict:
    """Sanity-check the adapter's camera and depth reading on `run.frames[::stride]`.

    1. Unoccluded records (`visib >= 0.95`, not `trunc`, with a `bbox`): their
       ground-truth 3D center, projected, must fall inside `bbox`.
    2. Ground pixels (`semantic == 7` with valid depth), back-projected: the
       recovered world `|z|` must be near 0 (the ground plane).
    """
    n_unoccluded = n_in_bbox = n_ground_px = 0
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
        rows, cols = np.nonzero((sem == _GROUND_SEMANTIC) & ~np.isnan(depth_m))
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
    passed = bool(
        center_in_bbox == 1.0 and n_unoccluded > 0 and n_ground_px > 0
        and median is not None and median <= 0.001
    )
    return {"center_in_bbox": center_in_bbox, "n_unoccluded": n_unoccluded,
            "ground_abs_z_median_m": median, "n_ground_px": n_ground_px, "passed": passed}
