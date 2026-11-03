"""Oracle stage values, the only place that feeds ground truth into a stage.

`track: oracle`, `center: truth`, `dedup: oracle` and `assign: truth` live here;
the real stage modules (tracking/, multiview/, geometry/) never see `SceneTruth`.
Oracle tracks and landmarks are aggregated by the real stages' own helpers
(`build_pass_tracks`, `landmarks_from_groups`), so only the grouping differs.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from counting.data.formats.orchard_synth import FrameInput, OrchardLayout, PassKey, SceneTruth
from counting.detection.base import Detections
from counting.multiview.assign import assign_trees, in_target_row
from counting.multiview.dedup import Landmarks, landmarks_from_groups
from counting.tracking.landmark import PassTracks, build_pass_tracks


def oracle_tracks(
    pass_key: PassKey,
    frames: Sequence[FrameInput],
    dets: Mapping[int, Detections],
    src: Mapping[int, np.ndarray],
    points: Mapping[int, np.ndarray],
) -> PassTracks:
    """One track per source id seen in the pass; every background (-1) detection is a track of its own.

    Tracks are numbered by their first observation in (frame idx, detection
    index) order, so the numbering never depends on the id values.
    """
    members: list[list[tuple[int, int]]] = []
    track_of: dict[int, int] = {}
    for f in sorted(frames, key=lambda f: f.idx):
        for d, g in enumerate(np.asarray(src[f.idx]).tolist()):
            if g < 0:
                members.append([(f.idx, d)])
                continue
            if g not in track_of:
                track_of[g] = len(members)
                members.append([])
            members[track_of[g]].append((f.idx, d))
    return build_pass_tracks(pass_key, members, dets, points, frames)


def truth_centers(majority: np.ndarray, X: np.ndarray, truth: SceneTruth) -> np.ndarray:
    """Each unit's truth center when its majority id is a known fruit; otherwise its own X (-1, unknown id)."""
    out = np.array(X, dtype=float).reshape(-1, 3)
    for i, m in enumerate(np.asarray(majority).tolist()):
        fruit = truth.fruits.get(m)
        if fruit is not None:
            out[i] = fruit.center
    return out


def truth_row_mask(majority: np.ndarray, X: np.ndarray, layout: OrchardLayout, truth: SceneTruth) -> np.ndarray:
    """Row filter by truth: a known fruit passes when its tree is a target-row tree; others use `in_target_row(X)`."""
    out = in_target_row(X, layout)
    target = set(layout.tree_ids)
    for i, m in enumerate(np.asarray(majority).tolist()):
        fruit = truth.fruits.get(m)
        if fruit is not None:
            out[i] = fruit.tree in target
    return out


def truth_trees(majority: np.ndarray, X: np.ndarray, layout: OrchardLayout, truth: SceneTruth) -> np.ndarray:
    """Tree by truth: a target-row fruit goes to its own tree; anything else to its grid cell (`assign_trees`)."""
    out = assign_trees(X, layout)
    target = set(layout.tree_ids)
    for i, m in enumerate(np.asarray(majority).tolist()):
        fruit = truth.fruits.get(m)
        if fruit is not None and fruit.tree in target:
            out[i] = fruit.tree
    return out


def oracle_dedup(
    passes: Sequence[PassTracks], keep: Sequence[np.ndarray], majorities: Sequence[np.ndarray]
) -> Landmarks:
    """Fuse kept tracks with the same majority id across passes only; splits inside a pass stay apart.

    The j-th landmark of id g gathers, from every pass, the j-th kept track
    (in track-number order) whose majority is g. A track with majority -1 is
    a landmark of its own. Landmarks are numbered by first member in
    (pass index, track number) order.
    """
    groups: list[list[tuple[int, int]]] = []
    slot: dict[tuple[int, int], int] = {}   # (id, j) -> landmark number
    for i, (_, k, maj) in enumerate(zip(passes, keep, majorities, strict=True)):
        seen: Counter[int] = Counter()
        for t in np.flatnonzero(np.asarray(k, dtype=bool)).tolist():
            g = int(maj[t])
            if g < 0:
                groups.append([(i, t)])
                continue
            key = (g, seen[g])
            seen[g] += 1
            if key not in slot:
                slot[key] = len(groups)
                groups.append([])
            groups[slot[key]].append((i, t))
    return landmarks_from_groups(passes, groups)
