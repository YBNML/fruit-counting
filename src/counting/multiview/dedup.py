"""Co-observation-guarded clustering of tracks: in-pass repair and multi-view dedup.

`cluster` joins items closest-first with union-find, but never joins two
groups that hold tracks of the same pass observed in a common frame: one
camera saw both, so they are different fruit. `repair_pass` uses it to join
the split pieces of a track inside one pass; `dedup` uses it to fuse tracks
of all passes into landmarks. Its `hybrid` mode keeps tracks within `band_m`
past the centre plane and joins tracks of different sides only when both lie
inside the band, so band 0 equals `self_side` and an unbounded band equals
`merge`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from counting.config.eval_schema import DedupConfig
from counting.data.formats.orchard_synth import OrchardLayout, PassKey
from counting.tracking.landmark import PassTracks


@dataclass
class Landmarks:
    """Fused fruit units. `members[l]` lists the `(pass key, track number)` tracks of landmark l."""

    X: np.ndarray       # (L,3) component-wise median of the member tracks' X
    label: np.ndarray   # (L,) int8 majority label of the member tracks, ties -> 0
    members: list[list[tuple[PassKey, int]]]


def _majority(labels: np.ndarray, weights: np.ndarray | None = None) -> int:
    """Label with the largest (weighted) count; ties go to the smaller label (0 for fruit/bag)."""
    values, inverse = np.unique(labels, return_inverse=True)
    return int(values[np.argmax(np.bincount(inverse, weights=weights))]) if len(values) else 0


def cluster(
    X: np.ndarray,
    frames: Sequence[Mapping[int, frozenset[int]]],
    radius_m: float,
    *,
    allow: Callable[[int, int], bool] | None = None,
) -> list[list[int]]:
    """Group items whose points lie within `radius_m`, joining the closest pairs first.

    `frames[i]` maps pass number -> frames item i was observed in. A pair is
    skipped when `allow(a, b)` is False or when the two groups share a pass
    with a common frame. Pairs are taken in (distance, a, b) order and the
    smaller index becomes the root. Items with a non-finite point stay alone.
    Groups come out in order of their smallest index, members ascending.
    """
    X = np.asarray(X, dtype=float).reshape(-1, 3)
    if len(frames) != len(X):
        raise ValueError(f"cluster: {len(X)} points but {len(frames)} frame maps")
    ok = np.flatnonzero(np.isfinite(X).all(axis=1))
    if len(ok) >= 2:
        pairs = ok[cKDTree(X[ok]).query_pairs(radius_m, output_type="ndarray")]   # a < b, original numbers
    else:
        pairs = np.empty((0, 2), dtype=int)
    pa, pb = pairs[:, 0], pairs[:, 1]
    order = np.lexsort((pb, pa, np.linalg.norm(X[pa] - X[pb], axis=1)))

    parent = list(range(len(X)))
    seen: list[dict[int, frozenset[int]]] = [dict(f) for f in frames]   # per root: pass -> observed frames

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs[order].tolist():
        if allow is not None and not allow(a, b):
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        fa, fb = seen[ra], seen[rb]
        if any(fa[p] & fb[p] for p in fa.keys() & fb.keys()):
            continue   # the same pass saw both groups in one frame: different fruit
        root, child = min(ra, rb), max(ra, rb)
        parent[child] = root
        seen[root] = {p: fa.get(p, frozenset()) | fb.get(p, frozenset()) for p in fa.keys() | fb.keys()}

    groups: dict[int, list[int]] = {}
    for i in range(len(X)):
        groups.setdefault(find(i), []).append(i)
    return [groups[r] for r in sorted(groups)]


def repair_pass(tracks: PassTracks, radius_m: float) -> PassTracks:
    """Join the split pieces of tracks inside one pass; `radius_m == 0` returns `tracks` itself.

    Joined tracks are renumbered in group order. X is the component-wise
    median of the member X, labels the `n_obs`-weighted majority (ties -> 0),
    `n_obs` the sum, `spread` the maximum and `frames` the union.
    """
    if radius_m == 0:
        return tracks
    groups = cluster(tracks.X, [{0: f} for f in tracks.frames], radius_m)
    new = np.empty(len(tracks.X), dtype=int)
    for g, members in enumerate(groups):
        new[members] = g
    return PassTracks(
        pass_key=tracks.pass_key,
        det_track={key: int(new[t]) for key, t in tracks.det_track.items()},
        X=np.array([np.median(tracks.X[m], axis=0) for m in groups], dtype=float).reshape(-1, 3),
        labels=np.array([_majority(tracks.labels[m], tracks.n_obs[m]) for m in groups], dtype=np.int8),
        n_obs=np.array([tracks.n_obs[m].sum() for m in groups], dtype=int),
        spread=np.array([np.max(tracks.spread[m]) for m in groups], dtype=float),
        frames=[frozenset().union(*(tracks.frames[t] for t in m)) for m in groups],
    )


def side_filter(tracks: PassTracks, layout: OrchardLayout, mode: str, band_m: float = 0.0) -> np.ndarray:
    """Tracks a pass contributes under `mode`.

    `self_side` keeps an aisle-A pass's tracks with `y >= row_y` and an
    aisle-B pass's tracks with `y < row_y`; `hybrid` moves both bounds
    `band_m` past the centre plane (A: `y >= row_y - band_m`, B:
    `y < row_y + band_m`); every other mode keeps all tracks.
    """
    y = np.asarray(tracks.X, dtype=float).reshape(-1, 3)[:, 1]
    if mode == "hybrid":
        return y >= layout.row_y - band_m if tracks.pass_key.side == "A" else y < layout.row_y + band_m
    if mode != "self_side":
        return np.ones(len(y), dtype=bool)
    return y >= layout.row_y if tracks.pass_key.side == "A" else y < layout.row_y


def landmarks_from_groups(
    passes: Sequence[PassTracks], groups: Sequence[Sequence[tuple[int, int]]]
) -> Landmarks:
    """Fuse groups of `(index into passes, track number)` into `Landmarks`, one per group.

    These are the rules `dedup` uses: X is the component-wise median of the
    member tracks' X and the label is the unweighted majority of their
    labels (ties -> 0). Members keep the given order.
    """
    X = [np.median(np.stack([passes[i].X[t] for i, t in g]), axis=0) for g in groups]
    label = [_majority(np.array([passes[i].labels[t] for i, t in g], dtype=np.int8)) for g in groups]
    return Landmarks(
        X=np.array(X, dtype=float).reshape(-1, 3),
        label=np.array(label, dtype=np.int8),
        members=[[(passes[i].pass_key, int(t)) for i, t in g] for g in groups],
    )


def dedup(
    passes: Sequence[PassTracks],
    keep: Sequence[np.ndarray],
    layout: OrchardLayout,
    cfg: DedupConfig,
    *,
    mode: Literal["self_side", "merge", "hybrid"],
) -> Landmarks:
    """Fuse the kept tracks of all passes into landmarks within `cfg.radius_m`.

    A track enters when its `keep` entry is True and `side_filter` passes it.
    `merge` joins tracks of any passes; `self_side` joins only tracks of
    passes on the same aisle side. `hybrid` keeps tracks within `cfg.band_m`
    past the centre plane and joins tracks of different sides only when
    both lie inside the band (`|y - row_y| < cfg.band_m`), so band 0 equals
    `self_side` and an unbounded band equals `merge`.
    """
    if mode not in ("self_side", "merge", "hybrid"):
        raise ValueError(f"unknown dedup mode: {mode!r}")
    items: list[tuple[int, int]] = []   # (pass index, track number); position = cluster item number
    for i, (tracks, k) in enumerate(zip(passes, keep, strict=True)):
        use = np.asarray(k, dtype=bool) & side_filter(tracks, layout, mode, cfg.band_m)
        items.extend((i, int(t)) for t in np.flatnonzero(use))
    X = np.array([passes[i].X[t] for i, t in items], dtype=float).reshape(-1, 3)
    side = [passes[i].pass_key.side for i, _ in items]
    allow: Callable[[int, int], bool] | None
    if mode == "self_side":
        allow = lambda a, b: side[a] == side[b]
    elif mode == "hybrid":
        near = (np.abs(X[:, 1] - layout.row_y) < cfg.band_m).tolist()   # inside the band around the plane
        allow = lambda a, b: side[a] == side[b] or (near[a] and near[b])
    else:                                                                # merge: no side rule
        allow = None
    groups = cluster(X, [{i: passes[i].frames[t]} for i, t in items], cfg.radius_m, allow=allow)
    return landmarks_from_groups(passes, [[items[j] for j in g] for g in groups])
