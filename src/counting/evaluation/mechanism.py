"""Report-only counts of the tracker and dedup mechanisms, read against the truth (scoring side).

`split_fragments` measures how far apart the tracks of one fruit stay in a pass after the in-pass repair,
`band_blocked_pairs` counts the cross-side pairs of one fruit that the dedup band keeps apart, and `mechanism_entry`
joins them with the tracker's `TrackTrace` counts into a result's `mechanism` entry. None of them changes a count.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from counting.config.eval_schema import DedupConfig
from counting.data.formats.orchard_synth import OrchardLayout
from counting.evaluation.match import unit_majority
from counting.multiview.dedup import dedup, side_filter
from counting.tracking.landmark import BIRTH_CAUSES, PassTracks, TrackTrace


def _majorities(tracks: PassTracks, src: Mapping[int, np.ndarray]) -> np.ndarray:
    """Each track's majority source id over its observations (`unit_majority`; -1 for no fruit)."""
    return unit_majority([[int(src[f][d]) for f, d in m] for m in tracks.members()])[0]


def _same_fruit_pairs(majority: Sequence[int]) -> list[tuple[int, int]]:
    """The position pairs `(a, b)`, a < b, of `majority` entries with the same source id other than -1, sorted."""
    groups: dict[int, list[int]] = {}
    for k, m in enumerate(majority):
        if m >= 0:
            groups.setdefault(int(m), []).append(k)
    return sorted((a, b) for g in groups.values() for i, a in enumerate(g) for b in g[i + 1:])


def split_fragments(tracks: Sequence[PassTracks], src: Mapping[int, np.ndarray]) -> np.ndarray:
    """The centre distances (m) of the track pairs of one pass with the same majority source id, `src` giving each
    frame's detections their source ids.

    `tracks` are the passes' tracks after the in-pass repair. A track whose majority is -1 (no fruit) or whose
    centre is not finite is left out. Pairs come pass by pass, in track order.
    """
    out: list[float] = []
    for tr in tracks:
        X = np.asarray(tr.X, dtype=float).reshape(-1, 3)
        idx = np.flatnonzero(np.isfinite(X).all(axis=1))
        majority = _majorities(tr, src)[idx]
        out.extend(float(np.linalg.norm(X[idx[a]] - X[idx[b]])) for a, b in _same_fruit_pairs(majority.tolist()))
    return np.array(out, dtype=float)


def band_blocked_pairs(
    tracks: Sequence[PassTracks], src: Mapping[int, np.ndarray], layout: OrchardLayout, cfg: DedupConfig, *,
    keep: Sequence[np.ndarray] | None = None,
) -> int:
    """The pairs of tracks from passes on different sides with the same majority source id and centres at most
    `cfg.radius_m` apart that the band keeps apart: one of the two or both lie outside the band
    (`|y - row_y| < cfg.band_m` is false), so the band rule does not join them, and the hybrid dedup of the same
    tracks leaves the two in different landmarks. A pair the dedup joins through other tracks (a same-side track
    inside the band that reaches the other side) does not count.

    A track counts when its `keep` entry is True (every track without `keep`), `side_filter` keeps it under the
    hybrid mode, its majority is not -1 and its centre is finite; these are the tracks the hybrid dedup takes.
    """
    if keep is None:
        keep = [np.ones(len(tr.X), dtype=bool) for tr in tracks]
    lm = dedup(tracks, keep, layout, cfg, mode="hybrid")
    pass_index = {tr.pass_key: i for i, tr in enumerate(tracks)}
    landmark_of = {(pass_index[k], t): u for u, m in enumerate(lm.members) for k, t in m}
    X, side, majority, landmark = [], [], [], []
    for i, tr in enumerate(tracks):
        P = np.asarray(tr.X, dtype=float).reshape(-1, 3)
        use = side_filter(tr, layout, "hybrid", cfg.band_m) & np.isfinite(P).all(axis=1)
        use &= np.asarray(keep[i], dtype=bool)
        idx = np.flatnonzero(use)
        X.extend(P[idx])
        side.extend([tr.pass_key.side] * len(idx))
        majority.extend(_majorities(tr, src)[idx].tolist())
        landmark.extend(landmark_of[(i, int(t))] for t in idx)
    near = [abs(float(x[1]) - layout.row_y) < cfg.band_m for x in X]
    return sum(1 for a, b in _same_fruit_pairs(majority)
               if side[a] != side[b] and not (near[a] and near[b]) and landmark[a] != landmark[b]
               and np.linalg.norm(X[a] - X[b]) <= cfg.radius_m)


def _percentile(values: np.ndarray, q: float) -> float | None:
    return float(np.percentile(values, q)) if len(values) else None


def mechanism_entry(traces: Sequence[TrackTrace], fragments: np.ndarray, band_blocked: int | None) -> dict:
    """A result's `mechanism` entry: the summed `TrackTrace` counts of the passes, the count, median and 90th
    percentile (m) of the `split_fragments` distances (None without any) and the `band_blocked_pairs` count (None
    where no hybrid dedup ran)."""
    fragments = np.asarray(fragments, dtype=float)
    return {"soft_depth_pairs": sum(t.soft_depth_pairs for t in traces),
            "births": {k: sum(t.births[k] for t in traces) for k in BIRTH_CAUSES},
            "fragments": {"n": len(fragments), "p50_m": _percentile(fragments, 50),
                          "p90_m": _percentile(fragments, 90)},
            "band_blocked_pairs": band_blocked}
