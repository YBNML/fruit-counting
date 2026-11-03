"""Per-pass landmark tracker with edge-aware association, and the last_box baseline.

Fruit do not move, so a track is a 3D landmark: its point `X` and physical size
`S` are projected into each new camera to predict a box.
A detection cut by the image border is compared on its inner (uncut) edge,
and a track ends only when its grown predicted box leaves the image. Tracks
without 3D, and every track under `method="last_box"`, chain by IoU with the
previous box instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from counting.config.eval_schema import TrackerConfig
from counting.data.formats.orchard_synth import FrameInput, PassKey
from counting.detection.base import Detections, box_iou
from counting.geometry.camera import Camera, box_centers
from counting.multiview.lift import complete_edge_boxes, edge_touch

_OUT = 1e6        # cost of a pair outside the gate; such pairs are never kept
_MIN_Z = 0.05     # metres; a landmark nearer than this to the camera plane ends its track
_MIN_IOU = 0.1    # IoU gate for tracks chained by box overlap


@dataclass
class PassTracks:
    """One pass's tracks, numbered in creation order, before repair and `min_obs` filtering."""

    pass_key: PassKey
    det_track: dict[tuple[int, int], int]   # (frame idx, det index) -> track number
    X: np.ndarray            # (T,3) world point; NaN when no observation has 3D
    labels: np.ndarray       # (T,) int8 majority label of the observations, ties -> 0
    n_obs: np.ndarray        # (T,) int
    spread: np.ndarray       # (T,) median distance from the X observations to X; NaN without 3D
    frames: list[frozenset[int]]   # frame idx each track was observed in

    def members(self) -> list[list[tuple[int, int]]]:
        """Each track's `(frame idx, det index)` observations, in idx order."""
        out: list[list[tuple[int, int]]] = [[] for _ in range(len(self.X))]
        for key in sorted(self.det_track):
            out[self.det_track[key]].append(key)
        return out


def predict_boxes(
    camera: Camera, X: np.ndarray, S: np.ndarray, grow: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project landmarks `X` with physical size `S` into `camera`.

    Returns `(uv, z, s_pred, alive)` with `s_pred = S * fx / z`. A landmark is
    alive while it is in front of the camera (`z > 0.05`) and its predicted box,
    grown on every side to half-size `(0.5 + grow) * s_pred`, still overlaps the
    image, so a fragment can remain visible even with the centre off-screen.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        uv, z = camera.project(np.asarray(X, dtype=float).reshape(-1, 3))
        s_pred = np.asarray(S, dtype=float) * camera.K[0, 0] / z
    h = (0.5 + grow) * s_pred
    u, v = uv[:, 0], uv[:, 1]
    alive = (
        (z > _MIN_Z)
        & (u + h > -0.5) & (u - h < camera.width - 0.5)
        & (v + h > -0.5) & (v - h < camera.height - 0.5)
    )
    return uv, z, s_pred, alive


def edge_cost(boxes: np.ndarray, touch: np.ndarray, uv: np.ndarray, s: float) -> np.ndarray:
    """Edge-aware distance from each box to the predicted box (centre `uv`, size `s`), over `s`.

    Per axis: a box cut on one side only is compared on its uncut inner edge,
    a box cut on both sides contributes 0, and an uncut box is compared by
    centre. `touch` columns are left, right, top, bottom (`edge_touch`).
    """
    return _edge_costs(boxes, touch, np.asarray(uv, dtype=float)[None], np.array([s], dtype=float))[0]


def _edge_costs(boxes: np.ndarray, touch: np.ndarray, uv: np.ndarray, s: np.ndarray) -> np.ndarray:
    """`edge_cost` for n predicted boxes at once: `uv` (n,2), `s` (n,) -> (n, M)."""
    boxes = np.asarray(boxes, dtype=float)
    touch = np.asarray(touch, dtype=bool)
    centre = box_centers(boxes)
    half = s[:, None] / 2
    d = []
    for a in (0, 1):
        lo, hi = touch[:, 2 * a], touch[:, 2 * a + 1]
        p = uv[:, a, None] + 0.5               # predicted centre in pixel-edge coordinates
        d.append(np.where(lo & ~hi, boxes[:, 2 + a] - (p + half),        # low side cut: inner high edges
                 np.where(hi & ~lo, boxes[:, a] - (p - half),            # high side cut: inner low edges
                 np.where(lo & hi, 0.0, centre[:, a] - uv[:, a, None]))))
    return np.hypot(d[0], d[1]) / s[:, None]


def _x_rows(P: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """Observations that define a track's X and S: interior ones with 3D, else every one with 3D."""
    finite = np.isfinite(P).all(axis=1)
    sel = finite & ~edge
    return sel if sel.any() else finite


def _majority(labels: np.ndarray) -> int:
    """Most frequent label; ties go to the smaller label (0 for fruit/bag)."""
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)]) if len(values) else 0


def build_pass_tracks(
    pass_key: PassKey,
    members: Sequence[Sequence[tuple[int, int]]],
    dets: Mapping[int, Detections],
    points: Mapping[int, np.ndarray],
    frames: Sequence[FrameInput],
) -> PassTracks:
    """Aggregate each track's `(frame idx, det index)` observations into `PassTracks`.

    These are the rules `track_pass` uses for its own output: X is the
    component-wise median of the interior observations with 3D (every
    observation with 3D when there is no interior one), `spread` the median
    distance of those observations to X, and `labels` the majority label.
    """
    sizes = {f.idx: (f.camera.width, f.camera.height) for f in frames}
    edges: dict[int, np.ndarray] = {}
    det_track: dict[tuple[int, int], int] = {}
    X, labels, n_obs, spread, seen = [], [], [], [], []
    for t, mem in enumerate(members):
        for f, d in mem:
            if f not in edges:
                edges[f] = edge_touch(dets[f].boxes, *sizes[f]).any(axis=1)
            det_track[(int(f), int(d))] = t
        P = np.array([points[f][d] for f, d in mem], dtype=float).reshape(-1, 3)
        sel = _x_rows(P, np.array([edges[f][d] for f, d in mem], dtype=bool))
        if sel.any():
            x = np.median(P[sel], axis=0)
            spread.append(float(np.median(np.linalg.norm(P[sel] - x, axis=1))))
        else:
            x = np.full(3, np.nan)
            spread.append(np.nan)
        X.append(x)
        labels.append(_majority(np.array([dets[f].labels[d] for f, d in mem], dtype=np.int8)))
        n_obs.append(len(mem))
        seen.append(frozenset(int(f) for f, _ in mem))
    return PassTracks(
        pass_key=pass_key,
        det_track=det_track,
        X=np.array(X, dtype=float).reshape(-1, 3),
        labels=np.array(labels, dtype=np.int8),
        n_obs=np.array(n_obs, dtype=int),
        spread=np.array(spread, dtype=float),
        frames=seen,
    )


class _Track:
    """Running state of one track while a pass is processed."""

    def __init__(self) -> None:
        self.keys: list[tuple[int, int]] = []
        self._P: list[np.ndarray] = []
        self._S: list[float] = []
        self._E: list[bool] = []
        self.X = np.full(3, np.nan)
        self.S = np.nan
        self.edge_only = True
        self.box = np.zeros(4)
        self.miss = 0
        self.alive = True
        self.pred: tuple[np.ndarray, float, float] | None = None   # (uv, s_pred, z_pred) this frame

    def add(self, key: tuple[int, int], p: np.ndarray, s: float, edge: bool, box: np.ndarray) -> None:
        self.keys.append(key)
        self._P.append(p)
        self._S.append(s)
        self._E.append(edge)
        P, E = np.array(self._P), np.array(self._E)
        sel = _x_rows(P, E)
        if sel.any():
            self.X = np.median(P[sel], axis=0)
            self.S = float(np.median(np.array(self._S)[sel]))
        else:
            self.X, self.S = np.full(3, np.nan), np.nan
        self.edge_only = bool(E.all())
        self.box = box
        self.miss = 0


def _cost_matrix(
    active: list[_Track], boxes: np.ndarray, touch: np.ndarray, zP: np.ndarray, cfg: TrackerConfig
) -> np.ndarray:
    """Track x detection cost; pairs outside the gate get `_OUT`.

    Tracks with a prediction this frame use the edge-aware cost and depth gate;
    the others use `1 - IoU` with their last box.
    """
    cost = np.full((len(active), len(boxes)), _OUT)
    pred = [i for i, t in enumerate(active) if t.pred is not None]
    by_iou = [i for i, t in enumerate(active) if t.pred is None]
    if pred:
        uv = np.stack([active[i].pred[0] for i in pred])
        s = np.array([active[i].pred[1] for i in pred])
        z = np.array([active[i].pred[2] for i in pred])
        edge_only = np.array([active[i].edge_only for i in pred])
        c = _edge_costs(boxes, touch, uv, s)
        # skip the depth gate for border boxes, edge-only tracks and detections without depth
        depth_ok = (
            touch.any(axis=1) | np.isnan(zP) | edge_only[:, None] | (np.abs(zP - z[:, None]) <= cfg.gate_m)
        )
        cost[pred] = np.where((c <= cfg.gate_px_frac) & depth_ok, c, _OUT)
    if by_iou:
        iou = box_iou(np.stack([active[i].box for i in by_iou]), boxes)
        cost[by_iou] = np.where(iou >= _MIN_IOU, 1.0 - iou, _OUT)
    return cost


def _associate(cost: np.ndarray, stages: Sequence[np.ndarray]) -> dict[int, int]:
    """Match tracks to detections stage by stage (ByteTrack style); returns track row -> det column."""
    pairs: dict[int, int] = {}
    for cols_mask in stages:
        rows = np.array([i for i in range(cost.shape[0]) if i not in pairs], dtype=int)
        cols = np.flatnonzero(cols_mask)
        if len(rows) == 0 or len(cols) == 0:
            continue
        sub = cost[np.ix_(rows, cols)]
        for r, c in zip(*linear_sum_assignment(sub)):
            if sub[r, c] < _OUT:
                pairs[int(rows[r])] = int(cols[c])
    return pairs


def track_pass(
    frames: Sequence[FrameInput],
    dets: Mapping[int, Detections],
    points: Mapping[int, np.ndarray],
    cfg: TrackerConfig,
    *,
    method: Literal["landmark", "last_box"] = "landmark",
) -> PassTracks:
    """Track one pass's detections frame by frame (idx order) into `PassTracks`.

    `points[idx]` holds each detection's lifted 3D point (NaN without depth).
    Every track is returned, including ones later repair or `min_obs` drops.
    """
    if method not in ("landmark", "last_box"):
        raise ValueError(f"unknown tracking method: {method!r}")
    if not frames:
        raise ValueError("track_pass needs at least one frame")
    frames = sorted(frames, key=lambda f: f.idx)
    tracks: list[_Track] = []
    active: list[_Track] = []
    for frame in frames:
        cam, D = frame.camera, dets[frame.idx]
        P = np.asarray(points[frame.idx], dtype=float).reshape(-1, 3)
        touch = edge_touch(D.boxes, cam.width, cam.height)
        edge = touch.any(axis=1)
        full = complete_edge_boxes(D.boxes, cam.width, cam.height)
        zP = cam.project(P)[1]
        S_d = np.sqrt((full[:, 2] - full[:, 0]) * (full[:, 3] - full[:, 1])) * zP / cam.K[0, 0]
        use = D.scores >= cfg.score_lo
        hi = use & (D.scores >= cfg.score_hi)

        for t in active:
            t.pred = None
        landmarks = [t for t in active if method == "landmark" and np.isfinite(t.X).all()]
        if landmarks:
            uv, z, s, alive = predict_boxes(
                cam, np.stack([t.X for t in landmarks]), np.array([t.S for t in landmarks]), cfg.gate_px_frac
            )
            for k, t in enumerate(landmarks):
                t.pred = (uv[k], float(s[k]), float(z[k]))
                t.alive = bool(alive[k])
        active = [t for t in active if t.alive]   # a landmark ends once its grown box leaves the image

        pairs = _associate(_cost_matrix(active, D.boxes, touch, zP, cfg), (hi, use & ~hi))
        nxt = []
        for i, t in enumerate(active):
            if i in pairs:
                j = pairs[i]
                t.add((frame.idx, j), P[j], float(S_d[j]), bool(edge[j]), D.boxes[j])
            else:
                t.miss += 1
                if t.pred is None:
                    if t.miss >= (1 if cfg.max_gap is None else cfg.max_gap):
                        continue
                elif cfg.max_gap is not None and t.miss >= cfg.max_gap:
                    continue
            nxt.append(t)
        used = set(pairs.values())
        for j in np.flatnonzero(hi):
            if int(j) not in used:
                t = _Track()
                t.add((frame.idx, int(j)), P[j], float(S_d[j]), bool(edge[j]), D.boxes[j])
                tracks.append(t)
                nxt.append(t)
        active = nxt
    return build_pass_tracks(frames[0].pass_key, [t.keys for t in tracks], dets, points, frames)
