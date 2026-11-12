"""Alignment of the tracked passes on the prediction side: segments along the tree row, the transforms of a
pass by segment, the rigid fits of a segment with their fallbacks, and the stages that align the passes by
the tracks they share and move them to the row centre plane.

The fits read predicted track centres, their positions along the row and the tree ledger of the target
row only; nothing here reads truth.

Segments. The centres start at the first tree of the target row and step by half the segment length L
until the last tree is covered, so neighbouring windows overlap by half. A point takes the transform of
its nearest centre (the first on a tie, the first segment for a NaN position), and the fit of a segment
uses the points within L/2 of its centre. `Segments.whole` is one segment of unbounded length.

One fit without known pairs (`fit_rigid`):
1. The candidate pairs are every reference point within the pair radius of each moving point, in moving
   then reference order.
2. Shift hypotheses: up to `RANSAC_DRAWS` candidates drawn without replacement, each proposing its
   displacement. A candidate whose displacement lies within `INLIER_M` of it is an inlier. The best
   hypothesis has the most inliers, then the smallest sum of inlier distances, then the earliest draw.
3. Three-pair rigid hypotheses, drawn from the same generator after the shifts when asked for (the fit
   of a whole group): three candidates with distinct moving and reference points whose moving points
   span at least `RIGID_AREA_M2` (cross product) give a rigid transform by Kabsch. One replaces the best
   with more inliers, or as many with a smaller distance sum. Over a long window the yaw carries the
   far ends past the inlier distance of a single shift.
4. Mutual nearest pairs among the inliers of the best hypothesis: each moving point picks its closest
   reference point, each reference point its closest moving point (the smaller index on a tie), and the
   pairs that pick each other stay. Fewer than `min_pairs` pairs give no fit.
5. Refinement by least squares: roll, pitch and yaw about the centroid of the moving pair points and a
   translation on top of the hypothesis shift, all starting at zero (the first trust radius of
   `least_squares` follows the size of the start, so a start near zero but not at zero stalls). Each
   component of a point residual goes through the Huber transform of scale `HUBER_M`; the roll and
   pitch prior terms `HUBER_M * angle / sigma` stay quadratic, so `sigma_rp_deg` keeps the meaning of
   a Gaussian sigma.
6. Gathering again, up to `REFIT_ROUNDS` times: the refined transform measures every candidate again,
   mutual nearest pairs are chosen among the new inliers and the refinement runs again. It stops, with
   the transform before, when the pairs stay the same or fall below `min_pairs`.
7. The check on held-out pairs (`_check`): the final pairs fall into two folds by alternate rank along the
   row; each fold is measured by the refinement of the other, started from its mean pair displacement.
   The fit stands only when the median held-out distance lies more than `CHECK_MARGIN_M` below the median
   pair distance at the identity; otherwise there is no fit, as with pairs that differ by noise alone.
The residual of a fit is the median distance of its pairs after the transform.

Fallbacks (`with_fallback`). A segment with a fit of its own uses it (`own`); a fit the check refuses is no
fit of its own. A segment without one takes the fit of the segment with its own fit whose centre is
nearest, the first on a tie (`nearest`). When no segment has its own fit, one fit of all the moving points
is tried (`group`, with the three-pair hypotheses), and without that every segment keeps the identity
(`identity`, no residual). Between the sides, a fit of the whole group short of pairs is tried once more
after a coarse shift (`_coarse_shift`): every displacement from a moving point to a reference point within
`COARSE_RADIUS_M` votes for those within `COARSE_VOTE_M` of it, and with at least `COARSE_MIN_VOTES` votes
the componentwise median around the most voted one moves the moving points before the fit. That fit goes
on with the same generator, passes the same check on the points before the move and stands only with more
pairs than the identity has (`_identity_pairs`: mutual nearest pairs within `INLIER_M` without any move).
The fits on one side and the fits of a segment take no coarse shift.

Used tracks. A stage reads the tracks of a pass with a finite centre, at least `min_obs` observations and a
centre less than half the row spacing from the target row across the row. Each is moved by the transform
its pass has so far and keeps its position along the row (`track_x`: the median position of the frames
that observed it). The transform of a pass then applies to all its tracks; one without 3D stays NaN.

Stages (`align_passes`). Every pass starts at the identity, the methods run in the order given, and a
stage puts its transform after the one each moved pass has.
- `tracks` on one side, without the rig only (the rig holds the pitches of a side together): on side A,
  then side B, each other pass of the side is fitted on its own to the pass at the reference pitch. A side
  without that pass is left as it is.
- `tracks` between the sides, when both sides have passes: side B is fitted to all the passes of side A,
  as one group under the rig (one body carries every camera) and pass by pass without it.
- `row_plane`: both sides move together, by one shift across the row per segment, so that the row centre
  plane lies on the target row. Where each side has at least `min_pairs` tracks in the window of a segment,
  the plane is the midpoint of the median across-row position of the side A tracks and that of the side B
  tracks (`own`), so a side with more tracks does not pull it. A Theil-Sen line along the row through
  these midpoints (the median slope over all pairs, the intercept median(y) - slope * median(x)) gives
  the plane of every segment (`line` in a segment without a midpoint of its own); a single midpoint holds
  for all of them. Without any, the midpoint of the two sides' medians over all their tracks serves every
  segment when each side has `min_pairs` tracks (`group`); otherwise nothing moves (`identity`).

Stress options. `row_bias_m` is added to the plane before the shift is taken. `pair_keep` keeps that share
of the moving tracks in each fit between the two sides, by a keyed uniform draw per track; the fits on one
side keep every track.

Determinism: every draw comes from a generator keyed by the caller (`noise_rng(*key, segment)`, the fit
of the whole group taking the segment count as its index), candidates and pairs are sorted, ties break by
fixed rules and the tolerances are fixed, so the same inputs give the same bytes. The check draws nothing,
and the fit after a coarse shift goes on with the generator of the fit of the whole group. `align_passes`
keys the fits of a group by (seed, noise seed, `STEP_ALIGN`, scene, stage, side, group) and its thinning by
the same parts under `STEP_ALIGN_THIN`: stage 0 on one side and 1 between the sides, side A 0 and B 1,
group 0 for the rig and the pitch in thousandths of a degree for a pass.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from counting.data.formats.orchard_synth import OrchardLayout, PassKey
from counting.geometry.noise import STEP_ALIGN, STEP_ALIGN_THIN, noise_rng

SEGMENT_M = 3.0              # segment length (m); the centres step by half of it
PAIR_RADIUS_M = 0.5          # radius of the candidate pairs (m)
MIN_PAIRS = 20               # fewest pairs a fit stands on
INLIER_M = 0.05              # inlier distance of a hypothesis (m)
HUBER_M = 0.02               # Huber scale of a point residual component, the noise assumed for a pair point (m)
RANSAC_DRAWS = 200           # shift hypotheses of a fit
RANSAC_DRAWS_RIGID = 1000    # three-pair rigid hypotheses of the fit of a whole group
RIGID_AREA_M2 = 0.05         # smallest cross product of the three moving points of a rigid hypothesis (m^2)
REFIT_ROUNDS = 2             # rounds of gathering the pairs again
CHECK_MARGIN_M = 0.005       # margin by which a fit must beat the identity on its held-out pairs (m)
COARSE_RADIUS_M = 1.5        # radius of the candidate displacements of a coarse shift (m)
COARSE_VOTE_M = 0.10         # distance within which two candidate displacements vote for each other (m)
COARSE_MIN_VOTES = 8         # fewest votes a coarse shift stands on
USES = ("own", "nearest", "line", "group", "identity")    # where the transform of a segment comes from
_METHODS = ("tracks", "row_plane")                         # the stages `align_passes` runs
_SIDES = {"A": 0, "B": 1}                                  # side part of a group key
_PITCH_TOL = 1e-6                                          # pitch tolerance of the reference pass (deg)


@dataclass(frozen=True, eq=False)
class Segments:
    """Segment centres along the row (x, m) and the segment length; `length_m` inf for one whole segment."""

    centres: np.ndarray
    length_m: float

    @classmethod
    def along(cls, tree_x: Sequence[float], length_m: float) -> Segments:
        """Centres from the first tree by `length_m / 2` until the last tree is covered."""
        lo, hi = float(min(tree_x)), float(max(tree_x))
        half = length_m / 2
        n = int(np.floor((hi - lo) / half + 1e-9)) + 1
        return cls(lo + half * np.arange(n), float(length_m))

    @classmethod
    def whole(cls) -> Segments:
        """One segment of unbounded length, centred at 0."""
        return cls(np.zeros(1), float("inf"))

    def index(self, x) -> np.ndarray:
        """The segment of each position `x`: the nearest centre, the first on a tie, 0 for NaN."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if len(self.centres) == 1:
            return np.zeros(len(x), dtype=np.int64)
        j = np.argmin(np.abs(x[:, None] - self.centres[None, :]), axis=1).astype(np.int64)
        j[np.isnan(x)] = 0
        return j

    def window(self, j: int, x) -> np.ndarray:
        """Whether each position `x` lies within half a segment length of centre `j` (all for one whole segment)."""
        x = np.asarray(x, dtype=np.float64)
        if np.isinf(self.length_m):
            return np.ones(x.shape, dtype=bool)
        return np.abs(x - self.centres[j]) <= self.length_m / 2 + 1e-9


def _same_segments(a: Segments, b: Segments) -> bool:
    return a is b or (a.length_m == b.length_m and np.array_equal(a.centres, b.centres))


@dataclass(frozen=True, eq=False)
class PassAlignment:
    """The rigid transforms of one pass, one per segment (S, 4, 4); a point takes the one of its segment."""

    segments: Segments
    matrices: np.ndarray

    @classmethod
    def identity(cls, segments: Segments) -> PassAlignment:
        return cls(segments, np.repeat(np.eye(4)[None], len(segments.centres), axis=0))

    def at(self, x) -> np.ndarray:
        """The transforms (n, 4, 4) for the positions `x` (n,)."""
        return self.matrices[self.segments.index(x)]

    def move(self, X, x) -> np.ndarray:
        """The points `X` (n, 3) at the positions `x` (n,), each moved by the transform of its segment; a NaN
        row stays NaN. With one segment this is the plain rigid move `X @ R.T + t`, byte for byte."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        idx = self.segments.index(x)
        if len(idx) != len(X):
            raise ValueError(f"{len(X)} points with {len(idx)} positions")
        out = np.empty_like(X)
        for j in np.unique(idx):
            sel = idx == j
            A = self.matrices[j]
            out[sel] = (X if sel.all() else X[sel]) @ A[:3, :3].T + A[:3, 3]
        return out

    def after(self, first: PassAlignment) -> PassAlignment:
        """`first`, then this alignment, segment by segment."""
        if not _same_segments(self.segments, first.segments):
            raise ValueError("alignments on different segments")
        return PassAlignment(self.segments, np.stack([a @ b for a, b in zip(self.matrices, first.matrices)]))


@dataclass(frozen=True)
class Fit:
    """The transform a segment uses. `n` is the segment's own count of pairs (or points, tracks, frames),
    `use` one of `USES`, `residual_m` the median pair residual of the fit it uses (None for the identity)."""

    matrix: np.ndarray
    n: int
    use: str
    residual_m: float | None


def _rotation(p: np.ndarray) -> np.ndarray:
    """The rotation `Rz(yaw) @ Ry(pitch) @ Rx(roll)` of `p = (roll, pitch, yaw, ...)` (rad)."""
    return Rotation.from_euler("ZYX", [p[2], p[1], p[0]]).as_matrix()


def _huber(e: np.ndarray) -> np.ndarray:
    """`e` mapped so that its square is the Huber cost of scale k = `HUBER_M`: e^2 within k, 2k|e| - k^2 beyond."""
    a = np.abs(e)
    outer = np.sign(e) * np.sqrt(np.maximum(2.0 * HUBER_M * a - HUBER_M**2, 0.0))
    return np.where(a <= HUBER_M, e, outer)


def _refine(M: np.ndarray, R: np.ndarray, d: np.ndarray, sigma_rp_deg: float | None) -> tuple[np.ndarray, float]:
    """The rigid transform moving the points `M` onto their pairs `R`, starting from the shift `d`, and the
    median pair distance after it.

    The variables are roll, pitch and yaw about the centroid c of `M` and a translation on top of `d`: a
    point moves to `(M - c) @ rot(p).T + c + d + dt`, `rot` being `_rotation`. Point residual components
    go through `_huber`; with `sigma_rp_deg` the quadratic prior terms `HUBER_M * roll / sigma` and
    `HUBER_M * pitch / sigma` follow.
    """
    c = M.mean(axis=0)
    sigma = None if sigma_rp_deg is None else np.radians(sigma_rp_deg)

    def residuals(p: np.ndarray) -> np.ndarray:
        e = _huber(((M - c) @ _rotation(p).T + c + d + p[3:] - R).ravel())
        if sigma is None:
            return e
        return np.concatenate([e, [HUBER_M * p[0] / sigma, HUBER_M * p[1] / sigma]])

    sol = least_squares(residuals, np.zeros(6), loss="linear", method="trf", ftol=1e-12, xtol=1e-12, gtol=1e-12)
    rot = _rotation(sol.x)
    A = np.eye(4)
    A[:3, :3] = rot
    A[:3, 3] = c + d + sol.x[3:] - rot @ c
    return A, float(np.median(np.linalg.norm(M @ rot.T + A[:3, 3] - R, axis=1)))


def _mutual(pm: np.ndarray, pr: np.ndarray, e: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """The kept candidates (indices, ascending, so in moving then reference order) that are mutual nearest:
    the closest reference point of their moving point and the closest moving point of their reference
    point, by `e`, the smaller index on a tie."""
    k = np.flatnonzero(keep)
    if len(k) == 0:
        return k
    by_m = k[np.lexsort((pr[k], e[k], pm[k]))]
    by_m = by_m[np.r_[True, pm[by_m][1:] != pm[by_m][:-1]]]
    by_r = k[np.lexsort((pm[k], e[k], pr[k]))]
    by_r = by_r[np.r_[True, pr[by_r][1:] != pr[by_r][:-1]]]
    return np.intersect1d(by_m, by_r)


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """The rigid transform (no reflection) moving the points `P` onto `Q` in least squares."""
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    s = 1.0 if np.linalg.det(Vt.T @ U.T) >= 0 else -1.0
    rot = Vt.T @ np.diag([1.0, 1.0, s]) @ U.T
    H = np.eye(4)
    H[:3, :3] = rot
    H[:3, 3] = cq - rot @ cp
    return H


def _candidates(moving: np.ndarray, reference: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Every reference point within `radius_m` of each moving point, as index arrays (moving, reference) in
    moving then reference order."""
    near = cKDTree(reference).query_ball_point(moving, radius_m)
    pm = np.array([m for m, rs in enumerate(near) for _ in rs], dtype=np.int64)
    pr = np.array([r for rs in near for r in sorted(rs)], dtype=np.int64)
    return pm, pr


def _check(M: np.ndarray, R: np.ndarray, sigma_rp_deg: float | None) -> bool:
    """Whether a fit beats the identity on held-out pairs. The pairs (moving points `M` before any move, row
    by row with their reference points `R`) fall into two folds by alternate rank along the row (x, stable
    order); each fold is measured by the refinement of the other started from its mean pair displacement.
    The median held-out distance must lie more than `CHECK_MARGIN_M` below the median pair distance at the
    identity."""
    rank = np.empty(len(M), dtype=np.int64)
    rank[np.argsort(M[:, 0], kind="stable")] = np.arange(len(M))
    fold = rank % 2
    held = np.empty(len(M))
    for f in (0, 1):
        train, test = fold != f, fold == f
        Af, _ = _refine(M[train], R[train], (R[train] - M[train]).mean(axis=0), sigma_rp_deg)
        held[test] = np.linalg.norm(M[test] @ Af[:3, :3].T + Af[:3, 3] - R[test], axis=1)
    return bool(np.median(held) < np.median(np.linalg.norm(R - M, axis=1)) - CHECK_MARGIN_M)


def fit_rigid(moving, reference, *, radius_m: float, sigma_rp_deg: float | None, min_pairs: int,
              rng: np.random.Generator, rigid_draws: int = 0,
              shift: np.ndarray | None = None) -> tuple[np.ndarray | None, int, float | None]:
    """The rigid transform moving the points `moving` onto `reference` without known pairs, its pair count
    and its median pair residual; `(None, pairs, None)` with fewer than `min_pairs` pairs or when the fit
    does not beat the identity on held-out pairs (`_check`), and `(None, 0, None)` without candidates.
    `rigid_draws` three-pair hypotheses follow the shift hypotheses. With `shift` (3,) the fit runs on the
    moving points moved by it and the transform returned includes it, so it applies to `moving` as given;
    the check reads the points before the move."""
    original = np.asarray(moving, dtype=np.float64).reshape(-1, 3)
    moving = original if shift is None else original + shift
    reference = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    if len(moving) == 0 or len(reference) == 0:
        return None, 0, None
    pm, pr = _candidates(moving, reference, radius_m)
    if len(pm) == 0:
        return None, 0, None

    D = reference[pr] - moving[pm]
    best = None                              # ((inliers, -distance sum), distances, start or None)
    for i in rng.choice(len(pm), size=min(RANSAC_DRAWS, len(pm)), replace=False):
        e = np.linalg.norm(D - D[i], axis=1)
        inliers = e <= INLIER_M
        score = (int(inliers.sum()), -float(e[inliers].sum()))
        if best is None or score > best[0]:
            best = (score, e, D[i])
    for _ in range(rigid_draws if len(pm) >= 3 else 0):
        k = rng.choice(len(pm), size=3, replace=False)
        if len(set(pm[k].tolist())) < 3 or len(set(pr[k].tolist())) < 3:
            continue
        P = moving[pm[k]]
        if np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0])) < RIGID_AREA_M2:
            continue
        H = _kabsch(P, reference[pr[k]])
        e = np.linalg.norm(moving[pm] @ H[:3, :3].T + H[:3, 3] - reference[pr], axis=1)
        inliers = e <= INLIER_M
        score = (int(inliers.sum()), -float(e[inliers].sum()))
        if score > best[0]:
            best = (score, e, None)

    _, e, start = best
    pairs = _mutual(pm, pr, e, e <= INLIER_M)
    if len(pairs) < min_pairs:
        return None, len(pairs), None
    M, R = moving[pm[pairs]], reference[pr[pairs]]
    A, residual = _refine(M, R, (R - M).mean(axis=0) if start is None else start, sigma_rp_deg)
    for _ in range(REFIT_ROUNDS):
        e = np.linalg.norm(moving[pm] @ A[:3, :3].T + A[:3, 3] - reference[pr], axis=1)
        again = _mutual(pm, pr, e, e <= INLIER_M)
        if len(again) < min_pairs or np.array_equal(again, pairs):
            break
        pairs = again
        M, R = moving[pm[pairs]], reference[pr[pairs]]
        A, residual = _refine(M, R, (R - M).mean(axis=0), sigma_rp_deg)
    if not _check(original[pm[pairs]], reference[pr[pairs]], sigma_rp_deg):
        return None, len(pairs), None
    if shift is not None:
        T = np.eye(4)
        T[:3, 3] = shift
        A = A @ T
    return A, len(pairs), residual


def _coarse_shift(moving: np.ndarray, reference: np.ndarray) -> np.ndarray | None:
    """The shift most of the candidate displacements agree on: every displacement from a moving point to a
    reference point within `COARSE_RADIUS_M` (moving then reference order) votes for each displacement
    within `COARSE_VOTE_M` of it, itself included; the componentwise median of the displacements around the
    most voted one (the first on a tie). None without candidates or below `COARSE_MIN_VOTES` votes."""
    moving = np.asarray(moving, dtype=np.float64).reshape(-1, 3)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    if len(moving) == 0 or len(reference) == 0:
        return None
    pm, pr = _candidates(moving, reference, COARSE_RADIUS_M)
    if len(pm) == 0:
        return None
    D = reference[pr] - moving[pm]
    tree = cKDTree(D)
    votes = tree.query_ball_point(D, COARSE_VOTE_M, return_length=True)
    best = int(np.argmax(votes))
    if votes[best] < COARSE_MIN_VOTES:
        return None
    return np.median(D[tree.query_ball_point(D[best], COARSE_VOTE_M)], axis=0)


def _identity_pairs(moving: np.ndarray, reference: np.ndarray) -> int:
    """The pairs the identity already has: the mutual nearest pairs among the candidates within `INLIER_M`
    (moving then reference order), by the distance without any move."""
    moving = np.asarray(moving, dtype=np.float64).reshape(-1, 3)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    if len(moving) == 0 or len(reference) == 0:
        return 0
    pm, pr = _candidates(moving, reference, INLIER_M)
    e = np.linalg.norm(reference[pr] - moving[pm], axis=1)
    return len(_mutual(pm, pr, e, np.ones(len(pm), dtype=bool)))


def fit_known(moving, reference, *, sigma_rp_deg: float | None) -> tuple[np.ndarray, float]:
    """The rigid transform moving each row of `moving` onto the same row of `reference`, and its median pair
    residual."""
    moving = np.asarray(moving, dtype=np.float64).reshape(-1, 3)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    if len(moving) == 0 or moving.shape != reference.shape:
        raise ValueError(f"known pairs need as many moving points as reference points, got {len(moving)} "
                         f"and {len(reference)}")
    return _refine(moving, reference, (reference - moving).mean(axis=0), sigma_rp_deg)


def with_fallback(own: Sequence[tuple[np.ndarray | None, int, float | None]], segments: Segments,
                  whole: Callable[[], tuple[np.ndarray | None, int, float | None]]) -> list[Fit]:
    """The fit each segment uses from the fits of its own (`own[j]` for segment j): its own, else that of the
    nearest segment with one; with none at all, `whole()` (called once) for every segment, else the
    identity. `n` is always the segment's own count."""
    has = [j for j, (A, _, _) in enumerate(own) if A is not None]
    if has:
        fits = []
        for j, (A, n, residual) in enumerate(own):
            if A is not None:
                fits.append(Fit(A, int(n), "own", residual))
            else:
                k = min(has, key=lambda i: (abs(segments.centres[i] - segments.centres[j]), i))
                fits.append(Fit(own[k][0], int(n), "nearest", own[k][2]))
        return fits
    A, _, residual = whole()
    if A is not None:
        return [Fit(A, int(n), "group", residual) for _, n, _ in own]
    return [Fit(np.eye(4), int(n), "identity", None) for _, n, _ in own]


def fit_segments(moving, x, reference, segments: Segments, *, radius_m: float, sigma_rp_deg: float | None,
                 min_pairs: int, key: Sequence[int], coarse: bool = False) -> list[Fit]:
    """The fit of each segment: the moving points at positions `x` inside its window onto all of
    `reference`, drawn from `noise_rng(*key, j)`, with `with_fallback`; the fit of the whole group uses
    every moving point, `noise_rng(*key, len(segments.centres))` and the three-pair hypotheses. With
    `coarse`, a fit of the whole group short of pairs runs once more after `_coarse_shift`, going on with
    the same generator, and stands only with more pairs than the identity has (`_identity_pairs`)."""
    moving = np.asarray(moving, dtype=np.float64).reshape(-1, 3)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    count = len(segments.centres)
    own = [fit_rigid(moving[segments.window(j, x)], reference, radius_m=radius_m, sigma_rp_deg=sigma_rp_deg,
                     min_pairs=min_pairs, rng=noise_rng(*key, j)) for j in range(count)]

    def whole() -> tuple[np.ndarray | None, int, float | None]:
        rng = noise_rng(*key, count)
        first = fit_rigid(moving, reference, radius_m=radius_m, sigma_rp_deg=sigma_rp_deg, min_pairs=min_pairs,
                          rng=rng, rigid_draws=RANSAC_DRAWS_RIGID)
        if coarse and first[0] is None and first[1] < min_pairs:
            s = _coarse_shift(moving, reference)
            if s is not None:
                A, n, residual = fit_rigid(moving, reference, radius_m=radius_m, sigma_rp_deg=sigma_rp_deg,
                                           min_pairs=min_pairs, rng=rng, rigid_draws=RANSAC_DRAWS_RIGID, shift=s)
                if A is not None and n > _identity_pairs(moving, reference):
                    return A, n, residual
        return first

    return with_fallback(own, segments, whole)


@dataclass(frozen=True, eq=False)
class AlignPass:
    """The tracks of one pass as the alignment reads them: centres `X` (T, 3) with NaN rows for tracks without
    3D, observation counts `n_obs` (T,) and positions along the row `x` (T,). No camera, pose or truth id."""

    key: PassKey
    X: np.ndarray
    n_obs: np.ndarray
    x: np.ndarray


def track_x(frames: Sequence[frozenset[int]], frame_x: Mapping[int, float]) -> np.ndarray:
    """The position along the row of each track: the median of `frame_x` over the frames that observed it
    (`frames[t]`), NaN for a track without frames."""
    out = np.full(len(frames), np.nan)
    for t, seen in enumerate(frames):
        if seen:
            out[t] = np.median([frame_x[k] for k in sorted(seen)])
    return out


@dataclass(frozen=True)
class AlignSettings:
    """The values `align_passes` runs with: the segment length (m), the pair radius (m), the roll and pitch
    prior sigma (deg, None for no prior), the fewest pairs of a fit and the fewest tracks per side of a row
    plane midpoint, the fewest observations of a used track, the bias (m) added to the row centre plane and
    the share of moving tracks kept in the fits between the two sides (None keeps all)."""

    segment_m: float = SEGMENT_M
    pair_radius_m: float = PAIR_RADIUS_M
    roll_pitch_sigma_deg: float | None = None
    min_pairs: int = MIN_PAIRS
    min_obs: int = 1
    row_bias_m: float = 0.0
    pair_keep: float | None = None


@dataclass(frozen=True)
class StageRecord:
    """The fits of one stage for one group, one per segment. `group` is the side letter for a rig group,
    pitch and side (`"10B"`) for a pass, `"AB"` for the row centre plane."""

    stage: str
    group: str
    segments: Segments
    fits: list[Fit]


@dataclass(frozen=True, eq=False)
class AlignResult:
    """The transform of every pass after all stages, in the order of the passes, and the stage records in
    the order they ran."""

    transforms: dict[PassKey, PassAlignment]
    stages: list[StageRecord]


def _theil_sen(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """The Theil-Sen line through the points (x, y): the median slope over all pairs i < j with distinct x,
    and the intercept median(y) - slope * median(x), as `scipy.stats.theilslopes` gives by default."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    keep = dx != 0
    slope = float(np.median((y[j] - y[i])[keep] / dx[keep]))
    return slope, float(np.median(y) - slope * np.median(x))


def _stack(segments: Segments, fits: Sequence[Fit]) -> PassAlignment:
    return PassAlignment(segments, np.stack([f.matrix for f in fits]))


def align_passes(passes: Sequence[AlignPass], layout: OrchardLayout, methods: Sequence[str],
                 settings: AlignSettings, *, rig: bool, ref_pitch: float, key: Sequence[int]) -> AlignResult:
    """The transforms that align the passes, stage by stage in the order of `methods` (`tracks`, `row_plane`),
    on the segments along the trees of `layout`; `key` = (seed, noise seed or 0, scene) keys the draws.
    `rig` fits side B as one group, and without it the passes of a side are first fitted to the one at
    `ref_pitch`."""
    for name in methods:
        if name not in _METHODS:
            raise ValueError(f"unknown alignment method: {name}")
    segments = Segments.along(layout.tree_x, settings.segment_m)
    transforms = {p.key: PassAlignment.identity(segments) for p in passes}
    stages: list[StageRecord] = []
    used = {}                                # pass key -> its used track centres (unmoved) and positions
    for p in passes:
        X = np.asarray(p.X, dtype=np.float64).reshape(-1, 3)
        keep = (np.isfinite(X).all(axis=1) & (np.asarray(p.n_obs) >= settings.min_obs)
                & (np.abs(X[:, 1] - layout.row_y) < layout.row_spacing / 2))
        used[p.key] = (X[keep], np.asarray(p.x, dtype=np.float64).reshape(-1)[keep])

    def points(group: Sequence[AlignPass]) -> tuple[np.ndarray, np.ndarray]:
        """The used tracks of `group`, moved by the transforms so far, and their positions along the row."""
        X, x = [np.empty((0, 3))], [np.empty(0)]
        for p in group:
            pX, px = used[p.key]
            X.append(transforms[p.key].move(pX, px))
            x.append(px)
        return np.concatenate(X), np.concatenate(x)

    def fit_group(stage: str, label: str, group: Sequence[AlignPass], reference: Sequence[AlignPass],
                  code: tuple[int, int, int]) -> None:
        moving, x = points(group)
        if stage == "both_sides" and settings.pair_keep is not None:
            keep = noise_rng(key[0], key[1], STEP_ALIGN_THIN, key[2], *code).random(len(moving)) < settings.pair_keep
            moving, x = moving[keep], x[keep]
        fits = fit_segments(moving, x, points(reference)[0], segments, radius_m=settings.pair_radius_m,
                            sigma_rp_deg=settings.roll_pitch_sigma_deg, min_pairs=settings.min_pairs,
                            key=(key[0], key[1], STEP_ALIGN, key[2], *code), coarse=stage == "both_sides")
        stages.append(StageRecord(stage, label, segments, fits))
        step = _stack(segments, fits)
        for p in group:
            transforms[p.key] = step.after(transforms[p.key])

    def row_plane() -> None:
        Xa, xa = points([p for p in passes if p.key.side == "A"])
        Xb, xb = points([p for p in passes if p.key.side == "B"])
        ya, yb = Xa[:, 1], Xb[:, 1]
        count = len(segments.centres)
        mids, own, ns = np.zeros(count), np.zeros(count, dtype=bool), []
        for j in range(count):
            wa, wb = segments.window(j, xa), segments.window(j, xb)
            n = int(min(wa.sum(), wb.sum()))
            ns.append(n)
            if n >= settings.min_pairs:
                mids[j], own[j] = (np.median(ya[wa]) + np.median(yb[wb])) / 2, True
        if own.any():
            if own.sum() >= 2:
                slope, intercept = _theil_sen(mids[own], segments.centres[own])
                plane = intercept + slope * segments.centres
            else:
                plane = np.full(count, mids[own][0])
            uses = ["own" if o else "line" for o in own]
        elif len(ya) >= settings.min_pairs and len(yb) >= settings.min_pairs:
            plane, uses = np.full(count, (np.median(ya) + np.median(yb)) / 2), ["group"] * count
        else:
            plane, uses = None, ["identity"] * count
        fits = []
        for j in range(count):
            A = np.eye(4)
            if plane is not None:
                A[1, 3] = layout.row_y - (plane[j] + settings.row_bias_m)
            fits.append(Fit(A, ns[j], uses[j], None))
        stages.append(StageRecord("row_plane", "AB", segments, fits))
        step = _stack(segments, fits)
        for p in passes:
            transforms[p.key] = step.after(transforms[p.key])

    for method in methods:
        if method == "row_plane":
            row_plane()
            continue
        if not rig:
            for side, s in _SIDES.items():
                on_side = [p for p in passes if p.key.side == side]
                ref = next((p for p in on_side if abs(p.key.pitch_deg - ref_pitch) <= _PITCH_TOL), None)
                if ref is None:
                    continue
                for p in on_side:
                    if p is not ref:
                        fit_group("same_side", f"{p.key.pitch_deg:g}{side}", [p], [ref],
                                  (0, s, round(p.key.pitch_deg * 1000)))
        side_a = [p for p in passes if p.key.side == "A"]
        side_b = [p for p in passes if p.key.side == "B"]
        if side_a and side_b:
            if rig:
                fit_group("both_sides", "B", side_b, side_a, (1, _SIDES["B"], 0))
            else:
                for p in side_b:
                    fit_group("both_sides", f"{p.key.pitch_deg:g}B", [p], side_a,
                              (1, _SIDES["B"], round(p.key.pitch_deg * 1000)))
    return AlignResult(transforms, stages)
