from dataclasses import replace

import numpy as np

from counting.config.eval_schema import DedupConfig
from counting.data.formats.orchard_synth import OrchardLayout, PassKey
from counting.multiview.assign import assign_trees, in_target_row
from counting.multiview.dedup import dedup, repair_pass
from counting.tracking.landmark import PassTracks

LAYOUT = OrchardLayout("apple", (0.0, 3.75, 7.5), 1, 3.75, (3, 4, 5), (0.0, 1.25, 2.5))
A30, B30, A50, B50 = PassKey(30.0, "A", 2), PassKey(30.0, "B", 1), PassKey(50.0, "A", 2), PassKey(50.0, "B", 1)
P = np.array([[0.0, 3.95, 1.5], [0.4, 3.55, 1.8], [0.9, 4.0, 1.65], [1.3, 3.6, 2.0]])
R4 = DedupConfig(radius_m=0.04)


def _pt(key, X, frames=None, labels=None):
    X = np.asarray(X, float)
    n = len(X)
    return PassTracks(pass_key=key, det_track={}, X=X,
                      labels=np.asarray(labels if labels is not None else [0] * n, np.int8),
                      n_obs=np.ones(n, int), spread=np.zeros(n),
                      frames=frames or [frozenset({i}) for i in range(n)])


def _all(*pts):
    return [np.ones(len(p.X), bool) for p in pts]


def _sets(lm):
    return {frozenset(m) for m in lm.members}


def test_merge_joins_same_points_across_passes():
    rng = np.random.default_rng(0)
    a, b = _pt(A30, P + rng.normal(0, 0.005, P.shape)), _pt(B30, P + rng.normal(0, 0.005, P.shape))
    lm = dedup([a, b], _all(a, b), LAYOUT, R4, mode="merge")
    assert len(lm.X) == 4 and all({k for k, _ in m} == {A30, B30} and len(m) == 2 for m in lm.members)
    r = _pt(B30, P[::-1])   # landmarks come out by smallest member number, members ascending
    assert dedup([a, r], _all(a, r), LAYOUT, R4, mode="merge").members == [
        [(A30, 0), (B30, 3)], [(A30, 1), (B30, 2)], [(A30, 2), (B30, 1)], [(A30, 3), (B30, 0)]]


def test_cooccurring_tracks_never_merge():
    a = _pt(A30, [[0, 3.9, 1.5], [0.01, 3.9, 1.5]], frames=[frozenset({1, 2}), frozenset({2, 3})])
    assert len(dedup([a], _all(a), LAYOUT, R4, mode="merge").X) == 2


def test_split_track_in_one_pass_is_repaired():
    a = _pt(A30, [[0, 3.9, 1.5], [0.01, 3.9, 1.5]], frames=[frozenset({1, 2}), frozenset({3, 4})])
    assert len(dedup([a], _all(a), LAYOUT, R4, mode="merge").X) == 1


def test_forbidden_merge_blocks_whole_component():
    a = _pt(A30, [[0, 3.9, 1.5], [0.03, 3.9, 1.5]], frames=[frozenset({1}), frozenset({1})])
    b = _pt(B30, [[0.01, 3.9, 1.5]])
    lm = dedup([a, b], _all(a, b), LAYOUT, R4, mode="merge")
    assert _sets(lm) == {frozenset({(A30, 0), (B30, 0)}), frozenset({(A30, 1)})}
    b2 = _pt(B30, [[0.02, 3.9, 1.5]])   # nearer to (A30, 1): the closest pair is joined first
    lm2 = dedup([a, b2], _all(a, b2), LAYOUT, R4, mode="merge")
    assert _sets(lm2) == {frozenset({(A30, 1), (B30, 0)}), frozenset({(A30, 0)})}


def test_self_side_keeps_own_half_and_never_crosses_sides():
    near, far, on = [0.5, 3.95, 1.6], [1.5, 3.55, 1.6], [2.0, 3.75, 1.6]
    a, b = _pt(A30, [near, far, on]), _pt(B30, [near, far, on])
    lm = dedup([a, b], _all(a, b), LAYOUT, DedupConfig(radius_m=0.09), mode="self_side")
    assert _sets(lm) == {frozenset({(A30, 0)}), frozenset({(B30, 1)}), frozenset({(A30, 2)})}
    lm2 = dedup([a, b], _all(a, b), LAYOUT, DedupConfig(radius_m=0.09), mode="merge")
    assert len(lm2.X) == 3 and all(len(m) == 2 for m in lm2.members)
    c, d = _pt(A30, [[1.0, 3.76, 1.6]]), _pt(B30, [[1.0, 3.74, 1.6]])   # 2 cm apart across the plane
    assert len(dedup([c, d], _all(c, d), LAYOUT, DedupConfig(radius_m=0.09), mode="self_side").X) == 2


def test_self_side_merges_pitches_within_side():
    a, c = _pt(A30, [[0.5, 3.95, 1.6]]), _pt(A50, [[0.52, 3.95, 1.6]])
    assert len(dedup([a, c], _all(a, c), LAYOUT, DedupConfig(radius_m=0.09), mode="self_side").X) == 1


def test_hybrid_band_filter_and_cross_side_rule():
    cfg = DedupConfig(radius_m=0.09, band_m=0.03)
    a = _pt(A30, [[1.0, 3.80, 1.6], [2.0, 3.73, 1.6]])                     # A: 5 cm on its own side, 2 cm across the plane
    b = _pt(B30, [[1.0, 3.77, 1.6], [2.0, 3.74, 1.6], [0.5, 3.85, 1.6]])   # B: 2 cm across, 1 cm on its own side, 10 cm across
    assert _sets(dedup([a, b], _all(a, b), LAYOUT, cfg, mode="hybrid")) == {
        frozenset({(A30, 0)}), frozenset({(B30, 0)}), frozenset({(A30, 1), (B30, 1)})}   # (A30, 0) lies outside the band
    assert _sets(dedup([a, b], _all(a, b), LAYOUT, cfg, mode="merge")) == {
        frozenset({(A30, 0), (B30, 0)}), frozenset({(A30, 1), (B30, 1)}), frozenset({(B30, 2)})}


def test_hybrid_band_zero_keeps_sides_apart():
    c, d = _pt(A30, [[1.0, 3.76, 1.6]]), _pt(B30, [[1.0, 3.74, 1.6]])     # 2 cm apart across the plane
    for band, n in ((0.0, 2), (0.05, 1)):
        assert len(dedup([c, d], _all(c, d), LAYOUT, DedupConfig(radius_m=0.09, band_m=band), mode="hybrid").X) == n


def test_hybrid_band_bounds_on_the_edge():
    """A keeps y = row_y - band, B drops y = row_y + band, and |y - row_y| = band is not inside the band."""
    cfg = DedupConfig(radius_m=0.09, band_m=0.25)                 # row_y 3.75: the bounds 3.5 and 4.0 are exact
    a = _pt(A30, [[1.0, 3.5, 1.6], [3.0, 3.6, 1.6]])
    b = _pt(B30, [[1.0, 3.5, 1.6], [2.0, 4.0, 1.6], [3.0, 3.6, 1.6]])
    assert _sets(dedup([a, b], _all(a, b), LAYOUT, cfg, mode="hybrid")) == {
        frozenset({(A30, 0)}), frozenset({(B30, 0)}),            # both on the band edge: not joined across sides
        frozenset({(A30, 1), (B30, 2)})}                         # both inside the band: joined; (B30, 1) is dropped


def test_hybrid_band_limits_match_self_side_and_merge():
    rng = np.random.default_rng(3)
    pts = [_pt(k, np.c_[rng.uniform(0, 2.5, 40), rng.normal(3.75, 0.06, 40), rng.uniform(1, 2, 40)])
           for k in (A30, B30, A50, B50)]
    keep = [rng.random(40) < 0.9 for _ in pts]
    for band, other in ((0.0, "self_side"), (1e9, "merge")):
        cfg = DedupConfig(radius_m=0.09, band_m=band)
        h, o = dedup(pts, keep, LAYOUT, cfg, mode="hybrid"), dedup(pts, keep, LAYOUT, cfg, mode=other)
        assert h.members == o.members and np.array_equal(h.X, o.X), band
    n = [len(dedup(pts, keep, LAYOUT, DedupConfig(radius_m=0.09, band_m=0.05), mode=m).X)
         for m in ("self_side", "hybrid", "merge")]
    assert n == [58, 82, 102]      # the limits differ, so the equalities above are not vacuous


def test_keep_mask_respected():
    a = _pt(A30, [[0, 3.9, 1.5], [1.0, 3.9, 1.5]])
    lm = dedup([a], [np.array([True, False])], LAYOUT, R4, mode="merge")
    assert _sets(lm) == {frozenset({(A30, 0)})}


def test_empty_input_gives_no_landmarks():
    a = _pt(A30, [[0, 3.9, 1.5], [1.0, 3.9, 1.5]])
    lm = dedup([a], [np.zeros(2, bool)], LAYOUT, R4, mode="merge")
    assert lm.X.shape == (0, 3) and lm.members == []


def test_landmark_median_and_label():
    a = _pt(A30, [[0, 3.9, 1.5]], labels=[1])
    b = _pt(B30, [[0.02, 3.9, 1.5]], labels=[1])
    c = _pt(A50, [[0.01, 3.9, 1.5]], labels=[0])
    lm = dedup([a, b, c], _all(a, b, c), LAYOUT, R4, mode="merge")
    np.testing.assert_allclose(lm.X, [[0.01, 3.9, 1.5]])
    assert lm.label.tolist() == [1]


def test_repair_pass_merges_split_track_only():
    a = PassTracks(pass_key=A30, det_track={(1, 0): 0, (2, 0): 0, (3, 0): 1, (4, 0): 1, (3, 1): 2, (9, 0): 3},
                   X=np.array([[0, 3.9, 1.5], [0.01, 3.9, 1.5], [0.02, 3.9, 1.5], [np.nan] * 3]),
                   labels=np.zeros(4, np.int8), n_obs=np.array([2, 2, 1, 1]), spread=np.zeros(4),
                   frames=[frozenset({1, 2}), frozenset({3, 4}), frozenset({3}), frozenset({9})])
    r = repair_pass(a, 0.04)
    assert r.n_obs.tolist() == [4, 1, 1] and r.frames == [frozenset({1, 2, 3, 4}), frozenset({3}), frozenset({9})]
    assert r.det_track == {(1, 0): 0, (2, 0): 0, (3, 0): 0, (4, 0): 0, (3, 1): 1, (9, 0): 2}
    np.testing.assert_allclose(r.X[0], [0.005, 3.9, 1.5])
    assert np.isnan(r.X[2]).all() and repair_pass(a, 0.0) is a
    r2 = repair_pass(replace(a, labels=np.array([1, 0, 0, 0], np.int8), n_obs=np.array([3, 2, 1, 1]),
                             spread=np.array([0.01, 0.02, 0.0, 0.0])), 0.04)
    assert r2.labels.tolist() == [1, 0, 0] and r2.spread[0] == 0.02   # n_obs-weighted label, max spread


def test_in_target_row_and_assign_trees():
    X = np.array([[0.62, 3.75, 1], [0.625, 3.75, 1], [-5, 3.75, 1], [99, 5.62, 1],
                  [0, 5.625, 1], [np.nan, 3.75, 1]], float)
    assert in_target_row(X, LAYOUT).tolist() == [True, True, True, True, False, False]
    assert assign_trees(X, LAYOUT).tolist() == [3, 4, 3, 5, -1, -1]
