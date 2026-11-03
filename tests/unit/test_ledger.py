import dataclasses

import numpy as np
import pytest

from counting.evaluation.ledger import UnitView, build_ledger, combine
from counting.evaluation.match import unit_majority


def _u(*obs):
    m, p = unit_majority([obs])
    return UnitView(int(m[0]), float(p[0]), tuple(obs))


UNITS = [_u(1, 1), _u(1), _u(2, 2, 3), _u(4), _u(-1), _u(99)]
KW = dict(truth_ids={1, 2, 3, 5}, target_ids={1, 2, 3, 4, 5, 6}, units=UNITS,
          track_majorities=[(0, 1), (1, 1), (0, 2), (0, 4), (0, -1), (0, 99)],
          kept_obs_ids={1, 2, 3, 4, -1, 99}, detected_ids={1, 2, 3, 4, 5, 99})


def test_terms_are_classified():
    L = build_ledger(**KW)
    assert (L.n_pred, L.n_truth, L.fp_bg, L.fp_off, L.boundary) == (6, 4, 1, 1, 1)
    assert (L.dup_track, L.dup_cross) == (0, 1)
    assert (L.miss_det, L.miss_drop, L.miss_merge, L.impure) == (0, 1, 1, 1)
    r = L.rates()
    assert r["excess_rate"] == 0.75 and r["miss_rate"] == 0.5 and r["merge_rate"] == 0.25


def test_rates_definitions():
    L = build_ledger(truth_ids={1, 2}, target_ids={1, 2, 3}, units=[_u(1), _u(1), _u(-1), _u(-1)],
                     track_majorities=[(0, 1), (1, 1), (0, -1), (1, -1)], kept_obs_ids={1, -1}, detected_ids={1, 2})
    assert (L.dup_cross, L.fp_bg, L.boundary, L.miss_drop) == (1, 2, 0, 1)
    r = L.rates()
    assert r["excess_rate"] == 1.5 and r["dup_rate"] == 0.5 and r["miss_rate"] == 0.5   # boundary is not excess
    M = build_ledger(truth_ids={1, 2, 3}, target_ids={1, 2, 3}, units=[_u(1, 2, 3)],
                     track_majorities=[(0, 1)], kept_obs_ids={1, 2, 3}, detected_ids={1, 2, 3})
    assert (M.impure, M.miss_merge) == (1, 2)
    assert M.rates()["merge_rate"] == pytest.approx(1 / 3) and M.rates()["miss_rate"] == pytest.approx(2 / 3)


def test_undetected_truth_is_miss_det():
    L = build_ledger(**{**KW, "detected_ids": {1, 2, 3, 4, 99}})
    assert (L.miss_det, L.miss_drop) == (1, 0)


def test_dup_in_same_pass_is_dup_track():
    L = build_ledger(truth_ids={1}, target_ids={1}, units=[_u(1), _u(1)],
                     track_majorities=[(0, 1), (0, 1)], kept_obs_ids={1}, detected_ids={1})
    assert (L.dup_track, L.dup_cross) == (1, 0)


def test_dup_split_counts_member_tracks_not_units():
    # Y6: landmark 0 fuses track 0 of pass 0 with track 0 of pass 1; landmark 1 is a second track of
    # pass 0. k_7 = 2 and k_{7,0} = 2, so the extra landmark is a split inside pass 0.
    L = build_ledger(truth_ids={7}, target_ids={7}, units=[_u(7, 7), _u(7)],
                     track_majorities=[(0, 7), (1, 7), (0, 7)], kept_obs_ids={7}, detected_ids={7})
    assert (L.dup_track, L.dup_cross) == (1, 0)


def test_truth_outside_target_rejected():
    with pytest.raises(ValueError):
        build_ledger(**{**KW, "truth_ids": {1, 7}})


def test_identity_holds_for_random_assignments():
    rng = np.random.default_rng(0)
    target = set(range(1, 21))
    for _ in range(300):
        truth = {int(x) for x in rng.choice(20, size=int(rng.integers(0, 21)), replace=False) + 1}
        pool = [-1, 98, 99] + sorted(target)
        units = [_u(*[int(x) for x in rng.choice(pool, size=int(rng.integers(1, 5)))])
                 for _ in range(int(rng.integers(0, 30)))]
        kept = {i for u in units for i in u.obs_ids}
        L = build_ledger(truth_ids=truth, target_ids=target, units=units,
                         track_majorities=[(int(rng.integers(0, 3)), u.majority) for u in units],
                         kept_obs_ids=kept, detected_ids=kept | {int(x) for x in rng.choice(20, 5) + 1})
        assert min(dataclasses.astuple(L)) >= 0


def test_combine_sums_fields():
    L = build_ledger(**KW)
    C = combine([L, L])
    assert C.n_pred == 12 and C.miss_merge == 2
    C.check()
