"""The exact error ledger and its rate definitions.

`build_ledger` classifies every counted unit and every truth id into the
decomposition's named terms, so that `n_pred - n_truth` equals the signed sum
of those terms exactly (`Ledger.check()`). `combine` sums ledgers field by
field, e.g. across passes or trees.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from typing import AbstractSet, Iterable, Sequence


@dataclass(frozen=True)
class UnitView:
    """One counted unit (a track, a landmark, ...): its majority source id,
    purity (majority count / observation count), and observed ids."""

    majority: int
    purity: float
    obs_ids: tuple[int, ...]


@dataclass(frozen=True)
class Ledger:
    """Exact decomposition of `n_pred - n_truth`."""

    n_pred: int
    n_truth: int
    fp_bg: int
    fp_off: int
    boundary: int
    dup_track: int
    dup_cross: int
    miss_det: int
    miss_drop: int
    miss_merge: int
    impure: int

    def check(self) -> None:
        """Raise `AssertionError` if the identity does not hold (kept explicit, not
        a bare `assert`, so it still runs under `python -O`)."""
        lhs = self.n_pred - self.n_truth
        rhs = (
            self.fp_bg + self.fp_off + self.boundary + self.dup_track + self.dup_cross
            - (self.miss_det + self.miss_drop + self.miss_merge)
        )
        if lhs != rhs:
            raise AssertionError(f"ledger identity violated: {lhs} != {rhs} ({self!r})")

    def rates(self) -> dict[str, float | None]:
        """Error rates over `n_truth`; every rate is `None` when `n_truth` is 0."""
        keys = (
            "dup_rate", "miss_rate", "excess_rate",
            "fp_off_rate", "fp_bg_rate", "boundary_rate", "merge_rate",
        )
        if self.n_truth == 0:
            return {k: None for k in keys}
        n = self.n_truth
        values = (
            (self.dup_track + self.dup_cross) / n,
            (self.miss_det + self.miss_drop + self.miss_merge) / n,
            (self.dup_track + self.dup_cross + self.fp_off + self.fp_bg) / n,
            self.fp_off / n,
            self.fp_bg / n,
            self.boundary / n,
            self.impure / n,
        )
        return dict(zip(keys, values))

    def as_dict(self) -> dict:
        """Fields plus `rates()`, for report and CSV building."""
        return {f.name: getattr(self, f.name) for f in fields(self)} | self.rates()


def build_ledger(
    *,
    truth_ids: AbstractSet[int],
    target_ids: AbstractSet[int],
    units: Sequence[UnitView],
    track_majorities: Sequence[tuple[int, int]],
    kept_obs_ids: AbstractSet[int],
    detected_ids: AbstractSet[int],
) -> Ledger:
    """Build the ledger for one pass/tree/run and `check()` it before returning.

    `truth_ids` must be a subset of `target_ids`, or `ValueError`.
    `track_majorities` lists `(pass_index, majority)` of every kept track behind
    `units`: one per unit in Y1, every member track of the landmarks in Y6. `k_g`
    comes from `units` and `k_{g,p}` from these pairs alone, which splits Dup into
    `dup_track` and `dup_cross`.
    """
    if not truth_ids <= target_ids:
        raise ValueError(f"truth_ids not a subset of target_ids: {sorted(truth_ids - target_ids)}")

    k_g: Counter[int] = Counter()
    k_gp: dict[int, Counter[int]] = defaultdict(Counter)
    fp_bg = 0
    fp_off = 0
    for unit in units:
        m = unit.majority
        if m == -1:
            fp_bg += 1
        elif m not in target_ids:
            fp_off += 1
        else:
            k_g[m] += 1
    for p, g in track_majorities:
        k_gp[g][p] += 1

    boundary = sum(1 for g in target_ids - truth_ids if k_g[g] >= 1)

    dup_track = 0
    dup_cross = 0
    for g in target_ids:
        d = max(k_g[g] - 1, 0)
        if d == 0:
            continue
        max_p = max(k_gp[g].values(), default=0)
        track = min(d, max(max_p - 1, 0))
        dup_track += track
        dup_cross += d - track

    miss_det = 0
    miss_drop = 0
    miss_merge = 0
    for g in truth_ids:
        if k_g[g] >= 1:
            continue
        if g not in detected_ids:
            miss_det += 1
        elif g in kept_obs_ids:
            miss_merge += 1
        else:
            miss_drop += 1

    impure = sum(1 for u in units if u.purity < 0.8)

    ledger = Ledger(
        n_pred=len(units),
        n_truth=len(truth_ids),
        fp_bg=fp_bg,
        fp_off=fp_off,
        boundary=boundary,
        dup_track=dup_track,
        dup_cross=dup_cross,
        miss_det=miss_det,
        miss_drop=miss_drop,
        miss_merge=miss_merge,
        impure=impure,
    )
    ledger.check()
    return ledger


def combine(ledgers: Iterable[Ledger]) -> Ledger:
    """Sum every field of `ledgers` (e.g. across passes or trees)."""
    totals = {f.name: 0 for f in fields(Ledger)}
    for ledger in ledgers:
        for name in totals:
            totals[name] += getattr(ledger, name)
    return Ledger(**totals)
