"""The error budget of a ladder: each rung step's change in count error, split into ledger terms.

`contrib_pairs` names the rung pairs of the contribution table and the budget. The main chain
(`L0`→…→`L3`) gives neighbour pairs; the `BRANCH_RUNGS` (`L4_base`, `L5`) branch off `L3` rather
than following each other, so `L3→L5` is the detector's share. The report-only pairs `L3_dd→L3`
and `L5→L5_L4base` come last. `path_pairs` is the part of those pairs that leads to `L5`; its
rows add up to `L5`'s error minus the first rung's.

A ladder with a pose rung (`POSE_ONLY_RUNGS`) takes instead the `POSE_PAIRS` whose two rungs both
ran (`ladder_pairs`).

`budget_rows` gives one row per pair and bundle, then a `pooled` row over the bundles. A row's
`d_e` is the change of (U - G) / G between the two rungs, and each ledger term of `TERMS` is its
change in counts over the truth count G, with sign -1 for the `miss_*` terms, so that the terms
add up to `d_e` (`residual` is what is left). The pooled row sums the counts before dividing
(Σ counts / Σ truth), rather than averaging the bundles' values. The Y6 rows read the row ledger
and the row's U and G6; the Y1 rows sum the pass ledgers and U1 and G1 of the passes at the given
pitches. `check_budget` adds the terms up again and refuses a row where they miss `d_e`.
`ladder_budget` builds the budget of a ladder with rung `L5` over its `ladder_pairs`: the Y6 rows,
then the Y1 rows per pitch, over the judged pitches and over every pass, all checked.
"""

from __future__ import annotations

import math
from typing import Literal, Mapping, Sequence

from counting.evaluation.verdict import JUDGED_PITCHES, pool_rung

# Rungs that branch off L3 instead of following each other on the main chain.
BRANCH_RUNGS = ("L4_base", "L5")
# The (from, to) rung pairs of a ladder with pose rungs, in table order.
POSE_PAIRS = (("L3", "L4_drift"), ("L3", "LVI"), ("L5", "L5VI"), ("LVI", "L5VI"), ("L4_drift_noimu", "L4_drift"),
              ("LV", "LVI"), ("LV_ref", "LV"), ("L4_drift_know", "L4_drift"), ("L4_drift_noal", "L4_drift"),
              ("LVI_first", "LVI"), ("LVI_fits", "LVI"), ("L4_drift_c1x", "L4_drift"), ("LVI_c1x", "LVI"),
              ("LVI_dd", "LVI"), ("L3", "L3_al"))
# The rungs that only a pose ladder has; any of them in a ladder's rungs selects `POSE_PAIRS`.
POSE_ONLY_RUNGS = frozenset(n for pair in POSE_PAIRS for n in pair) - {"L3", "L5"}
# The ledger terms of a budget row; `n_pred - n_truth` is their sum with `_SIGN`.
TERMS = ("fp_bg", "fp_off", "boundary", "dup_track", "dup_cross", "miss_det", "miss_drop", "miss_merge")
_SIGN = {t: -1 if t.startswith("miss_") else 1 for t in TERMS}


def _main_chain(chain: Sequence[str]) -> list[str]:
    return [name for name in chain if name not in BRANCH_RUNGS]


def contrib_pairs(chain: Sequence[str], rungs: Mapping[str, object]) -> list[tuple[str, str]]:
    """The (from, to) rung pairs of the contribution table and the budget, in table order.

    With `L3` in `chain`: the neighbour pairs of `chain` without the `BRANCH_RUNGS`, then `("L3", b)`
    for each `BRANCH_RUNGS` rung `b` in `chain`. Without `L3`: the neighbour pairs of `chain`.
    Then `("L3_dd", "L3")` when both ran (keys of `rungs`), and `("L5", "L5_L4base")` when both ran.
    """
    if "L3" in chain:
        main = _main_chain(chain)
        pairs = [*zip(main, main[1:]), *(("L3", b) for b in BRANCH_RUNGS if b in chain)]
    else:
        pairs = list(zip(chain, chain[1:]))
    for a, b in (("L3_dd", "L3"), ("L5", "L5_L4base")):
        if a in rungs and b in rungs:
            pairs.append((a, b))
    return pairs


def pose_pairs(rungs: Mapping[str, object]) -> list[tuple[str, str]]:
    """The `POSE_PAIRS` whose two rungs both ran (keys of `rungs`), in `POSE_PAIRS` order."""
    return [(a, b) for a, b in POSE_PAIRS if a in rungs and b in rungs]


def ladder_pairs(chain: Sequence[str], rungs: Mapping[str, object]) -> list[tuple[str, str]]:
    """The pairs of a ladder's contribution table and budget: `pose_pairs(rungs)` when a name of
    `rungs` is one of `POSE_ONLY_RUNGS`, else `contrib_pairs(chain, rungs)`. Only the names count,
    not the rungs' results."""
    if any(name in POSE_ONLY_RUNGS for name in rungs):
        return pose_pairs(rungs)
    return contrib_pairs(chain, rungs)


def path_pairs(chain: Sequence[str]) -> list[tuple[str, str]]:
    """The pairs from `chain[0]` to `L5`: the main chain's neighbour pairs up to `L3`, then `("L3", "L5")`;
    without `L3`, the neighbour pairs of `chain` up to `L5`. `[]` when `L5` is not in `chain`."""
    if "L5" not in chain:
        return []
    if "L3" in chain:
        main = _main_chain(chain)
        upto = main[: main.index("L3") + 1]
        return [*zip(upto, upto[1:]), ("L3", "L5")]
    upto = list(chain[: list(chain).index("L5") + 1])
    return list(zip(upto, upto[1:]))


def _counts(rung: Mapping, kind: str, pitches: Sequence[float] | None) -> tuple[float, float, dict]:
    """(U, G, {term: count}) of one bundle's rung: the Y6 row, or the sum over the passes at `pitches`."""
    if kind == "y6":
        y6 = rung["y6"]
        return y6["row"]["U"], y6["row"]["G6"], {t: y6["ledger"][t] for t in TERMS}
    passes = [p for p in rung["y1"]["passes"] if pitches is None or p["pitch"] in pitches]
    return (sum(p["U1"] for p in passes), sum(p["G1"] for p in passes),
            {t: sum(p["ledger"][t] for p in passes) for t in TERMS})


def _delta(a, b):
    return None if a is None or b is None else b - a


def _pooled_deltas(kind: str, items_a: Sequence[Mapping], items_b: Sequence[Mapping],
                   pitches: Sequence[float] | None) -> tuple[float | None, float | None]:
    """(d_tree_mae, d_e1_median) from `pool_rung` over one rung's `items_a` and the other's `items_b`:
    the change of `y6.tree_mae_over_mean` (y6), or of `y1.e1_median` at `pitches` (y1)."""
    if kind == "y6":
        return _delta(*(pool_rung(items)["y6"]["tree_mae_over_mean"] for items in (items_a, items_b))), None
    return None, _delta(*(pool_rung(items, pitches)["y1"]["e1_median"] for items in (items_a, items_b)))


def _row(run_id: str, kind: str, group: str, pair: str, n_truth, d_e: float, d_tree_mae, d_e1_median,
         d_counts: Mapping[str, float]) -> dict:
    # `+ 0.0` turns a -0.0 (a miss term that did not change) into 0.0
    terms = {t: _SIGN[t] * d_counts[t] / n_truth + 0.0 for t in TERMS}
    return {"run_id": run_id, "kind": kind, "group": group, "pair": pair, "n_truth": n_truth, "d_e": d_e,
            "d_tree_mae": d_tree_mae, "d_e1_median": d_e1_median, **terms,
            "residual": d_e - math.fsum(terms.values())}


def budget_rows(
    run_entries: Sequence[Mapping], pairs: Sequence[tuple[str, str]], *, kind: Literal["y6", "y1"],
    pitches: Sequence[float] | None = None, group: str | None = None,
) -> list[dict]:
    """For each pair, one row per bundle of `run_entries`, then the `run_id == "pooled"` row.

    `kind="y6"` reads the row ledger and the row's U and G6; `d_tree_mae` is the change of
    `metrics.tree_mae_over_mean` (pooled: of `pool_rung`'s) and `d_e1_median` is None.
    `kind="y1"` sums the pass ledgers and U1 and G1 of the passes at `pitches` (every pass when
    None); `d_e1_median` is the change of `pool_rung(..., pitches)`'s `y1.e1_median` over the
    bundle's rung (pooled: over every bundle's) and `d_tree_mae` is None. `group` defaults to
    `"row"` (y6) or `"all"` (y1). A row whose truth count is 0 is left out. Two rungs that score a
    different truth count on one bundle raise `ValueError`.
    """
    group = group or ("row" if kind == "y6" else "all")
    rows = []
    for a, b in pairs:
        n_sum, du_sum, dc_sum = 0, 0, dict.fromkeys(TERMS, 0)
        for entry in run_entries:
            run_id, ra, rb = entry["meta"]["run_id"], entry["rungs"][a], entry["rungs"][b]
            (ua, n, ca), (ub, m, cb) = _counts(ra, kind, pitches), _counts(rb, kind, pitches)
            if n != m:
                raise ValueError(f"{run_id}: rungs {a} and {b} score different truth ({n} and {m})")
            d_counts = {t: cb[t] - ca[t] for t in TERMS}
            n_sum, du_sum = n_sum + n, du_sum + (ub - ua)
            dc_sum = {t: dc_sum[t] + d_counts[t] for t in TERMS}
            if not n:
                continue
            if kind == "y6":
                extra = _delta(*(r["y6"]["metrics"]["tree_mae_over_mean"] for r in (ra, rb))), None
            else:
                extra = _pooled_deltas(kind, [ra], [rb], pitches)
            rows.append(_row(run_id, kind, group, f"{a}→{b}", n, (ub - ua) / n, *extra, d_counts))
        if n_sum:
            extra = _pooled_deltas(kind, [entry["rungs"][a] for entry in run_entries],
                                   [entry["rungs"][b] for entry in run_entries], pitches)
            rows.append(_row("pooled", kind, group, f"{a}→{b}", n_sum, du_sum / n_sum, *extra, dc_sum))
    return rows


def check_budget(rows: Sequence[Mapping], tol: float = 1e-9) -> None:
    """Add each row's signed terms up again (the stored `residual` is not trusted) and compare them
    to its `d_e`; the first row off by more than `tol` raises `ValueError`, naming the row but no value."""
    for r in rows:
        if not abs(r["d_e"] - math.fsum(r[t] for t in TERMS)) <= tol:     # `not <=` also catches NaN
            raise ValueError(f"budget row {r['run_id']} {r['kind']}/{r['group']} {r['pair']}: "
                             "ledger terms do not add up to the error change")


def ladder_budget(run_entries: Sequence[Mapping], crop: str) -> list[dict] | None:
    """The checked budget of a ladder whose first bundle's `chain` has `L5`, or None (no `L5` or no pair).

    In order: the Y6 rows (only when every rung of the pairs has its Y6 on every bundle), then the
    Y1 rows per pitch of those rungs' passes (ascending, `group="pitch_<p>"`), over the
    `JUDGED_PITCHES[crop]` passes (`group="judged"`) and over every pass (`group="all"`).
    """
    chain, rungs = run_entries[0]["chain"], run_entries[0]["rungs"]
    pairs = ladder_pairs(chain, rungs)
    if "L5" not in chain or not pairs:
        return None
    names = {name for pair in pairs for name in pair}
    items = [entry["rungs"][name] for entry in run_entries for name in names]
    rows = []
    if all(item["y6"] is not None for item in items):
        rows += budget_rows(run_entries, pairs, kind="y6")
    for pitch in sorted({p["pitch"] for item in items for p in item["y1"]["passes"]}):
        rows += budget_rows(run_entries, pairs, kind="y1", pitches=[pitch], group=f"pitch_{pitch:g}")
    rows += budget_rows(run_entries, pairs, kind="y1", pitches=JUDGED_PITCHES[crop], group="judged")
    rows += budget_rows(run_entries, pairs, kind="y1", group="all")
    check_budget(rows)
    return rows
