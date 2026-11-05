"""Pool scene-seed results and judge ladder rungs against the criteria table.

`pool_views` combines evaluate-shaped views (`y1.passes`, `y6.trees`/`ledger`) from
several scenes into one summary (the metrics in `POOLED_KEYS`):

- y1: `e1_median`/`e1_max` from every pass's non-None |e1|, and the rates
  `dup_rate`, `miss_rate`, `merge_rate` and `fp_off_rate` of every pass's ledger
  combined (`ledger.combine`).
- y6: over every scene's trees, `tree_mae_over_mean` = Σ|U - G6| / ΣG6 and
  `attribution_error` = Σ|U - U_truth_attr| / ΣG6 (both `None` when ΣG6 is 0),
  and the rates `miss_rate` and `excess_rate` of the scenes' row ledgers combined.

`pool_rung` pools one rung across bundles; for a noisy rung (every bundle sharing
the same noise `seeds`, or `ValueError`) it pools each noise seed across the
bundles first (their `per_seed[i]`), then averages the seeds' pooled summaries
(`seeds._mean`), so the result is a seed mean rather than a single median over
every seed's passes.
`ladder_pooled` runs `pool_rung` over every rung of a set of bundles, in the
first bundle's rung order.

`ladder_verdicts` compares `ladder_pooled`'s summaries to `criteria[kind][rung][crop]`
(`kind` is `y1` then `y6`; a rung whose pooled `kind` is `None`, e.g. under
`stop_after=y1`, skips that half). A metric named in `POOLED_KEYS` is judged on
its pooled value (`unit="pooled"`); every other metric is judged scene by scene
(`unit="scene"`, the row's own `value` stays `None`), and a scene row's `ok` is
`False` if any scene's is `False`, else `None` if any scene's is `None`, else
`True`. Every row keeps each scene's own value, and, for a noisy rung, its
noise-seed spread, under `per_run` (in the given bundle order), so a report can
be built from `results.json` alone without rereading the bundles.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Mapping, Sequence

import numpy as np

from counting.evaluation.ledger import Ledger, combine
from counting.evaluation.seeds import _mean

POOLED_KEYS = {
    "y1": ("e1_median", "e1_max", "dup_rate", "miss_rate", "merge_rate", "fp_off_rate"),
    "y6": ("tree_mae_over_mean", "attribution_error", "miss_rate", "excess_rate"),
}

_LEDGER_FIELDS = tuple(f.name for f in fields(Ledger))


def _ledger_from(d: Mapping) -> Ledger:
    """A `Ledger` from the `Ledger` fields of `d`, which may carry extra keys (e.g. `as_dict()`'s rates)."""
    return Ledger(**{name: d[name] for name in _LEDGER_FIELDS})


def pool_views(views: Sequence[Mapping]) -> dict:
    """Pool noise-free evaluate-shaped `views`: a ladder rung per bundle, or one noise seed's
    per-bundle views. Shape: `{"n_runs", "y1": {...}, "y6": {...} | None}`.
    """
    passes = [p for v in views for p in v["y1"]["passes"]]
    e1s = [abs(p["e1"]) for p in passes if p["e1"] is not None]
    rates = combine(_ledger_from(p["ledger"]) for p in passes).rates()
    y1 = {"n_passes": len(passes), "e1_median": float(np.median(e1s)) if e1s else None,
          "e1_max": max(e1s) if e1s else None, "dup_rate": rates["dup_rate"],
          "miss_rate": rates["miss_rate"], "merge_rate": rates["merge_rate"],
          "fp_off_rate": rates["fp_off_rate"]}

    y6 = None
    if not any(v["y6"] is None for v in views):
        trees = [t for v in views for t in v["y6"]["trees"]]
        g6_sum = sum(t["G6"] for t in trees)
        mae = attribution_error = None
        if g6_sum:
            mae = sum(abs(t["U"] - t["G6"]) for t in trees) / g6_sum
            attribution_error = sum(abs(t["U"] - t["U_truth_attr"]) for t in trees) / g6_sum
        row_rates = combine(_ledger_from(v["y6"]["ledger"]) for v in views).rates()
        y6 = {"n_trees": len(trees), "tree_mae_over_mean": mae, "attribution_error": attribution_error,
              "miss_rate": row_rates["miss_rate"], "excess_rate": row_rates["excess_rate"]}

    return {"n_runs": len(views), "y1": y1, "y6": y6}


def pool_rung(items: Sequence[Mapping]) -> dict:
    """Pool one rung's `items`, one per bundle (`pool_views(items)` without noise).

    A noisy rung's bundles must share `noise.seeds`, or `ValueError`. Otherwise `pool_views`
    runs once per noise-seed index across the bundles' `per_seed[i]`, and the seeds' pooled
    summaries are averaged with `seeds._mean`.
    """
    if not items[0].get("noisy"):
        return pool_views(items)
    all_seeds = [it["seeds"] for it in items]
    if any(s != all_seeds[0] for s in all_seeds[1:]):
        raise ValueError(f"bundles disagree on the noise seeds: {all_seeds}")
    seeds = all_seeds[0]
    return _mean([pool_views([it["per_seed"][i] for it in items]) for i in range(len(seeds))])


def ladder_pooled(run_entries: Sequence[Mapping]) -> dict[str, dict]:
    """`{rung: pool_rung([every bundle's that rung])}`, in the first bundle's rung order.

    Every bundle must name the same set of rungs, or `ValueError`.
    """
    names = list(run_entries[0]["rungs"])
    if any(set(entry["rungs"]) != set(names) for entry in run_entries[1:]):
        raise ValueError("bundles disagree on the rungs")
    return {name: pool_rung([entry["rungs"][name] for entry in run_entries]) for name in names}


def _ok(value, op: str, limit) -> bool | None:
    if value is None:
        return None
    if op == "le":
        return value <= limit
    if op == "ge":
        return value >= limit
    raise ValueError(f"unknown criteria op: {op!r}")


def ladder_verdicts(
    run_entries: Sequence[Mapping], pooled: Mapping[str, dict], criteria: Mapping, crop: str,
) -> dict[str, list[dict]]:
    """Judge every rung of `pooled` against `criteria[kind][rung][crop]`, in the first bundle's
    rung order and `criteria`'s metric order (`kind` is `y1` then `y6`).

    A rung whose pooled `kind` is `None` skips that half; a rung or crop missing from `criteria`
    contributes no rows for that half. Each row is `{metric, op, limit, unit, value, per_run, ok}`:
    `unit` is `"pooled"` (`value` is the pooled value) or `"scene"` (`value` is `None`); `per_run`
    is `[{run_id, seed, value, spread, ok}]` in `run_entries` order, `spread` being the noisy
    rung's matching `spread` entry or `None`. A row's `ok` is `None` wherever its compared value
    is `None`; a `"scene"` row's `ok` is `False` if any scene's is `False`, else `None` if any is
    `None`, else `True`. A rung left with no rows at all is left out of the result.
    """
    out: dict[str, list[dict]] = {}
    for rung_name in run_entries[0]["rungs"]:
        rows = []
        for kind, metrics_key in (("y1", "summary"), ("y6", "metrics")):
            if pooled[rung_name].get(kind) is None:
                continue
            spec = ((criteria.get(kind) or {}).get(rung_name) or {}).get(crop)
            if not spec:
                continue
            pooled_vals = pooled[rung_name][kind]
            for key, lim in spec.items():
                (op, limit), = lim.items()
                per_run = []
                for entry in run_entries:
                    rung = entry["rungs"][rung_name]
                    per_value = rung[kind][metrics_key].get(key)
                    spread = rung["spread"][kind][metrics_key].get(key) if rung.get("noisy") else None
                    per_run.append({"run_id": entry["meta"]["run_id"], "seed": entry["meta"]["seed"],
                                    "value": per_value, "spread": None if spread is None else list(spread),
                                    "ok": _ok(per_value, op, limit)})
                if key in POOLED_KEYS[kind]:
                    unit, value = "pooled", pooled_vals.get(key)
                    ok = _ok(value, op, limit)
                else:
                    oks = [p["ok"] for p in per_run]
                    unit, value = "scene", None
                    ok = False if any(o is False for o in oks) else \
                        (None if any(o is None for o in oks) else True)
                rows.append({"metric": f"{kind}.{key}", "op": op, "limit": limit, "unit": unit,
                            "value": value, "per_run": per_run, "ok": ok})
        if rows:
            out[rung_name] = rows
    return out


def verdict_counts(verdicts: Mapping[str, list[dict]] | None) -> tuple[int, int]:
    """(# rows across every rung whose `ok` is `True`, total # rows). `verdicts` may be `None`."""
    rows = [row for rung_rows in (verdicts or {}).values() for row in rung_rows]
    return sum(1 for row in rows if row["ok"] is True), len(rows)


def verdict_summary(verdicts: Mapping[str, list[dict]] | None) -> str:
    """`f"{n_ok}/{n}"`, or `"-"` with no rows at all (including `verdicts=None`)."""
    n_ok, n = verdict_counts(verdicts)
    return f"{n_ok}/{n}" if n else "-"
