"""Pool scene-seed results and judge ladder rungs against the criteria table.

`pool_views` combines evaluate-shaped views (`y1.passes`, `y6.trees`/`ledger`) from
several scenes into one summary (the metrics in `POOLED_KEYS`):

- y1: `e1_median`/`e1_max` from every pass's non-None |e1|, and the rates
  `dup_rate`, `miss_rate`, `merge_rate` and `fp_off_rate` of every pass's ledger
  combined (`ledger.combine`).
- y6: over every scene's trees, `tree_mae_over_mean` = Σ|U - G6| / ΣG6 and
  `attribution_error` = Σ|U - U_truth_attr| / ΣG6 (both `None` when ΣG6 is 0),
  and the rates `miss_rate` and `excess_rate` of the scenes' row ledgers combined.

With `pitches`, the y1 half pools only the passes at those camera pitches.

`pool_rung` pools one rung across bundles; for a noisy rung (every bundle sharing
the same noise `seeds`, or `ValueError`) it pools each noise seed across the
bundles first (their `per_seed[i]`), then averages the seeds' pooled summaries
(`seeds._mean`), so the result is a seed mean rather than a single median over
every seed's passes.
`ladder_pooled` runs `pool_rung` over every rung of a set of bundles, in the
first bundle's rung order. When a rung of `PITCH_JUDGED_RUNGS` ran and the crop is
given, every rung also gets `y1_judged`, its y1 pooled over the `JUDGED_PITCHES`
passes alone (judged on the `PITCH_JUDGED_RUNGS` rungs, reported on the others).

`ladder_verdicts` compares `ladder_pooled`'s summaries to `criteria[kind][rung][crop]`
(`kind` is `y1` then `y6`; a rung whose pooled `kind` is `None`, e.g. under
`stop_after=y1`, skips that half). A metric named in `POOLED_KEYS` is judged on
its pooled value (`unit="pooled"`); every other metric is judged scene by scene
(`unit="scene"`, the row's own `value` stays `None`), and a scene row's `ok` is
`False` if any scene's is `False`, else `None` if any scene's is `None`, else
`True`. Every row keeps each scene's own value, and, for a noisy rung, its
noise-seed spread, under `per_run` (in the given bundle order), so a report can
be built from `results.json` alone without rereading the bundles. The y1 rows of a
`PITCH_JUDGED_RUNGS` rung take the `JUDGED_PITCHES` values instead (`y1_judged` and
`judged_y1`) and name the pitches. For a noisy rung these are the seed means of each seed's
judged-pitch pool, and the spread is over those pools.

`PITCH_JUDGED_RUNGS` are rungs L5, L5VI and L5VI_ld (L5VI with the LiDAR box depth). `judged_pitches`
gives the pitches a row names: every `JUDGED_PITCHES[crop]` pitch for rung L5; for the `RAN_PITCH_RUNGS`
L5VI and L5VI_ld only those their passes ran (the pitches of every bundle's `y1.passes`), and `[]` when
they ran none of them (a camera configuration without a judged pitch), the value and `ok` then being None.

A pose criteria table names its judged camera configuration (`judged_config`). When the base
config's camera configuration (`ladder_verdicts`' `config_label`, from `pose.config_label`) is
named and differs from it, every row gets `report_config` (that name) and keeps the `ok` it
computes; `verdict_counts`, and so `verdict_summary`, leaves such rows out. A config without a
configuration name, or a table without `judged_config`, is judged as before.

`subset_verdicts` judges rung L5's pass subsets (`runner._score_subsets`'s table) the
same way, each against its own truth: `row_abs_e_pipe` scene by scene (the same scene
rule, shared via `_scene_ok`), `tree_mae_over_mean`/`miss_rate`/`excess_rate` pooled.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Iterable, Mapping, Sequence

import numpy as np

from counting.evaluation.ledger import Ledger, combine
from counting.evaluation.seeds import _mean, _range

POOLED_KEYS = {
    "y1": ("e1_median", "e1_max", "dup_rate", "miss_rate", "merge_rate", "fp_off_rate"),
    "y6": ("tree_mae_over_mean", "attribution_error", "miss_rate", "excess_rate"),
}

# The camera pitches (deg) of the passes and detection groups judged on a learned detector, per crop.
JUDGED_PITCHES = {"pear": (45.0, 70.0), "apple": (30.0, 50.0)}
# Rungs whose y1 is judged on the `JUDGED_PITCHES` passes alone; the other pitches are report-only there.
PITCH_JUDGED_RUNGS = ("L5", "L5VI", "L5VI_ld")
# The `PITCH_JUDGED_RUNGS` rungs whose rows name only the judged pitches their passes ran (`judged_pitches`).
RAN_PITCH_RUNGS = ("L5VI", "L5VI_ld")

_LEDGER_FIELDS = tuple(f.name for f in fields(Ledger))


def _ledger_from(d: Mapping) -> Ledger:
    """A `Ledger` from the `Ledger` fields of `d`, which may carry extra keys (e.g. `as_dict()`'s rates)."""
    return Ledger(**{name: d[name] for name in _LEDGER_FIELDS})


def pool_views(views: Sequence[Mapping], pitches: Sequence[float] | None = None) -> dict:
    """Pool noise-free evaluate-shaped `views`: a ladder rung per bundle, or one noise seed's
    per-bundle views. Shape: `{"n_runs", "y1": {...}, "y6": {...} | None}`.

    With `pitches`, y1 pools only the passes at those pitches (y6 is unchanged); with no pass
    there, `n_passes` is 0 and every other y1 value is None.
    """
    passes = [p for v in views for p in v["y1"]["passes"] if pitches is None or p["pitch"] in pitches]
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


def pool_rung(items: Sequence[Mapping], pitches: Sequence[float] | None = None) -> dict:
    """Pool one rung's `items`, one per bundle (`pool_views(items, pitches)` without noise).

    A noisy rung's bundles must share `noise.seeds`, or `ValueError`. Otherwise `pool_views`
    runs once per noise-seed index across the bundles' `per_seed[i]` (with `pitches`), and
    the seeds' pooled summaries are averaged with `seeds._mean`.
    """
    if not items[0].get("noisy"):
        return pool_views(items, pitches)
    all_seeds = [it["seeds"] for it in items]
    if any(s != all_seeds[0] for s in all_seeds[1:]):
        raise ValueError(f"bundles disagree on the noise seeds: {all_seeds}")
    seeds = all_seeds[0]
    return _mean([pool_views([it["per_seed"][i] for it in items], pitches) for i in range(len(seeds))])


def judged_y1(item: Mapping, crop: str) -> dict:
    """The y1 half of `pool_views` over one bundle's rung or sweep cell `item`, at `JUDGED_PITCHES[crop]`
    alone. Only `item["y1"]` is read, so a sweep cell that keeps no `y6.trees` works too.

    A noisy `item` gives its seeds' mean (`seeds._mean`) of each seed view's judged y1, as `pool_rung`
    does; only `y1` is read there too.
    """
    if item.get("noisy"):
        return _mean([judged_y1(view, crop) for view in item["per_seed"]])
    return pool_views([{"y1": item["y1"], "y6": None}], JUDGED_PITCHES[crop])["y1"]


def judged_pitches(rung_name: str, rungs: Sequence[Mapping], crop: str) -> list[float]:
    """The pitches that the y1 rows of `PITCH_JUDGED_RUNGS` rung `rung_name` name, given that rung of every
    bundle (`rungs`): for a `RAN_PITCH_RUNGS` rung, the `JUDGED_PITCHES[crop]` pitches its passes ran in some
    bundle (ascending, `[]` when none); for any other rung, every `JUDGED_PITCHES[crop]` pitch."""
    pitches = list(JUDGED_PITCHES[crop])
    if rung_name in RAN_PITCH_RUNGS:
        ran = {float(p["pitch"]) for r in rungs for p in r["y1"]["passes"]}
        pitches = sorted(set(pitches) & ran)
    return pitches


def ladder_pooled(run_entries: Sequence[Mapping], crop: str | None = None) -> dict[str, dict]:
    """`{rung: pool_rung([every bundle's that rung])}`, in the first bundle's rung order.

    With `crop`, when a rung of `PITCH_JUDGED_RUNGS` is among them, every rung also gets
    `y1_judged`: the y1 of `pool_rung` over the `JUDGED_PITCHES[crop]` passes alone.
    Every bundle must name the same set of rungs, or `ValueError`.
    """
    names = list(run_entries[0]["rungs"])
    if any(set(entry["rungs"]) != set(names) for entry in run_entries[1:]):
        raise ValueError("bundles disagree on the rungs")
    judged = crop is not None and any(name in PITCH_JUDGED_RUNGS for name in names)
    out = {}
    for name in names:
        items = [entry["rungs"][name] for entry in run_entries]
        out[name] = pool_rung(items)
        if judged:
            out[name]["y1_judged"] = pool_rung(items, JUDGED_PITCHES[crop])["y1"]
    return out


def _ok(value, op: str, limit) -> bool | None:
    if value is None:
        return None
    if op == "le":
        return value <= limit
    if op == "ge":
        return value >= limit
    raise ValueError(f"unknown criteria op: {op!r}")


def _scene_ok(oks: Iterable[bool | None]) -> bool | None:
    """The scene-row combination rule: `False` if any scene's `ok` is `False`, else `None` if any is
    `None`, else `True`. Shared by `ladder_verdicts` and `subset_verdicts`."""
    oks = list(oks)
    return False if any(o is False for o in oks) else (None if any(o is None for o in oks) else True)


def ladder_verdicts(
    run_entries: Sequence[Mapping], pooled: Mapping[str, dict], criteria: Mapping, crop: str, *,
    config_label: str | None = None,
) -> dict[str, list[dict]]:
    """Judge every rung of `pooled` against `criteria[kind][rung][crop]`, in the first bundle's
    rung order and `criteria`'s metric order (`kind` is `y1` then `y6`).

    A rung whose pooled `kind` is `None` skips that half; a rung or crop missing from `criteria`
    contributes no rows for that half. Each row is `{metric, op, limit, unit, value, per_run, ok}`:
    `unit` is `"pooled"` (`value` is the pooled value) or `"scene"` (`value` is `None`); `per_run`
    is `[{run_id, seed, value, spread, ok}]` in `run_entries` order, `spread` being the noisy
    rung's matching `spread` entry or `None` (on a judged y1 row, the [min, max] over the seeds of
    each seed view's `judged_y1`). A row's `ok` is `None` wherever its compared value
    is `None`; a `"scene"` row's `ok` is `False` if any scene's is `False`, else `None` if any is
    `None`, else `True`. A rung left with no rows at all is left out of the result.

    The y1 rows of a `PITCH_JUDGED_RUNGS` rung are judged on the `JUDGED_PITCHES[crop]` passes
    alone (pooled: `pooled[rung]["y1_judged"]`, from `ladder_pooled(..., crop=crop)`; per scene:
    `judged_y1`) and carry `"pitches"` (`judged_pitches`: rung L5's is `JUDGED_PITCHES[crop]`; a
    `RAN_PITCH_RUNGS` rung's keeps only the pitches its passes ran in some bundle, ascending, `[]` when
    none); without such a pass, their values and `ok` are None.

    With `criteria["judged_config"]` set and `config_label` a name that differs from it, every row
    also gets `"report_config": config_label` (its `ok` as computed); otherwise no row has the key.
    """
    judged_config, report_config = criteria.get("judged_config"), None
    if judged_config and isinstance(config_label, str) and config_label != judged_config:
        report_config = config_label
    out: dict[str, list[dict]] = {}
    for rung_name in run_entries[0]["rungs"]:
        rows = []
        for kind, metrics_key in (("y1", "summary"), ("y6", "metrics")):
            if pooled[rung_name].get(kind) is None:
                continue
            spec = ((criteria.get(kind) or {}).get(rung_name) or {}).get(crop)
            if not spec:
                continue
            judged = kind == "y1" and rung_name in PITCH_JUDGED_RUNGS
            rungs = [entry["rungs"][rung_name] for entry in run_entries]
            if judged:
                pooled_vals, scene_vals = pooled[rung_name]["y1_judged"], [judged_y1(r, crop) for r in rungs]
                spreads = [_range([judged_y1(v, crop) for v in r["per_seed"]]) if r.get("noisy") else None
                           for r in rungs]
                pitches = judged_pitches(rung_name, rungs, crop)
            else:
                pooled_vals, scene_vals = pooled[rung_name][kind], [r[kind][metrics_key] for r in rungs]
                spreads = [r["spread"][kind][metrics_key] if r.get("noisy") else None for r in rungs]
            for key, lim in spec.items():
                (op, limit), = lim.items()
                per_run = []
                for entry, values, seed_spread in zip(run_entries, scene_vals, spreads):
                    per_value = values.get(key)
                    spread = None if seed_spread is None else seed_spread.get(key)
                    per_run.append({"run_id": entry["meta"]["run_id"], "seed": entry["meta"]["seed"],
                                    "value": per_value, "spread": None if spread is None else list(spread),
                                    "ok": _ok(per_value, op, limit)})
                if key in POOLED_KEYS[kind]:
                    unit, value = "pooled", pooled_vals.get(key)
                    ok = _ok(value, op, limit)
                else:
                    unit, value = "scene", None
                    ok = _scene_ok(p["ok"] for p in per_run)
                row = {"metric": f"{kind}.{key}", "op": op, "limit": limit, "unit": unit,
                       "value": value, "per_run": per_run, "ok": ok}
                if judged:
                    row["pitches"] = list(pitches)
                if report_config is not None:
                    row["report_config"] = report_config
                rows.append(row)
        if rows:
            out[rung_name] = rows
    return out


def subset_verdicts(
    run_entries: Sequence[Mapping], criteria: Mapping, crop: str, names: Sequence[str],
) -> dict[str, list[dict]]:
    """Judge pass subsets `names` of rung L5 against `criteria["y6"]["L5"][crop]`, each subset scored
    against its own truth (rung L5's `subsets` table, from `evaluate(..., subsets=True)`).

    Keyed `f"L5[{name}]"`, in `criteria`'s metric order. `row_abs_e_pipe` is judged scene by scene
    (`abs(e_pipe)`, the same scene rule as `ladder_verdicts`'s, via `_scene_ok`); `tree_mae_over_mean`,
    `miss_rate` and `excess_rate` are pooled over the run entries: `tree_mae_over_mean` as
    Sum(value * G6) / Sum(G6) over the subset's scenes with a pass, the rates from the subset
    rows' ledgers `combine`d. A `name` missing from a bundle's rung L5 subset table raises
    `ValueError` (defence only; the CLI's `check_verdict_subsets` catches this first).
    """
    spec = ((criteria.get("y6") or {}).get("L5") or {}).get(crop) or {}
    out: dict[str, list[dict]] = {}
    for name in names:
        subset_rows = []
        for entry in run_entries:
            by_s = {r["S"]: r for r in entry["rungs"]["L5"]["subsets"]}
            if name not in by_s:
                raise ValueError(f"pass subset {name!r} is not in rung L5's table of {entry['meta']['run_id']}")
            subset_rows.append(by_s[name])
        rows = []
        for key, lim in spec.items():
            (op, limit), = lim.items()
            per_run = []
            for entry, s in zip(run_entries, subset_rows):
                if key == "row_abs_e_pipe":
                    per_value = None if s["e_pipe"] is None else abs(s["e_pipe"])
                elif key == "tree_mae_over_mean":
                    per_value = s["tree_mae_over_mean"]
                else:
                    per_value = None if s["ledger"] is None else _ledger_from(s["ledger"]).rates()[key]
                per_run.append({"run_id": entry["meta"]["run_id"], "seed": entry["meta"]["seed"],
                                "value": per_value, "spread": None, "ok": _ok(per_value, op, limit)})
            if key == "row_abs_e_pipe":
                unit, value = "scene", None
                ok = _scene_ok(p["ok"] for p in per_run)
            elif key == "tree_mae_over_mean":
                weighted = [(s["tree_mae_over_mean"], s["G6"]) for s in subset_rows
                           if s["G6"] and s["tree_mae_over_mean"] is not None]
                den = sum(g for _, g in weighted)
                unit, value = "pooled", (sum(v * g for v, g in weighted) / den if den else None)
                ok = _ok(value, op, limit)
            else:
                ledgers = [_ledger_from(s["ledger"]) for s in subset_rows if s["ledger"] is not None]
                unit, value = "pooled", (combine(ledgers).rates()[key] if ledgers else None)
                ok = _ok(value, op, limit)
            rows.append({"metric": f"y6.{key}", "op": op, "limit": limit, "unit": unit,
                        "value": value, "per_run": per_run, "ok": ok})
        out[f"L5[{name}]"] = rows
    return out


def verdict_counts(verdicts: Mapping[str, list[dict]] | None) -> tuple[int, int]:
    """(# rows across every rung whose `ok` is `True`, total # rows), leaving out the rows of a report
    configuration (`report_config`). `verdicts` may be `None`."""
    rows = [row for rung_rows in (verdicts or {}).values() for row in rung_rows if "report_config" not in row]
    return sum(1 for row in rows if row["ok"] is True), len(rows)


def verdict_summary(verdicts: Mapping[str, list[dict]] | None) -> str:
    """`f"{n_ok}/{n}"`, or `"-"` with no rows at all (including `verdicts=None`)."""
    n_ok, n = verdict_counts(verdicts)
    return f"{n_ok}/{n}" if n else "-"
