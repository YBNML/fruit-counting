"""Eval sweeps: grid files, sweep points over ladder rungs, and one `prepare` per group of cells.

A grid file is a YAML mapping with only these keys:

    bases: [L1, L3]          # ladder rung or baseline names; leave out to use the config as it is
    common: {stages: {det: oracle_noisy}, detector: {noise: {}}}   # under every point but the baseline
    factors:                 # one factor at a time: each value is one point
      lift.far_min_frac: [0.2, 0.25, 0.3]
    product:                 # a small full product; not together with factors
      lift.depth: [box_median, box_far]
      tracker.depth_gate: [hard, soft]
    combos:                  # named nested patches
      tight: {dedup: {radius_m: 0.03}}

A `factors`/`product` key is a dot path whose first part is a stage setting
section (truth, passes, stages, detector, lift, noise, tracker, dedup); a
`common` or `combo` key is checked by the top-level keys of its mapping, which
must equal one of those sections exactly (a nested mapping, never a dot path).

Points, in order: `baseline` (no patch; the name is reserved); `factors` in
file order and value order, named `key=<value>`; `product` by
`itertools.product` (the last key changes fastest), named
`k1=<value>,k2=<value>`; `combos` in file order, named by the combo. A string
`<value>` is written as it is, any other value as compact sorted JSON (`0.2`,
`null`, `true`, `{"fruit":0.015}`). A point's patch is a nested dict
(`lift.far_min_frac` = 0.2 gives `{"lift": {"far_min_frac": 0.2}}`) built
straight from the YAML values, never through `--set` strings. Every point but
`baseline` carries `common` under it too (`grid.common` deep-merged with the
point's own patch, the point's values winning).

Cells: points are the outer loop and `bases` the inner one (without `bases`,
one cell per point with base None). A cell's config is the point patch merged
into the config first, in the same place as `--set`, and then the base's rung
patch (`rung_config`) on top. So a key the rung sets keeps the rung's value,
and a key the rung leaves alone keeps the point's value. Merging is deep: a
mapping value in a point merges into the config's mapping
(`lift.center_offset_m: [{fruit: 0.015}]` keeps `bag`), while
`--set key={...}` replaces the whole mapping.

A cell's `overridden` lists the point's leaf paths (a leaf is a value that is
not a mapping, an empty mapping included) that equal a leaf path of the
rung's patch, are a prefix of one, or have one as a prefix, even when both
give the same value. They are dot-joined and sorted.

`run_sweep` groups cells, and a cell's detection noise seeds, by
`_prepared_settings` (the settings `prepare` checks or fixes in the context,
such as `passes` and, under `stages.det=oracle_noisy`, the detection noise
seed), in order of first appearance. It runs `prepare` once per group and
evaluates all of the group's cells before the next group, so it holds one
context at a time; a pose or depth noise seed needs no separate `prepare` and
runs on its cell's own group. Each cell keeps the Y1 summary, each pass's
counts and ledger, and the Y6 mode, row, ledger and metrics; only the
baseline point keeps the per-tree Y6 table. A noisy cell also keeps `noisy`,
`seeds`, `noise`, `spread` and `per_seed`, same as a noisy ladder rung.
"""

from __future__ import annotations

import copy
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from counting.config.eval_schema import EvalAppConfig, SplitsConfig
from counting.data.formats.orchard_synth import SynthRun, select_frames
from counting.evaluation.runner import (
    _merge, _patched, _prepared_settings, _unsupported_setting, check_ladder_request, describe_run, evaluate,
    evaluate_seeds, prepare, rung_config,
)
from counting.evaluation.seeds import combine_seeds

_GRID_KEYS = ("bases", "common", "factors", "product", "combos")
_STAGE_SECTIONS = ("truth", "passes", "stages", "detector", "lift", "noise", "tracker", "dedup")
_PASS_KEYS = ("pitch", "side", "aisle", "n_frames", "U1", "G1", "e1", "ledger")
_Y6_KEYS = ("mode", "n_landmarks", "row", "ledger", "metrics")


@dataclass(frozen=True)
class Grid:
    """A checked grid file; `path` is the file it was read from, for error messages."""

    path: Path
    bases: tuple[str, ...]
    factors: dict[str, list]
    product: dict[str, list]
    combos: dict[str, dict]
    common: dict


@dataclass(frozen=True)
class SweepPoint:
    """One grid point: its name and nested patch; `factor` and `value` are set for a `factors` point only."""

    name: str
    patch: dict
    factor: str | None
    value: object


@dataclass(frozen=True)
class SweepCell:
    """One point on one base (None without `bases`), with its config and the point keys the rung overrode."""

    point: SweepPoint
    base: str | None
    cfg: EvalAppConfig
    overridden: tuple[str, ...]


def _check_section_keys(p: Path, label: str, patch: Mapping) -> None:
    """Refuse a `common`/`combo` key that is not exactly one of `_STAGE_SECTIONS` (no dot paths)."""
    for key in patch:
        if key not in _STAGE_SECTIONS:
            raise ValueError(f"{p}: {label} key {key!r} must be one of {', '.join(_STAGE_SECTIONS)}; "
                             "write nested mappings, not dot paths")


def load_grid(path: str | Path) -> Grid:
    """Read and check a grid file.

    A bad file raises `ValueError` starting with its path; a missing one raises `FileNotFoundError`.
    """
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"{p}: not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: a grid file needs a mapping")
    for key in raw:
        if key not in _GRID_KEYS:
            raise ValueError(f"{p}: unknown grid key {key!r}; allowed: {', '.join(_GRID_KEYS)}")
    if raw.get("factors") is not None and raw.get("product") is not None:
        raise ValueError(f"{p}: factors and product cannot be used together")
    common = raw.get("common")
    if common is not None and not isinstance(common, dict):
        raise ValueError(f"{p}: common needs a mapping")
    common = common or {}
    _check_section_keys(p, "common", common)
    sections: dict[str, dict] = {}
    for name in ("factors", "product", "combos"):
        value = raw.get(name)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{p}: {name} needs a mapping")
        sections[name] = value or {}
    if not any(sections.values()):
        raise ValueError(f"{p}: a grid needs factors, product or combos")
    bases = raw.get("bases")
    bases = [] if bases is None else bases
    if not isinstance(bases, list) or not all(isinstance(b, str) for b in bases):
        raise ValueError(f"{p}: bases needs a list of rung names")
    for i, b in enumerate(bases):
        if b in bases[:i]:
            raise ValueError(f"{p}: duplicate base {b!r}")
    for name in ("factors", "product"):
        for key, values in sections[name].items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"{p}: {key} needs a non-empty list of values")
    for name, patch in sections["combos"].items():
        if not isinstance(name, str):            # YAML reads an unquoted 0.05, 7 or true as a number or bool
            raise ValueError(f"{p}: combo names must be strings; quote the name {name!r}")
        if not isinstance(patch, dict):
            raise ValueError(f"{p}: combo {name!r} needs a mapping")
        _check_section_keys(p, f"combo {name!r}", patch)
    keys = [*sections["factors"], *sections["product"]]
    for key in keys:
        if not isinstance(key, str) or key.split(".")[0] not in _STAGE_SECTIONS:
            raise ValueError(f"{p}: {key} is not a stage setting; grid keys start with one of "
                             f"{', '.join(_STAGE_SECTIONS)}")
    return Grid(path=p, bases=tuple(bases), factors=sections["factors"], product=sections["product"],
                combos=sections["combos"], common=common)


def _value_text(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, separators=(",", ":"))


def _nested(key: str, value: object) -> dict:
    """`{"a": {"b": value}}` for `key` `a.b` (a deep copy of `value`)."""
    out = copy.deepcopy(value)
    for part in reversed(key.split(".")):
        out = {part: out}
    return out


def grid_points(grid: Grid) -> list[SweepPoint]:
    """The grid's points in order: `baseline`, then `factors` or `product`, then `combos`.

    `baseline`'s patch is `{}`; every other point's patch is `grid.common` deep-merged
    with its own patch (the point's values win).
    """
    points = [SweepPoint("baseline", {}, None, None)]
    for key, values in grid.factors.items():
        for v in values:
            points.append(SweepPoint(f"{key}={_value_text(v)}", _merge(grid.common, _nested(key, v)), key,
                                     copy.deepcopy(v)))
    if grid.product:
        keys = list(grid.product)
        for combo in itertools.product(*grid.product.values()):
            patch: dict = {}
            for k, v in zip(keys, combo):
                patch = _merge(patch, _nested(k, v))
            name = ",".join(f"{k}={_value_text(v)}" for k, v in zip(keys, combo))
            points.append(SweepPoint(name, _merge(grid.common, patch), None, None))
    for name, patch in grid.combos.items():
        points.append(SweepPoint(name, _merge(grid.common, patch), None, None))
    seen: set[str] = set()
    for point in points:
        if point.name in seen:
            raise ValueError(f"{grid.path}: duplicate point name {point.name!r}")
        seen.add(point.name)
    return points


def _leaf_paths(patch: Mapping) -> list[tuple[str, ...]]:
    """Paths to the values of `patch` that are not mappings (an empty mapping is a leaf), in key order."""
    out: list[tuple[str, ...]] = []
    for k, v in patch.items():
        if isinstance(v, Mapping) and v:
            out.extend((str(k), *rest) for rest in _leaf_paths(v))
        else:
            out.append((str(k),))
    return out


def _related(p: tuple[str, ...], q: tuple[str, ...]) -> bool:
    """True when `p` and `q` are the same path or one is a prefix of the other."""
    return p[:len(q)] == q or q[:len(p)] == p


def _rung_patch(ladder: Mapping, name: str) -> Mapping:
    """The patch `rung_config` merged for `name` (from `rungs`, else `baselines`); `name` is known."""
    return next(ladder[s][name] for s in ("rungs", "baselines") if name in (ladder.get(s) or {}))


def sweep_cells(cfg: EvalAppConfig, grid: Grid, ladder: Mapping | None) -> list[SweepCell]:
    """Every (point, base) cell of `grid` on `cfg`, points outer and bases inner; no bundle is opened.

    With `bases`, `ladder` must be given (`ValueError` naming the grid file),
    and the ladder request is checked next (`check_ladder_request`, its errors
    as they are). A cell's config is `cfg` with the point patch and then the
    rung patch merged in. A config error (`ValueError`, pydantic's included) or
    an unsupported setting (`NotImplementedError`) names the grid file, the
    point and the base.
    """
    if grid.bases and ladder is None:
        raise ValueError(f"{grid.path}: bases need a ladder")
    if grid.bases:
        check_ladder_request(cfg, ladder, grid.bases)
    cells: list[SweepCell] = []
    for point in grid_points(grid):
        leaves = _leaf_paths(point.patch)
        for base in grid.bases or (None,):
            where = f"{grid.path}: point {point.name!r}" + ("" if base is None else f" on {base}")
            try:
                c = _patched(cfg, point.patch)
                if base is not None:
                    c = rung_config(c, ladder, base)
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from e
            refused = _unsupported_setting(c)
            if refused is not None:
                raise NotImplementedError(f"{where}: {refused} is not supported in this version")
            overridden: tuple[str, ...] = ()
            if base is not None:
                rung_leaves = _leaf_paths(_rung_patch(ladder, base))
                overridden = tuple(sorted({".".join(p) for p in leaves if any(_related(p, q) for q in rung_leaves)}))
            cells.append(SweepCell(point=point, base=base, cfg=c, overridden=overridden))
    return cells


def check_set_overlap(set_keys: Sequence[str], points: Sequence[SweepPoint]) -> None:
    """Refuse a `--set` key (the path before `=`) equal to a grid key, a prefix of one or under one.

    Stops at the first clash: `set_keys` in order, then points, then each point's leaf paths.
    """
    point_leaves = [_leaf_paths(point.patch) for point in points]
    for key in set_keys:
        k = tuple(key.split("."))
        for leaves in point_leaves:
            for leaf in leaves:
                if _related(k, leaf):
                    raise ValueError(f"--set {key} overlaps the grid key {'.'.join(leaf)}; drop one of them")


def check_frame_selection(run: SynthRun, cells: Sequence[SweepCell]) -> None:
    """Select `run`'s frames once per distinct `passes` setting of `cells`, so a bad one fails before `prepare`."""
    seen: set[str] = set()
    for cell in cells:
        passes = cell.cfg.passes
        key = json.dumps(passes.model_dump(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        select_frames(run.frames, base_step_m=run.meta.base_step_m, pitches=passes.pitches, sides=passes.sides,
                      step_m=passes.step_m)


def _kept_view(cell: SweepCell, y1: dict, y6: dict | None) -> dict:
    """The `y1`/`y6` part of one evaluate result (or one of its seed views) a sweep keeps for `cell`."""
    kept_y6 = None
    if y6 is not None:
        kept_y6 = {k: y6[k] for k in _Y6_KEYS}
        if cell.point.name == "baseline":
            kept_y6["trees"] = y6["trees"]
    return {"y1": {"summary": y1["summary"], "passes": [{k: p[k] for k in _PASS_KEYS} for p in y1["passes"]]},
            "y6": kept_y6}


def _kept(cell: SweepCell, res: dict) -> dict:
    """The part of `evaluate`'s or `evaluate_seeds`'s result a sweep keeps for `cell`.

    A noisy `res` also keeps `noisy`, `seeds`, `noise`, `spread` and a `per_seed` trimmed the same way.
    """
    out = {"point": cell.point.name, "base": cell.base, "overridden": list(cell.overridden),
           "stages": res["stages"], "lift_depth": res["lift_depth"], **_kept_view(cell, res["y1"], res["y6"])}
    if res.get("noisy"):
        out.update(noisy=True, seeds=res["seeds"], noise=res["noise"], spread=res["spread"],
                   per_seed=[{"noise_seed": p["noise_seed"], **_kept_view(cell, p["y1"], p["y6"])}
                             for p in res["per_seed"]])
    return out


def run_sweep(run: SynthRun, cells: Sequence[SweepCell], splits: SplitsConfig) -> tuple[dict, dict]:
    """Evaluate `cells` on `run`, with one `prepare` per group of cells that share `_prepared_settings`.

    A cell under `stages.det=oracle_noisy` belongs to one group per noise seed;
    pose and depth noise cells run all their seeds on their group's context.
    Returns `{"meta": describe_run(first group's context), "cells": [kept result per cell, in input order]}`
    and the seconds spent, `{"prepare": ..., "evaluate": ...}`.
    """
    check_frame_selection(run, cells)
    groups: dict[str, list[tuple[int, int | None]]] = {}
    for i, cell in enumerate(cells):
        seeds = cell.cfg.noise.seeds if cell.cfg.stages.det == "oracle_noisy" else [None]
        for s in seeds:
            groups.setdefault(json.dumps(_prepared_settings(cell.cfg, s), sort_keys=True), []).append((i, s))
    out: list[dict | None] = [None] * len(cells)
    det_seed_results: dict[int, dict[int, dict]] = {}
    meta = None
    t_prepare = t_evaluate = 0.0
    for members in groups.values():
        i0, s0 = members[0]
        t0 = time.perf_counter()
        ctx = prepare(run, cells[i0].cfg, splits, det_seed=s0)
        t1 = time.perf_counter()
        t_prepare += t1 - t0
        if meta is None:
            meta = describe_run(ctx)
        for i, s in members:
            if s is None:
                out[i] = _kept(cells[i], evaluate_seeds(ctx, cells[i].cfg))
            else:
                det_seed_results.setdefault(i, {})[s] = evaluate(ctx, cells[i].cfg, noise_seed=s)
        t_evaluate += time.perf_counter() - t1
        del ctx   # one context at a time
    for i, per_seed in det_seed_results.items():
        cfg = cells[i].cfg
        out[i] = _kept(cells[i], combine_seeds(cfg, [(s, per_seed[s]) for s in cfg.noise.seeds]))
    return {"meta": meta, "cells": out}, {"prepare": t_prepare, "evaluate": t_evaluate}
