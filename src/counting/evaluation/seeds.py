"""Noise seeds: one stage setting run once per seed and combined (mean, [min, max] spread, per-seed results)."""

from __future__ import annotations

import math
from typing import Sequence

from counting.config.eval_schema import EvalAppConfig


def noise_settings(cfg: EvalAppConfig) -> dict:
    """The noise of `cfg`: its pose and depth factors, and `detector`.

    `detector` is the `DetNoise` dump under `stages.det=oracle_noisy`, else None.
    """
    det = cfg.detector.noise.model_dump() if cfg.stages.det == "oracle_noisy" else None
    return {**cfg.noise.model_dump(exclude={"seeds"}), "detector": det}


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _mean(values: list):
    """The seeds' combined value of one field.

    Equal values stay as they are; None if any is None; dicts go by key and lists
    by element (only with the same keys or length for every seed); numbers (not
    bool) give their mean. Anything else raises `ValueError`.
    """
    first = values[0]
    if all(v == first for v in values[1:]):
        return first
    if any(v is None for v in values):
        return None
    if all(isinstance(v, dict) for v in values) and all(v.keys() == first.keys() for v in values):
        return {k: _mean([v[k] for v in values]) for k in first}
    if all(isinstance(v, list) for v in values) and all(len(v) == len(first) for v in values):
        return [_mean(list(xs)) for xs in zip(*values)]
    if all(_is_number(v) for v in values):
        return math.fsum(values) / len(values)
    raise ValueError(f"noise seeds disagree on {values!r}")


def _range(values: list):
    """[min, max] of numbers, None if any is None, dicts by key; `values` already passed `_mean`."""
    if any(v is None for v in values):
        return None
    if isinstance(values[0], dict):
        return {k: _range([v[k] for v in values]) for k in values[0]}
    return [min(values), max(values)]


def combine_seeds(cfg: EvalAppConfig, per_seed: Sequence[tuple[int, dict]]) -> dict:
    """Combine `evaluate` results, one per noise seed (`per_seed` is `(seed, result)` in run order).

    `y1` and `y6` are the seeds' means (`_mean`; `y6` is None when every seed's is);
    `stages` and `lift_depth` are the first result's. `spread` is the [min, max]
    of `y1.summary`, `y6.row` and `y6.metrics` over the seeds, and `per_seed` keeps
    each seed's `y1` and `y6`.
    """
    results = [r for _, r in per_seed]
    y1 = _mean([r["y1"] for r in results])
    y6 = _mean([r["y6"] for r in results])
    spread = {"y1": {"summary": _range([r["y1"]["summary"] for r in results])}, "y6": None}
    if y6 is not None:
        spread["y6"] = {k: _range([r["y6"][k] for r in results]) for k in ("row", "metrics")}
    first = results[0]
    return {"stages": first["stages"], "lift_depth": first["lift_depth"], "y1": y1, "y6": y6,
            "noisy": True, "seeds": [s for s, _ in per_seed], "noise": noise_settings(cfg), "spread": spread,
            "per_seed": [{"noise_seed": s, "y1": r["y1"], "y6": r["y6"]} for s, r in per_seed]}
