"""YAML config loading with dot-path CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, TypeVar

import yaml
from pydantic import BaseModel

from counting.config.schema import AppConfig

M = TypeVar("M", bound=BaseModel)


def load_yaml_model(
    path: str | Path,
    model: type[M],
    *,
    overrides: Iterable[str] | None = None,
) -> M:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping: {p}")
    if overrides:
        raw = apply_overrides(raw, list(overrides))
    return model.model_validate(raw)


def load_config(
    path: str | Path,
    *,
    overrides: Iterable[str] | None = None,
) -> AppConfig:
    return load_yaml_model(path, AppConfig, overrides=overrides)


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"Override must be key.path=value: {spec!r}")
        key, _, raw_value = spec.partition("=")
        _set_dot_path(out, key.split("."), _coerce(raw_value))
    return out


def _set_dot_path(node: dict[str, Any], parts: list[str], value: Any) -> None:
    cur: Any = node
    for i, part in enumerate(parts[:-1]):
        if not isinstance(cur, dict):
            raise KeyError(
                f"Cannot descend into non-dict at: {'.'.join(parts[: i + 1])}"
            )
        if part not in cur or cur[part] is None:
            cur[part] = {}
        elif not isinstance(cur[part], dict):
            raise KeyError(
                f"Cannot descend: existing value at "
                f"{'.'.join(parts[: i + 1])} is not a dict"
            )
        cur = cur[part]
    if not isinstance(cur, dict):
        raise KeyError(f"Cannot set key on non-dict at: {'.'.join(parts[:-1])}")
    cur[parts[-1]] = value


def _coerce(raw: str) -> Any:
    """Coerce a raw `--set` value string.

    A value starting with `[` or `{` (after stripping surrounding whitespace) is
    parsed as a YAML flow collection, so `--set noise.seeds=[0, 1, 2]` and
    `--set lift.center_offset_m={fruit: 0.0}` work. Inside the brackets, plain
    YAML 1.1 scalar rules apply, not these `--set` rules, so e.g. `[1e-3]` stays
    the string `"1e-3"` there, not a float. A malformed flow value raises
    `ValueError`. Everything else keeps the scalar rules below (`1e-3` is a float).
    """
    s = raw.strip()
    if s.startswith(("[", "{")):
        try:
            return yaml.safe_load(s)
        except yaml.YAMLError as e:
            raise ValueError(f"Override value is not valid YAML flow syntax: {raw}") from e
    lowered = s.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        if "." in raw or "e" in lowered:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw
