"""Stable hash of the semantic (non-runtime) parts of a config."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel

_RUNTIME_KEYS = {"device", "output_dir", "runs"}


def config_hash(cfg: BaseModel) -> str:
    data = cfg.model_dump(mode="json")
    pruned = {k: v for k, v in data.items() if k not in _RUNTIME_KEYS}
    blob = json.dumps(pruned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
