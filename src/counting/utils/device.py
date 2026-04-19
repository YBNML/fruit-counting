"""Device selection with safe fallbacks."""

from __future__ import annotations

import os
import sys

import torch

_VALID = {"auto", "cpu", "mps", "cuda"}

# Must be set before any torch MPS op is invoked. Setting at import time
# guarantees downstream modules that import torch later still see it.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_device(requested: str) -> str:
    """Return one of 'cpu' | 'mps' | 'cuda'.

    'auto' picks cuda → mps → cpu depending on availability.
    An explicit device that is not available raises RuntimeError.
    """
    req = requested.lower()
    if req not in _VALID:
        raise ValueError(f"Unknown device: {requested!r}. Valid: {sorted(_VALID)}")

    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"

    if req == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return "mps"

    return "cpu"
