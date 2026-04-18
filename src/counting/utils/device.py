"""Device selection with safe fallbacks."""

from __future__ import annotations

import os

import torch

_VALID = {"auto", "cpu", "mps", "cuda"}


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
            _enable_mps_fallback()
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
        _enable_mps_fallback()
        return "mps"

    return "cpu"


def _enable_mps_fallback() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
