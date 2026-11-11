"""Prediction-side reading of a pose cache: one bundle's relative camera poses, chained from a given start.

`PoseFile` reads a pose cache (`counting.odometry.cache`) once and, before any pose is used, checks
that it belongs to the bundle being scored: the bundle fingerprint and the run id in its meta, then
its content digest against a pinned digest. Each check runs only when its value is given, and a
mismatch raises `ValueError` naming the cache path.

The cache holds every pass's poses relative to the pass's first cached frame. The pose of the
pass's start comes from the caller; this module reads no truth, frame pose or planned pose. A pass's
camera-to-world pose of frame idx is `start @ rel_c2w[row of idx]`, so the anchor frame (the first
cached frame of the pass) gets `start` itself. A frame missing from the cache raises `ValueError`
naming the cache path, the frame idx and the pass; no other pose stands in for it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from counting.odometry.cache import PassOdometry, load_pose_cache


def _code(code: Sequence[int]) -> tuple[int, int, int]:
    """A pass code as a tuple of Python ints, the key `load_pose_cache` uses."""
    return tuple(int(v) for v in code)


class PoseFile:
    """One bundle's pose cache, read and checked once.

    `fingerprint`, `run_id` and `digest`, when given, are checked in that order against the meta
    `fingerprint`, the meta `run_id` and the content digest. Attributes: `path`, `meta` (the cache
    meta, with its `format`) and `digest` (the content digest).
    """

    def __init__(self, path: Path, *, fingerprint: str | None = None, run_id: str | None = None,
                 digest: str | None = None) -> None:
        cache = load_pose_cache(path)
        self.path = cache.path
        self.meta = cache.meta
        self.digest = cache.digest
        if fingerprint is not None and self.meta.get("fingerprint") != fingerprint:
            raise ValueError(f"{self.path}: pose cache was made for bundle fingerprint "
                             f"{str(self.meta.get('fingerprint'))[:12]}, this bundle has {fingerprint[:12]}")
        if run_id is not None and self.meta.get("run_id") != run_id:
            raise ValueError(f"{self.path}: pose cache is for run {self.meta.get('run_id')}, this bundle is {run_id}")
        if digest is not None and self.digest != digest:
            raise ValueError(f"{self.path}: pose cache digest {self.digest[:16]} differs from the pinned {digest[:16]}")
        self._passes = cache.passes
        self._rows = {code: {int(i): row for row, i in enumerate(p.idx)} for code, p in cache.passes.items()}

    def _pass(self, code: Sequence[int]) -> PassOdometry:
        key = _code(code)
        if key not in self._passes:
            raise ValueError(f"{self.path}: no poses for pass {key}")
        return self._passes[key]

    def anchor(self, code: tuple[int, int, int]) -> int:
        """The frame idx of the pass's first cached frame, whose pose is the start."""
        return int(self._pass(code).idx[0])

    def idxs(self, code: tuple[int, int, int]) -> np.ndarray:
        """The pass's cached frame idx, ascending (a copy)."""
        return self._pass(code).idx.copy()

    def poses(self, code: tuple[int, int, int], start: np.ndarray, idxs: Sequence[int]) -> np.ndarray:
        """Camera-to-world poses of the frames `idxs` of the pass, shape (len(idxs), 4, 4): `start @ rel_c2w`
        of each frame's row, with `start` the pose of the pass's anchor frame."""
        p = self._pass(code)
        rows_of = self._rows[_code(code)]
        rows = []
        for idx in idxs:
            if int(idx) not in rows_of:
                raise ValueError(f"{self.path}: no pose for frame {int(idx)} of pass {_code(code)}")
            rows.append(rows_of[int(idx)])
        return np.asarray(start, dtype=np.float64) @ p.rel_c2w[np.asarray(rows, dtype=np.intp)]
