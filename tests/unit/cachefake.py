"""Detection caches made from a fake bundle's truth, for the cached-detector tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from counting.data.formats.orchard_synth import SynthRun
from counting.detection.base import Detections
from counting.detection.cached import cache_path, write_cache
from counting.evaluation.truth import is_positive

COPY_SHIFT = 0.15


def write_truth_cache(run_root: Path, cache_dir: Path, *, shift: float = 0.0, low_score: float | None = None,
                      main: bool = True, score: float = 0.9) -> str:
    """Write a cache of every frame of the bundle at `run_root` into `cache_dir` and return its digest.

    Per count1 record, the main box is its `bbox` moved by `shift` x width along x (score `score`,
    label: bagged or not), left out when `main` is False; with `low_score`, a copy of the `bbox`
    moved by +`COPY_SHIFT` x width (whatever `shift` is) is added at that score.
    """
    run = SynthRun.open(run_root)
    truth = run.truth()
    dets = {}
    for f in run.frames:
        rows = []
        for r in truth.records.get(f.idx, ()):
            if not is_positive(r, "count1", truth.count1):
                continue
            x0, y0, x1, y1 = r.bbox
            w, lab = x1 - x0, int(truth.fruits[r.id].bagged)
            if main:
                rows.append((x0 + shift * w, y0, x1 + shift * w, y1, score, lab))
            if low_score is not None:
                rows.append((x0 + COPY_SHIFT * w, y0, x1 + COPY_SHIFT * w, y1, low_score, lab))
        a = np.array(rows, np.float64).reshape(-1, 6)
        dets[f.idx] = Detections(a[:, :4].astype(np.float32), a[:, 4].astype(np.float32), a[:, 5].astype(np.int8))
    return write_cache(cache_path(cache_dir, run.meta.run_id), dets,
                       {"run_id": run.meta.run_id, "fingerprint": run.meta.fingerprint})
