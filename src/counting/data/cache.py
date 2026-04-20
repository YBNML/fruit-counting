"""Sharded fp16 embedding cache.

Layout:
    <cache_dir>/
        meta.json              # dtype, shape, count, user-provided meta fields
        index.json             # {image_id: [shard_id, row]}
        shards/
            00000.npz          # arr_0..arr_N-1 (each an ndarray)
            00001.npz
            ...

`.npz` is used for simplicity and portability. Each shard holds up to
`shard_size` embeddings under keys "0", "1", ... matching row order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class _ShardBuffer:
    shard_id: int
    arrays: list[np.ndarray]

    def save(self, out_dir: Path, dtype: np.dtype) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.shard_id:05d}.npz"
        kwargs = {str(i): a.astype(dtype, copy=False) for i, a in enumerate(self.arrays)}
        np.savez_compressed(path, **kwargs)


class FeatureCacheWriter:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        meta: dict[str, Any],
        shard_size: int = 256,
        dtype: str = "float16",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.meta = dict(meta)
        self.shard_size = shard_size
        self.dtype_name = dtype
        self.dtype = np.dtype(dtype)
        self._index: dict[str, list[int]] = {}
        self._buf: _ShardBuffer | None = None
        self._shape: tuple[int, ...] | None = None
        self._count = 0
        self._shard_counter = 0

    def open(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "shards").mkdir(exist_ok=True)
        self._buf = _ShardBuffer(shard_id=0, arrays=[])

    def write(self, image_id: str, embedding: np.ndarray) -> None:
        if self._buf is None:
            raise RuntimeError("FeatureCacheWriter used before open()")
        if self._shape is None:
            self._shape = tuple(embedding.shape)
        elif tuple(embedding.shape) != self._shape:
            raise ValueError(
                f"shape mismatch: got {embedding.shape}, expected {self._shape}"
            )

        if image_id in self._index:
            raise ValueError(f"duplicate image_id: {image_id}")

        self._index[image_id] = [self._buf.shard_id, len(self._buf.arrays)]
        self._buf.arrays.append(embedding)
        self._count += 1

        if len(self._buf.arrays) >= self.shard_size:
            self._flush()

    def close(self) -> None:
        if self._buf is None:
            raise RuntimeError("FeatureCacheWriter used before open()")
        if self._buf.arrays:
            self._flush()
        meta = {
            **self.meta,
            "dtype": self.dtype_name,
            "shape": list(self._shape) if self._shape is not None else [],
            "count": self._count,
        }
        (self.cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        (self.cache_dir / "index.json").write_text(json.dumps(self._index))
        self._buf = None

    def _flush(self) -> None:
        assert self._buf is not None
        self._buf.save(self.cache_dir / "shards", dtype=self.dtype)
        self._shard_counter += 1
        self._buf = _ShardBuffer(shard_id=self._shard_counter, arrays=[])


class FeatureCacheReader:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        index_path = self.cache_dir / "index.json"
        if not meta_path.exists() or not index_path.exists():
            raise FileNotFoundError(f"cache not found at {self.cache_dir}")
        self.meta: dict[str, Any] = json.loads(meta_path.read_text())
        self._index: dict[str, list[int]] = json.loads(index_path.read_text())
        self._shard_cache: dict[int, dict[str, np.ndarray]] = {}

    def keys(self):
        return list(self._index.keys())

    def __len__(self) -> int:
        return len(self._index)

    def read(self, image_id: str) -> np.ndarray:
        if image_id not in self._index:
            raise KeyError(image_id)
        shard_id, row = self._index[image_id]
        shard = self._load_shard(shard_id)
        return shard[str(row)]

    def assert_compatible(self, *, expected_hash: str) -> None:
        got = self.meta.get("hash", "")
        if got != expected_hash:
            raise ValueError(
                f"cache hash mismatch: expected {expected_hash}, got {got!r}"
            )

    def _load_shard(self, shard_id: int) -> dict[str, np.ndarray]:
        if shard_id not in self._shard_cache:
            p = self.cache_dir / "shards" / f"{shard_id:05d}.npz"
            with np.load(p) as npz:
                self._shard_cache[shard_id] = {k: npz[k] for k in npz.files}
        return self._shard_cache[shard_id]


def compute_cache_meta_hash(meta: dict[str, Any]) -> str:
    """Deterministic 16-char hash of a meta dict (sorted keys, compact JSON)."""
    blob = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
