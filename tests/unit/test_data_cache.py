import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from counting.data.cache import (
    FeatureCacheReader,
    FeatureCacheWriter,
    compute_cache_meta_hash,
)


def _fake_embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((256, 64, 64)).astype(np.float16)


def test_write_and_read_roundtrip(tmp_path):
    writer = FeatureCacheWriter(
        cache_dir=tmp_path,
        meta={"source": "test", "sam_ckpt_hash": "abc"},
        shard_size=2,
    )
    writer.open()
    ids = ["img_a.jpg", "img_b.jpg", "img_c.jpg"]
    for i, name in enumerate(ids):
        writer.write(name, _fake_embedding(i))
    writer.close()

    shards = sorted(p.name for p in (tmp_path / "shards").iterdir())
    assert shards == ["00000.npz", "00001.npz"]

    reader = FeatureCacheReader(tmp_path)
    assert set(reader.keys()) == set(ids)
    a = reader.read("img_a.jpg")
    b = reader.read("img_b.jpg")
    c = reader.read("img_c.jpg")
    assert a.shape == (256, 64, 64)
    assert a.dtype == np.float16
    assert not np.array_equal(a, b)
    assert not np.array_equal(b, c)


def test_meta_contains_expected_fields(tmp_path):
    writer = FeatureCacheWriter(
        cache_dir=tmp_path,
        meta={"source": "test", "sam_ckpt_hash": "abc"},
        shard_size=2,
    )
    writer.open()
    writer.write("x.jpg", _fake_embedding(0))
    writer.close()
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["source"] == "test"
    assert meta["sam_ckpt_hash"] == "abc"
    assert meta["dtype"] == "float16"
    assert meta["shape"] == [256, 64, 64]
    assert meta["count"] == 1


def test_reader_detects_hash_mismatch(tmp_path):
    writer = FeatureCacheWriter(
        cache_dir=tmp_path,
        meta={"source": "test", "sam_ckpt_hash": "abc"},
        shard_size=2,
    )
    writer.open()
    writer.write("x.jpg", _fake_embedding(0))
    writer.close()

    reader = FeatureCacheReader(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        reader.assert_compatible(expected_hash="WRONG")


def test_compute_hash_is_stable():
    a = compute_cache_meta_hash({"sam": "abc", "image_size": 1024})
    b = compute_cache_meta_hash({"image_size": 1024, "sam": "abc"})
    assert a == b
    assert len(a) == 16


def test_missing_key_raises(tmp_path):
    writer = FeatureCacheWriter(
        cache_dir=tmp_path,
        meta={"source": "test", "sam_ckpt_hash": "abc"},
        shard_size=2,
    )
    writer.open()
    writer.write("only.jpg", _fake_embedding(0))
    writer.close()

    reader = FeatureCacheReader(tmp_path)
    with pytest.raises(KeyError):
        reader.read("missing.jpg")
