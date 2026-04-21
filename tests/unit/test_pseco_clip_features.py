from pathlib import Path

import pytest
import torch

from counting.models.pseco.clip_features import (
    load_text_features,
    save_text_features,
)


def test_save_and_load_roundtrip(tmp_path):
    features = {
        "apples": torch.randn(512),
        "sea shells": torch.randn(512),
        "hot air balloons": torch.randn(512),
    }
    out = tmp_path / "clip_text_features.pt"
    save_text_features(features, out)
    assert out.exists()

    loaded = load_text_features(out)
    assert set(loaded.keys()) == set(features.keys())
    for name, tensor in features.items():
        torch.testing.assert_close(loaded[name], tensor)


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_text_features(tmp_path / "missing.pt")


def test_load_rejects_nondict(tmp_path):
    out = tmp_path / "bogus.pt"
    torch.save(torch.randn(3), out)
    with pytest.raises(ValueError, match="mapping"):
        load_text_features(out)


def test_load_rejects_wrong_dim(tmp_path):
    out = tmp_path / "wrong.pt"
    torch.save({"apples": torch.randn(256)}, out)
    with pytest.raises(ValueError, match="512"):
        load_text_features(out)


@pytest.mark.slow
def test_encode_with_open_clip_real():
    """Actually load open_clip ViT-B/32 and encode three class names.

    Skipped in quick test runs; run explicitly via `pytest -m slow` after the
    weights are cached locally.
    """
    open_clip = pytest.importorskip("open_clip")  # noqa: F841
    from counting.models.pseco.clip_features import encode_class_names

    features = encode_class_names(
        ["apples", "sea shells", "hot air balloons"],
        device="cpu",
    )
    assert set(features.keys()) == {"apples", "sea shells", "hot air balloons"}
    for tensor in features.values():
        assert tensor.shape == (512,)
        assert tensor.dtype == torch.float32
