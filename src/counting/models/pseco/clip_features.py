"""CLIP ViT-B/32 text feature extraction and caching."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

_CLIP_DIM = 512
_CLIP_MODEL_NAME = "ViT-B-32"
_CLIP_PRETRAINED = "openai"


def encode_class_names(
    class_names: Iterable[str],
    *,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Return {class_name: tensor(512,)} by forwarding class names through
    open_clip's ViT-B/32 text tower. Requires the `open_clip_torch` package."""
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        _CLIP_MODEL_NAME, pretrained=_CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(_CLIP_MODEL_NAME)
    model.to(device).eval()

    names = list(class_names)
    tokens = tokenizer(names).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens).float().cpu()

    return {name: feats[i] for i, name in enumerate(names)}


def save_text_features(
    features: dict[str, torch.Tensor],
    path: str | Path,
) -> None:
    """Save `{name: tensor(512,)}` dict to a `.pt` file, creating parents."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, out)


def load_text_features(path: str | Path) -> dict[str, torch.Tensor]:
    """Load `.pt` cache, validating structure: dict[str, Tensor(512,)]."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CLIP feature cache not found: {p}")
    data = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"CLIP cache at {p} is not a mapping (got {type(data).__name__})")
    for name, tensor in data.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (_CLIP_DIM,):
            raise ValueError(
                f"CLIP cache entry {name!r} has shape {tuple(tensor.shape)}; expected (512,)"
            )
    return data
