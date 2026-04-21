from pathlib import Path

import numpy as np
import pytest

from counting.data.cache import FeatureCacheWriter

ROOT = Path(__file__).resolve().parents[2]


def _weights_available() -> bool:
    import yaml

    cfg_path = ROOT / "configs" / "train" / "pseco_head.yaml"
    if not cfg_path.exists():
        return False
    cfg = yaml.safe_load(cfg_path.read_text())
    need = [cfg["model"]["sam_checkpoint"], cfg["model"]["init_decoder"]]
    return all((ROOT / p).exists() for p in need)


def _external_available() -> bool:
    return (ROOT / "external" / "PseCo" / "models.py").exists()


@pytest.mark.slow
@pytest.mark.skipif(
    not (_weights_available() and _external_available()),
    reason="PseCo weights or external/ not present",
)
def test_train_pseco_head_smoke(tmp_path):
    """Run 2 epochs on a tiny synthetic cached dataset with a fake CLIP cache."""
    import json

    import torch
    from PIL import Image

    from counting.config.train_schema import TrainAppConfig
    from counting.models.pseco.clip_features import save_text_features
    from counting.models.pseco.trainer import train_pseco_head

    dataset_root = tmp_path / "fsc"
    (dataset_root / "images_384_VarV2").mkdir(parents=True)

    names = ["1.jpg", "2.jpg"]
    for n in names:
        Image.fromarray(
            (np.random.rand(64, 64, 3) * 255).astype("uint8")
        ).save(dataset_root / "images_384_VarV2" / n)
    (dataset_root / "annotation_FSC147_384.json").write_text(json.dumps({
        "1.jpg": {"points": [[5, 5], [10, 10]], "box_examples_coordinates": []},
        "2.jpg": {"points": [[20, 20]], "box_examples_coordinates": []},
    }))
    (dataset_root / "Train_Test_Val_FSC_147.json").write_text(json.dumps({
        "train": ["1.jpg"], "val": ["2.jpg"], "test": [],
    }))
    (dataset_root / "ImageClasses_FSC147.txt").write_text(
        "1.jpg\ttestfruit\n2.jpg\ttestfruit\n"
    )

    cache_dir = tmp_path / "cache"
    writer = FeatureCacheWriter(
        cache_dir=cache_dir,
        meta={"source": "smoke", "hash": "smoke0000smoke00"},
        shard_size=4,
    )
    writer.open()
    for n in names:
        writer.write(n, np.random.randn(256, 64, 64).astype(np.float16))
    writer.close()

    # Fake CLIP features — random 512-d tensor for the test class
    clip_cache_path = tmp_path / "clip.pt"
    save_text_features({"testfruit": torch.randn(512)}, clip_cache_path)

    cfg = TrainAppConfig.model_validate({
        "run_name": "smoke",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 0,
        "output_dir": str(tmp_path / "runs"),
        "data": {
            "format": "fsc147", "root": str(dataset_root),
            "train_split": "train", "val_split": "val", "image_size": 1024,
        },
        "model": {
            "sam_checkpoint": str(ROOT / "models" / "PseCo" / "sam_vit_h.pth"),
            "init_decoder": str(ROOT / "models" / "PseCo" / "point_decoder_vith.pth"),
            "init_mlp": "",
            "clip_features_cache": str(clip_cache_path),
        },
        "cache": {
            "enabled": True, "dir": str(cache_dir),
            "dtype": "float16", "augment_variants": 1,
        },
        "train": {
            "batch_size": 1, "epochs": 2, "lr": 1e-4,
            "weight_decay": 1e-4, "warmup_steps": 0,
            "scheduler": "cosine",
            "loss_weights": {"cls": 1.0, "count": 0.1},
            "early_stopping": {"patience": 5, "metric": "val_mae", "mode": "min"},
        },
        "logging": {
            "tensorboard": False, "log_every_n_steps": 1, "save_every_n_epochs": 1,
        },
    })

    train_pseco_head(cfg)

    ckpt_dir = tmp_path / "runs" / "smoke" / "checkpoints"
    assert (ckpt_dir / "best.ckpt").exists()
    assert (ckpt_dir / "last.ckpt").exists()
