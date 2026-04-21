import pytest
from pydantic import ValidationError

from counting.config.train_schema import TrainAppConfig


def _minimal():
    return {
        "run_name": "pseco_head_v1",
        "device": "cuda",
        "seed": 42,
        "output_dir": "./runs",
        "data": {
            "format": "fsc147",
            "root": "./datasets/fsc147",
            "train_split": "train",
            "val_split": "val",
            "image_size": 1024,
        },
        "model": {
            "sam_checkpoint": "models/PseCo/sam_vit_h.pth",
            "init_decoder": "models/PseCo/point_decoder_vith.pth",
            "init_mlp": "models/PseCo/MLP_small_box_w1_zeroshot.tar",
            "clip_features_cache": "models/PseCo/clip_text_features.pt",
        },
        "cache": {
            "enabled": True,
            "dir": "./feature_cache/fsc147_vit_h",
            "dtype": "float16",
            "augment_variants": 1,
        },
        "train": {
            "batch_size": 8,
            "epochs": 30,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "warmup_steps": 500,
            "scheduler": "cosine",
            "loss_weights": {"cls": 1.0, "count": 0.1},
            "early_stopping": {"patience": 5, "metric": "val_mae", "mode": "min"},
        },
        "logging": {
            "tensorboard": True,
            "log_every_n_steps": 20,
            "save_every_n_epochs": 1,
        },
    }


def test_valid_parses():
    cfg = TrainAppConfig.model_validate(_minimal())
    assert cfg.run_name == "pseco_head_v1"
    assert cfg.data.format == "fsc147"
    assert cfg.train.scheduler == "cosine"
    assert cfg.model.clip_features_cache.endswith("clip_text_features.pt")


def test_scheduler_must_be_known():
    d = _minimal()
    d["train"]["scheduler"] = "tangent"
    with pytest.raises(ValidationError):
        TrainAppConfig.model_validate(d)


def test_dtype_must_be_float16_or_float32():
    d = _minimal()
    d["cache"]["dtype"] = "int8"
    with pytest.raises(ValidationError):
        TrainAppConfig.model_validate(d)


def test_augment_variants_positive():
    d = _minimal()
    d["cache"]["augment_variants"] = 0
    with pytest.raises(ValidationError):
        TrainAppConfig.model_validate(d)


def test_early_stopping_mode_literal():
    d = _minimal()
    d["train"]["early_stopping"]["mode"] = "sideways"
    with pytest.raises(ValidationError):
        TrainAppConfig.model_validate(d)


def test_clip_features_cache_required_non_empty():
    d = _minimal()
    d["model"]["clip_features_cache"] = ""
    with pytest.raises(ValidationError, match="clip_features_cache"):
        TrainAppConfig.model_validate(d)
