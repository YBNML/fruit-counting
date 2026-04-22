import pytest
from pydantic import ValidationError

from counting.config.schema import (
    AppConfig,
    ClassifierStageConfig,
    DeblurStageConfig,
    IOConfig,
    PipelineConfig,
    PseCoStageConfig,
    SRStageConfig,
    StageSet,
)


def _minimal_dict():
    return {
        "device": "auto",
        "seed": 42,
        "output_dir": "./runs",
        "pipeline": {
            "stages": {
                "deblur": {"enabled": True, "weights": "models/deblur/fpn.pth"},
                "pseco": {
                    "enabled": True,
                    "prompt": "protective fruit bag",
                    "sam_checkpoint": "models/PseCo/sam_vit_h.pth",
                    "decoder_checkpoint": "models/PseCo/point_decoder_vith.pth",
                    "mlp_checkpoint": "models/PseCo/MLP_small_box_w1_zeroshot.tar",
                },
                "sr": {"enabled": False, "scale": 2.0, "max_crop_side": 500},
                "classifier": {
                    "enabled": False,
                    "checkpoint": "models/classification/classification_model.pt",
                    "threshold": 0.5,
                },
            }
        },
        "io": {"output_format": "json", "save_visualizations": False},
    }


def test_valid_config_parses():
    cfg = AppConfig.model_validate(_minimal_dict())
    assert cfg.device == "auto"
    assert cfg.pipeline.stages.pseco.prompt == "protective fruit bag"


def test_invalid_device_rejected():
    d = _minimal_dict()
    d["device"] = "tpu"
    with pytest.raises(ValidationError):
        AppConfig.model_validate(d)


def test_pseco_requires_checkpoints_when_enabled():
    d = _minimal_dict()
    d["pipeline"]["stages"]["pseco"]["sam_checkpoint"] = ""
    with pytest.raises(ValidationError):
        AppConfig.model_validate(d)


def test_sr_disabled_allows_missing_fields():
    d = _minimal_dict()
    d["pipeline"]["stages"]["sr"] = {"enabled": False}
    cfg = AppConfig.model_validate(d)
    assert cfg.pipeline.stages.sr.enabled is False


def test_classifier_threshold_bounds():
    d = _minimal_dict()
    d["pipeline"]["stages"]["classifier"]["threshold"] = 1.5
    with pytest.raises(ValidationError):
        AppConfig.model_validate(d)


def test_io_format_options():
    d = _minimal_dict()
    d["io"]["output_format"] = "both"
    cfg = AppConfig.model_validate(d)
    assert cfg.io.output_format == "both"

    d["io"]["output_format"] = "xml"
    with pytest.raises(ValidationError):
        AppConfig.model_validate(d)


def test_pseco_hyperparameters_validate():
    from counting.config.schema import AppConfig

    d = _minimal_dict()
    d["pipeline"]["stages"]["pseco"].update({
        "clip_features_cache": "models/PseCo/clip_text_features.pt",
        "point_threshold": 0.1,
        "max_points": 500,
        "anchor_size": 16,
        "nms_threshold": 0.3,
        "score_threshold": 0.2,
    })
    cfg = AppConfig.model_validate(d)
    assert cfg.pipeline.stages.pseco.clip_features_cache.endswith(".pt")
    assert cfg.pipeline.stages.pseco.anchor_size == 16

    d["pipeline"]["stages"]["pseco"]["anchor_size"] = 0
    import pytest
    with pytest.raises(Exception):  # ValidationError on gt=0 constraint
        AppConfig.model_validate(d)
