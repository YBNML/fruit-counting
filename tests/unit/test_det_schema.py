from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from counting.config.det_schema import (
    INFER_KEYS, DetDataConfig, DetInferConfig, DetTrainConfig, FrcnnConfig, OptimConfig, model_mismatch,
)
from counting.config.loader import load_yaml_model

ROOT = Path(__file__).resolve().parents[2]


def test_detector_defaults_follow_the_chosen_settings():
    m = FrcnnConfig()
    assert m.arch == "fasterrcnn_resnet50_fpn_v2"
    assert m.anchor_sizes == [16, 32, 64, 128, 256] and m.aspect_ratios == [0.5, 1.0, 2.0]
    assert (m.train_size, m.infer_min_size, m.infer_max_size) == (1024, 1080, 1920)
    assert (m.max_dets, m.score_min, m.rpn_pre_nms_test, m.rpn_post_nms_test) == (1000, 0.05, 2000, 2000)
    assert (m.rpn_pre_nms_train, m.rpn_post_nms_train, m.box_batch_per_image) == (2000, 2000, 512)
    assert m.trainable_backbone_layers == 3 and m.freeze_backbone_bn is True
    t = DetTrainConfig()
    assert t.data.crops == ["apple", "pear"] and t.data.seeds is None
    assert t.data.root == "~/datasets/orchard_synth/{crop}/train_s{seed}"
    assert (t.optim.epochs, t.optim.batch_size) == (12, 2) and t.optim.base_lr() == pytest.approx(0.0025)
    assert OptimConfig(batch_size=1).base_lr() == pytest.approx(0.00125) and OptimConfig(lr=0.01).base_lr() == 0.01
    assert (t.aug.hflip_p, t.aug.brightness, t.aug.contrast, t.aug.saturation, t.aug.hue) == (0.5, 0.2, 0.2, 0.2, 0.02)
    assert t.pretrained is True and (t.device, t.output_dir, t.run_name) == ("cuda", "runs/train", "frcnn_synth")
    d = DetInferConfig()
    assert d.weights_sha256 is None and d.output_dir == "runs/detect" and d.model == m


def test_committed_detector_configs_spell_out_every_setting():
    loaded = {}
    for rel, cls in (("configs/train/frcnn_synth.yaml", DetTrainConfig), ("configs/detect/frcnn_synth.yaml",
                                                                         DetInferConfig)):
        raw = yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))
        loaded[cls] = load_yaml_model(ROOT / rel, cls)
        assert set(raw) == set(cls.model_fields), rel
        for name, field in cls.model_fields.items():
            if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
                assert set(raw[name]) == set(field.annotation.model_fields), (rel, name)
    assert model_mismatch(loaded[DetTrainConfig].model, loaded[DetInferConfig].model) == []
    assert model_mismatch(FrcnnConfig(), FrcnnConfig(arch="fasterrcnn_resnet50_fpn", anchor_sizes=[8, 16, 32, 64, 128])) \
        == ["arch", "anchor_sizes"]
    assert INFER_KEYS == ("arch", "anchor_sizes", "aspect_ratios", "freeze_backbone_bn")


def test_detector_configs_refuse_unknown_keys_and_bad_values():
    with pytest.raises(ValidationError):
        FrcnnConfig(anchors=[16])
    with pytest.raises(ValidationError):
        FrcnnConfig(anchor_sizes=[16, 32, 64, 128])
    with pytest.raises(ValidationError, match="model.infer_min_size must be <= model.infer_max_size"):
        FrcnnConfig(infer_min_size=2000)
    with pytest.raises(ValidationError, match=r"data.root needs the \{crop\} and \{seed\} fields"):
        DetDataConfig(root="~/datasets/train_s{seed}")
    for root in ("~/d/{crop}/s{seed}/{run}", "~/d/{crop}/s{seed}/{0}", "~/d/{crop}/s{seed}/{"):
        with pytest.raises(ValidationError, match=r"data.root may use only the \{crop\} and \{seed\} fields, got "):
            DetDataConfig(root=root)
    assert DetDataConfig(root="~/d/{crop}/{{x}}/s{seed}").root == "~/d/{crop}/{{x}}/s{seed}"
    with pytest.raises(ValidationError, match="data.crops repeats a crop"):
        DetDataConfig(crops=["apple", "apple"])
    with pytest.raises(ValidationError):
        DetInferConfig(weights_sha256="abc")


def test_cuda_environment_pins_torch_and_torchvision():
    text = (ROOT / "environment-cuda.yml").read_text(encoding="utf-8")
    assert "- torch==2.15.0" in text and "- torchvision==0.29.2" in text and "--extra-index-url" not in text


def test_frozen_detection_config_pins_only_the_weights():
    rel = "configs/detect/frozen/frcnn_synth_d1.yaml"
    raw = yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))
    assert set(raw) == set(DetInferConfig.model_fields) and set(raw["model"]) == set(FrcnnConfig.model_fields)
    frozen = load_yaml_model(ROOT / rel, DetInferConfig)
    base = load_yaml_model(ROOT / "configs/detect/frcnn_synth.yaml", DetInferConfig)
    assert len(frozen.weights_sha256) == 64 and int(frozen.weights_sha256, 16) >= 0
    assert frozen.model_dump(exclude={"weights_sha256"}) == base.model_dump(exclude={"weights_sha256"})
