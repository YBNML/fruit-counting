"""The torchvision Faster R-CNN detector: model builder, weight loading and single-frame inference.

`build_model` makes a COCO-shaped torchvision Faster R-CNN (v2 or v1, per `FrcnnConfig.arch`) and
then swaps in the configured anchors, a 3-way box predictor (background, fruit, bag) and, if
asked, frozen batch norm in the backbone. `FrcnnDetector` runs one RGB frame through a model and
returns its raw output as `Detections` (clipped, classes mapped, no NMS or other postprocessing),
ready for `counting.detection.cached.detect_to_cache`.

torch and torchvision are imported only inside functions, so importing this module stays cheap
and torch-free. Only `FrameInput` is imported from `counting.data.formats.orchard_synth`: the
detector never sees ground truth.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from counting.config.det_schema import FrcnnConfig
from counting.data.formats.orchard_synth import FrameInput
from counting.detection.base import Detections
from counting.detection.cached import file_sha256

COCO_TERMS = "research use only; COCO annotations CC BY 4.0, COCO images under Flickr terms"
# Detector classes: 0 is background, 1 fruit and 2 bag; `Detections.labels` uses 0 fruit and 1 bag.
NUM_CLASSES = 3


def to_detections(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, *, width: int, height: int
) -> Detections:
    """Raw detector output (numpy arrays) as `Detections` for a `width` x `height` frame.

    Boxes are clipped to `[0, width] x [0, height]`; class 1 (fruit) becomes label 0 and class 2
    (bag) label 1. Any other class raises `ValueError`. No box is dropped or reordered.
    """
    labels = np.asarray(labels)
    bad = labels[(labels != 1) & (labels != 2)]
    if bad.size:
        raise ValueError(f"detector class {int(bad[0])} is not fruit (1) or bag (2)")
    b = np.array(boxes, dtype=np.float64).reshape(-1, 4)
    b[:, (0, 2)] = np.clip(b[:, (0, 2)], 0.0, float(width))
    b[:, (1, 3)] = np.clip(b[:, (1, 3)], 0.0, float(height))
    return Detections(b.astype(np.float32), np.asarray(scores, dtype=np.float32).reshape(-1),
                      (labels.reshape(-1) - 1).astype(np.int8))


def _arch(arch: str):
    """(torchvision builder, its weights enum) for `FrcnnConfig.arch`."""
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn,
        fasterrcnn_resnet50_fpn_v2,
    )

    if arch == "fasterrcnn_resnet50_fpn_v2":
        return fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    if arch == "fasterrcnn_resnet50_fpn":
        return fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    raise ValueError(f"unknown detector arch {arch!r}")


def _freeze_batch_norm(module) -> None:
    """Replace every `nn.BatchNorm2d` under `module` with a `FrozenBatchNorm2d` of the same values and eps."""
    import torch
    from torchvision.ops.misc import FrozenBatchNorm2d

    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            with torch.no_grad():
                frozen.weight.copy_(child.weight)
                frozen.bias.copy_(child.bias)
                frozen.running_mean.copy_(child.running_mean)
                frozen.running_var.copy_(child.running_var)
            setattr(module, name, frozen)
        else:
            _freeze_batch_norm(child)


def build_model(
    cfg: FrcnnConfig, *, pretrained: bool, mode: Literal["train", "infer"], seed: int | None = None
):
    """A torchvision Faster R-CNN built from `cfg`, with a 3-class predictor (background, fruit, bag).

    `pretrained` loads the COCO weights (from the torch.hub cache, downloading them if missing)
    before the predictor is replaced. `mode` picks the resize: `train_size` for both sides in
    "train", `infer_min_size`/`infer_max_size` in "infer". With `seed`, `torch.manual_seed(seed)`
    runs first, so the freshly initialised layers are reproducible.
    """
    import torch
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if mode == "train":
        min_size, max_size = cfg.train_size, cfg.train_size
    elif mode == "infer":
        min_size, max_size = cfg.infer_min_size, cfg.infer_max_size
    else:
        raise ValueError(f"mode must be 'train' or 'infer', got {mode!r}")
    builder, weights = _arch(cfg.arch)
    if seed is not None:
        torch.manual_seed(seed)
    model = builder(
        weights=weights.COCO_V1 if pretrained else None, weights_backbone=None, num_classes=91,
        trainable_backbone_layers=cfg.trainable_backbone_layers if pretrained else None,
        min_size=min_size, max_size=max_size, box_detections_per_img=cfg.max_dets, box_score_thresh=cfg.score_min,
        rpn_pre_nms_top_n_train=cfg.rpn_pre_nms_train, rpn_pre_nms_top_n_test=cfg.rpn_pre_nms_test,
        rpn_post_nms_top_n_train=cfg.rpn_post_nms_train, rpn_post_nms_top_n_test=cfg.rpn_post_nms_test,
        box_batch_size_per_image=cfg.box_batch_per_image,
    )
    # The v2 builder passes its own anchor generator to FasterRCNN, so anchors cannot go in as kwargs.
    model.rpn.anchor_generator = AnchorGenerator(tuple((s,) for s in cfg.anchor_sizes), (tuple(cfg.aspect_ratios),) * 5)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,
                                                      NUM_CLASSES)
    if cfg.freeze_backbone_bn:
        _freeze_batch_norm(model.backbone)
    return model


def pretrained_info(cfg: FrcnnConfig) -> dict:
    """Provenance of the COCO weights `build_model(cfg, pretrained=True, ...)` starts from.

    `sha256` is that of the file in the torch.hub cache, or None when it is not there (nothing is
    downloaded here).
    """
    import torch

    _, weights = _arch(cfg.arch)
    w = weights.COCO_V1
    file = Path(urlparse(w.url).path).name
    cached = Path(torch.hub.get_dir()) / "checkpoints" / file
    return {"weights": f"{weights.__name__}.{w.name}", "file": file,
            "sha256": file_sha256(cached) if cached.is_file() else None,
            "recipe": w.meta["recipe"], "terms": COCO_TERMS}


def _load_error_detail(e: RuntimeError) -> str:
    """The first line of a `load_state_dict` error that names a key (torch puts a header line first)."""
    lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
    if len(lines) > 1 and lines[0].startswith("Error(s) in loading state_dict"):
        return lines[1]
    return lines[0] if lines else type(e).__name__


def load_weights(model, path: Path) -> None:
    """Load the state dict at `path` into `model`; `ValueError` (naming `path`) if the file is not a torch state
    dict or any key is missing, unexpected or misshapen."""
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError, ValueError) as e:
        raise ValueError(f"{path}: not a readable torch weights file ({type(e).__name__})") from None
    if not isinstance(state, Mapping):
        raise ValueError(f"{path}: not a model state dict (the file holds a {type(state).__name__})")
    try:
        result = model.load_state_dict(state, strict=False)
    except RuntimeError as e:
        raise ValueError(f"{path}: weights do not fit this model: {_load_error_detail(e)}") from None
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    if missing or unexpected:
        key = (missing + unexpected)[0]
        raise ValueError(f"{path}: weights do not fit this model: {len(missing)} missing and "
                         f"{len(unexpected)} unexpected keys (first: {key})")


class FrcnnDetector:
    """A `Detector` that runs a Faster R-CNN model on one frame and returns its raw detections."""

    name = "frcnn"

    def __init__(self, model, *, device: str) -> None:
        self.device = device
        self.model = model.to(device).eval()

    def detect(self, frame: FrameInput) -> Detections:
        import torch

        width, height = frame.camera.width, frame.camera.height
        with Image.open(frame.rgb_path) as im:
            if im.size != (width, height):
                raise ValueError(f"{frame.rgb_path}: image is {im.size[0]}x{im.size[1]}, "
                                 f"the frame camera is {width}x{height}")
            rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / np.float32(255)
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(self.device)
        with torch.inference_mode():
            out = self.model([x])[0]
        return to_detections(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["labels"].cpu().numpy(),
                             width=width, height=height)


def load_detector(weights: Path, cfg: FrcnnConfig, *, device: str) -> FrcnnDetector:
    """A `FrcnnDetector` for the weights at `weights`, built for inference from `cfg`."""
    model = build_model(cfg, pretrained=False, mode="infer")
    load_weights(model, weights)
    return FrcnnDetector(model, device=device)
