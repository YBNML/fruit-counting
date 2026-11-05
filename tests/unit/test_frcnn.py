from pathlib import Path

import numpy as np
import pytest

from counting.config.det_schema import DetInferConfig, FrcnnConfig
from counting.config.loader import load_yaml_model
from counting.data.formats.orchard_synth import SynthRun
from counting.detection.cached import postprocess
from counting.detection.frcnn import to_detections
from tests.unit.synthfake import write_run

ROOT = Path(__file__).resolve().parents[2]
# The committed inference model (so a changed arch or anchors follow it) with small sampling sizes, which keep a CPU
# training step near two seconds.
SMALL = load_yaml_model(ROOT / "configs" / "detect" / "frcnn_synth.yaml", DetInferConfig).model.model_copy(update=dict(
    train_size=128, infer_min_size=360, infer_max_size=640, max_dets=50, rpn_pre_nms_test=200, rpn_post_nms_test=100,
    rpn_pre_nms_train=200, rpn_post_nms_train=100, box_batch_per_image=32))
OTHER_ARCH = {"fasterrcnn_resnet50_fpn_v2": "fasterrcnn_resnet50_fpn",
              "fasterrcnn_resnet50_fpn": "fasterrcnn_resnet50_fpn_v2"}
TRAIN_CFG = str(ROOT / "configs" / "train" / "frcnn_synth.yaml")
SPLITS = str(ROOT / "configs" / "eval" / "splits.yaml")


def _tiny_model(nan_grad=False):
    """A stand-in for the detector in the training loop: one weight `w` (0) and the four torchvision losses, all
    finite. With `nan_grad` the gradient is NaN (d/dw of 0 * sqrt(w) at w = 0)."""
    import torch

    from counting.training.detector import LOSS_KEYS

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, images, targets):
            loss = (0.0 * torch.sqrt(self.w)).sum() if nan_grad else ((self.w - 1.0) ** 2).sum()
            return {k: loss for k in LOSS_KEYS}

    return Tiny()


def _tiny_sets(tmp_path, *, epochs=3):
    """`--set` values of a CPU training on one two-frame train bundle (written here), one frame per step."""
    write_run(tmp_path / "apple_s100", crop="apple", pitches=(30.0,), seed=100, max_frames=2, images=True)
    return [f"data.root={tmp_path}/{{crop}}_s{{seed}}", "data.crops=[apple]", "data.seeds=[100]",
            "model.train_size=128", "optim.batch_size=1", f"optim.epochs={epochs}", "optim.warmup_iters=1",
            "log_every=1", "pretrained=false", "num_workers=0", "device=cpu", f"output_dir={tmp_path}/train"]


def _first_epoch_then_stop(cfg, monkeypatch, **kwargs):
    """Run `train_detector(cfg, ...)` until its first epoch checkpoint is saved, then stop it; return its folder."""
    import counting.training.detector as td
    from tests.unit.synthfake import load_splits

    real = td.save_epoch_checkpoint

    def stop_after_saving(*args, **kw):
        real(*args, **kw)
        raise RuntimeError("stopped")

    monkeypatch.setattr(td, "save_epoch_checkpoint", stop_after_saving)
    with pytest.raises(RuntimeError, match="stopped"):
        td.train_detector(cfg, load_splits(), **kwargs)
    monkeypatch.setattr(td, "save_epoch_checkpoint", real)
    return td.run_folder(cfg, load_splits(), max_iters=kwargs.get("max_iters"), densest_first=False)[0]


def test_to_detections_maps_classes_and_clips():
    d = to_detections(np.array([[-3.0, 5, 20, 400], [10, 10, 30, 30]]), np.array([0.9, 0.2]), np.array([2, 1]),
                      width=640, height=360)
    assert d.boxes.tolist() == [[0, 5, 20, 360], [10, 10, 30, 30]] and d.labels.tolist() == [1, 0]
    assert (d.boxes.dtype, d.scores.dtype, d.labels.dtype, d.depth_m) == (np.float32, np.float32, np.int8, None)
    with pytest.raises(ValueError, match=r"detector class 3 is not fruit \(1\) or bag \(2\)"):
        to_detections(np.zeros((1, 4)), np.ones(1), np.array([3]), width=640, height=360)


@pytest.mark.slow
def test_build_model_follows_the_config():
    import torch

    from counting.detection.frcnn import build_model, pretrained_info
    m = build_model(FrcnnConfig(), pretrained=False, mode="infer", seed=0)
    assert m.rpn.anchor_generator.sizes == ((16,), (32,), (64,), (128,), (256,))
    assert m.rpn.anchor_generator.aspect_ratios == ((0.5, 1.0, 2.0),) * 5
    assert (m.roi_heads.box_predictor.cls_score.out_features, m.roi_heads.box_predictor.bbox_pred.out_features) == (3, 12)
    assert (m.transform.min_size, m.transform.max_size) == ((1080,), 1920)
    assert (m.roi_heads.detections_per_img, m.roi_heads.score_thresh, m.roi_heads.nms_thresh) == (1000, 0.05, 0.5)
    assert (m.rpn._pre_nms_top_n["testing"], m.rpn._post_nms_top_n["testing"]) == (2000, 2000)
    assert m.roi_heads.fg_bg_sampler.batch_size_per_image == 512
    assert not any(isinstance(x, torch.nn.BatchNorm2d) for x in m.backbone.modules())
    assert any(isinstance(x, torch.nn.BatchNorm2d) for x in m.roi_heads.box_head.modules())
    t = build_model(FrcnnConfig(), pretrained=False, mode="train", seed=0)
    assert (t.transform.min_size, t.transform.max_size) == ((1024,), 1024)
    assert torch.equal(t.roi_heads.box_predictor.cls_score.weight, m.roi_heads.box_predictor.cls_score.weight)
    info = pretrained_info(FrcnnConfig())
    assert info["file"] == "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth" and info["weights"].endswith(".COCO_V1")
    assert info["recipe"] and info["terms"].startswith("research use only")


@pytest.mark.slow
def test_untrained_detector_gives_cache_ready_detections(tmp_path):
    from PIL import Image

    from counting.detection.frcnn import FrcnnDetector, build_model
    run = SynthRun.open(write_run(tmp_path / "b", crop="apple", pitches=(30.0,), seed=200, max_frames=2, images=True))
    det = FrcnnDetector(build_model(SMALL, pretrained=False, mode="infer", seed=0), device="cpu")
    for f in run.frames:
        d = det.detect(f)
        assert det.name == "frcnn" and d.depth_m is None and len(d) <= 50
        assert (d.boxes.dtype, d.scores.dtype, d.labels.dtype) == (np.float32, np.float32, np.int8)
        assert (d.boxes[:, [0, 2]] >= 0).all() and (d.boxes[:, [0, 2]] <= 640).all()
        assert (d.boxes[:, [1, 3]] >= 0).all() and (d.boxes[:, [1, 3]] <= 360).all()
        assert set(d.labels.tolist()) <= {0, 1} and (d.scores >= 0.05).all()
        postprocess(d, width=640, height=360)
    Image.new("RGB", (320, 180)).save(run.frames[0].rgb_path)
    with pytest.raises(ValueError, match="image is 320x180, the frame camera is 640x360"):
        det.detect(run.frames[0])


@pytest.mark.slow
def test_load_weights_refuses_other_architectures(tmp_path):
    import torch
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    from counting.detection.frcnn import build_model, load_weights
    torch.save(build_model(SMALL, pretrained=False, mode="infer").state_dict(), tmp_path / "w.pt")
    load_weights(build_model(SMALL, pretrained=False, mode="infer"), tmp_path / "w.pt")
    other = build_model(SMALL.model_copy(update={"arch": OTHER_ARCH[SMALL.arch]}), pretrained=False, mode="infer")
    with pytest.raises(ValueError, match="weights do not fit this model"):
        load_weights(other, tmp_path / "w.pt")
    coco_head = build_model(SMALL, pretrained=False, mode="infer")
    coco_head.roi_heads.box_predictor = FastRCNNPredictor(coco_head.roi_heads.box_predictor.cls_score.in_features, 91)
    torch.save(coco_head.state_dict(), tmp_path / "w91.pt")
    with pytest.raises(ValueError, match="weights do not fit this model"):
        load_weights(build_model(SMALL, pretrained=False, mode="infer"), tmp_path / "w91.pt")


@pytest.mark.slow
def test_train_detector_writes_weights_log_and_record_and_resumes(tmp_path, monkeypatch):
    import csv
    import hashlib
    import json
    import math
    from pathlib import Path

    import torch
    from typer.testing import CliRunner

    import counting.training.detector as td
    from counting.cli import app
    from counting.config.det_schema import DetTrainConfig
    from counting.config.loader import load_yaml_model
    from tests.unit.synthfake import load_splits

    root = Path(__file__).resolve().parents[2]
    train_cfg = str(root / "configs" / "train" / "frcnn_synth.yaml")
    write_run(tmp_path / "apple_s100", crop="apple", pitches=(30.0,), seed=100, max_frames=2, images=True)
    sets = [f"data.root={tmp_path}/{{crop}}_s{{seed}}", "data.crops=[apple]", "data.seeds=[100]",
            "model.train_size=128", "model.rpn_pre_nms_train=200", "model.rpn_post_nms_train=100",
            "model.box_batch_per_image=32", "optim.batch_size=1", "optim.epochs=2", "optim.warmup_iters=1",
            "log_every=1", "pretrained=false", "num_workers=0", "device=cpu", f"output_dir={tmp_path}/train"]
    cfg = load_yaml_model(train_cfg, DetTrainConfig, overrides=sets)
    real = td.save_epoch_checkpoint

    def stop_after_saving(*args, **kwargs):     # an interrupted run: the first epoch is saved, then it stops
        real(*args, **kwargs)
        raise RuntimeError("stopped")

    monkeypatch.setattr(td, "save_epoch_checkpoint", stop_after_saving)
    with pytest.raises(RuntimeError, match="stopped"):
        td.train_detector(cfg, load_splits())
    monkeypatch.setattr(td, "save_epoch_checkpoint", real)
    folder = td.run_folder(cfg, load_splits(), max_iters=None, densest_first=False)[0]
    assert (folder / "last.pt").is_file() and not (folder / "weights.pt").exists()
    # Sentinels the resumed run must start from: a classifier bias and the momentum of the last parameter (the
    # box regression bias); neither is near what a fresh start or two more steps give.
    ckpt = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
    ckpt["model"]["roi_heads.box_predictor.cls_score.bias"] = torch.tensor([4.0, -4.0, -4.0])
    last = max(ckpt["optimizer"]["state"])
    assert ckpt["optimizer"]["state"][last]["momentum_buffer"].shape == (12,)
    ckpt["optimizer"]["state"][last]["momentum_buffer"] = torch.full((12,), 100.0)
    torch.save(ckpt, folder / "last.pt")
    with open(folder / "log.csv", "a", encoding="utf-8") as fh:   # a row written after the checkpoint
        fh.write("99,1,0.0,1.0,0.0,0.0,0.0,0.0,0.0,\n")
    r = CliRunner().invoke(app, ["train", "detector", "--config", train_cfg, "--splits",
                                 str(root / "configs" / "eval" / "splits.yaml"), "--resume", str(folder),
                                 *[x for s in sets for x in ("--set", s)]])
    assert r.exit_code == 0, r.output
    with open(folder / "log.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [int(x["step"]) for x in rows] == [1, 2, 3, 4] and [int(x["epoch"]) for x in rows] == [1, 1, 2, 2]
    assert all(math.isfinite(float(x["loss"])) for x in rows)
    sha = hashlib.sha256((folder / "weights.pt").read_bytes()).hexdigest()
    assert (folder / "weights.sha256").read_text(encoding="utf-8").split()[0] == sha and f"weights {sha}" in r.output
    rec = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    assert rec["weights_sha256"] == sha and rec["data"][0]["seed"] == 100 and rec["pretrained"] is None
    assert rec["timings"]["steps"] == 4 and rec["config_hash"] == td.train_config_hash(cfg)
    assert rec["args"]["resume"] is True and [s["from_step"] for s in rec["segments"]] == [0, 2]
    state = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    assert "roi_heads.box_predictor.cls_score.weight" in state
    assert torch.allclose(state["roi_heads.box_predictor.cls_score.bias"], torch.tensor([4.0, -4.0, -4.0]), atol=0.5)
    momentum = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)["optimizer"]["state"][last]
    assert (momentum["momentum_buffer"] > 50).all()      # 100 * 0.9 ** 2 plus two clipped gradients


@pytest.mark.slow
def test_detect_cli_writes_a_cache_that_detmetrics_reads(tmp_path):
    import hashlib
    import json
    from pathlib import Path

    import torch
    import yaml
    from typer.testing import CliRunner

    from counting.cli import app
    from counting.config.det_schema import DetTrainConfig
    from counting.detection.cached import CachedDetector, cache_path, rgb_fingerprint
    from counting.detection.frcnn import build_model
    from counting.evaluation.cli import eval_app

    root = Path(__file__).resolve().parents[2]
    splits = str(root / "configs" / "eval" / "splits.yaml")
    val = write_run(tmp_path / "val_s200", crop="apple", pitches=(30.0,), seed=200, max_frames=2, images=True)
    torch.save(build_model(SMALL, pretrained=False, mode="infer", seed=0).state_dict(), tmp_path / "w.pt")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(DetTrainConfig(model=SMALL).model_dump(mode="json")),
                                          encoding="utf-8")
    sha = hashlib.sha256((tmp_path / "w.pt").read_bytes()).hexdigest()
    sets = ["device=cpu", "model.infer_min_size=360", "model.infer_max_size=640", "model.max_dets=5",
            "model.rpn_pre_nms_test=200", "model.rpn_post_nms_test=100", f"output_dir={tmp_path}/detect"]
    args = ["detect", "--config", str(root / "configs" / "detect" / "frcnn_synth.yaml"), "--weights",
            str(tmp_path / "w.pt"), "--run", str(val), "--splits", splits, *[x for s in sets for x in ("--set", s)]]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    run = SynthRun.open(val)
    path = cache_path(tmp_path / "detect" / sha[:16], run.meta.run_id)
    det = CachedDetector(path, fingerprint=run.meta.fingerprint)
    assert (det.meta["weights_sha256"], det.meta["score_min"], det.meta["max_dets"]) == (sha, 0.05, 5)
    assert det.meta["rgb_sha256"] == rgb_fingerprint(run.frames) and det.digest[:16] in r.output
    assert det.meta["train_id"] == tmp_path.name and det.meta["pretrained"] is None
    side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    n_cap = len(side["saturated"])                  # untrained: random boxes reach the cap of 5
    assert side["n_frames"] == 2 and n_cap >= 1 and set(side["saturated"]) <= {f.idx for f in run.frames}
    assert (f"warning: {run.meta.run_id}: {n_cap} frame(s) reached max_dets=5; detections past the cap are not in the "
            "cache" in r.output)
    for f in run.frames:
        det.detect(f)
    raw = yaml.safe_load((root / "configs" / "eval" / "apple_val.yaml").read_text(encoding="utf-8"))
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump({**raw, "runs": [str(val)]}), encoding="utf-8")
    r2 = CliRunner().invoke(eval_app, ["detmetrics", "--config", str(tmp_path / "cfg.yaml"), "--cache", str(path.parent),
                                       "--splits", splits, "--out", str(tmp_path / "dm")])
    assert r2.exit_code == 0, r2.output
    r3 = CliRunner().invoke(app, args)
    assert r3.exit_code == 1 and "cache exists" in r3.output


@pytest.mark.slow
def test_training_stops_on_a_non_finite_gradient_before_the_step(tmp_path, monkeypatch):
    from counting.config.det_schema import DetTrainConfig
    from counting.detection import frcnn
    from counting.evaluation.runner import CheckFailed
    from counting.training.detector import train_detector
    from tests.unit.synthfake import load_splits

    cfg = load_yaml_model(TRAIN_CFG, DetTrainConfig, overrides=_tiny_sets(tmp_path))
    for clip in (10.0, None):                       # clipping on, and off
        model = _tiny_model(nan_grad=True)
        monkeypatch.setattr(frcnn, "build_model", lambda *args, **kwargs: model)
        run = cfg.model_copy(update={"optim": cfg.optim.model_copy(update={"clip_grad_norm": clip}),
                                     "output_dir": str(tmp_path / f"train_{clip}")})
        with pytest.raises(CheckFailed, match=r"^non-finite gradient at step 1$"):
            train_detector(run, load_splits())
        assert float(model.w) == 0.0                # the optimizer step was not taken


@pytest.mark.slow
def test_a_resumed_training_takes_no_step_past_max_iters(tmp_path, monkeypatch):
    import torch

    from counting.config.det_schema import DetTrainConfig
    from counting.detection import frcnn
    from counting.training.detector import train_detector
    from tests.unit.synthfake import load_splits

    cfg = load_yaml_model(TRAIN_CFG, DetTrainConfig, overrides=_tiny_sets(tmp_path))
    monkeypatch.setattr(frcnn, "build_model", lambda *args, **kwargs: _tiny_model())
    folder = _first_epoch_then_stop(cfg, monkeypatch, max_iters=5)       # last.pt after step 2
    ckpt = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
    torch.save({**ckpt, "step": 5}, folder / "last.pt")                  # as if all 5 steps were done
    result = train_detector(cfg, load_splits(), max_iters=5, resume=folder)
    assert result.steps == 5 and (folder / "weights.pt").is_file()


@pytest.mark.slow
def test_resume_refuses_an_unreadable_checkpoint(tmp_path, monkeypatch):
    import io

    import torch
    from typer.testing import CliRunner

    from counting.cli import app
    from counting.config.det_schema import DetTrainConfig
    from counting.detection import frcnn

    sets = _tiny_sets(tmp_path)
    cfg = load_yaml_model(TRAIN_CFG, DetTrainConfig, overrides=sets)
    monkeypatch.setattr(frcnn, "build_model", lambda *args, **kwargs: _tiny_model())
    folder = _first_epoch_then_stop(cfg, monkeypatch)
    ckpt = folder / "last.pt"
    good = ckpt.read_bytes()
    listed = io.BytesIO()
    torch.save([1, 2], listed)
    for content in (b"not a checkpoint", good[:len(good) // 2], listed.getvalue()):
        ckpt.write_bytes(content)
        r = CliRunner().invoke(app, ["train", "detector", "--config", TRAIN_CFG, "--splits", SPLITS, "--resume",
                                     str(folder), *[x for s in sets for x in ("--set", s)]])
        assert r.exit_code == 1 and f"error: {ckpt}: not a readable training checkpoint" in r.output, r.output
    assert not (folder / "weights.pt").exists()


@pytest.mark.slow
def test_load_weights_refuses_files_that_are_not_state_dicts(tmp_path):
    import io
    import re

    import torch

    from counting.detection.frcnn import load_weights

    model = torch.nn.Linear(2, 2)
    whole = io.BytesIO()
    torch.save(model.state_dict(), whole)
    listed = io.BytesIO()
    torch.save([1, 2], listed)
    for name, content, reason in (("text.pt", b"weights", "not a readable torch weights file"),
                                  ("cut.pt", whole.getvalue()[:-60], "not a readable torch weights file"),
                                  ("list.pt", listed.getvalue(), "not a model state dict")):
        (tmp_path / name).write_bytes(content)
        with pytest.raises(ValueError, match=f"^{re.escape(str(tmp_path / name))}: {reason}"):
            load_weights(model, tmp_path / name)


@pytest.mark.slow
def test_freeze_batch_norm_keeps_the_learned_values():
    import torch
    from torchvision.ops.misc import FrozenBatchNorm2d

    from counting.detection.frcnn import _freeze_batch_norm

    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.Sequential(torch.nn.BatchNorm2d(4, eps=1e-3),
                                                                          torch.nn.ReLU()))
    bn = net[1][0]
    with torch.no_grad():                           # values a pretrained network would carry, not the defaults
        bn.weight.uniform_(0.5, 2.0)
        bn.bias.normal_()
        bn.running_mean.normal_()
        bn.running_var.uniform_(0.5, 2.0)
    learned = {k: getattr(bn, k).detach().clone() for k in ("weight", "bias", "running_mean", "running_var")}
    x = torch.randn(2, 3, 8, 8)
    before = net.eval()(x)
    _freeze_batch_norm(net)
    frozen = net[1][0]
    assert isinstance(frozen, FrozenBatchNorm2d) and frozen.eps == 1e-3
    for name, value in learned.items():
        assert torch.equal(getattr(frozen, name), value), name
    torch.testing.assert_close(net(x), before)
