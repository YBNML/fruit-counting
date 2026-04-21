"""PseCo ROIHeadMLP fine-tuning: orchestrates cache reader, proposer, and trainer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from counting.config.train_schema import TrainAppConfig
from counting.data.cache import FeatureCacheReader
from counting.data.formats.fsc147 import FSC147Dataset
from counting.models.pseco.clip_features import load_text_features
from counting.models.pseco.labeling import assign_targets_from_points
from counting.models.pseco.losses import pseco_head_loss
from counting.models.pseco.proposals import PointDecoderProposer
from counting.training.callbacks import CosineWithWarmup, EarlyStopping
from counting.training.runner import Runner
from counting.utils.device import resolve_device


def _ensure_external_on_path() -> None:
    root = Path(__file__).resolve().parents[4]
    ext = root / "external" / "PseCo"
    if str(ext) not in sys.path:
        sys.path.insert(0, str(ext))


class _CachedFSC147(Dataset):
    """Yields cached embeddings + proposals + class_name + GT points."""

    def __init__(
        self,
        fsc: FSC147Dataset,
        reader: FeatureCacheReader,
        proposer: PointDecoderProposer,
    ):
        self._recs = list(fsc)
        self._reader = reader
        self._proposer = proposer

    def __len__(self) -> int:
        return len(self._recs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self._recs[idx]
        emb = self._reader.read(rec.relpath)
        proposals = self._proposer.propose(emb)
        # GT points come from FSC-147 annotations in the ORIGINAL image
        # coordinate system (longest side = 384 in this dataset). PointDecoder
        # predictions and our proposal boxes live in SAM's preprocessed space
        # (longest side resized to 1024, then padded to 1024x1024). Scale GT
        # points to that 1024 space so containment checks match.
        from PIL import Image
        with Image.open(rec.path) as im:
            w, h = im.size
        scale = 1024.0 / float(max(w, h))
        scaled_points = [(x * scale, y * scale) for (x, y) in rec.points]
        return {
            "image_id": rec.relpath,
            "class_name": rec.class_name,
            "points": scaled_points,
            "embedding": torch.from_numpy(emb).to(torch.float32),
            "proposals": proposals,
            "gt_count": rec.count,
        }


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_ids": [b["image_id"] for b in batch],
        "class_names": [b["class_name"] for b in batch],
        "points_per_image": [b["points"] for b in batch],
        "embeddings": torch.stack([b["embedding"] for b in batch]),
        "proposals": [b["proposals"] for b in batch],
        "gt_counts": torch.tensor([b["gt_count"] for b in batch], dtype=torch.float32),
    }


def _require_classes_in_cache(
    records: list[Any], clip_cache: dict[str, torch.Tensor], split: str
) -> None:
    """Fail fast if any record's class_name is missing from the CLIP cache."""
    unknown: set[str] = set()
    for rec in records:
        if rec.class_name and rec.class_name not in clip_cache:
            unknown.add(rec.class_name)
    if unknown:
        missing = ", ".join(sorted(unknown)[:5])
        raise KeyError(
            f"{split}: {len(unknown)} class(es) missing from CLIP feature cache: "
            f"{missing}{' ...' if len(unknown) > 5 else ''}. "
            "Re-run `counting extract-clip-features`."
        )


def train_pseco_head(cfg: TrainAppConfig) -> None:
    _ensure_external_on_path()
    from models import ROIHeadMLP

    device = resolve_device(cfg.device)

    fsc_train = FSC147Dataset(cfg.data.root, split=cfg.data.train_split)
    fsc_val = FSC147Dataset(cfg.data.root, split=cfg.data.val_split)

    # --- CLIP text features: load and move to device once ---
    clip_cache_cpu = load_text_features(cfg.model.clip_features_cache)
    clip_cache = {name: tensor.to(device) for name, tensor in clip_cache_cpu.items()}

    _require_classes_in_cache(list(fsc_train), clip_cache, cfg.data.train_split)
    _require_classes_in_cache(list(fsc_val), clip_cache, cfg.data.val_split)

    reader = FeatureCacheReader(cfg.cache.dir)

    proposer = PointDecoderProposer(
        sam_checkpoint=cfg.model.sam_checkpoint,
        point_decoder_checkpoint=cfg.model.init_decoder,
        device=device,
    )
    proposer.prepare()

    train_ds = _CachedFSC147(fsc_train, reader, proposer)
    val_ds = _CachedFSC147(fsc_val, reader, proposer)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=0, collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=0, collate_fn=_collate_batch,
    )

    head = ROIHeadMLP().to(device)
    if cfg.model.init_mlp:
        state = torch.load(cfg.model.init_mlp, map_location="cpu", weights_only=False)
        # PseCo MLP checkpoints store the state_dict under "cls_head"; other
        # generic conventions use "model" or "state_dict". Walk whichever
        # wrapper key is present before passing to load_state_dict.
        if isinstance(state, dict):
            for wrapper_key in ("cls_head", "model", "state_dict"):
                inner = state.get(wrapper_key)
                if isinstance(inner, dict) and any(
                    isinstance(k, str) and "." in k for k in inner.keys()
                ):
                    state = inner
                    break
        head.load_state_dict(state, strict=False)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    total_steps = cfg.train.epochs * max(1, len(train_loader))
    scheduler = CosineWithWarmup(
        base_lr=cfg.train.lr,
        warmup_steps=cfg.train.warmup_steps,
        total_steps=total_steps,
        min_lr=max(1e-6, cfg.train.lr * 0.01),
    )
    es = EarlyStopping(
        patience=cfg.train.early_stopping.patience,
        mode=cfg.train.early_stopping.mode,
    )

    run_dir = Path(cfg.output_dir) / cfg.run_name
    runner = Runner(
        model=head,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=es,
        device=device,
        run_dir=run_dir,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        save_every_n_epochs=cfg.logging.save_every_n_epochs,
        config_snapshot=cfg.model_dump(mode="json"),
        tensorboard=cfg.logging.tensorboard,
    )

    def _build_prompts(class_names: list[str]) -> list[torch.Tensor]:
        """Return one (1, 1, 512) prompt tensor per image based on its class."""
        return [clip_cache[cn].view(1, 1, 512) for cn in class_names]

    def step_fn(batch, model):
        embs = batch["embeddings"].to(device)
        gt_counts = batch["gt_counts"].to(device)
        B = embs.size(0)

        bboxes_per_image = _topk_points_as_boxes(
            batch["proposals"], k=16, image_size=1024, device=device,
        )
        prompts = _build_prompts(batch["class_names"])

        raw_scores = model(embs, bboxes_per_image, prompts)  # (B, 16)
        raw_scores = raw_scores.reshape(B * 16)

        # Convert single-class scores into binary logits [-s, s]
        logits = torch.stack([-raw_scores, raw_scores], dim=1)  # (B*16, 2)

        # Real pos/neg targets from GT points
        targets_cpu = assign_targets_from_points(
            bboxes_per_image, batch["points_per_image"]
        )
        targets = targets_cpu.to(device)

        pred_counts = (raw_scores.detach().reshape(B, 16) > 0).sum(dim=1).float()

        return pseco_head_loss(
            logits=logits,
            targets=targets,
            pred_counts=pred_counts,
            gt_counts=gt_counts,
            cls_weight=cfg.train.loss_weights.get("cls", 1.0),
            count_weight=cfg.train.loss_weights.get("count", 0.1),
        )

    def eval_fn(batch, model) -> dict[str, float]:
        embs = batch["embeddings"].to(device)
        gt = batch["gt_counts"]
        B = embs.size(0)
        bboxes = _topk_points_as_boxes(
            batch["proposals"], k=16, image_size=1024, device=device,
        )
        prompts = _build_prompts(batch["class_names"])
        raw_scores = model(embs, bboxes, prompts).reshape(B, 16)  # (B, 16)
        pred_counts = (raw_scores > 0).sum(dim=1).float().cpu()
        mae = F.l1_loss(pred_counts, gt).item()
        return {"mae": mae}

    for epoch in range(cfg.train.epochs):
        runner.state.epoch = epoch
        runner.train_epoch(train_loader, step_fn)
        val_metrics = runner.validate(val_loader, eval_fn)
        runner.maybe_save(val_metrics, metric_key="mae")
        if runner.early_stopping.update(val_metrics.get("mae", float("inf"))):
            break

    runner.close()
    proposer.cleanup()


def _topk_points_as_boxes(
    proposals: list[dict[str, torch.Tensor]],
    *,
    k: int,
    image_size: int,
    half_side: float = 16.0,
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    """Return per-image (1, k, 4) boxes centered on the top-k predicted points."""
    boxes_per_image: list[torch.Tensor] = []
    for p in proposals:
        pts = p["pred_points"].squeeze(0)[:k]
        if pts.size(0) < k:
            pad = torch.zeros((k - pts.size(0), 2))
            pts = torch.cat([pts, pad], dim=0)
        x = pts[:, 0].clamp(0, image_size)
        y = pts[:, 1].clamp(0, image_size)
        boxes = torch.stack([
            (x - half_side).clamp(0),
            (y - half_side).clamp(0),
            (x + half_side).clamp(max=image_size),
            (y + half_side).clamp(max=image_size),
        ], dim=1).unsqueeze(0)
        boxes_per_image.append(boxes.to(device))
    return boxes_per_image
