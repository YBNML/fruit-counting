"""PseCo ROIHeadMLP fine-tuning: orchestrates cache reader, proposer, and trainer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from counting.config.train_schema import TrainAppConfig
from counting.data.cache import FeatureCacheReader
from counting.data.formats.fsc147 import FSC147Dataset
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
    """Yields (embedding_tensor, proposals, gt_points) tuples."""

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
        return {
            "image_id": rec.relpath,
            "embedding": torch.from_numpy(emb).to(torch.float32),
            "proposals": proposals,
            "gt_count": rec.count,
        }


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_ids": [b["image_id"] for b in batch],
        "embeddings": torch.stack([b["embedding"] for b in batch]),
        "proposals": [b["proposals"] for b in batch],
        "gt_counts": torch.tensor([b["gt_count"] for b in batch], dtype=torch.float32),
    }


def train_pseco_head(cfg: TrainAppConfig) -> None:
    _ensure_external_on_path()
    from models import ROIHeadMLP

    device = resolve_device(cfg.device)

    fsc_train = FSC147Dataset(cfg.data.root, split=cfg.data.train_split)
    fsc_val = FSC147Dataset(cfg.data.root, split=cfg.data.val_split)

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
        if "model" in state:
            state = state["model"]
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

    def step_fn(batch, model):
        embs = batch["embeddings"].to(device)
        gt_counts = batch["gt_counts"].to(device)
        B = embs.size(0)

        # Build per-proposal ROI features and logits via the ROIHead.
        # Placeholder simplification (documented in plan): top-16 proposals,
        # unit-vector prompt embeddings. Real CLIP text wiring lands in Plan 3.
        bboxes_per_image = _topk_points_as_boxes(batch["proposals"], k=16, image_size=1024)
        prompts = _unit_prompts(batch_size=B, num_proposals=16, device=device)

        # ROIHeadMLP.forward returns (B, num_proposals) dot-product scores —
        # one score per (image, proposal) pair when each image has 1 text class.
        # We treat positive/background as a binary task by thresholding at 0.
        raw_scores = model(embs, bboxes_per_image, prompts)  # (B, 16)
        raw_scores = raw_scores.reshape(B * 16)

        # Cast to 2-class logits: [neg_score, pos_score] = [-s, s]
        logits = torch.stack([-raw_scores, raw_scores], dim=1)  # (B*16, 2)
        targets = torch.ones(B * 16, dtype=torch.long, device=device)

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
        bboxes = _topk_points_as_boxes(batch["proposals"], k=16, image_size=1024)
        prompts = _unit_prompts(batch_size=B, num_proposals=16, device=device)
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
        boxes_per_image.append(boxes)
    return boxes_per_image


def _unit_prompts(*, batch_size: int, num_proposals: int, device) -> list[torch.Tensor]:
    """Placeholder text prompt embeddings: one unit-vector text class per image.

    ROIHeadMLP expects a list of B tensors each of shape (N_text, C, 512).
    We use N_text=1 and C=num_proposals to match the ROI embedding shape.
    The dot product then yields (B, num_proposals) — one score per proposal.

    Note: Real CLIP text embeddings are wired in Plan 3.
    """
    # Shape: (1, num_proposals, 512) repeated B times.
    # N_text=1 means one text query class; C=num_proposals matches ROI count.
    prompt = torch.ones(1, num_proposals, 512, device=device) / (512 ** 0.5)
    return [prompt] * batch_size
