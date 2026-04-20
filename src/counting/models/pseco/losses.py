"""PseCo ROIHeadMLP training loss.

The upstream loss is a 2-class cross-entropy between ROIHead logits and the
pseudo labels from PointDecoder proposals (positive = matches CLIP text
prompt, negative = background/other). We add a small count-L1 auxiliary so
the model is penalized for disagreeing with the annotated image count.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pseco_head_loss(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    pred_counts: torch.Tensor,
    gt_counts: torch.Tensor,
    cls_weight: float = 1.0,
    count_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    cls_loss = F.cross_entropy(logits, targets)
    count_loss = F.l1_loss(pred_counts.float(), gt_counts.float())
    total = cls_weight * cls_loss + count_weight * count_loss
    return {"total": total, "cls": cls_loss.detach(), "count": count_loss.detach()}
