import torch

from counting.models.pseco.losses import pseco_head_loss


def test_loss_is_scalar_and_finite():
    logits = torch.randn(4, 2)
    targets = torch.tensor([0, 1, 0, 1])
    pred_counts = torch.tensor([2.0, 3.0])
    gt_counts = torch.tensor([2, 3])
    out = pseco_head_loss(
        logits=logits,
        targets=targets,
        pred_counts=pred_counts,
        gt_counts=gt_counts,
        cls_weight=1.0,
        count_weight=0.1,
    )
    assert out["total"].ndim == 0
    assert torch.isfinite(out["total"])
    assert "cls" in out and "count" in out


def test_loss_weights_affect_total():
    torch.manual_seed(0)
    logits = torch.randn(8, 2)
    targets = torch.randint(0, 2, (8,))
    pred_counts = torch.tensor([1.0, 2.0])
    gt_counts = torch.tensor([2, 3])

    small = pseco_head_loss(
        logits=logits, targets=targets,
        pred_counts=pred_counts, gt_counts=gt_counts,
        cls_weight=1.0, count_weight=0.0,
    )["total"]
    big = pseco_head_loss(
        logits=logits, targets=targets,
        pred_counts=pred_counts, gt_counts=gt_counts,
        cls_weight=1.0, count_weight=10.0,
    )["total"]
    assert big.item() > small.item()


def test_count_loss_zero_when_perfect():
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    targets = torch.tensor([0, 1])
    pred_counts = torch.tensor([2.0, 3.0])
    gt_counts = torch.tensor([2, 3])
    out = pseco_head_loss(
        logits=logits, targets=targets,
        pred_counts=pred_counts, gt_counts=gt_counts,
        cls_weight=1.0, count_weight=1.0,
    )
    assert out["count"].item() == 0.0
    assert out["cls"].item() < 0.01
