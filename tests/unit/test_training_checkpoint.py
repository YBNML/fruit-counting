import torch

from counting.training.checkpoint import load_checkpoint, save_checkpoint


def test_save_and_load_roundtrip(tmp_path):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        path=tmp_path / "ckpt.pt",
        model=model,
        optimizer=opt,
        epoch=7,
        best_metric=0.123,
        config_snapshot={"run_name": "test"},
    )

    model2 = torch.nn.Linear(4, 2)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    meta = load_checkpoint(tmp_path / "ckpt.pt", model=model2, optimizer=opt2)

    assert meta["epoch"] == 7
    assert meta["best_metric"] == 0.123
    assert meta["config_snapshot"] == {"run_name": "test"}

    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_load_without_optimizer_is_ok(tmp_path):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    save_checkpoint(
        path=tmp_path / "ckpt.pt",
        model=model,
        optimizer=opt,
        epoch=1,
        best_metric=0.0,
        config_snapshot={},
    )

    model2 = torch.nn.Linear(4, 2)
    meta = load_checkpoint(tmp_path / "ckpt.pt", model=model2)
    assert meta["epoch"] == 1
