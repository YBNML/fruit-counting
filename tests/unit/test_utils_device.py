import pytest

from counting.utils.device import resolve_device


def test_resolve_device_cpu_explicit():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_picks_something():
    d = resolve_device("auto")
    assert d in {"cpu", "mps", "cuda"}


def test_resolve_device_cuda_without_cuda_raises(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested"):
        resolve_device("cuda")


def test_resolve_device_invalid_raises():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")
