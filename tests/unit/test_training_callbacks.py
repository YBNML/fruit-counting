import math

from counting.training.callbacks import (
    CosineWithWarmup,
    EarlyStopping,
)


def test_cosine_warmup_monotonic_then_decay():
    sched = CosineWithWarmup(base_lr=1e-3, warmup_steps=5, total_steps=25, min_lr=1e-5)
    lrs = [sched.lr_at(s) for s in range(26)]
    assert lrs[0] < lrs[4] < lrs[5]
    assert math.isclose(lrs[5], 1e-3, rel_tol=1e-6)
    for i in range(5, 25):
        assert lrs[i] >= lrs[i + 1] - 1e-12
    assert math.isclose(lrs[25], 1e-5, rel_tol=1e-6, abs_tol=1e-9)


def test_early_stopping_min_mode_triggers_after_patience():
    es = EarlyStopping(patience=2, mode="min")
    assert es.update(1.0) is False
    assert es.update(0.9) is False
    assert es.update(0.95) is False
    assert es.update(0.95) is True


def test_early_stopping_max_mode():
    es = EarlyStopping(patience=1, mode="max")
    assert es.update(0.5) is False
    assert es.update(0.4) is True


def test_early_stopping_invalid_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        EarlyStopping(patience=1, mode="sideways")
