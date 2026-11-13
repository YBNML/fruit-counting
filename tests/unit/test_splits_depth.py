"""The seed splits of the depth evaluation."""

from pathlib import Path

import pytest

from counting.config.eval_schema import SplitsConfig
from counting.config.loader import load_yaml_model

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]


def test_depth_splits_hold_new_held_out_seeds():
    splits = load_yaml_model(ROOT / "configs" / "eval" / "splits_depth.yaml", SplitsConfig)   # overlaps are refused
    assert splits.dev == [5] and splits.val == [200, 201] and splits.exp == [300]
    assert splits.test == [1006, 1007, 1008]
    assert splits.train == list(range(100, 120))
    assert all(splits.split_of(seed) is None for seed in range(1000, 1006))
