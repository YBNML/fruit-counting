"""The seed splits of the pose ladder."""

from pathlib import Path

import pytest

from counting.config.eval_schema import SplitsConfig
from counting.config.loader import load_yaml_model

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]


def test_pose_splits_hold_new_held_out_seeds():
    splits = load_yaml_model(ROOT / "configs" / "eval" / "splits_pose.yaml", SplitsConfig)   # overlaps are refused
    assert splits.dev == [5] and splits.val == [200] and splits.exp == [300]
    assert splits.test == [1003, 1004, 1005]
    assert splits.train == list(range(100, 120))
    assert all(splits.split_of(seed) is None for seed in (1000, 1001, 1002))
