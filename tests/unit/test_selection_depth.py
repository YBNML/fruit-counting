"""The rules that choose the LiDAR depth values on val: their items, candidates, vetoes, stops and the threshold
numbers written in their text."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]


def test_depth_selection_rules_hold_their_values():
    rules = yaml.safe_load((ROOT / "configs" / "eval" / "selection_depth.yaml").read_text(encoding="utf-8"))
    assert set(rules) == {"order", "fixed", "items", "vetoes", "stops", "disclosed", "detail", "tuning"}
    assert rules["order"] == ["match_window", "lidar_far", "combine"]
    items = rules["items"]
    assert set(items) == set(rules["order"])
    fixed = rules["fixed"]
    assert fixed["start"] == {"match": "gate", "window_s": 0.5, "combine": "ivw", "min_points": 3, "gate_k": 3}
    assert fixed["belief"] == {"belief_cam_rel": 0.02, "belief_range_m": 0.02}
    assert fixed["lidar_far_scale"] == 1.0
    match_window = items["match_window"]
    assert match_window["candidates"] == {"match": ["far", "gate", "map"], "window_s": [0.15, 0.5, 1.0],
                                          "map_window_s": [1.0, "pass"]}
    assert match_window["steps"][1]["within"] == 0.005
    combine = items["combine"]
    assert combine["candidates"] == {"combine": ["replace", "ivw"], "min_points": [1, 3, 5], "gate_k": [2, 3, 5]}
    assert combine["steps"][1] == {"within": 0.002, "tie": "simplest"}
    assert items["lidar_far"]["candidates"] == {"far_min_frac": [0.1, 0.25], "far_gap_m": [0.03, 0.06]}
    vetoes, stops = rules["vetoes"], rules["stops"]
    assert set(vetoes) == {"drift", "second_scene", "pear", "learned_detector", "none_left"}
    assert set(stops) == {"when", "depth_tail", "gain", "second_scene", "no_candidate"}
    assert rules["tuning"] is True
    # the threshold numbers live in the rule text, so the text fragments that carry them are held here
    assert "more than 0.01 above the best candidate" in vetoes["drift"]
    assert "on val seed 201 is above that of L4_drift or LVI" in vetoes["second_scene"]
    assert "more than 0.005 above that of its camera-only rung" in vetoes["pear"]
    assert "more than 0.005 above that of L5VI" in vetoes["learned_detector"]
    assert "or above 0.12" in vetoes["learned_detector"]
    for fragment in ("before combining", "more than 0.10 m off", "more than twice", "more than 0.10 below the coverage"):
        assert fragment in stops["depth_tail"]
    assert "not more than 0.01 below" in stops["gain"]
    for fragment in ("above 0.08", "its own line of 0.12", "more than 0.01 below it"):
        assert fragment in stops["second_scene"]
    # only values known when the file was committed are disclosed
    assert "known when this file was committed" in rules["disclosed"]
    assert "simulation check" not in rules["disclosed"]
