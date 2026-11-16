"""The val base config of the cross-side join rule (apple_val_join.yaml): the depth ladder's val base config with
four values changed (run name, output folder, configuration label c2 and dedup.band_join), the committed configs
without the rule keep their config hashes, and on the pose ladder the rule reaches the rungs on oracle detections
while the learned-detector rungs keep the band rule."""

import copy

import pytest
import yaml

from counting.config.eval_schema import EvalAppConfig
from counting.config.hashing import config_hash
from counting.config.loader import load_yaml_model
from counting.evaluation.runner import load_ladder, rung_config
from tests.unit.synthfake import ROOT

pytestmark = pytest.mark.pose

EVAL = ROOT / "configs" / "eval"
JOIN = EVAL / "apple_val_join.yaml"
DEPTH = EVAL / "apple_val_depth.yaml"
LADDER = load_ladder(EVAL / "sweeps_pose" / "ladder_pose.yaml")
# Config hashes of the committed configs that test_eval_schema.py does not pin, as they were before the join rule keys.
KEYLESS_HASHES = {
    "apple_val_depth": "e0927207de5c99c4",
    "pear_val_depth": "78070a71d3cb5c2a",
    "frozen/apple_test_depth": "e1a8c75e51687a1a",
    "frozen/pear_test_depth": "c3f0d842eeba7484",
    "apple_dev_pose_c1": "862708b06e58ccd1",
    "apple_dev_pose_c2": "e5f1d600adb6ec4b",
    "pear_dev_pose_c1": "79dcf89582d046c3",
    "pear_dev_pose_c2": "95360828f0162369",
}
ORACLE_RUNGS = ("L3", "L4_drift", "LVI")
CACHED_RUNGS = ("L5", "L5VI")


def test_val_join_config_is_the_depth_config_with_the_join_rule():
    depth = yaml.safe_load(DEPTH.read_text(encoding="utf-8"))
    join = yaml.safe_load(JOIN.read_text(encoding="utf-8"))
    expected = copy.deepcopy(depth)
    expected["run_name"] = "apple_val_join"
    expected["output_dir"] = "runs/eval_join_val"
    expected["pose"]["config_label"] = "c2"
    expected["dedup"]["band_join"] = "either_mutual"
    assert join == expected
    assert join["output_dir"] not in ("runs/eval_pose", "runs/eval_depth_val", "runs/eval_join")


def test_keyless_configs_keep_their_hashes():
    for name, h in KEYLESS_HASHES.items():
        assert config_hash(load_yaml_model(EVAL / f"{name}.yaml", EvalAppConfig)) == h, name


def test_val_join_config_takes_the_rule_on_oracle_detections_only():
    cfg = load_yaml_model(JOIN, EvalAppConfig)
    assert config_hash(cfg) != KEYLESS_HASHES["apple_val_depth"]
    assert cfg.dedup.band_join == "either_mutual" and cfg.dedup.band_join_cached is None
    for name in (*ORACLE_RUNGS, *CACHED_RUNGS):
        rung = rung_config(cfg, LADDER, name)
        cached = name in CACHED_RUNGS
        assert (rung.stages.det == "cached") is cached, name
        assert rung.dedup.band_join_for(rung.stages.det) == (None if cached else "either_mutual"), name
        assert rung.dedup.band_join == "either_mutual", name
