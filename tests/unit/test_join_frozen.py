"""The frozen held-out configs of the cross-side join rule: apple_test_join.yaml (configuration c2, judged) is the
frozen pose config apple_test_pose.yaml with dedup.band_join, apple_test_join_ref.yaml (c2_ref, report rows) the same
config without it, and pear_test_join.yaml (c2, judged) the frozen pear pose config. All three run on the held-out
scenes of the depth splits, write into runs/eval_join and never into runs/eval_pose, and pin the content digests of
their detection caches, pose caches and judged IMU files. Their protect sections pin the pose ladder, the depth
splits, the pose criteria and, for the two judged configs, the camera depth grid; their requests pass the held-out
checks of the rung and grid cell configs, and the join rule reaches the apple rungs on oracle detections only.

Every file is read inside the tests. Nothing here opens a bundle, a cache or an IMU file."""

import hashlib
import re
from pathlib import Path

import pytest
import yaml

from counting.config.eval_schema import EvalAppConfig, SplitsConfig
from counting.config.loader import load_yaml_model
from counting.evaluation.protect import (
    check_test_frozen_request, check_test_lidar_rungs, check_test_lift, check_test_pose_rungs, check_test_rung_protect,
    check_test_rungs, judges_lidar,
)
from counting.evaluation.runner import ProtectionError, check_ladder_request, load_ladder, rung_config
from counting.evaluation.sweep import load_grid, sweep_cells

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "configs" / "eval"
# Repository paths of the files a frozen config of the join rule pins, and its output folder.
LADDER = "configs/eval/sweeps_pose/ladder_pose.yaml"
SPLITS = "configs/eval/splits_depth.yaml"
CRITERIA = "configs/eval/criteria_pose.yaml"
GRID = "configs/eval/sweeps_depth/depth_cam.yaml"
OUTPUT = "runs/eval_join"
# Each frozen config: its crop, the frozen pose config it is built on, its configuration label, its join rule and
# whether it pins the camera depth grid.
CONFIGS = {
    "apple_test_join": ("apple", "apple_test_pose", "c2", "either_mutual", True),
    "apple_test_join_ref": ("apple", "apple_test_pose", "c2_ref", None, False),
    "pear_test_join": ("pear", "pear_test_pose", "c2", None, True),
}
# The one rung list a held-out ladder request may name: every rung and baseline of the pose ladder off its val list.
HELD = ["L3", "L4_drift", "LVI", "L5", "L5VI", "L3_al", "L4_drift_noimu", "L4_drift_noal", "L4_drift_fitp",
        "L4_drift_c1x", "L4_drift_dvo", "L3_pass", "L3_pass_al", "L3_pass_c1x", "LV", "L5V", "LVI_first", "LVI_fitp",
        "LVI_fits", "LVI_dd", "LVI_rowbias5", "LVI_rowbias10", "LVI_thin50", "LVI_thin80", "LVI_oracle_struct",
        "L4_base", "L5_L4base", "B_drift", "B_drift_fitp", "B_pass", "B_pass_c1x"]
# The rungs of that list on cached detections: they keep the band rule whatever dedup.band_join holds.
CACHED = ("L5", "L5VI", "L5V", "L5_L4base")
# The cells of the camera depth grid on its base B_drift.
GRID_CELLS = 11
# The content digests the frozen configs pin, per crop and bundle run id: detection caches (det), pose caches (vo,
# vo_dd) and judged IMU files (imu_ext).
DIGESTS = {
    "apple": {"det": {}, "vo": {}, "vo_dd": {}, "imu_ext": {}},
    "pear": {"det": {}, "vo": {}, "vo_dd": {}, "imu_ext": {}},
}


def _sha256(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def _load(name: str) -> EvalAppConfig:
    return load_yaml_model(EVAL / "frozen" / f"{name}.yaml", EvalAppConfig)


def _pins(dump: dict) -> dict:
    """The digests the config dump `dump` pins, taken out of it: detection caches, pose caches vo and vo_dd and IMU
    files imu_ext."""
    return {"det": dump["detector"].pop("digests", None) or {},
            "vo": dump["pose"]["caches"]["vo"].pop("digests", None) or {},
            "vo_dd": dump["pose"]["caches"]["vo_dd"].pop("digests", None) or {},
            "imu_ext": dump["pose"]["imu_files"]["imu_ext"].pop("digests", None) or {}}


def test_frozen_join_configs_follow_the_frozen_pose_configs():
    seeds = load_yaml_model(ROOT / SPLITS, SplitsConfig).test
    for name, (crop, base, label, rule, _) in CONFIGS.items():
        dump = _load(name).model_dump()
        expected = _load(base).model_dump()
        _pins(expected)
        runs = [f"~/datasets/orchard_synth/{crop}/test_s{seed}_shake" for seed in seeds]
        expected.update(run_name=name, runs=runs, output_dir=OUTPUT, protect=dump["protect"])
        expected["pose"]["config_label"] = label
        if rule is not None:
            expected["dedup"]["band_join"] = rule
        pins = _pins(dump)
        assert dump == expected, name
        assert pins == DIGESTS[crop], name
        for pinned in pins.values():
            assert sorted(int(k.split("_")[1][1:]) for k in pinned) in ([], sorted(seeds)), name
            assert all(re.fullmatch(rf"{crop}_s[0-9]+_[0-9a-f]{{8}}", k) for k in pinned), name
            assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in pinned.values()), name
        assert dump["output_dir"] == dump["protect"]["output_dir"] == OUTPUT != "runs/eval_pose"


def test_frozen_join_protect_pins_the_committed_files():
    ladder = load_ladder(ROOT / LADDER)
    assert HELD == [n for n in [*ladder["rungs"], *ladder["baselines"]] if n not in ladder["val_only"]]
    criteria = yaml.safe_load((ROOT / CRITERIA).read_text(encoding="utf-8"))
    assert criteria["judged_config"] == "c2" and not judges_lidar(criteria)
    for name, (_, _, label, _, grid) in CONFIGS.items():
        cfg = _load(name)
        section = cfg.protect
        assert (section.ladder.path, section.splits.path, section.criteria.path) == (LADDER, SPLITS, CRITERIA)
        for pinned in (section.ladder, section.splits, section.criteria):
            assert pinned.sha256 == _sha256(pinned.path), (name, pinned.path)
        assert section.grids == ({GRID: _sha256(GRID)} if grid else {}), name
        assert section.judged == ["L4_drift", "LVI", "L5VI"] and section.ladders == [HELD], name
        assert section.output_dir == cfg.output_dir == OUTPUT, name
        assert cfg.pose.config_label == label, name


def _rung_checks(cfg: EvalAppConfig, ladder: dict, configs: dict, criteria: dict) -> None:
    """The held-out checks of the rung or grid cell configs `configs` (name -> (its ladder rung, config)), in the
    order the eval commands run them."""
    plain = {name: c for name, (_, c) in configs.items()}
    check_test_rung_protect(cfg, plain)
    check_test_rungs(cfg, plain)
    check_test_pose_rungs(cfg, configs, ladder)
    check_test_lift(cfg, plain)
    check_test_lidar_rungs(cfg, plain, lidar_judged=judges_lidar(criteria))


def test_frozen_join_requests_pass_the_held_out_checks():
    ladder = load_ladder(ROOT / LADDER)
    criteria = yaml.safe_load((ROOT / CRITERIA).read_text(encoding="utf-8"))
    files = {"--ladder": Path(LADDER), "--splits": Path(SPLITS), "--criteria": Path(CRITERIA)}
    pins = {"ladder_sha256": _sha256(LADDER), "criteria_sha256": _sha256(CRITERIA), "splits_sha256": _sha256(SPLITS),
            "with_2x2": False, "subsets": False, "criteria": criteria}
    sweep = {"kind": "sweep", "files": {**files, "--grid": Path(GRID)}, "grid_sha256": _sha256(GRID), "rungs": None}
    for name, (_, _, _, _, grid) in CONFIGS.items():
        cfg = _load(name)
        check_test_frozen_request(cfg, kind="ladder", files=files, grid_sha256=None, rungs=HELD, **pins)
        check_ladder_request(cfg, ladder, HELD)
        _rung_checks(cfg, ladder, {n: (n, rung_config(cfg, ladder, n)) for n in HELD}, criteria)
        if grid:
            check_test_frozen_request(cfg, **sweep, **pins)
            cells = sweep_cells(cfg, load_grid(ROOT / GRID), ladder)
            _rung_checks(cfg, ladder, {f"{c.point.name} @ {c.base}": (c.base, c.cfg) for c in cells}, criteria)
            assert len(cells) == GRID_CELLS, name
        else:
            refusal = f"split=test: --grid {GRID} is not one of protect.grids"
            with pytest.raises(ProtectionError, match=re.escape(refusal)):
                check_test_frozen_request(cfg, **sweep, **pins)


def test_frozen_join_rule_reaches_the_apple_rungs_on_oracle_detections_only():
    ladder = load_ladder(ROOT / LADDER)
    for name, (_, _, _, rule, _) in CONFIGS.items():
        cfg = _load(name)
        for rung in HELD:
            c = rung_config(cfg, ladder, rung)
            cached = c.stages.det == "cached"
            assert cached == (rung in CACHED), (name, rung)
            assert (c.dedup.band_join, c.dedup.band_join_cached) == (rule, None), (name, rung)
            assert c.dedup.band_join_for(c.stages.det) == (None if cached else rule), (name, rung)
