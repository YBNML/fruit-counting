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
    "apple": {
        "det": {"apple_s1006_95468d66": "92453329458dfc1910795c93059c2525ca20947d03d6325f08dbf92a61749692",
                "apple_s1007_42067b5d": "fa647119d382fced9f62adba7c2711d056d274a1d709e2e2d54a0de8f2654dda",
                "apple_s1008_92e79245": "8c97a96ce9abc772081265746d056af3de8399be1d25195d5f8086112d916858"},
        "vo": {"apple_s1006_95468d66": "9aaef4fe161a79fa825b4a883a1b0568b294233263ef6dd9b8c872945fbb6b2b",
               "apple_s1007_42067b5d": "f941cffe5d3e8dc3d3ac97f4431821a652b51ba0617f322ba61374a3a78c3262",
               "apple_s1008_92e79245": "7eb1c56de7bb6a468ce25d85cbce5b46e846f3708f8797894953c5b9a74515c5"},
        "vo_dd": {"apple_s1006_95468d66": "827d73f841299912708126ab9e068e7331499d37d36834666e00366e2440f96f",
                  "apple_s1007_42067b5d": "27d68dbf265fa574395b96b460e2e1771bf4df53b8e9d66a90b08f3b9cbe2049",
                  "apple_s1008_92e79245": "fd18ce2a5dfc843b4e616b139d444bcf26d672c3410064d332536eef35b44894"},
        "imu_ext": {"apple_s1006_95468d66": "14340d1939dae09b3b1970b370875403d922bf60c23e6be8f94c16f54b3c144b",
                    "apple_s1007_42067b5d": "c8fec84cd28a0e8000fdf8f4398f8b82d75bf9ac8148fb40c69b143c025ad77d",
                    "apple_s1008_92e79245": "93f604f2a59ffcfe67921698104e062ee25a5908d8b8fba54a4311dd67a7b5cf"},
    },
    "pear": {
        "det": {"pear_s1006_9373990c": "91aaaeec522f64c160defa9657625bc7c62c7fb122aec45bad7877c57da71095",
                "pear_s1007_49e403db": "866be756aef0c8e0d74ad3aea654500a6fee0f37ab320f0a2172c005491859de",
                "pear_s1008_673e55e8": "b6a1d692d9f05f83cb585927c955f358bd85a99be1b57edbfb0386e325a475da"},
        "vo": {"pear_s1006_9373990c": "8d5771fbff4c37fcecb90c25aa4e77ffd0784150eb375fe90d816a95af7832bf",
               "pear_s1007_49e403db": "8838a4c68b731ff4bbcccfbd24ffe1aa23a4c841472798319548562f6da6a772",
               "pear_s1008_673e55e8": "efaf9cae9c1a21588ae4093f7d2172947f056f9ea3ff8f47e4d8b5d9a05b6c11"},
        "vo_dd": {"pear_s1006_9373990c": "b1bc0d7daeb3baa0dcc53531a2c9d7014e88126702d30fcafbce3efd9b9a6020",
                  "pear_s1007_49e403db": "1fa7848e4f6266c0c109faca3c6f0cdd15aa3feb3736d921b23ae7fd2df3fc7c",
                  "pear_s1008_673e55e8": "3339b72be10601535b6ad0c5b7e3588a91c9b2b18c2d696808fe0ea686a1a336"},
        "imu_ext": {"pear_s1006_9373990c": "aa0e2c8908943d939aa264e975adc0c27e9aa787c7c87726f9c5e367f4755214",
                    "pear_s1007_49e403db": "3f67da9df9841eb9fb97c288c4b2c20777b872f27358491534535aa5aca54798",
                    "pear_s1008_673e55e8": "783f3549cc8a5dc4fa124e9fb2b81ce6773d1ec9be2294ec78a366091b909991"},
    },
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
