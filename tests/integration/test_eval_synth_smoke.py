import os
from pathlib import Path

import pytest

from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.report import results_bytes
from counting.evaluation.runner import evaluate, load_ladder, prepare, rung_config
from counting.evaluation.truth import check_geometry, check_summary, truth_sets
from tests.unit.synthfake import fake_cfg, load_splits
from tests.unit.test_runner import _without_mode, assert_exact

pytestmark = [pytest.mark.slow, pytest.mark.synth]
ROOT = Path(__file__).resolve().parents[2]
BASE = Path(os.environ.get("COUNTING_SYNTH_DEV", str(ROOT.parent / "orchard_synth" / "out"))).expanduser()
LADDER = load_ladder(ROOT / "configs" / "eval" / "sweeps" / "ladder.yaml")
# Recounted from the dev frame.json files (count1, target row, seed 5); matches summary.json.
DEV = {
    "apple/robot3": ("apple", (1102, 1089, 1253), 1306,
                     {(10, "A"): 833, (10, "B"): 813, (30, "A"): 752, (30, "B"): 751, (50, "A"): 343, (50, "B"): 350}),
    "pear/robot3": ("pear", (465, 473, 679), 680,
                    {(20, "A"): 284, (20, "B"): 299, (45, "A"): 215, (45, "B"): 219, (70, "A"): 172, (70, "B"): 168}),
    "pear/robot3_occl": ("pear", (373, 408, 626), 632,
                         {(20, "A"): 207, (20, "B"): 225, (45, "A"): 189, (45, "B"): 192, (70, "A"): 158, (70, "B"): 176}),
}


def _open(rel):
    if not (BASE / rel / "manifest.json").exists():
        pytest.skip(f"dev render not present: {rel}")
    return SynthRun.open(BASE / rel)


@pytest.mark.parametrize("rel", sorted(DEV))
def test_dev_truth_tables(rel):
    crop, (a, b, u), total, g1 = DEV[rel]
    run = _open(rel)
    t, passes = run.truth(), run.passes()
    tt = truth_sets(t, run.layout, passes)
    assert {(int(k.pitch_deg), k.side): len(v) for k, v in tt.g1.items()} == g1
    side = lambda s: len(frozenset().union(*(v for k, v in tt.g1.items() if k.side == s)))   # noqa: E731
    assert (side("A"), side("B"), len(tt.g6)) == (a, b, u) and sum(tt.g6_total.values()) == total
    check_summary(t, run.layout, passes)


@pytest.mark.parametrize("rel", sorted(DEV))
def test_dev_geometry(rel):
    run = _open(rel)
    res = check_geometry(run, run.truth(), stride=20)
    assert res["passed"] and res["ground"] == "ok"      # these renders see the ground, so its check runs


@pytest.mark.parametrize("rel", sorted(DEV))
def test_dev_L0_exact_and_reproducible(rel):
    crop = DEV[rel][0]
    run = _open(rel)
    cfg = rung_config(fake_cfg(BASE / rel, crop), LADDER, "L0")
    r1 = evaluate(prepare(run, cfg, load_splits()), cfg)
    r2 = evaluate(prepare(SynthRun.open(BASE / rel), cfg, load_splits()), cfg)
    assert_exact(r1)
    assert r1["y6"]["row"]["U"] == DEV[rel][1][2] and r1["y6"]["row"]["e_pipe"] == 0.0
    assert results_bytes(r1) == results_bytes(r2)


@pytest.mark.parametrize("rel", sorted(DEV))
def test_dev_L1_tracker_sanity(rel):
    """Regression guard, not a verdict: L1 e1 is about 1-2 % on these bundles."""
    crop = DEV[rel][0]
    run = _open(rel)
    cfg = rung_config(fake_cfg(BASE / rel, crop), LADDER, "L1")
    res = evaluate(prepare(run, cfg, load_splits()), cfg)
    assert res["y1"]["summary"]["e1_median"] <= 0.05


LEDGER = ("fp_bg", "fp_off", "boundary", "dup_track", "dup_cross", "miss_det", "miss_drop", "miss_merge", "impure")
# Default box depth (box_median 0.5, hard depth gate), measured at bf72709
# (L3_dd: the L3 setting with det_depth): row U and row ledger.
DEPTH_BASELINE = {
    "apple/robot3": {"L2c": (2497, (0, 0, 0, 912, 333, 0, 0, 1, 89)),
                     "L3": (3154, (0, 668, 0, 904, 330, 0, 0, 1, 139)),
                     "L3_dd": (1374, (0, 0, 0, 56, 66, 0, 0, 1, 8))},
    "pear/robot3": {"L2c": (883, (0, 0, 0, 129, 75, 0, 0, 0, 23)),
                    "L3": (960, (0, 79, 0, 129, 75, 0, 2, 0, 63)),
                    "L3_dd": (708, (0, 0, 0, 22, 10, 0, 3, 0, 1))},
    "pear/robot3_occl": {"L2c": (841, (0, 0, 0, 177, 40, 0, 2, 0, 6)),
                         "L3": (887, (0, 46, 0, 177, 40, 0, 2, 0, 10)),
                         "L3_dd": (637, (0, 1, 0, 6, 7, 0, 3, 0, 0))},
}
# box_far + soft depth gate + repair_m x1.5 of the crop default:
# (repair override, L3 row e_pipe, L1 Y1 |e1| median) measured on the dev renders.
REMEDY = {"apple/robot3": ("tracker.repair_m=0.06", 0.156, 0.007),
          "pear/robot3": ("tracker.repair_m=0.135", 0.060, 0.004),
          "pear/robot3_occl": ("tracker.repair_m=0.135", 0.045, 0.000)}


@pytest.mark.parametrize("rel", sorted(DEV))
def test_dev_depth_baselines_unchanged(rel):
    crop = DEV[rel][0]
    cfg = fake_cfg(BASE / rel, crop)
    ctx = prepare(_open(rel), cfg, load_splits())
    for name, (U, ledger) in DEPTH_BASELINE[rel].items():
        y6 = evaluate(ctx, rung_config(cfg, LADDER, name))["y6"]
        assert (y6["row"]["U"], tuple(y6["ledger"][k] for k in LEDGER)) == (U, ledger), name


@pytest.mark.parametrize("rel", sorted(DEV))
def test_dev_depth_tail_remedy(rel):
    """Regression guard, not a verdict: within 0.5 pp of the measured values."""
    crop = DEV[rel][0]
    repair, e_pipe, e1 = REMEDY[rel]
    cfg = fake_cfg(BASE / rel, crop, "lift.depth=box_far", "tracker.depth_gate=soft", repair)
    ctx = prepare(_open(rel), cfg, load_splits())
    l3 = evaluate(ctx, rung_config(cfg, LADDER, "L3"))
    l1 = evaluate(ctx, rung_config(cfg, LADDER, "L1"))
    assert abs(l3["y6"]["row"]["e_pipe"] - e_pipe) <= 0.005
    assert abs(l1["y1"]["summary"]["e1_median"] - e1) <= 0.005


def test_dev_hybrid_band_limits():
    """Regression guard, not a verdict: hybrid on the dev apple render at L3 (default box depth).

    Band 0 equals self_side and band 1e9 equals merge; row U measured at 8141780 with this rule.
    """
    rel = "apple/robot3"
    ctx = prepare(_open(rel), fake_cfg(BASE / rel, "apple"), load_splits())

    def y6(*sets):
        return evaluate(ctx, rung_config(fake_cfg(BASE / rel, "apple", *sets), LADDER, "L3"))["y6"]

    side, merge = y6("stages.dedup=self_side"), y6()
    band0, band10, band_inf = (y6("stages.dedup=hybrid", f"dedup.band_m={b}") for b in ("0.0", "0.10", "1e9"))
    assert (side["row"]["U"], band10["row"]["U"], merge["row"]["U"]) == (2313, 2561, 3154)
    assert _without_mode(band0) == _without_mode(side) and _without_mode(band_inf) == _without_mode(merge)
