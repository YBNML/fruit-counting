import os
from pathlib import Path

import pytest

from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.report import results_bytes
from counting.evaluation.runner import evaluate, load_ladder, prepare, rung_config
from counting.evaluation.truth import check_geometry, check_summary, truth_sets
from tests.unit.synthfake import fake_cfg, load_splits
from tests.unit.test_runner import assert_exact

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
    assert check_geometry(run, run.truth(), stride=20)["passed"]


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
