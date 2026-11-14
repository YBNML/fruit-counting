"""The rung names of the LiDAR depth ladder in the judgement, the report and the error budget.

Rung L5VI_ld (L5VI with the LiDAR box depth) is judged on its judged pitches as L5VI is, in the verdicts, in the
report's criteria tables and in a grid cell on it. The budget pairs of the pose ladder keep their order, and the pairs
of the LiDAR depth rungs follow them with their labels in the contribution table, while a ladder without pose rungs
keeps its own pairs. A criteria table with the pose numbers under the LiDAR depth rung names judges fifteen lines per
crop. The run entries are hand-made dicts; nothing here opens a bundle."""

import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest
import yaml

from counting.evaluation.budget import (
    POSE_ONLY_RUNGS, POSE_PAIRS, SHARED_RUNGS, check_budget, contrib_pairs, ladder_budget, ladder_pairs, pose_pairs,
)
from counting.evaluation.ledger import Ledger
from counting.evaluation.report import _contrib_table, _sweep_criteria_cell, criteria_rows
from counting.evaluation.verdict import (
    JUDGED_PITCHES, PITCH_JUDGED_RUNGS, RAN_PITCH_RUNGS, judged_pitches, ladder_pooled, ladder_verdicts, verdict_counts,
    verdict_summary,
)
from tests.unit.test_verdict_bytes import NOISE_SEEDS, PITCHES, PLAIN_RUNGS, _ladder_entries

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]
CRITERIA_POSE = ROOT / "configs" / "eval" / "criteria_pose.yaml"
ZERO_LEDGER = {f.name: 0 for f in dataclasses.fields(Ledger)} | {"n_pred": 10, "n_truth": 10}

CRIT = {"y1": {name: {"apple": {"e1_median": {"le": 0.10}}, "pear": {"e1_median": {"le": 0.08}}}
               for name in ("L5", "L5VI", "L5VI_ld")}}
# The budget pairs of the pose ladder, in their table order.
POSE_LADDER_PAIRS = (("L3", "L4_drift"), ("L3", "LVI"), ("L5", "L5VI"), ("LVI", "L5VI"), ("L4_drift_noimu", "L4_drift"),
                     ("LV", "LVI"), ("LV_ref", "LV"), ("L4_drift_know", "L4_drift"), ("L4_drift_noal", "L4_drift"),
                     ("LVI_first", "LVI"), ("LVI_fits", "LVI"), ("L4_drift_c1x", "L4_drift"), ("LVI_c1x", "LVI"),
                     ("LVI_dd", "LVI"), ("L3", "L3_al"))
# The budget pairs of the LiDAR depth rungs, in their table order, with their contribution table labels.
DEPTH_LABELS = {
    ("L4_drift", "L4_drift_ld"): "L4_drift→L4_drift_ld (LiDAR 깊이)",
    ("LVI", "LVI_ld"): "LVI→LVI_ld (LiDAR 깊이)",
    ("L5VI", "L5VI_ld"): "L5VI→L5VI_ld (LiDAR 깊이)",
    ("L4_drift_ld", "L4_drift_ldn"): "L4_drift_ld→L4_drift_ldn (항법 LiDAR 공용)",
    ("LVI_ld", "LVI_ldn"): "LVI_ld→LVI_ldn (항법 LiDAR 공용)",
    ("L3", "L3_ld"): "L3→L3_ld (LiDAR 깊이)",
    ("LVI_ld", "L5VI_ld"): "LVI_ld→L5VI_ld (검출)",
    ("L4_drift_ld_orc", "L4_drift_ld"): "L4_drift_ld_orc→L4_drift_ld (짝짓기)",
    ("LVI_ddd", "LVI"): "LVI_ddd→LVI (상자 깊이)",
    ("LVI_ddv", "LVI"): "LVI_ddv→LVI (VO 깊이)",
    ("L3_dd", "L3"): "L3_dd→L3 (상자 깊이)",
}
# The rungs of a LiDAR depth ladder: the judgement chain, the judged rungs, their camera-only pairs, the report rungs.
DEPTH_LADDER = ["L3", "L5", "L3_dd", "L3_ld", "L4_drift_ld", "LVI_ld", "L5VI_ld", "L4_drift", "LVI", "L5VI", "LVI_dd",
                "LVI_ddd", "LVI_ddv", "L4_drift_ld_orc", "L4_drift_ldn", "LVI_ldn"]
# The judged rungs of the pose criteria and the LiDAR depth rungs that take their numbers.
LIDAR_NAMES = {"L4_drift": "L4_drift_ld", "LVI": "LVI_ld", "L5VI": "L5VI_ld"}


def _entry(run_id, seed, e1s, names):
    """A run entry whose rungs `names` each have a pass (pitch, e1) per item of `e1s` and no Y6."""
    passes = [{"pitch": p, "e1": e, "ledger": dict(ZERO_LEDGER)} for p, e in e1s]
    summary = {"e1_median": float(np.median([abs(e) for _, e in e1s]))}
    rung = {"y1": {"passes": passes, "summary": summary}, "y6": None}
    return {"meta": {"run_id": run_id, "seed": seed}, "rungs": {name: copy.deepcopy(rung) for name in names}}


def _verdicts(entries, crop="apple", criteria=CRIT, **kw):
    return ladder_verdicts(entries, ladder_pooled(entries, crop=crop), criteria, crop, **kw)


def test_learned_detector_lidar_rung_is_judged_on_its_pitches():
    names = ("L5", "L5VI", "L5VI_ld")
    pair = [_entry("a", 1006, [(10.0, 0.5), (50.0, 0.04)], names),
            _entry("b", 1007, [(10.0, 0.6), (50.0, 0.06)], names)]
    verdicts = _verdicts(pair)
    (row,) = verdicts["L5VI_ld"]
    assert (row["value"], row["ok"], row["pitches"]) == (pytest.approx(0.05), True, [50.0])   # every pass: 0.28
    assert [p["value"] for p in row["per_run"]] == pytest.approx([0.04, 0.06])
    assert verdicts["L5VI_ld"] == verdicts["L5VI"]
    assert verdicts["L5"][0]["pitches"] == [30.0, 50.0]                 # rung L5 names every judged pitch
    one = [_entry("c", 1006, [(10.0, 0.02)], names)]                      # one camera at 10 degrees: nothing to judge
    (row,) = _verdicts(one)["L5VI_ld"]
    assert (row["value"], row["ok"], row["pitches"]) == (None, None, [])
    pear = [_entry("d", 1008, [(20.0, 0.3), (70.0, 0.05)], names)]
    (row,) = _verdicts(pear, "pear")["L5VI_ld"]
    assert (row["value"], row["ok"], row["pitches"]) == (pytest.approx(0.05), True, [70.0])


def test_criteria_rows_follow_the_judged_pitches():
    (rung,) = _entry("a", 1006, [(10.0, 0.5), (50.0, 0.04)], ["L5VI_ld"])["rungs"].values()
    (row,) = criteria_rows(rung, CRIT, "L5VI_ld", "apple")
    assert (row["value"], row["ok"]) == (pytest.approx(0.04), True)     # every pass: 0.27, over the limit
    cell = {"point": "p0", "base": "L5VI_ld", "overridden": ["noise.lidar_range_sigma_m"], **rung}
    assert criteria_rows(cell, CRIT, cell["base"], "apple") == [row]
    assert _sweep_criteria_cell(cell, CRIT, "apple") == "1/1"
    view = lambda e1s: _entry("n", 1006, e1s, ["L5VI_ld"])["rungs"]["L5VI_ld"]
    noisy = {**cell, **view([(10.0, 0.5), (50.0, -0.02)]), "noisy": True, "seeds": [0, 1],
             "per_seed": [{"noise_seed": 0, **view([(10.0, 0.5), (50.0, 0.04)])},
                          {"noise_seed": 1, **view([(10.0, 0.5), (50.0, -0.08)])}]}
    (row,) = criteria_rows(noisy, CRIT, noisy["base"], "apple")         # each seed's judged |e1|, then their mean
    assert (row["value"], row["ok"]) == (pytest.approx(0.06), True)
    assert _sweep_criteria_cell(noisy, CRIT, "apple") == "1/1"


def test_ledger_pairs_keep_their_order_and_add_the_depth_pairs():
    pose_rungs = dict.fromkeys(n for pair in POSE_LADDER_PAIRS for n in pair)
    assert pose_pairs(pose_rungs) == list(POSE_LADDER_PAIRS)
    assert POSE_PAIRS == (*POSE_LADDER_PAIRS, *DEPTH_LABELS)
    assert SHARED_RUNGS == {"L3", "L5", "L3_dd"}
    assert POSE_ONLY_RUNGS == {n for pair in POSE_PAIRS for n in pair} - SHARED_RUNGS
    kept = [pair for pair in POSE_LADDER_PAIRS if set(pair) <= set(DEPTH_LADDER)]
    entries = _ladder_entries(DEPTH_LADDER, "apple", PITCHES["apple"], NOISE_SEEDS)
    chain, rungs = entries[0]["chain"], entries[0]["rungs"]
    assert ladder_pairs(chain, rungs) == [*kept, *DEPTH_LABELS]
    budget = ladder_budget(entries, "apple")                                # its own check_budget passed
    check_budget(budget)
    assert list(dict.fromkeys(tuple(r["pair"].split("→")) for r in budget)) == [*kept, *DEPTH_LABELS]
    assert {r["group"] for r in budget} == {"row", "pitch_10", "pitch_30", "pitch_50", "judged", "all"}
    table = _contrib_table(chain, rungs, crop="apple")
    labels = [line.split(" | ")[0].removeprefix("| ") for line in table[2:]]
    assert labels == [*(f"{a}→{b}" for a, b in kept), *DEPTH_LABELS.values()]
    plain = dict.fromkeys(PLAIN_RUNGS)
    chain = PLAIN_RUNGS[:8]
    assert not POSE_ONLY_RUNGS & set(PLAIN_RUNGS)
    assert ladder_pairs(chain, plain) == contrib_pairs(chain, plain)
    assert contrib_pairs(chain, plain)[-2:] == [("L3_dd", "L3"), ("L5", "L5_L4base")]


def test_fifteen_judged_lines_per_crop():
    pose = yaml.safe_load(CRITERIA_POSE.read_text(encoding="utf-8"))
    criteria = {"judged_config": "c2l",
                **{kind: {LIDAR_NAMES[name]: spec for name, spec in pose[kind].items()} for kind in ("y1", "y6")}}
    y6 = ["y6.row_abs_e_pipe", "y6.tree_mae_over_mean", "y6.miss_rate", "y6.excess_rate"]
    for crop in ("apple", "pear"):
        entries = _ladder_entries(DEPTH_LADDER, crop, PITCHES[crop], NOISE_SEEDS)
        verdicts = _verdicts(entries, crop, criteria, config_label="c2l")
        assert {name: [row["metric"] for row in rows] for name, rows in verdicts.items()} == {
            "L4_drift_ld": y6, "LVI_ld": ["y1.e1_median", *y6, "y6.attribution_error"],
            "L5VI_ld": ["y1.e1_median", *y6]}
        assert verdict_counts(verdicts)[1] == 15 and verdict_summary(verdicts).endswith("/15")
        assert verdicts["L5VI_ld"][0]["pitches"] == list(JUDGED_PITCHES[crop])
        assert "pitches" not in verdicts["LVI_ld"][0]
        assert verdict_summary(_verdicts(entries, crop, criteria, config_label="c2")) == "-"


def test_judged_pitches_helper():
    assert PITCH_JUDGED_RUNGS == ("L5", "L5VI", "L5VI_ld") and RAN_PITCH_RUNGS == ("L5VI", "L5VI_ld")
    apple = [_entry("a", 1006, [(10.0, 0.5), (50.0, 0.04)], ["L5VI_ld"])["rungs"]["L5VI_ld"]]
    assert judged_pitches("L5VI_ld", apple, "apple") == [50.0]
    assert judged_pitches("L5VI", apple, "apple") == [50.0]
    assert judged_pitches("L5", apple, "apple") == [30.0, 50.0]
    pear = [_entry("b", 1007, [(20.0, 0.1), (70.0, 0.02)], ["L5VI_ld"])["rungs"]["L5VI_ld"]]
    assert judged_pitches("L5VI_ld", pear, "pear") == [70.0]
    both = [_entry(f"c{i}", 1006 + i, e1s, ["L5VI_ld"])["rungs"]["L5VI_ld"]       # the pitches of every bundle
            for i, e1s in enumerate([[(10.0, 0.1), (50.0, 0.1)], [(10.0, 0.1), (30.0, 0.1)]])]
    assert judged_pitches("L5VI_ld", both, "apple") == [30.0, 50.0]
    none = [_entry("d", 1008, [(10.0, 0.1)], ["L5VI_ld"])["rungs"]["L5VI_ld"]]
    assert judged_pitches("L5VI_ld", none, "apple") == []
