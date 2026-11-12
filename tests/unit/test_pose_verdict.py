"""The judged camera configuration of the pose criteria, the judged pitches of rung L5VI and the pass coverage
that pose results record."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from counting.data.formats.orchard_synth import SynthRun, select_frames
from counting.evaluation.cli import eval_app
from counting.evaluation.ledger import Ledger
from counting.evaluation.report import criteria_rows
from counting.evaluation.truth import pass_coverage, truth_sets
from counting.evaluation.verdict import ladder_pooled, ladder_verdicts, verdict_counts, verdict_summary

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]
POSE = ROOT / "configs" / "eval" / "sweeps_pose"
SPLITS = str(ROOT / "configs" / "eval" / "splits_pose.yaml")
CRITERIA_POSE = str(ROOT / "configs" / "eval" / "criteria_pose.yaml")
ZERO_LEDGER = {f.name: 0 for f in dataclasses.fields(Ledger)} | {"n_pred": 10, "n_truth": 10}

CRIT = {"y1": {"L5VI": {"apple": {"e1_median": {"le": 0.10}}}}}


def _entry(run_id, seed, e1s, *, rung):
    """A run entry whose one rung `rung` has a pass (pitch, e1) per item of `e1s` and no Y6."""
    passes = [{"pitch": p, "e1": e, "ledger": dict(ZERO_LEDGER)} for p, e in e1s]
    summary = {"e1_median": float(np.median([abs(e) for _, e in e1s]))}
    return {"meta": {"run_id": run_id, "seed": seed}, "rungs": {rung: {"y1": {"passes": passes, "summary": summary},
                                                                     "y6": None}}}


def _by_pass(frames):
    """`frames` grouped by pass, passes in first-appearance order."""
    out = {}
    for f in frames:
        out.setdefault(f.pass_key, []).append(f)
    return out


def _pose_cfg(tmp_path: Path, run: Path, *, pitches) -> str:
    """apple_dev_pose_c2.yaml on `run`, with the passes at `pitches` and the reference camera at the first."""
    raw = yaml.safe_load((ROOT / "configs" / "eval" / "apple_dev_pose_c2.yaml").read_text(encoding="utf-8"))
    raw.update(runs=[str(run)], output_dir=str(tmp_path / "out"))
    raw["passes"]["pitches"] = list(pitches)
    raw["pose"]["rig_ref"] = pitches[0]
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(p)


def _only_folder(tmp_path: Path) -> Path:
    (d,) = (tmp_path / "out" / "apple_dev_pose_c2").iterdir()
    return d


def _only_results(tmp_path: Path) -> dict:
    return json.loads((_only_folder(tmp_path) / "results.json").read_text(encoding="utf-8"))


def _only_report(tmp_path: Path) -> str:
    return (_only_folder(tmp_path) / "report.md").read_text(encoding="utf-8")


def test_l5vi_judges_y1_on_the_judged_pitches_it_ran():
    pair = [_entry("a", 1003, [(10.0, 0.5), (50.0, 0.04)], rung="L5VI"),
            _entry("b", 1004, [(10.0, 0.6), (50.0, 0.06)], rung="L5VI")]
    (row,) = ladder_verdicts(pair, ladder_pooled(pair, crop="apple"), CRIT, "apple")["L5VI"]
    assert (row["value"], row["ok"], row["pitches"]) == (pytest.approx(0.05), True, [50.0])
    one = [_entry("c", 1003, [(10.0, 0.02)], rung="L5VI")]           # one camera at 10 degrees: nothing to judge
    (row,) = ladder_verdicts(one, ladder_pooled(one, crop="apple"), CRIT, "apple")["L5VI"]
    assert (row["value"], row["ok"], row["pitches"]) == (None, None, [])
    three = [_entry("d", 1000, [(10.0, 0.5), (30.0, 0.02), (50.0, -0.04)], rung="L5")]
    crit_l5 = {"y1": {"L5": {"apple": {"e1_median": {"le": 0.10}}}}}
    (row,) = ladder_verdicts(three, ladder_pooled(three, crop="apple"), crit_l5, "apple")["L5"]
    assert row["pitches"] == [30.0, 50.0]                               # the stage-wide L5 rows stay as they were
    view = lambda e1s: _entry("n", 1005, e1s, rung="L5VI")["rungs"]["L5VI"]
    noisy = {**view([(10.0, 0.5), (50.0, -0.02)]), "noisy": True, "seeds": [0, 1],
             "spread": {"y1": {"summary": {"e1_median": [0.27, 0.29]}}, "y6": None},
             "per_seed": [{"noise_seed": 0, **view([(10.0, 0.5), (50.0, 0.04)])},
                          {"noise_seed": 1, **view([(10.0, 0.5), (50.0, -0.08)])}]}
    seeded = [{"meta": {"run_id": "n", "seed": 1005}, "rungs": {"L5VI": noisy}}]
    (row,) = ladder_verdicts(seeded, ladder_pooled(seeded, crop="apple"), CRIT, "apple")["L5VI"]
    (scene,) = row["per_run"]                                   # each seed's |e1| first, then the seed mean
    assert row["value"] == pytest.approx(0.06) and scene["value"] == pytest.approx(0.06)
    assert scene["spread"] == pytest.approx([0.04, 0.08])
    assert criteria_rows(noisy, CRIT, "L5VI", "apple")[0]["value"] == pytest.approx(0.06)


def test_rows_of_a_report_configuration_are_not_judged():
    pair = [_entry("a", 1003, [(50.0, 0.04)], rung="L5VI")]
    pooled = ladder_pooled(pair, crop="apple")
    crit = {"judged_config": "c2", **CRIT}
    judged = ladder_verdicts(pair, pooled, crit, "apple", config_label="c2")
    report = ladder_verdicts(pair, pooled, crit, "apple", config_label="c1")
    assert "report_config" not in judged["L5VI"][0] and verdict_counts(judged) == (1, 1)
    assert report["L5VI"][0]["report_config"] == "c1" and report["L5VI"][0]["ok"] is True
    assert verdict_counts(report) == (0, 0) and verdict_summary(report) == "-"
    plain = ladder_verdicts(pair, pooled, CRIT, "apple", config_label="c1")     # no judged_config: judged as before
    assert "report_config" not in plain["L5VI"][0]


def test_pose_runs_record_the_coverage_of_their_pitches(apple_run, tmp_path):
    run = SynthRun.open(apple_run)
    every = select_frames(run.frames, base_step_m=run.meta.base_step_m)
    chosen = [f for f in every if f.pass_key.pitch_deg == 10.0]
    cov = pass_coverage(run.truth(), run.layout, _by_pass(chosen), _by_pass(every), variant="count1")
    n10 = len(truth_sets(run.truth(), run.layout, _by_pass(chosen), variant="count1").g6)
    n_all = len(truth_sets(run.truth(), run.layout, _by_pass(every), variant="count1").g6)
    assert cov == {"pitches": [10.0], "g6": n10, "g6_all": n_all, "ratio": pytest.approx(n10 / n_all)}
    r = CliRunner().invoke(eval_app, ["ladder", "--config", _pose_cfg(tmp_path, apple_run, pitches=[10.0]),
                                      "--ladder", str(POSE / "ladder_pose.yaml"), "--rungs", "L3",
                                      "--splits", SPLITS, "--criteria", CRITERIA_POSE])
    assert r.exit_code == 0, r.output
    res = _only_results(tmp_path)
    assert res["schema_version"] == 11 and res["runs"][0]["meta"]["coverage"] == cov
    assert "> 덮음 " in _only_report(tmp_path)
