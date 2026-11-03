import json
import re
from pathlib import Path

import yaml
from typer.testing import CliRunner

from counting.evaluation.cli import eval_app
from tests.unit.synthfake import python_c, write_run

ROOT = Path(__file__).resolve().parents[2]
SPLITS = str(ROOT / "configs" / "eval" / "splits.yaml")
runner = CliRunner()


def test_truth_ok(apple_run):
    res = runner.invoke(eval_app, ["truth", str(apple_run), "--subsets", "--check-geometry",
                                   "--split", "dev", "--splits", SPLITS])
    assert res.exit_code == 0, res.output
    for text in ("summary OK", "geometry OK", "split OK", "drop 30"):
        assert text in res.output


def test_truth_summary_mismatch_exits_1(tmp_path):
    run = write_run(tmp_path, crop="apple", pitches=(30.0,))
    p = run / "summary.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    s["summary"]["trees"]["3"]["count1"]["A"] += 1
    p.write_text(json.dumps(s), encoding="utf-8")
    res = runner.invoke(eval_app, ["truth", str(run)])
    assert res.exit_code == 1 and "summary MISMATCH" in res.output


def test_truth_unknown_pitch_lists_available(apple_run):
    res = runner.invoke(eval_app, ["truth", str(apple_run), "--pitches", "45"])
    assert res.exit_code == 2 and "available pitches" in res.output


def test_truth_wrong_split_exits_1(apple_run):
    res = runner.invoke(eval_app, ["truth", str(apple_run), "--split", "test", "--splits", SPLITS])
    assert res.exit_code == 1 and "split MISMATCH" in res.output


def test_truth_on_non_sequence_bundle(tmp_path):
    run = write_run(tmp_path, crop="apple", max_frames=10)
    res = runner.invoke(eval_app, ["truth", str(run)])
    assert res.exit_code == 0 and "is_sequence: false" in res.output


def test_cli_import_does_not_load_torch():
    code = "import sys, counting.cli; print('torch' in sys.modules)"
    assert python_c(code).strip() == "False"


LADDER = str(ROOT / "configs" / "eval" / "sweeps" / "ladder.yaml")
CRIT = str(ROOT / "configs" / "eval" / "criteria.yaml")


def _cfg_file(tmp_path, run, crop):
    raw = yaml.safe_load((ROOT / "configs" / "eval" / f"{crop}_dev.yaml").read_text(encoding="utf-8"))
    raw.update(runs=[str(run)], output_dir=str(tmp_path / "out"))
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(p)


def test_eval_run_writes_experiment_and_is_reproducible(apple_run, tmp_path):
    args = ["run", "--config", _cfg_file(tmp_path, apple_run, "apple"), "--splits", SPLITS, "--criteria", CRIT]
    r1 = runner.invoke(eval_app, args)
    assert r1.exit_code == 0, r1.output
    dirs = list((tmp_path / "out" / "apple_dev").iterdir())
    assert len(dirs) == 1 and re.fullmatch(r"[0-9a-f]{12}", dirs[0].name)
    b1 = (dirs[0] / "results.json").read_bytes()
    r2 = runner.invoke(eval_app, args)
    assert r2.exit_code == 0 and "reproduced (same results.json)" in r2.output
    assert list((tmp_path / "out" / "apple_dev").iterdir()) == dirs
    assert (dirs[0] / "results.json").read_bytes() == b1


def test_eval_run_set_overrides(apple_run, tmp_path):
    args = ["run", "--config", _cfg_file(tmp_path, apple_run, "apple"), "--splits", SPLITS,
            "--set", "stages.track=last_box", "--set", "stop_after=y1"]
    assert runner.invoke(eval_app, args).exit_code == 0
    (d,) = (tmp_path / "out" / "apple_dev").iterdir()
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    rung = res["runs"][0]["rungs"]["run"]
    assert res["kind"] == "run" and rung["y6"] is None and rung["stages"]["track"] == "last_box"


def test_eval_ladder_rungs_and_2x2(apple_run, tmp_path):
    args = ["ladder", "--config", _cfg_file(tmp_path, apple_run, "apple"), "--rungs", "L0,L1,last_box",
            "--with-2x2", "--ladder", LADDER, "--splits", SPLITS, "--criteria", CRIT]
    r = runner.invoke(eval_app, args)
    assert r.exit_code == 0, r.output
    (d,) = (tmp_path / "out" / "apple_dev").iterdir()
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    run0 = res["runs"][0]
    assert res["kind"] == "ladder" and set(run0["rungs"]) == {"L0", "L1", "last_box"}
    assert run0["chain"] == ["L0", "L1"] and len(run0["interaction_2x2"]["cells"]) == 4
    assert "L0→L1 (추적)" in (d / "report.md").read_text(encoding="utf-8")


def test_eval_run_reordered_runs_stay_reproducible(apple_run, tmp_path):
    """A rerun with the same bundles in a different `--run` order reports as reproduced
    (results.json's `runs` list is sorted by bundle identity, not by cfg.runs/--run order)."""
    run_b = write_run(tmp_path / "run_b", crop="apple", pitches=(30.0,))
    cfg = _cfg_file(tmp_path, apple_run, "apple")
    forward = ["run", "--config", cfg, "--splits", SPLITS, "--run", str(apple_run), "--run", str(run_b)]
    backward = ["run", "--config", cfg, "--splits", SPLITS, "--run", str(run_b), "--run", str(apple_run)]
    r1 = runner.invoke(eval_app, forward)
    assert r1.exit_code == 0, r1.output
    (d,) = (tmp_path / "out" / "apple_dev").iterdir()
    b1 = (d / "results.json").read_bytes()
    r2 = runner.invoke(eval_app, backward)
    assert r2.exit_code == 0 and "reproduced (same results.json)" in r2.output, r2.output
    assert (d / "results.json").read_bytes() == b1


def test_eval_run_unsupported_option_exits_2(apple_run, tmp_path):
    args = ["run", "--config", _cfg_file(tmp_path, apple_run, "apple"), "--splits", SPLITS,
            "--set", "stages.dedup=hybrid"]
    r = runner.invoke(eval_app, args)
    assert r.exit_code == 2 and "is not supported in this version" in r.output


def test_truth_malformed_summary_exits_2(tmp_path):
    run = write_run(tmp_path, crop="apple", pitches=(30.0,), max_frames=4)
    p = run / "summary.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    del s["summary"]["trees"]
    p.write_text(json.dumps(s), encoding="utf-8")
    res = runner.invoke(eval_app, ["truth", str(run)])
    assert res.exit_code == 2 and "summary.json: no 'summary.trees' mapping" in res.output, res.output


def test_truth_corrupt_depth_png_exits_2(tmp_path):
    run = write_run(tmp_path, crop="apple", pitches=(30.0,), max_frames=4)
    (run / "000000_depth16.png").write_bytes(b"not a png")
    res = runner.invoke(eval_app, ["truth", str(run), "--check-geometry"])
    assert res.exit_code == 2 and "000000_depth16.png: unreadable PNG" in res.output, res.output


def test_eval_run_malformed_manifest_exits_2(tmp_path):
    run = write_run(tmp_path / "bundle", crop="apple", pitches=(30.0,), max_frames=4)
    p = run / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    del m["trees"]
    p.write_text(json.dumps(m), encoding="utf-8")
    res = runner.invoke(eval_app, ["run", "--config", _cfg_file(tmp_path, run, "apple"), "--splits", SPLITS])
    assert res.exit_code == 2 and "manifest.json: missing field 'trees'" in res.output, res.output
