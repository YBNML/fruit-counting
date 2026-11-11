import csv
import json
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from counting.evaluation.cli import eval_app
from tests.unit.synthfake import move_unoccluded_box, python_c, write_run

ROOT = Path(__file__).resolve().parents[2]
SPLITS = str(ROOT / "configs" / "eval" / "splits.yaml")
runner = CliRunner()


def test_truth_ok(apple_run):
    res = runner.invoke(eval_app, ["truth", str(apple_run), "--subsets", "--check-geometry",
                                   "--split", "dev", "--splits", SPLITS])
    assert res.exit_code == 0, res.output
    for text in ("summary OK", "geometry OK", "split OK", "drop 30"):
        assert text in res.output


def test_truth_geometry_without_ground(leaf_runs, tmp_path):
    ok = runner.invoke(eval_app, ["truth", str(leaf_runs[0]), "--check-geometry"])
    assert ok.exit_code == 0, ok.output
    assert ("geometry OK (center_in_bbox=1.0, n_unoccluded=41, ground=n/a (no ground pixels), "
            "ground_abs_z_median_m=None, n_ground_px=0)") in ok.output.splitlines()
    root = write_run(tmp_path, crop="apple", pitches=(30.0,))
    move_unoccluded_box(root / "000000_frame.json")
    bad = runner.invoke(eval_app, ["truth", str(root), "--check-geometry"])
    line = next(line for line in bad.output.splitlines() if line.startswith("geometry "))
    assert bad.exit_code == 1 and line.startswith("geometry FAIL (") and "ground=n/a (no ground pixels)" in line, \
        bad.output


def test_truth_geometry_fails_when_ground_has_no_depth(no_depth_run):
    res = runner.invoke(eval_app, ["truth", str(no_depth_run), "--check-geometry"])
    line = next(line for line in res.output.splitlines() if line.startswith("geometry "))
    assert res.exit_code == 1 and "summary OK" in res.output and line.startswith("geometry FAIL (") \
        and "ground=fail," in line and line.endswith("n_ground_px=0)"), res.output


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


def test_eval_run_writes_experiment_and_is_reproducible(leaf_runs, tmp_path):
    args = ["run", "--config", _cfg_file(tmp_path, leaf_runs[0], "apple"), "--splits", SPLITS, "--criteria", CRIT]
    r1 = runner.invoke(eval_app, args)
    assert r1.exit_code == 0, r1.output
    dirs = list((tmp_path / "out" / "apple_dev").iterdir())
    assert len(dirs) == 1 and re.fullmatch(r"[0-9a-f]{12}", dirs[0].name)
    b1 = (dirs[0] / "results.json").read_bytes()
    res = json.loads(b1)
    assert res["verdicts"] is None and list(res["pooled"]) == ["run"]
    r2 = runner.invoke(eval_app, [*args[:-1], str(tmp_path / "none.yaml")])     # the run id has no criteria hash
    assert r2.exit_code == 0 and "reproduced (same results.json)" in r2.output
    assert list((tmp_path / "out" / "apple_dev").iterdir()) == dirs
    assert (dirs[0] / "results.json").read_bytes() == b1


def test_eval_run_set_overrides(leaf_runs, tmp_path):
    args = ["run", "--config", _cfg_file(tmp_path, leaf_runs[0], "apple"), "--splits", SPLITS,
            "--set", "stages.track=last_box", "--set", "stop_after=y1"]
    assert runner.invoke(eval_app, args).exit_code == 0
    (d,) = (tmp_path / "out" / "apple_dev").iterdir()
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    rung = res["runs"][0]["rungs"]["run"]
    assert res["kind"] == "run" and rung["y6"] is None and rung["stages"]["track"] == "last_box"


def test_eval_run_set_list_and_null_sections(apple_run, tmp_path):
    base = ["run", "--config", _cfg_file(tmp_path, apple_run, "apple"), "--splits", SPLITS, "--set", "stop_after=y1"]
    ok = runner.invoke(eval_app, base + ["--set", "passes.pitches=[30]"])
    assert ok.exit_code == 0, ok.output
    (d,) = (tmp_path / "out" / "apple_dev").iterdir()
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    assert {p["pitch"] for p in res["runs"][0]["rungs"]["run"]["y1"]["passes"]} == {30.0}
    r = runner.invoke(eval_app, base + ["--set", "detector.noise.miss_p=0.1"])
    assert r.exit_code == 2 and "detector.noise needs stages.det=oracle_noisy" in r.output, r.output
    r = runner.invoke(eval_app, base + ["--set", "lift.depth.x=1"])
    assert r.exit_code == 2 and "error: --set: " in r.output and "not a dict" in r.output, r.output


def test_box_far_run_records_depth_fingerprint(leaf_runs, tmp_path):
    args = ["run", "--config", _cfg_file(tmp_path, leaf_runs[0], "apple"), "--splits", SPLITS,
            "--set", "lift.depth=box_far", "--set", "stop_after=y1"]
    r = runner.invoke(eval_app, args)
    assert r.exit_code == 0, r.output
    (d,) = (tmp_path / "out" / "apple_dev").iterdir()
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    assert res["runs"][0]["rungs"]["run"]["lift_depth"] == "box_far" and rec["data"][0]["depth_fingerprint"]


def test_eval_run_with_noise_writes_seed_rows(leaf_runs, tmp_path):
    args = ["run", "--config", _cfg_file(tmp_path, leaf_runs[0], "apple"), "--splits", SPLITS,
            "--set", "stages.det=oracle_noisy", "--set", "detector.noise.miss_p=0.1",
            "--set", "noise.depth_dropout=0.05", "--set", "noise.seeds=[0,1]"]
    r = runner.invoke(eval_app, args)
    assert r.exit_code == 0, r.output
    (d,) = (tmp_path / "out" / "apple_dev").iterdir()
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    rung = res["runs"][0]["rungs"]["run"]
    assert res["schema_version"] == 10 and rung["noisy"] and rung["seeds"] == [0, 1]
    assert rung["noise"]["detector"]["miss_p"] == 0.1 and rung["noise"]["depth_dropout"] == 0.05
    line = next(line for line in r.output.splitlines() if line.startswith("run: row U="))
    assert line.endswith(" (mean over noise seeds 0,1)")
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    assert rec["detector"]["name"] == "oracle_noisy" and rec["detector"]["noise"]["miss_p"] == 0.1
    assert rec["seeds"]["noise_seeds"] == [0, 1]
    rows = list(csv.reader((d / "per_pass.csv").read_text(encoding="utf-8").splitlines()))
    assert [row[2] for row in rows[1:]] == ["0", "0", "1", "1", "mean", "mean"]      # two passes per seed, then the mean
    assert (d / "report.md").read_text(encoding="utf-8").splitlines()[1] == \
        "> 합성 렌더 · 정답 자세 · 보정 전(U). 실제 정확도가 아니다."               # detection and depth noise only


def test_box_far_ladder_records_depth_fingerprint(leaf_runs, tmp_path):
    args = ["ladder", "--config", _cfg_file(tmp_path, leaf_runs[0], "apple"), "--rungs", "L3", "--splits", SPLITS,
            "--set", "lift.depth=box_far", "--set", "stop_after=y1"]
    r = runner.invoke(eval_app, args)
    assert r.exit_code == 0, r.output
    (d,) = (tmp_path / "out" / "apple_dev").iterdir()
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    assert res["runs"][0]["rungs"]["L3"]["lift_depth"] == "box_far" and rec["data"][0]["depth_fingerprint"]
    assert rec["detector"] == {"name": "oracle", "noise": None, "weights_sha256": None, "cache_sha256": None,
                               "pretrained": None}


def test_eval_ladder_rungs_and_2x2(leaf_runs, tmp_path):
    args = ["ladder", "--config", _cfg_file(tmp_path, leaf_runs[0], "apple"), "--rungs", "L0,L1,last_box",
            "--with-2x2", "--ladder", LADDER, "--splits", SPLITS, "--criteria", CRIT]
    r = runner.invoke(eval_app, args)
    assert r.exit_code == 0, r.output
    (d,) = (tmp_path / "out" / "apple_dev").iterdir()
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    run0 = res["runs"][0]
    assert res["kind"] == "ladder" and set(run0["rungs"]) == {"L0", "L1", "last_box"}
    assert run0["chain"] == ["L0", "L1"] and len(run0["interaction_2x2"]["cells"]) == 4
    report = (d / "report.md").read_text(encoding="utf-8")
    assert "L0→L1 (추적)" in report and "**판정" not in report          # a dev ladder: verdicts in results.json only
    assert list(res["verdicts"]) == ["L1"] and [r["unit"] for r in res["verdicts"]["L1"]] == ["pooled"] * 5
    assert list(res["pooled"]) == ["L0", "L1", "last_box"] and res["pooled"]["L1"]["n_runs"] == 1


@pytest.mark.slow
def test_eval_run_reordered_runs_stay_reproducible(leaf_runs, tmp_path):
    """A rerun with the same bundles in a different `--run` order reports as reproduced
    (results.json's `runs` list is sorted by bundle identity, not by cfg.runs/--run order)."""
    run_a = leaf_runs[0]
    run_b = write_run(tmp_path / "small", crop="apple", pitches=(30.0,), trees_per_row=2)   # another fingerprint
    cfg = _cfg_file(tmp_path, run_a, "apple")
    forward = ["run", "--config", cfg, "--splits", SPLITS, "--run", str(run_a), "--run", str(run_b)]
    backward = ["run", "--config", cfg, "--splits", SPLITS, "--run", str(run_b), "--run", str(run_a)]
    r1 = runner.invoke(eval_app, forward)
    assert r1.exit_code == 0, r1.output
    (d,) = (tmp_path / "out" / "apple_dev").iterdir()
    b1 = (d / "results.json").read_bytes()
    r2 = runner.invoke(eval_app, backward)
    assert r2.exit_code == 0 and "reproduced (same results.json)" in r2.output, r2.output
    assert (d / "results.json").read_bytes() == b1


def test_eval_run_cached_without_a_cache_exits_2(apple_run, tmp_path):
    base = ["run", "--config", _cfg_file(tmp_path, apple_run, "apple"), "--splits", SPLITS, "--set", "stages.det=cached"]
    r = runner.invoke(eval_app, base)
    assert r.exit_code == 2 and "stages.det=cached needs detector.cache" in r.output, r.output
    (tmp_path / "empty").mkdir()
    r = runner.invoke(eval_app, [*base, "--set", f"detector.cache={tmp_path / 'empty'}"])
    assert r.exit_code == 2 and "no detection cache for this bundle" in r.output, r.output


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


def test_eval_run_bad_flow_value_exits_2(tmp_path):
    cfg = _cfg_file(tmp_path, tmp_path / "missing", "apple")
    r = runner.invoke(eval_app, ["run", "--config", cfg, "--splits", SPLITS, "--set", "passes.pitches=[30"])
    assert r.exit_code == 2 and "error: Override value is not valid YAML flow syntax: [30" in r.output, r.output
    assert "--set:" not in r.output


def test_eval_run_malformed_manifest_exits_2(tmp_path):
    run = write_run(tmp_path / "bundle", crop="apple", pitches=(30.0,), max_frames=4)
    p = run / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    del m["trees"]
    p.write_text(json.dumps(m), encoding="utf-8")
    res = runner.invoke(eval_app, ["run", "--config", _cfg_file(tmp_path, run, "apple"), "--splits", SPLITS])
    assert res.exit_code == 2 and "manifest.json: missing field 'trees'" in res.output, res.output


def test_eval_ladder_refuses_bad_request_before_reading_bundles(tmp_path):
    cfg = _cfg_file(tmp_path, tmp_path / "missing", "apple")
    no_2x2 = tmp_path / "no_2x2.yaml"
    no_2x2.write_text("rungs: {L3: {}}\n", encoding="utf-8")
    for extra, phrase in ((["--set", "lift.depth=det_depth"], "ladder base config has lift.depth=det_depth"),
                          (["--set", "noise.depth_dropout=0.05"], "ladder base config has noise set"),
                          (["--rungs", "L3,L9"], "unknown rung 'L9'"),
                          (["--ladder", str(no_2x2), "--with-2x2"], "with_2x2 needs an 'interaction_2x2' mapping"),
                          (["--rungs", "L1", "--subsets"], "--subsets needs rung L3"),
                          (["--rungs", "L3", "--subsets", "--set", "stop_after=y1"], "--subsets needs stop_after=y6")):
        r = runner.invoke(eval_app, ["ladder", "--config", cfg, "--splits", SPLITS, *extra])
        assert r.exit_code == 2 and phrase in r.output and "manifest.json" not in r.output, r.output


def test_eval_ladder_subsets_need_L3_and_get_their_own_id(leaf_runs, tmp_path):
    base = ["ladder", "--config", _cfg_file(tmp_path, leaf_runs[0], "apple"), "--splits", SPLITS]
    bad = runner.invoke(eval_app, base + ["--rungs", "L1", "--subsets"])
    assert bad.exit_code == 2 and "--subsets needs rung L3" in bad.output, bad.output
    for extra in (["--subsets"], []):
        r = runner.invoke(eval_app, base + ["--rungs", "L1,L3", *extra])
        assert r.exit_code == 0, r.output             # a second id, not a reproducibility failure
    found = {}
    for d in (tmp_path / "out" / "apple_dev").iterdir():
        rungs = json.loads((d / "results.json").read_text(encoding="utf-8"))["runs"][0]["rungs"]
        found["subsets" in rungs["L3"]] = (rungs, (d / "report.md").read_text(encoding="utf-8"))
    assert set(found) == {True, False}
    (with_s, report), (without, _) = found[True], found[False]
    assert "subsets" not in with_s["L1"] and [r["S"] for r in with_s["L3"]["subsets"]][-1] == "drop 30"
    assert "**패스 묶음 (L3)**" in report and with_s["L3"]["y6"] == without["L3"]["y6"]


def test_eval_ladder_opens_every_bundle_before_scoring(leaf_runs, tmp_path, monkeypatch):
    import counting.evaluation.runner as runner_module
    calls = []
    monkeypatch.setattr(runner_module, "prepare", lambda *a, **k: calls.append(1))
    args = ["ladder", "--config", _cfg_file(tmp_path, leaf_runs[0], "apple"), "--rungs", "L1", "--splits", SPLITS,
            "--run", str(leaf_runs[0]), "--run", str(tmp_path / "missing")]
    r = runner.invoke(eval_app, args)
    assert r.exit_code == 2 and "no manifest.json" in r.output and calls == [], r.output
