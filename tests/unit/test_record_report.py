import copy
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.record import experiment_id, git_state
from counting.evaluation.report import (
    SCHEMA_VERSION, ReproducibilityError, _pitch_rows, criteria_rows, render_report, results_bytes,
    write_experiment,
)
from counting.evaluation.runner import describe_run, evaluate, load_ladder, prepare, run_ladder
from tests.unit.synthfake import fake_cfg, load_splits, write_run

ROOT = Path(__file__).resolve().parents[2]
CRIT = yaml.safe_load((ROOT / "configs" / "eval" / "criteria.yaml").read_text(encoding="utf-8"))
CODE = {"commit": "a" * 40, "dirty": False, "diff_sha256": None}


def _results(root, crop="apple"):
    cfg = fake_cfg(root, crop)
    ctx = prepare(SynthRun.open(root), cfg, load_splits())
    return cfg, {"schema_version": SCHEMA_VERSION, "kind": "run", "run_name": "t", "experiment_id": "abc123abc123",
                 "split": cfg.split, "crop": cfg.crop,
                 "runs": [{"meta": describe_run(ctx), "chain": [], "rungs": {"run": evaluate(ctx, cfg)},
                           "interaction_2x2": None}]}


def test_experiment_id_stable_and_sensitive():
    kw = dict(config_hash="c" * 16, fingerprints=["f2", "f1"], code=CODE, seed=0)
    a = experiment_id(**kw)
    assert len(a) == 12 and a == experiment_id(**{**kw, "fingerprints": ["f1", "f2"]})
    assert a != experiment_id(**{**kw, "seed": 1})
    assert a != experiment_id(**{**kw, "code": {**CODE, "dirty": True, "diff_sha256": "d" * 64}})
    assert a != experiment_id(**kw, extra="ladder")


def test_git_state_temp_repo(tmp_path):
    g = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.txt").write_text("1")
    subprocess.run(g + ["add", "a.txt"], check=True)
    subprocess.run(g + ["commit", "-qm", "x"], check=True)
    s = git_state(tmp_path)
    assert len(s["commit"]) == 40 and s["dirty"] is False and s["diff_sha256"] is None
    (tmp_path / "b.txt").write_text("new")                       # untracked files count too
    s1 = git_state(tmp_path)
    (tmp_path / "b.txt").write_text("newer")
    s2 = git_state(tmp_path)
    assert s1["dirty"] and len(s1["diff_sha256"]) == 64 and s1["diff_sha256"] != s2["diff_sha256"]


def test_git_state_outside_repo(tmp_path):
    assert git_state(tmp_path)["commit"] is None


def test_results_bytes_sorted_rounded_nan_null_numpy():
    assert results_bytes({"b": 1 / 3, "a": [float("nan"), 2]}) == \
        b'{\n  "a": [\n    null,\n    2\n  ],\n  "b": 0.333333\n}\n'
    assert results_bytes({"n": np.int64(3), "f": np.float32(0.5), "t": np.bool_(True), "v": np.array([1, 2]),
                          "p": (1, 2)}) == results_bytes({"n": 3, "f": 0.5, "t": True, "v": [1, 2], "p": [1, 2]})


def test_evaluate_bytes_reproducible(apple_run):
    assert results_bytes(_results(apple_run)[1]) == results_bytes(_results(apple_run)[1])


def test_write_experiment_reproducible_and_guarded(apple_run, tmp_path):
    cfg, res = _results(apple_run)
    d1, rep1 = write_experiment(tmp_path, cfg=cfg, results=res, record={"x": 1}, criteria=None)
    b1 = (d1 / "results.json").read_bytes()
    d2, rep2 = write_experiment(tmp_path, cfg=cfg, results=copy.deepcopy(res), record={"x": 2}, criteria=None)
    assert d1 == d2 == tmp_path / "t" / "abc123abc123" and (rep1, rep2) == (False, True)
    assert (d2 / "results.json").read_bytes() == b1
    for name in ("config.yaml", "record.json", "results.json", "per_pass.csv", "per_tree.csv", "report.md"):
        assert (d1 / name).exists()
    assert str(tmp_path).encode() not in b1 and b"/Users/" not in b1 and b"/home/" not in b1
    bad = copy.deepcopy(res)
    bad["runs"][0]["rungs"]["run"]["y6"]["row"]["U"] += 1
    with pytest.raises(ReproducibilityError):
        write_experiment(tmp_path, cfg=cfg, results=bad, record={}, criteria=None)


def test_report_null_cells(tmp_path):
    _, res = _results(write_run(tmp_path, crop="apple", pitches=(30.0, 80.0)))
    lines = render_report(res, criteria=CRIT).splitlines()
    assert lines[0] == "# Y1·Y6 합성 채점 — t (abc123abc123)"
    assert lines[1] == "> 합성 렌더 · 정답 자세 · 보정 전(U). 실제 정확도가 아니다."
    assert "> dev 묶음: 개발용 수치다. 판정에 쓰지 않는다." in lines
    assert "| 80 | A | 2 | 31 | 0 | 0 | - | 0 | 0 | 0 | 0 |" in lines
    passes = res["runs"][0]["rungs"]["run"]["y1"]["passes"]
    assert [p["depth_err"] is None for p in passes] == [False, False, True, True]   # no detection at 80 deg
    e30 = [abs(p["e1"]) for p in passes if p["pitch"] == 30.0]
    assert "| 기울기 | 패스 | e1 중앙값 | e1 최대 |" in lines
    assert f"| 30 | 2 | {float(np.median(e30)):+.1%} | {max(e30):+.1%} |" in lines
    assert "| 80 | 2 | - | - |" in lines


def test_pitch_rows_use_abs_e1():
    passes = [{"pitch": 30.0, "e1": -0.04}, {"pitch": 30.0, "e1": 0.02}]
    assert _pitch_rows(passes) == [["30", "2", "+3.0%", "+4.0%"]]     # median and max of |e1|


def test_report_depth_err_line(leaf_runs, l1_l3_report):
    _, res = _results(leaf_runs[1])
    s = res["runs"][0]["rungs"]["run"]["y1"]["summary"]["depth_err"]
    text = render_report(res, criteria=None)
    assert (f"- 깊이 오차(상자 깊이 - 검출 깊이): n {s['n']}, 절댓값 p50 {s['p50_m']:.3f} m, "
            f"p90 {s['p90_m']:.3f} m, p95 {s['p95_m']:.3f} m, 10 cm 넘게 가까움 {s['near_10cm']:+.1%}, "
            f"멂 {s['far_10cm']:+.1%}") in text
    assert l1_l3_report[1].count("- 깊이 오차(") == 1          # L1 (det_depth) has no line, L3 has one
    assert SCHEMA_VERSION == 3


def test_contribution_row_names(apple_run):
    cfg = fake_cfg(apple_run, "apple")
    ctx = prepare(SynthRun.open(apple_run), cfg, load_splits())
    out = run_ladder(ctx, cfg, load_ladder(ROOT / "configs" / "eval" / "sweeps" / "ladder.yaml"), ["L0", "L1", "L3"])
    res = {"schema_version": SCHEMA_VERSION, "kind": "ladder", "run_name": "t", "experiment_id": "abc123abc123",
           "split": "dev", "crop": "apple", "runs": [{"meta": describe_run(ctx), **out}]}
    text = render_report(res, criteria=None)
    assert "| L0→L1 (추적) |" in text and "| L1→L3 |" in text


def test_contrib_table_has_e1_max_column(l1_l3_report):
    out, text = l1_l3_report
    a, b = (out["rungs"][n]["y1"]["summary"]["e1_max"] for n in ("L1", "L3"))
    assert "| 구간 | Δ Y1 e1 중앙값 | Δ Y1 e1 최대 | Δ 줄 e_pipe | Δ 그루 MAE÷평균 |" in text
    row = next(line for line in text.splitlines() if line.startswith("| L1→L3 |"))
    assert row.split("|")[3].strip() == f"{b - a:+.1%}"


def test_criteria_rows_compare():
    rung = {"y1": {"summary": {"e1_median": 0.03, "fp_off_rate": 0.0}},
            "y6": {"metrics": {"row_abs_e_pipe": 0.06}}}
    rows = {r["metric"]: r for r in criteria_rows(rung, CRIT, "L3", "apple")}
    assert rows["y1.e1_median"]["ok"] is True and rows["y6.row_abs_e_pipe"]["ok"] is False
    assert rows["y6.row_abs_e_pipe"]["limit"] == 0.05 and rows["y6.miss_rate"]["ok"] is None


def test_row_sum_line_shows_naive_count_and_multiple(apple_run):
    cfg, res = _results(apple_run)
    row = res["runs"][0]["rungs"]["run"]["y6"]["row"]
    text = render_report(res, criteria=None)
    m = re.search(r"naive (\d+) \((-|\d+\.\d\d배)\)", text)
    assert m is not None, text
    assert int(m.group(1)) == row["naive_U"]
    expected = "-" if row["naive_ratio"] is None else f"{row['naive_ratio']:.2f}배"
    assert m.group(2) == expected
    # the old "signed percent of a multiple" rendering must be gone
    assert "naive +" not in text and "naive -" not in text


@pytest.fixture(scope="module")
def l1_l3_report(apple_run):
    """(ladder output, report.md text) for rungs L1 and L3 with the 2 x 2 table, against the repo criteria."""
    cfg = fake_cfg(apple_run, "apple")
    ctx = prepare(SynthRun.open(apple_run), cfg, load_splits())
    out = run_ladder(ctx, cfg, load_ladder(ROOT / "configs" / "eval" / "sweeps" / "ladder.yaml"),
                     ["L1", "L3"], with_2x2=True)
    res = {"schema_version": SCHEMA_VERSION, "kind": "ladder", "run_name": "t", "experiment_id": "abc123abc123",
           "split": "dev", "crop": "apple", "runs": [{"meta": describe_run(ctx), **out}]}
    return out, render_report(res, criteria=CRIT)


def test_criteria_table_once_after_contrib_and_2x2(l1_l3_report):
    _, text = l1_l3_report
    heading = "**기준과 비교(가정값, 참고)**"
    assert text.count(heading) == 1
    i_contrib, i_2x2, i_crit = text.index("**기여**"), text.index("**2 × 2**"), text.index(heading)
    assert i_contrib < i_2x2 < i_crit
    assert "| L1 | y1." in text and "| L3 | y1." in text


def test_criteria_table_formats_ratios_as_percent(l1_l3_report):
    """Ratio values and limits use the report's percent format; metres keep 3 decimals."""
    out, text = l1_l3_report
    cells = {tuple(c.strip() for c in line.split("|")[1:3]): [c.strip() for c in line.split("|")[3:6]]
             for line in text.splitlines() if re.match(r"\| L[13] \| y[16]\.", line)}
    e1 = out["rungs"]["L1"]["y1"]["summary"]["e1_median"]
    assert cells[("L1", "y1.e1_median")][:2] == [f"{e1:+.1%}", "≤+2.0%"]
    assert f"e1 중앙값 {e1:+.1%}" in text   # the same number as in the Y1 summary line
    p90 = out["rungs"]["L3"]["y6"]["metrics"]["pos_err_p90_m"]
    assert cells[("L3", "y6.pos_err_p90_m")][:2] == ["-" if p90 is None else f"{p90:.3f}", "≤0.040"]
    for (rung, metric), (value, limit, _) in cells.items():
        if not metric.endswith("_m"):
            assert re.fullmatch(r"-|[+-]\d+\.\d%", value) and re.fullmatch(r"[≤≥][+-]\d+\.\d%", limit), (rung, metric)


def test_stop_after_y1_skips_y6_sections_and_criteria_rows(apple_run):
    # A rung run under stop_after=y1 has rung["y6"] None; render_report, the
    # per_tree.csv writer and criteria_rows must all skip the Y6 content for it.
    cfg = fake_cfg(apple_run, "apple", "stop_after=y1")
    ctx = prepare(SynthRun.open(apple_run), cfg, load_splits())
    out = run_ladder(ctx, cfg, load_ladder(ROOT / "configs" / "eval" / "sweeps" / "ladder.yaml"), ["L3"])
    rung = out["rungs"]["L3"]
    assert rung["y6"] is None

    rows = criteria_rows(rung, CRIT, "L3", "apple")
    assert rows and all(r["metric"].startswith("y1.") for r in rows)

    res = {"schema_version": SCHEMA_VERSION, "kind": "ladder", "run_name": "t", "experiment_id": "abc123abc123",
           "split": "dev", "crop": "apple", "runs": [{"meta": describe_run(ctx), **out}]}
    text = render_report(res, criteria=CRIT)
    assert "**Y6**" not in text and "y6." not in text
