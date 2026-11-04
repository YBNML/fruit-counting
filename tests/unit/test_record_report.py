import copy
import csv
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.record import experiment_id, git_state
from counting.evaluation.report import (
    SCHEMA_VERSION, ReproducibilityError, _depth_err_line, _pitch_rows, _sweep_rows, _y1_summary_line,
    criteria_rows, render_report, results_bytes, write_experiment,
)
from counting.evaluation.runner import describe_run, evaluate, load_ladder, prepare, run_ladder
from tests.unit.synthfake import fake_cfg, load_splits, write_run

ROOT = Path(__file__).resolve().parents[2]
CRIT = yaml.safe_load((ROOT / "configs" / "eval" / "criteria.yaml").read_text(encoding="utf-8"))
LADDER = load_ladder(ROOT / "configs" / "eval" / "sweeps" / "ladder.yaml")
CODE = {"commit": "a" * 40, "dirty": False, "diff_sha256": None}


def _results(root, crop="apple", ctx=None):
    cfg = fake_cfg(root, crop)
    if ctx is None:
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


def test_evaluate_bytes_reproducible(apple_run, ctx_of):
    assert results_bytes(_results(apple_run)[1]) == results_bytes(_results(apple_run, ctx=ctx_of["apple"])[1])


def test_write_experiment_reproducible_and_guarded(apple_run, ctx_of, tmp_path):
    cfg, res = _results(apple_run, ctx=ctx_of["apple"])
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
    assert SCHEMA_VERSION == 4


def test_contribution_row_names(apple_ladder, ctx_of):
    out = {"chain": ["L0", "L1", "L3"], "rungs": {n: apple_ladder["rungs"][n] for n in ("L0", "L1", "L3")},
           "interaction_2x2": None}
    res = {"schema_version": SCHEMA_VERSION, "kind": "ladder", "run_name": "t", "experiment_id": "abc123abc123",
           "split": "dev", "crop": "apple", "runs": [{"meta": describe_run(ctx_of["apple"]), **out}]}
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


def test_row_sum_line_shows_naive_count_and_multiple(apple_run, ctx_of):
    cfg, res = _results(apple_run, ctx=ctx_of["apple"])
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
def l1_l3_report(apple_ladder, ctx_of):
    """(ladder output, report.md text) for rungs L1 and L3 with the 2 x 2 table, against the repo criteria."""
    out = {"chain": ["L1", "L3"], "rungs": {n: apple_ladder["rungs"][n] for n in ("L1", "L3")},
           "interaction_2x2": apple_ladder["interaction_2x2"]}
    res = {"schema_version": SCHEMA_VERSION, "kind": "ladder", "run_name": "t", "experiment_id": "abc123abc123",
           "split": "dev", "crop": "apple", "runs": [{"meta": describe_run(ctx_of["apple"]), **out}]}
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


def test_stop_after_y1_skips_y6_sections_and_criteria_rows(apple_run, ctx_of):
    # A rung run under stop_after=y1 has rung["y6"] None; render_report, the
    # per_tree.csv writer and criteria_rows must all skip the Y6 content for it.
    cfg = fake_cfg(apple_run, "apple", "stop_after=y1")
    ctx = ctx_of["apple"]
    out = run_ladder(ctx, cfg, load_ladder(ROOT / "configs" / "eval" / "sweeps" / "ladder.yaml"), ["L3"])
    rung = out["rungs"]["L3"]
    assert rung["y6"] is None

    rows = criteria_rows(rung, CRIT, "L3", "apple")
    assert rows and all(r["metric"].startswith("y1.") for r in rows)

    res = {"schema_version": SCHEMA_VERSION, "kind": "ladder", "run_name": "t", "experiment_id": "abc123abc123",
           "split": "dev", "crop": "apple", "runs": [{"meta": describe_run(ctx), **out}]}
    text = render_report(res, criteria=CRIT)
    assert "**Y6**" not in text and "y6." not in text


META = {"run_id": "r1", "crop": "apple", "seed": 5, "n_frames": 4, "fingerprint": "f" * 64, "synth_commit": None}


def _sweep_results(y6):
    cell = {"point": "baseline", "base": "L3", "overridden": [], "stages": {"dedup": "merge"},
            "lift_depth": "box_median", "y6": y6,
            "y1": {"summary": {"e1_median": 0.01, "e1_max": 0.02, "dup_rate": 0.0, "miss_rate": 0.0,
                               "merge_rate": 0.0, "fp_off_rate": 0.0, "naive_U": 3, "depth_err": None},
                   "passes": []}}
    point = copy.deepcopy({**cell, "point": "tracker.gate_m=0.1"})
    return {"schema_version": SCHEMA_VERSION, "kind": "sweep", "run_name": "t", "experiment_id": "abc123abc123",
            "split": "dev", "crop": "apple",
            "sweep": {"grid": "g.yaml", "bases": ["L3"], "points": [
                {"name": "baseline", "patch": {}, "factor": None, "value": None},
                {"name": "tracker.gate_m=0.1", "patch": {"tracker": {"gate_m": 0.1}}, "factor": "tracker.gate_m",
                 "value": 0.1}]},
            "runs": [{"meta": META, "cells": [copy.deepcopy(cell), point]}]}


def test_sweep_rows_and_report_without_y6(tmp_path):
    res = _sweep_results(None)
    folder, _ = write_experiment(tmp_path, cfg=fake_cfg(Path("bundle"), "apple"), results=res, record={}, criteria=None)
    rows = list(csv.reader((folder / "sweep.csv").read_text(encoding="utf-8").splitlines()))
    assert [r[2] for r in rows[1:]] == ["baseline", "tracker.gate_m=0.1"] and all(r[14:] == [""] * 18 for r in rows[1:])
    assert "| baseline | L3 | - | +1.0% | +2.0% | - | - | - | - | - |" in render_report(res, criteria=None).splitlines()
    lines = render_report(res, criteria=CRIT).splitlines()
    assert "| baseline | L3 | - | +1.0% | +2.0% | - | - | - | - | 2/2 |" in lines and "- L3 · tracker.gate_m: -" in lines


@pytest.fixture(scope="module")
def noisy_ladder(leaf_ctx, leaf_runs):
    """(results, report text) of rungs L3 and L4_base with noise seeds 0 and 1 on the clean leaf bundle."""
    out = run_ladder(leaf_ctx[0], fake_cfg(leaf_runs[0], "apple", "noise.seeds=[0,1]"), LADDER, ["L3", "L4_base"])
    res = {"schema_version": SCHEMA_VERSION, "kind": "ladder", "run_name": "t", "experiment_id": "abc123abc123",
           "split": "dev", "crop": "apple", "runs": [{"meta": describe_run(leaf_ctx[0]), **out}]}
    return res, render_report(res, criteria=CRIT)


def test_report_head_line_follows_the_pose_noise(noisy_ladder):
    res, text = noisy_ladder
    assert text.splitlines()[1] == "> 합성 렌더 · 자세 잡음 칸 있음(칸마다 적음) · 보정 전(U). 실제 정확도가 아니다."
    one = copy.deepcopy(res)
    one["kind"], one["runs"][0]["chain"] = "run", []
    one["runs"][0]["rungs"] = {"run": one["runs"][0]["rungs"]["L4_base"]}
    assert render_report(one, criteria=None).splitlines()[1] == \
        "> 합성 렌더 · 자세 잡음 σ_t 2 cm, σ_r 0.5°, b 2 cm · 보정 전(U). 실제 정확도가 아니다."


def test_report_noisy_rung_section_and_contrib_row(noisy_ladder):
    res, text = noisy_ladder
    l4, lines = res["runs"][0]["rungs"]["L4_base"], text.splitlines()
    assert "- 잡음(시드 0, 1의 평균): σ_t 2 cm, σ_r 0.5°, b 2 cm, σ_rel 2%, 결측 5%, 검출 없음" in lines
    assert "**잡음 시드별 (L4_base)**" in lines
    assert "| 시드 | e1 중앙값 | e1 최대 | U | G6 | e_pipe | e_pipe 절댓값 | 그루 MAE÷평균 |" in lines
    for p in l4["per_seed"]:
        s, row, m = p["y1"]["summary"], p["y6"]["row"], p["y6"]["metrics"]
        assert (f"| {p['noise_seed']} | {s['e1_median']:+.1%} | {s['e1_max']:+.1%} | {row['U']} | {row['G6']} | "
                f"{row['e_pipe']:+.1%} | {m['row_abs_e_pipe']:+.1%} | {m['tree_mae_over_mean']:+.1%} |") in lines
    assert any(line.startswith("| 평균 |") for line in lines)
    U = l4["y6"]["row"]["U"]
    assert any(line.startswith(f"- 줄 합: U {U if isinstance(U, int) else f'{U:.1f}'}, ") for line in lines)
    assert any(line.startswith("| L3→L4_base (자세·깊이 잡음(기준 조합)) |") for line in lines)
    assert any(re.match(r"\| L4_base \| y6\.row_abs_e_pipe \| ", line) for line in lines)
    s = l4["y1"]["summary"]                                   # one pitch: its row is the summary's seed mean
    n = len(l4["y1"]["passes"])
    assert f"| 30 | {n} | {s['e1_median']:+.1%} | {s['e1_max']:+.1%} |" in lines
    third = {"e1_median": 0.1, "e1_max": 0.2, "dup_rate": 0.0, "miss_rate": 0.0, "merge_rate": 0.0,
             "fp_off_rate": 0.0, "naive_U": 64 / 3}
    assert _y1_summary_line(third).endswith(", naive U 21.3")
    assert _depth_err_line({"n": 10 / 3, "p50_m": 0.01, "p90_m": 0.02, "p95_m": 0.03, "near_10cm": 0.0,
                            "far_10cm": 0.0}).startswith("- 깊이 오차(상자 깊이 - 검출 깊이): n 3.3, ")


def test_per_pass_and_per_tree_rows_per_seed_then_mean(noisy_ladder, tmp_path):
    res, _ = noisy_ladder
    folder, _ = write_experiment(tmp_path, cfg=fake_cfg(Path("bundle"), "apple"), results=res, record={}, criteria=None)
    rows = list(csv.reader((folder / "per_pass.csv").read_text(encoding="utf-8").splitlines()))
    n = len(res["runs"][0]["rungs"]["L3"]["y1"]["passes"])
    assert rows[0][:4] == ["run_id", "rung", "noise_seed", "pitch"]
    assert [(r[1], r[2]) for r in rows[1:]] == [("L3", "")] * n + [("L4_base", "0")] * n + \
        [("L4_base", "1")] * n + [("L4_base", "mean")] * n
    trees = list(csv.reader((folder / "per_tree.csv").read_text(encoding="utf-8").splitlines()))
    assert trees[0][:4] == ["run_id", "rung", "noise_seed", "tree"] and {r[2] for r in trees[1:]} == {"", "0", "1", "mean"}


def test_sweep_rows_per_seed_then_mean():
    res = _sweep_results(None)
    cell = res["runs"][0]["cells"][1]
    per = [copy.deepcopy(cell["y1"]) for _ in range(2)]
    per[1]["summary"]["e1_median"] = 0.03
    pose = {"pose_trans_sigma_m": 0.02, "pose_rot_sigma_deg": 0.0, "pass_bias_m": 0.0, "depth_rel_sigma": 0.0,
            "depth_dropout": 0.0, "detector": None}
    cell.update(noisy=True, seeds=[0, 1], noise=pose, spread={},
                per_seed=[{"noise_seed": s, "y1": y1, "y6": None} for s, y1 in zip((0, 1), per)])
    cell["y1"]["summary"]["e1_median"] = 0.02
    assert [(r[2], r[5], r[8]) for r in _sweep_rows(res)] == [
        ("baseline", "", 0.01), ("tracker.gate_m=0.1", 0, 0.01), ("tracker.gate_m=0.1", 1, 0.03),
        ("tracker.gate_m=0.1", "mean", 0.02)]
    assert render_report(res, criteria=None).splitlines()[1] == \
        "> 합성 렌더 · 자세 잡음 점 있음(점마다 다름, sweep.csv의 patch 열) · 보정 전(U). 실제 정확도가 아니다."


def test_report_head_names_the_base_when_pose_noise_comes_from_it():
    """dedup_modes.yaml-style sweep: bases L3 and L4_base, the same point on both, with no noise key in
    any point's patch. The noise comes from the L4_base rung itself, so the head must not say it varies
    per point."""
    base_cell = {"point": "baseline", "base": "L3", "overridden": [], "stages": {"dedup": "merge"},
                 "lift_depth": "box_median", "y6": None,
                 "y1": {"summary": {"e1_median": 0.01, "e1_max": 0.02, "dup_rate": 0.0, "miss_rate": 0.0,
                                    "merge_rate": 0.0, "fp_off_rate": 0.0, "naive_U": 3, "depth_err": None},
                        "passes": []}}
    noisy_cell = copy.deepcopy(base_cell)
    pose = {"pose_trans_sigma_m": 0.02, "pose_rot_sigma_deg": 0.5, "pass_bias_m": 0.02, "depth_rel_sigma": 0.0,
            "depth_dropout": 0.0, "detector": None}
    noisy_cell.update(base="L4_base", noisy=True, seeds=[0], noise=pose, spread={},
                      per_seed=[{"noise_seed": 0, "y1": noisy_cell["y1"], "y6": None}])
    res = {"schema_version": SCHEMA_VERSION, "kind": "sweep", "run_name": "t", "experiment_id": "abc123abc123",
           "split": "dev", "crop": "apple",
           "sweep": {"grid": "dedup_modes.yaml", "bases": ["L3", "L4_base"],
                     "points": [{"name": "baseline", "patch": {}, "factor": None, "value": None}]},
           "runs": [{"meta": META, "cells": [base_cell, noisy_cell]}]}
    assert render_report(res, criteria=None).splitlines()[1] == \
        "> 합성 렌더 · 자세 잡음 칸 있음(sweep.csv의 base·patch 열) · 보정 전(U). 실제 정확도가 아니다."
