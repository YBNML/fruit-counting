import copy
import csv
import dataclasses
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

import counting.evaluation.report as report_module
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.ledger import Ledger
from counting.evaluation.record import experiment_id, git_state, request_extra
from counting.evaluation.report import (
    SCHEMA_VERSION, ReproducibilityError, _depth_err_line, _pitch_rows, _SUBSET_VERDICT_NOTE, _sweep_rows,
    _verdict_lines, _y1_summary_line, criteria_rows, render_report, results_bytes, write_experiment,
)
from counting.evaluation.runner import describe_run, evaluate, load_ladder, prepare, run_ladder
from counting.evaluation.verdict import (
    POOLED_KEYS, ladder_pooled, ladder_verdicts, pool_rung, verdict_counts, verdict_summary,
)
from tests.unit.synthfake import fake_cfg, load_splits

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


def test_evaluate_bytes_reproducible(leaf_runs, leaf_ctx):
    assert results_bytes(_results(leaf_runs[0])[1]) == results_bytes(_results(leaf_runs[0], ctx=leaf_ctx[0])[1])


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


def test_report_null_cells(pitch80_ctx):
    _, res = _results(pitch80_ctx.run.root, ctx=pitch80_ctx)
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
    assert SCHEMA_VERSION == 8


def test_contribution_row_names(apple_ladder, apple_ladder_ctx):
    out = {"chain": ["L0", "L1", "L3"], "rungs": {n: apple_ladder["rungs"][n] for n in ("L0", "L1", "L3")},
           "interaction_2x2": None}
    res = {"schema_version": SCHEMA_VERSION, "kind": "ladder", "run_name": "t", "experiment_id": "abc123abc123",
           "split": "dev", "crop": "apple", "runs": [{"meta": describe_run(apple_ladder_ctx), **out}]}
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
def l1_l3_report(apple_ladder, apple_ladder_ctx):
    """(ladder output, report.md text) for rungs L1 and L3 with the 2 x 2 table, against the repo criteria."""
    out = {"chain": ["L1", "L3"], "rungs": {n: apple_ladder["rungs"][n] for n in ("L1", "L3")},
           "interaction_2x2": apple_ladder["interaction_2x2"]}
    res = {"schema_version": SCHEMA_VERSION, "kind": "ladder", "run_name": "t", "experiment_id": "abc123abc123",
           "split": "dev", "crop": "apple", "runs": [{"meta": describe_run(apple_ladder_ctx), **out}]}
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


def test_write_experiment_writes_nothing_when_the_report_fails(tmp_path, monkeypatch):
    def broken(results, *, criteria):
        raise RuntimeError("report failed")

    monkeypatch.setattr(report_module, "render_report", broken)
    with pytest.raises(RuntimeError, match="report failed"):
        write_experiment(tmp_path, cfg=fake_cfg(Path("bundle"), "apple"), results=_sweep_results(None), record={},
                         criteria=None)
    assert list(tmp_path.iterdir()) == []


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


def _L(**kw):
    """A ledger dict with every field 0 but `kw`."""
    return {f.name: 0 for f in dataclasses.fields(Ledger)} | kw


def _view(passes, trees, row_ledger, metrics=None, summary=None):
    """An evaluate-shaped result: `passes` as (e1, ledger), `trees` as (U, U_truth_attr, G6)."""
    return {"y1": {"passes": [{"e1": e1, "ledger": led} for e1, led in passes], "summary": summary or {}},
            "y6": {"trees": [{"U": u, "U_truth_attr": a, "G6": g} for u, a, g in trees], "ledger": row_ledger,
                   "metrics": metrics or {}}}


def _entry(seed, rung_name, rung):
    return {"meta": {"run_id": f"apple_s{seed}_x", "seed": seed}, "rungs": {rung_name: rung}}


TWO = [
    _entry(1000, "L3", _view([(0.10, _L(n_truth=10, dup_track=1)), (-0.04, _L(n_truth=25, miss_det=1)),
                              (0.00, _L(n_truth=5))],
                             [(10, 9, 8), (4, 4, 5)], _L(n_truth=13, miss_det=1, dup_cross=2, fp_off=1),
                             metrics={"row_abs_e_pipe": 0.077, "pos_err_p90_m": 0.03},
                             summary={"e1_median": 0.04, "fp_off_rate": 0.0})),
    _entry(1001, "L3", _view([(0.20, _L(n_truth=20, dup_track=2, fp_off=1, impure=1)), (None, _L())],
                             [(7, 7, 7)], _L(n_truth=7, fp_bg=1),
                             metrics={"row_abs_e_pipe": 0.0, "pos_err_p90_m": 0.05},
                             summary={"e1_median": 0.2, "fp_off_rate": 0.05})),
]


def test_pooled_and_verdicts_on_two_hand_made_bundles():
    pooled = ladder_pooled(TWO)
    assert pooled["L3"]["n_runs"] == 2
    assert pooled["L3"]["y1"] == pytest.approx({"n_passes": 5, "e1_median": 0.07, "e1_max": 0.20, "dup_rate": 0.05,
                                                "miss_rate": 1 / 60, "merge_rate": 1 / 60, "fp_off_rate": 1 / 60})
    assert pooled["L3"]["y6"] == {"n_trees": 3, "tree_mae_over_mean": 0.15, "attribution_error": 0.05,
                                  "miss_rate": 0.05, "excess_rate": 0.2}
    rows = ladder_verdicts(TWO, pooled, CRIT, "apple")["L3"]
    assert [(r["metric"], r["unit"], r["ok"]) for r in rows] == [
        ("y1.e1_median", "pooled", False), ("y1.fp_off_rate", "pooled", True),
        ("y6.row_abs_e_pipe", "scene", False), ("y6.tree_mae_over_mean", "pooled", False),
        ("y6.attribution_error", "pooled", False), ("y6.pos_err_p90_m", "scene", False),
        ("y6.miss_rate", "pooled", True), ("y6.excess_rate", "pooled", False)]
    by = {r["metric"]: r for r in rows}
    assert by["y6.row_abs_e_pipe"]["value"] is None and \
        [(p["seed"], p["value"], p["spread"], p["ok"]) for p in by["y6.row_abs_e_pipe"]["per_run"]] == \
        [(1000, 0.077, None, False), (1001, 0.0, None, True)]
    assert by["y1.e1_median"]["value"] == pytest.approx(0.07) and \
        [p["value"] for p in by["y1.e1_median"]["per_run"]] == [0.04, 0.2]
    assert by["y6.miss_rate"]["per_run"][0]["value"] is None          # not in the hand-made metrics
    assert verdict_counts({"L3": rows}) == (2, 8) and verdict_summary({"L3": rows}) == "2/8"
    assert verdict_summary({}) == verdict_summary(None) == "-"


def _noisy(views, metrics, spread):
    """A noisy rung over noise seeds 0 and 1 whose seed views are `views`; its seed-mean metrics and spread as given."""
    return {"noisy": True, "seeds": [0, 1], "per_seed": [{"noise_seed": s, **v} for s, v in zip((0, 1), views)],
            "y1": {"summary": {}}, "y6": {"metrics": metrics, "row": {"e_pipe": 0.0}},
            "spread": {"y1": {"summary": {}}, "y6": {"metrics": spread, "row": {}}}}


NOISY_TWO = [
    _entry(1000, "L4_base", _noisy([_view([(0.10, _L(n_truth=10, dup_track=1))], [(9, 9, 10)], _L(n_truth=10, miss_det=1)),
                                    _view([(-0.30, _L(n_truth=10))], [(13, 12, 10)], _L(n_truth=10, dup_track=3))],
                                   {"row_abs_e_pipe": 0.2}, {"row_abs_e_pipe": [0.15, 0.25]})),
    _entry(1001, "L4_base", _noisy([_view([(0.00, _L(n_truth=10))], [(10, 10, 10)], _L(n_truth=10)),
                                    _view([(0.10, _L(n_truth=10, miss_det=1))], [(11, 11, 10)], _L(n_truth=10, fp_bg=1))],
                                   {"row_abs_e_pipe": 0.05}, {"row_abs_e_pipe": [0.0, 0.1]})),
]


def test_pooled_noisy_rung_is_the_seed_mean_and_its_verdict_keeps_the_spread():
    pooled = ladder_pooled(NOISY_TWO)["L4_base"]
    assert pooled["y1"] == pytest.approx({"n_passes": 2, "e1_median": 0.125, "e1_max": 0.20, "dup_rate": 0.025,
                                          "miss_rate": 0.025, "merge_rate": 0.0, "fp_off_rate": 0.0})
    assert pooled["y6"] == pytest.approx({"n_trees": 2, "tree_mae_over_mean": 0.125, "attribution_error": 0.025,
                                          "miss_rate": 0.025, "excess_rate": 0.1})
    (row,) = ladder_verdicts(NOISY_TWO, {"L4_base": pooled}, CRIT, "apple")["L4_base"]
    assert (row["metric"], row["unit"], row["limit"], row["ok"]) == ("y6.row_abs_e_pipe", "scene", 0.10, False)
    assert [(p["value"], p["spread"], p["ok"]) for p in row["per_run"]] == [(0.2, [0.15, 0.25], False),
                                                                          (0.05, [0.0, 0.1], True)]
    for p, e in zip(row["per_run"], NOISY_TWO):            # a copy: editing a row leaves the rung's spread alone
        assert p["spread"] is not e["rungs"]["L4_base"]["spread"]["y6"]["metrics"]["row_abs_e_pipe"]
    with pytest.raises(ValueError, match="bundles disagree on the noise seeds"):
        pool_rung([NOISY_TWO[0]["rungs"]["L4_base"], {**NOISY_TWO[1]["rungs"]["L4_base"], "seeds": [0, 2]}])


def test_pooled_of_one_bundle_is_its_own_summary(apple_ladder):
    for name in ("L1", "L3", "L4_base"):
        rung = apple_ladder["rungs"][name]
        pooled = pool_rung([rung])
        assert pooled["y1"] == {"n_passes": len(rung["y1"]["passes"]),
                                **{k: rung["y1"]["summary"][k] for k in POOLED_KEYS["y1"]}}, name
        assert pooled["y6"] == {"n_trees": len(rung["y6"]["trees"]),
                                **{k: rung["y6"]["metrics"][k] for k in POOLED_KEYS["y6"]}}, name


def test_pooled_over_two_fake_bundles_sums_trees_and_ledgers(apple_ladder, noisy_ladder):
    a, b = apple_ladder["rungs"]["L3"], noisy_ladder[0]["runs"][0]["rungs"]["L3"]
    pooled = pool_rung([a, b])
    trees = a["y6"]["trees"] + b["y6"]["trees"]
    g6 = sum(t["G6"] for t in trees)
    assert pooled["y6"]["tree_mae_over_mean"] == pytest.approx(sum(abs(t["U"] - t["G6"]) for t in trees) / g6)
    rows = [a["y6"]["ledger"], b["y6"]["ledger"]]
    excess = sum(r["dup_track"] + r["dup_cross"] + r["fp_off"] + r["fp_bg"] for r in rows)
    assert pooled["y6"]["excess_rate"] == pytest.approx(excess / sum(r["n_truth"] for r in rows))
    e1s = sorted(abs(p["e1"]) for r in (a, b) for p in r["y1"]["passes"] if p["e1"] is not None)
    assert pooled["y1"]["e1_max"] == e1s[-1] and pooled["y1"]["n_passes"] == len(a["y1"]["passes"]) + len(b["y1"]["passes"])


def test_verdicts_skip_y6_under_stop_after_y1_and_count_missing_values_as_not_passed():
    y1_only = [{**e, "rungs": {"L3": {**e["rungs"]["L3"], "y6": None}}} for e in TWO]
    assert ladder_pooled(y1_only)["L3"]["y6"] is None
    rows = ladder_verdicts(y1_only, ladder_pooled(y1_only), CRIT, "apple")["L3"]
    assert [r["metric"] for r in rows] == ["y1.e1_median", "y1.fp_off_rate"]
    empty = [_entry(1000, "L3", _view([(None, _L())], [(0, 0, 0)], _L()))]
    rows = ladder_verdicts(empty, ladder_pooled(empty), CRIT, "apple")["L3"]
    assert all(r["ok"] is None for r in rows) and verdict_counts({"L3": rows}) == (0, len(rows)) == (0, 8)
    assert ladder_verdicts(TWO, ladder_pooled(TWO), CRIT, "pear").keys() == {"L3"}
    assert ladder_verdicts(TWO, ladder_pooled(TWO), CRIT, "plum") == {}
    assert ladder_verdicts(TWO, ladder_pooled(TWO), {"y1": {}}, "apple") == {}

    def p90_row(values):
        """The y6.pos_err_p90_m verdict row of TWO with the two scenes' p90 set to `values`."""
        two = [{**e, "rungs": {"L3": {**e["rungs"]["L3"], "y6": {**e["rungs"]["L3"]["y6"], "metrics": {
            **e["rungs"]["L3"]["y6"]["metrics"], "pos_err_p90_m": v}}}}} for e, v in zip(TWO, values)]
        return next(r for r in ladder_verdicts(two, ladder_pooled(two), CRIT, "apple")["L3"]
                    if r["metric"] == "y6.pos_err_p90_m")

    assert [p90_row(v)["ok"] for v in ((None, 0.05), (None, 0.03), (0.03, 0.03))] == [False, None, True]


def test_criteria_sha_goes_into_the_ladder_and_sweep_ids():
    kw = dict(config_hash="c" * 16, fingerprints=["f1"], code=CODE, seed=0)
    lad = dict(ladder_sha256="1" * 64, rungs=["L1", "L3"], with_2x2=True)
    a, b = (request_extra("ladder", **lad, criteria_sha256=s) for s in ("a" * 64, "b" * 64))
    assert a == f"ladder:{'1' * 16}:L1,L3:True:criteria:{'a' * 16}"
    assert experiment_id(**kw, extra=a) != experiment_id(**kw, extra=b)
    assert request_extra("ladder", **lad, subsets=True) == f"ladder:{'1' * 16}:L1,L3:True:subsets:criteria:-"
    sw = dict(grid_name="g.yaml", grid_sha256="2" * 64)
    assert request_extra("sweep", **sw, criteria_sha256="a" * 64) == f"sweep:g.yaml:{'2' * 16}:-:criteria:{'a' * 16}"
    assert request_extra("sweep", **sw, ladder_sha256="1" * 64) == f"sweep:g.yaml:{'2' * 16}:{'1' * 16}:criteria:-"
    assert request_extra("run", criteria_sha256="a" * 64) == ""


def test_verdict_lines_name_unit_values_spread_and_count():
    res = {"split": "test", "runs": [*TWO], "verdicts": {}}
    res["verdicts"] = ladder_verdicts(TWO, ladder_pooled(TWO), CRIT, "apple")
    lines = _verdict_lines(res)
    assert lines[:3] == ["**판정 (test, 장면 시드 2개)**", "| 칸 | 지표 | 단위 | 합친 값 | 장면 시드별 | 기준 | 판정 |",
                         "|---|---|---|---|---|---|---|"]
    assert "| L3 | y1.e1_median | 합침 | +7.0% | s1000 +4.0%, s1001 +20.0% | ≤+4.0% | FAIL |" in lines
    assert "| L3 | y6.row_abs_e_pipe | 장면마다 | - | s1000 +7.7%, s1001 +0.0% | ≤+5.0% | FAIL |" in lines
    assert "| L3 | y6.pos_err_p90_m | 장면마다 | - | s1000 0.030, s1001 0.050 | ≤0.040 | FAIL |" in lines
    assert "| L3 | y6.miss_rate | 합침 | +5.0% | s1000 -, s1001 - | ≤+5.0% | OK |" in lines
    assert lines[-2:] == ["- 통과 2 / 항목 8 (값이 없는 항목은 통과로 세지 않는다)", ""]
    noisy = {"split": "test", "runs": NOISY_TWO,
             "verdicts": ladder_verdicts(NOISY_TWO, ladder_pooled(NOISY_TWO), CRIT, "apple")}
    assert "| L4_base | y6.row_abs_e_pipe | 장면마다 | - | s1000 +20.0% (+15.0%~+25.0%), s1001 +5.0% (+0.0%~+10.0%) " \
           "| ≤+10.0% | FAIL |" in _verdict_lines(noisy)


def test_verdict_lines_puts_a_blank_line_between_the_subset_note_and_the_table():
    row = {"metric": "y6.row_abs_e_pipe", "unit": "scene", "value": None, "op": "le", "limit": 0.05, "ok": True,
           "per_run": [{"seed": 1000, "value": 0.05, "spread": None, "ok": True}]}
    res = {"split": "test", "runs": [{"meta": {"run_id": "a", "seed": 1000}, "rungs": {}}],
           "verdicts": {"L5[A]": [row]}}
    lines = _verdict_lines(res)
    assert lines[0] == "**판정 (test, 장면 시드 1개)**"
    assert lines[1] == _SUBSET_VERDICT_NOTE
    assert lines[2] == ""          # a blank line, or CommonMark reads the table as the note's lazy continuation
    assert lines[3] == "| 칸 | 지표 | 단위 | 합친 값 | 장면 시드별 | 기준 | 판정 |"
    assert lines[4] == "|---|---|---|---|---|---|---|"
    assert "| L5[A] | y6.row_abs_e_pipe | 장면마다 | - | s1000 +5.0% | ≤+5.0% | OK |" in lines


def test_report_draws_the_verdict_table_for_test_runs_only(apple_ladder, apple_ladder_ctx):
    entries = [{"meta": describe_run(apple_ladder_ctx), **apple_ladder}]
    res = {"schema_version": SCHEMA_VERSION, "kind": "ladder", "run_name": "t", "experiment_id": "abc123abc123",
           "split": "test", "crop": "apple", "runs": entries, "pooled": ladder_pooled(entries)}
    res["verdicts"] = ladder_verdicts(entries, res["pooled"], CRIT, "apple")
    assert list(res["verdicts"]) == ["L1", "L3", "L4_base"]
    text = render_report(res, criteria=CRIT)
    assert text.endswith("\n".join(_verdict_lines(res)) + "\n")
    assert text.index("**기준과 비교(가정값, 참고)**") < text.index("**판정 (test, 장면 시드 1개)**")
    for split in ("dev", "val"):
        assert "**판정" not in render_report({**res, "split": split}, criteria=CRIT)
