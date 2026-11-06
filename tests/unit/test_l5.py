import copy
import dataclasses
import json
import re
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

import counting.evaluation.runner as runner_module
import counting.evaluation.sweep as sweep_module
from counting.data.formats.orchard_synth import SynthRun
from counting.detection.cached import cache_digest, cache_path
from counting.evaluation.budget import budget_rows, check_budget, contrib_pairs, ladder_budget
from counting.evaluation.cli import eval_app
from counting.evaluation.detmetrics import detection_rows
from counting.evaluation.ledger import Ledger
from counting.evaluation.report import SCHEMA_VERSION, _contrib_table, criteria_rows, render_report
from counting.evaluation.runner import (
    check_verdict_request, check_verdict_subsets, describe_run, evaluate, evaluate_seeds, load_ladder, prepare,
    rung_config, run_ladder,
)
from counting.evaluation.sweep import Grid, run_sweep, sweep_cells
from counting.evaluation.verdict import ladder_pooled, ladder_verdicts, subset_verdicts
from tests.unit.cachefake import write_truth_cache
from tests.unit.synthfake import fake_cfg, load_splits

ROOT = Path(__file__).resolve().parents[2]
L5LADDER = load_ladder(ROOT / "configs" / "eval" / "sweeps" / "ladder_l5.yaml")
STEP = "passes.step_m=0.45"          # 22 of the clean leaf bundle's 62 frames, both passes kept
SPLITS = str(ROOT / "configs" / "eval" / "splits.yaml")
CRIT_FILE = str(ROOT / "configs" / "eval" / "criteria.yaml")
CRIT = yaml.safe_load((ROOT / "configs" / "eval" / "criteria.yaml").read_text(encoding="utf-8"))


def l5cfg(root, cache_dir, *sets):
    """The apple dev config on `root`, thinned, with det=oracle and `cache_dir` as the cache folder for rung L5."""
    return fake_cfg(root, "apple", STEP, f"detector.cache={cache_dir}", *sets)


@cache
def _l5_once(root: str, cache_dir: str, sets: tuple[str, ...]):
    cfg = rung_config(l5cfg(Path(root), cache_dir, *sets), L5LADDER, "L5")
    return prepare(SynthRun.open(Path(root)), cfg, load_splits()), cfg


def _l5(root, cache_dir, *sets):
    """(context, config) of rung L5 on `root` with `cache_dir`; prepared once per arguments (contexts are read-only)."""
    return _l5_once(str(root), str(cache_dir), tuple(sets))


def _counts(res):
    """Everything `evaluate` scores but the box-depth diagnostics."""
    return ([{k: v for k, v in p.items() if k != "depth_err"} for p in res["y1"]["passes"]],
            {k: v for k, v in res["y1"]["summary"].items() if k != "depth_err"}, res["y6"])


@pytest.fixture(scope="module")
def caches(leaf_runs, tmp_path_factory):
    """{name: folder} of caches of the clean leaf bundle: `perfect`, `shifted` (moved left by 0.2 x width),
    `low` (`shifted` plus 0.07 copies moved right by 0.15 x width) and `copies` (the 0.07 copies alone)."""
    specs = {"perfect": {}, "shifted": {"shift": -0.2}, "low": {"shift": -0.2, "low_score": 0.07},
             "copies": {"low_score": 0.07, "main": False}}
    out = {}
    for name, kw in specs.items():
        out[name] = tmp_path_factory.mktemp(f"cache_{name}")
        write_truth_cache(leaf_runs[0], out[name], **kw)
    return out


def test_l5_ladder_extends_the_oracle_ladder():
    oracle = load_ladder(ROOT / "configs" / "eval" / "sweeps" / "ladder.yaml")
    assert list(L5LADDER["rungs"]) == [*oracle["rungs"], "L5"]
    assert list(L5LADDER["baselines"]) == [*oracle["baselines"], "L5_L4base"]
    assert {k: v for k, v in L5LADDER["rungs"].items() if k != "L5"} == dict(oracle["rungs"])
    assert {k: v for k, v in L5LADDER["baselines"].items() if k != "L5_L4base"} == dict(oracle["baselines"])
    assert L5LADDER["interaction_2x2"] == oracle["interaction_2x2"]
    l5 = {"stages": {"det": "cached", "track": "landmark", "center": "measured", "assign": "grid"}}
    assert L5LADDER["rungs"]["L5"] == l5
    assert L5LADDER["baselines"]["L5_L4base"] == {**l5, "noise": oracle["rungs"]["L4_base"]["noise"]}


def test_perfect_cache_L5_equals_L3(leaf_runs, caches):
    root, post = leaf_runs[0], ("detector.post.nms_iou=1.0", "detector.post.edge_snap_px=0")
    base = l5cfg(root, caches["perfect"], *post)
    l3 = evaluate(prepare(SynthRun.open(root), base, load_splits()), rung_config(base, L5LADDER, "L3"))
    l5 = evaluate(*_l5(root, caches["perfect"], *post))
    assert _counts(l5) == _counts(l3) and (l3["y6"]["row"]["U"], l3["y6"]["row"]["G6"]) == (18, 12)
    assert l5["stages"]["det"] == "cached" and l5["y1"]["summary"]["depth_err"] is None
    run_id = SynthRun.open(root).meta.run_id
    assert l5["cache"] == {"file": f"{run_id}.npz", "digest": cache_digest(cache_path(caches["perfect"], run_id)),
                           "weights_sha256": None, "pretrained": None}


def test_cached_boxes_below_score_lo_never_reach_scoring(leaf_runs, caches):
    root = leaf_runs[0]
    (ca, fa), (cb, fb) = _l5(root, caches["shifted"]), _l5(root, caches["low"])
    a, b = evaluate(ca, fa), evaluate(cb, fb)
    assert _counts(a) == _counts(b) and min(float(d.scores.min()) for d in cb.dets.values() if len(d)) >= 0.1
    L = b["y6"]["ledger"]
    assert (L["n_pred"], L["n_truth"], L["fp_bg"], L["dup_track"], L["dup_cross"], L["miss_drop"],
            L["miss_merge"]) == (28, 12, 0, 4, 12, 0, 0)
    cc, fc = _l5(root, caches["low"], "tracker.score_lo=0.05")      # copies kept, matched after the main boxes
    low = np.concatenate([cc.src[i][d.scores < 0.5] for i, d in cc.dets.items()])
    high = np.concatenate([cc.src[i][d.scores >= 0.5] for i, d in cc.dets.items()])
    assert sum(map(len, cc.dets.values())) == 252 and set(low.tolist()) == {-1}
    assert np.array_equal(high, np.concatenate([cb.src[i] for i in cb.dets])) and _counts(evaluate(cc, fc)) == _counts(b)
    for sets, n_boxes, misses in (((), 0, (12, 0)), (("tracker.score_lo=0.05",), 130, (0, 12))):
        ctx, cfg = _l5(root, caches["copies"], *sets)        # the floor decides what counts as detected
        res = evaluate(ctx, cfg)
        L = res["y6"]["ledger"]
        assert sum(map(len, ctx.dets.values())) == n_boxes and [p["U1"] for p in res["y1"]["passes"]] == [0, 0]
        assert (L["n_pred"], L["miss_det"], L["miss_drop"]) == (0, *misses)


def test_min_obs_cached_changes_cached_rungs_only(leaf_runs, caches):
    root = leaf_runs[0]
    base, tuned = l5cfg(root, caches["shifted"]), l5cfg(root, caches["shifted"], "tracker.min_obs_cached=2")
    ctx = prepare(SynthRun.open(root), base, load_splits())
    assert evaluate(ctx, rung_config(tuned, L5LADDER, "L3")) == evaluate(ctx, rung_config(base, L5LADDER, "L3"))
    got = evaluate(*_l5(root, caches["shifted"], "tracker.min_obs_cached=2"))
    want = evaluate(*_l5(root, caches["shifted"], "tracker.min_obs=2"))
    assert _counts(got) == _counts(want) and got["y6"]["row"]["U"] == 24
    assert [p["U1"] for p in got["y1"]["passes"]] == [12, 12]


def test_cached_context_checks_its_cache(leaf_runs, caches, tmp_path):
    root = leaf_runs[0]
    run = SynthRun.open(root)
    rid = run.meta.run_id
    ctx, c5 = _l5(root, caches["shifted"])
    for key, value, had in (("tracker.score_lo", 0.2, 0.1), ("tracker.score_hi", 0.7, 0.5)):
        with pytest.raises(ValueError, match=f"{key}={value} differs from the prepared {key}={had}"):
            evaluate(ctx, rung_config(l5cfg(root, caches["shifted"], f"{key}={value}"), L5LADDER, "L5"))
    with pytest.raises(ValueError, match=r"detector.post=.* differs from the prepared detector.post="):
        evaluate(ctx, rung_config(l5cfg(root, caches["shifted"], "detector.post.min_short_px=12"), L5LADDER, "L5"))
    with pytest.raises(ValueError, match="stages.det=cached differs from the prepared stages.det=oracle"):
        evaluate(prepare(run, l5cfg(root, caches["shifted"]), load_splits()), c5)
    with pytest.raises(FileNotFoundError, match=re.escape(f"{tmp_path / rid}.npz: no detection cache for this bundle")):
        _l5(root, tmp_path)
    digest = cache_digest(cache_path(caches["shifted"], rid))
    with pytest.raises(ValueError, match=f"cache digest {digest[:16]} differs from the pinned {'0' * 16}"):
        _l5(root, caches["shifted"], f"detector.digests={{{rid}: '{'0' * 64}'}}")
    with pytest.raises(ValueError, match=f"detector.digests pins no cache for run {rid}"):
        _l5(root, caches["shifted"], f"detector.digests={{other: '{digest}'}}")
    assert _l5(root, caches["shifted"], f"detector.digests={{{rid}: '{digest}'}}")[0].cache["digest"] == digest
    for cfg, phrase in ((l5cfg(root, caches["low"]),                       # another folder, the same settings
                         f"detector.cache={caches['low']} differs from the prepared detector.cache={caches['shifted']}"),
                        (l5cfg(root, caches["shifted"], f"detector.digests={{{rid}: '{digest}'}}"),
                         f"detector.digests={{'{rid}': '{digest}'}} differs from the prepared detector.digests=None")):
        with pytest.raises(ValueError, match=re.escape(phrase)):
            evaluate(ctx, rung_config(cfg, L5LADDER, "L5"))
    noisy = evaluate_seeds(ctx, rung_config(l5cfg(root, caches["shifted"], "noise.seeds=[0]"), L5LADDER, "L5_L4base"))
    assert noisy["noisy"] and noisy["cache"] == ctx.cache


@pytest.fixture(scope="module")
def l5_ladder(leaf_runs, caches):
    """(oracle context, base config, `run_ladder` of L0..L3 and L5 on the `low` cache with the pass-subset
    tables, the stages.det of each `context_for` call)."""
    root = leaf_runs[0]
    run, cfg, splits = SynthRun.open(root), l5cfg(root, caches["low"]), load_splits()
    calls = []

    def context_for(c):
        calls.append(c.stages.det)
        return prepare(run, c, splits)

    ctx = prepare(run, cfg, splits)
    out = run_ladder(ctx, cfg, L5LADDER, ["L0", "L1", "L2a", "L2b", "L2c", "L3", "L5"], subsets=True,
                     context_for=context_for)
    return ctx, cfg, out, calls


def test_ladder_runs_L5_on_its_own_context(leaf_runs, caches, l5_ladder, monkeypatch):
    ctx, cfg, out, calls = l5_ladder
    assert calls == ["cached"] and out["chain"] == ["L0", "L1", "L2a", "L2b", "L2c", "L3", "L5"]
    assert out["rungs"]["L5"] == evaluate(*_l5(leaf_runs[0], caches["low"]), subsets=True)
    assert out["rungs"]["L3"] == evaluate(ctx, rung_config(cfg, L5LADDER, "L3"), subsets=True)
    assert [r["S"] for r in out["rungs"]["L5"]["subsets"]] == ["all", "A", "B", "only 30", "drop 30"]
    seen = []
    monkeypatch.setattr(runner_module, "evaluate", lambda *a, **k: seen.append(a[1]) or evaluate(*a, **k))
    with pytest.raises(ValueError, match="stages.det=cached differs from the prepared stages.det=oracle"):
        run_ladder(ctx, cfg, L5LADDER, ["L3", "L5"])
    bad = {"rungs": {**L5LADDER["rungs"], "L5p": {"stages": {"det": "cached"}, "passes": {"pitches": [30.0]}}}}
    with pytest.raises(ValueError, match="passes=.* differs from the prepared passes=.*a rung may change the detector"):
        run_ladder(ctx, cfg, bad, ["L3", "L5p"], context_for=lambda c: pytest.fail("prepared a context"))
    assert seen == []


def test_sweep_cells_on_rung_L5_keep_their_cache(leaf_runs, caches, monkeypatch):
    root = leaf_runs[0]
    grid = Grid(path=Path("g.yaml"), bases=("L5",), factors={}, product={}, combos={}, common={})
    cells = sweep_cells(l5cfg(root, caches["low"]), grid, L5LADDER)
    monkeypatch.setattr(sweep_module, "prepare", lambda *a, **k: _l5(root, caches["low"])[0])   # the same settings
    entry, _ = run_sweep(SynthRun.open(root), cells, load_splits())
    digest = cache_digest(cache_path(caches["low"], SynthRun.open(root).meta.run_id))
    assert [(c["point"], c["base"], c["cache"]["digest"]) for c in entry["cells"]] == [("baseline", "L5", digest)]


def test_eval_ladder_with_L5_records_the_cache(leaf_runs, caches, tmp_path):
    root = leaf_runs[0]
    raw = yaml.safe_load((ROOT / "configs" / "eval" / "apple_dev.yaml").read_text(encoding="utf-8"))
    raw.update(runs=[str(root)], output_dir=str(tmp_path / "out"))
    raw["passes"]["step_m"] = 0.45
    raw["detector"]["cache"] = str(caches["low"])
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    r = CliRunner().invoke(eval_app, ["ladder", "--config", str(cfg_path), "--ladder",
                                      str(ROOT / "configs" / "eval" / "sweeps" / "ladder_l5.yaml"),
                                      "--rungs", "L3,L5", "--subsets", "--set", "verdict_subsets=[A]",
                                      "--splits", SPLITS, "--criteria", CRIT_FILE])
    assert r.exit_code == 0, r.output
    folder = Path(r.output.splitlines()[0])
    res = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    rec = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    rid = SynthRun.open(root).meta.run_id
    digest = cache_digest(cache_path(caches["low"], rid))
    rungs = res["runs"][0]["rungs"]
    assert rungs["L5"]["cache"]["digest"] == digest and "cache" not in rungs["L3"]
    assert rec["detector"]["name"] == "cached" and rec["detector"]["cache_sha256"] == digest
    assert rec["detector"]["caches"] == [{"run_id": rid, "file": f"{rid}.npz", "digest": digest, "weights_sha256": None}]
    assert rec["data"][0]["cache_digests"] == [digest] and rec["extra"]["detectors"] == ["cached", "oracle"]
    assert "L5: row U=28 G6=12" in r.output
    assert res["schema_version"] == 6 and res["detection"]["rung"] == "L5"
    assert res["detection"]["thresholds"] == {"operating": 0.5, "low": 0.1}
    assert res["detection"]["postprocess"] == {"nms_iou": 0.5, "edge_snap_px": 2.0, "min_short_px": 0.0}
    det_rows = res["verdicts"]["detection"]
    assert [(v["group"], v["metric"]) for v in det_rows] == [("pitch_30", "detection.recall"),
        ("pitch_30", "detection.precision"), ("pitch_50", "detection.recall"), ("pitch_50", "detection.precision")]
    assert det_rows[0]["value"] is not None and det_rows[2]["value"] is None
    assert res["pooled"]["L5"]["y1_judged"]["n_passes"] == res["pooled"]["L3"]["y1_judged"]["n_passes"] == 2
    assert next(v for v in res["verdicts"]["L5"] if v["metric"] == "y1.e1_median")["pitches"] == [30.0, 50.0]
    assert [v["metric"] for v in res["verdicts"]["L5[A]"]] == [
        "y6.row_abs_e_pipe", "y6.tree_mae_over_mean", "y6.miss_rate", "y6.excess_rate"]
    assert (folder / "report.md").read_text(encoding="utf-8").splitlines()[2] == \
        "> 검출: 학습 검출기 캐시 · 가중치 - · 사전학습 정보 없음"
    report = (folder / "report.md").read_text(encoding="utf-8")
    assert "**검출 (L5와 같은 프레임, 운영점 0.5)**" in report
    assert "**트랙이 된 오검출 (L3→L5, Y1 패스 장부, 기울기별)**" in report
    header = (folder / "budget.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == ("run_id,kind,group,pair,n_truth,d_e,d_tree_mae,d_e1_median,fp_bg,fp_off,boundary,dup_track,"
                      "dup_cross,miss_det,miss_drop,miss_merge,residual")
    assert {r["group"] for r in res["budget"]} == {"row", "pitch_30", "judged", "all"}


def test_eval_refuses_verdict_subsets_requests_before_scoring(leaf_runs, tmp_path, monkeypatch):
    raw = yaml.safe_load((ROOT / "configs" / "eval" / "apple_dev.yaml").read_text(encoding="utf-8"))
    raw.update(runs=[str(leaf_runs[0])], output_dir=str(tmp_path / "out"))
    raw["detector"]["cache"] = str(tmp_path / "cache")
    cfg_path, grid = tmp_path / "cfg.yaml", tmp_path / "grid.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    grid.write_text("factors: {tracker.gate_m: [0.1]}\n", encoding="utf-8")

    def scoring(*args, **kw):
        raise runner_module.CheckFailed("scoring reached")

    monkeypatch.setattr(runner_module, "prepare", scoring)
    l5 = ["ladder", "--ladder", str(ROOT / "configs" / "eval" / "sweeps" / "ladder_l5.yaml"), "--rungs", "L3,L5"]
    only = "verdict_subsets is judged by eval ladder only"
    for args, phrase in (([*l5, "--subsets", "--set", "verdict_subsets=[nope]"],
                          "verdict_subsets names 'nope', not a pass subset of this bundle (all, A, B, only 30, drop 30)"),
                         ([*l5, "--set", "verdict_subsets=[A]"], "verdict_subsets needs --subsets and rung L5"),
                         (["run", "--set", "verdict_subsets=[A]"], f"{only}; eval run does not read it"),
                         (["sweep", "--grid", str(grid), "--set", "verdict_subsets=[A]"], f"{only}; eval sweep")):
        r = CliRunner().invoke(eval_app, [*args, "--config", str(cfg_path), "--splits", SPLITS, "--criteria", CRIT_FILE])
        assert r.exit_code == 2 and phrase in r.output, (args, r.output)
    assert not (tmp_path / "out").exists()


ZERO_LEDGER ={f.name: 0 for f in dataclasses.fields(Ledger)} | {"n_pred": 10, "n_truth": 10}


def test_verdict_subsets_judge_the_L5_subset_row(leaf_runs, caches):
    def entry(run_id, seed, G6=None, U=None, mae=None, **terms):
        """A run entry whose rung L5 has the subset `drop 20` against `G6` truth fruits; no pass without `G6`."""
        row = {"S": "drop 20", "n_passes": 0, "U": None, "G6": None, "e_pipe": None, "e_all": None,
               "tree_mae_over_mean_all": None, "tree_mae_over_mean": None, "ledger": None}
        if G6 is not None:
            row.update(n_passes=4, U=U, G6=G6, e_pipe=(U - G6) / G6, tree_mae_over_mean=mae,
                       ledger={**ZERO_LEDGER, "n_pred": U, "n_truth": G6, **terms})
        return {"meta": {"run_id": run_id, "seed": seed}, "rungs": {"L5": {"subsets": [row]}}}

    # Scenes of different size, one failing |e_pipe|, one with no pass: pooling weighs by G6 and sums the ledgers.
    a = entry("a", 1000, 100, 105, 0.06, dup_track=7, miss_det=2)
    b = entry("b", 1001, 50, 56, 0.20, fp_bg=14, miss_drop=8)
    rows = subset_verdicts([a, b, entry("c", 1002)], CRIT, "pear", ["drop 20"])["L5[drop 20]"]
    by = {r["metric"]: r for r in rows}
    e = by["y6.row_abs_e_pipe"]
    assert [p["value"] for p in e["per_run"]] == [pytest.approx(0.05), pytest.approx(0.12), None]
    assert [p["ok"] for p in e["per_run"]] == [True, False, None] and (e["value"], e["ok"]) == (None, False)
    pooled = {k: (by[f"y6.{k}"]["value"], by[f"y6.{k}"]["ok"])
              for k in ("tree_mae_over_mean", "miss_rate", "excess_rate")}
    assert pooled == {"tree_mae_over_mean": (pytest.approx(16 / 150), True),      # plain scene means: 0.13,
                      "miss_rate": (pytest.approx(10 / 150), True),               # 0.09
                      "excess_rate": (pytest.approx(21 / 150), False)}            # and 0.175
    lone = subset_verdicts([a, entry("c", 1002)], CRIT, "pear", ["drop 20"])["L5[drop 20]"]
    assert [(r["value"], r["ok"]) for r in lone] == [(None, None), (pytest.approx(0.06), True),
                                                     (pytest.approx(0.02), True), (pytest.approx(0.07), True)]
    with pytest.raises(ValueError, match="pass subset 'drop 45' is not in rung L5's table of a"):
        subset_verdicts([a, b], CRIT, "pear", ["drop 45"])
    root = leaf_runs[0]
    cfg = l5cfg(root, caches["low"], "verdict_subsets=[drop 30]")
    with pytest.raises(ValueError, match="verdict_subsets needs --subsets and rung L5"):
        check_verdict_request(cfg, ["L3"], subsets=True)
    check_verdict_request(cfg, ["L3", "L5"], subsets=True)
    check_verdict_subsets([SynthRun.open(root)], cfg)
    with pytest.raises(ValueError, match=re.escape("verdict_subsets names 'drop 20', not a pass subset of this bundle "
                                                   "(all, A, B, only 30, drop 30)")):
        check_verdict_subsets([SynthRun.open(root)], l5cfg(root, caches["low"], "verdict_subsets=[drop 20]"))


def _entry(run_id, seed, e1s):
    """A run entry whose one rung L5 has a pass (pitch, e1) per item of `e1s` and no Y6."""
    passes = [{"pitch": p, "e1": e, "ledger": dict(ZERO_LEDGER)} for p, e in e1s]
    summary = {"e1_median": float(np.median([abs(e) for _, e in e1s]))}
    return {"meta": {"run_id": run_id, "seed": seed}, "rungs": {"L5": {"y1": {"passes": passes, "summary": summary},
                                                                     "y6": None}}}


def _results(ctx, out, **extra):
    """A one-bundle ladder results dict around `run_ladder`'s `out`, as `_run_eval` builds it."""
    return {"schema_version": SCHEMA_VERSION, "kind": "ladder", "run_name": "t", "experiment_id": "abc123abc123",
            "split": "dev", "crop": "apple", "runs": [{"meta": describe_run(ctx), **out}], "pooled": None,
            "verdicts": None, **extra}


def test_L5_y1_verdict_uses_the_judged_pitches_only():
    crit = {"y1": {"L5": {"apple": {"e1_median": {"le": 0.10}}}}}
    entries = [_entry("a", 1000, [(10.0, 0.5), (30.0, 0.02), (50.0, -0.04)]),
               _entry("b", 1001, [(10.0, 0.6), (30.0, 0.06)])]
    pooled = ladder_pooled(entries, crop="apple")
    assert pooled["L5"]["y1"]["e1_median"] == pytest.approx(0.06)
    assert pooled["L5"]["y1_judged"]["e1_median"] == pytest.approx(0.04)
    (row,) = ladder_verdicts(entries, pooled, crit, "apple")["L5"]
    assert (row["value"], row["ok"], row["pitches"]) == (pytest.approx(0.04), True, [30.0, 50.0])
    assert [p["value"] for p in row["per_run"]] == pytest.approx([0.03, 0.06])
    (ref,) = criteria_rows(entries[0]["rungs"]["L5"], crit, "L5", "apple")      # the report's reference table
    assert (ref["value"], ref["ok"]) == (pytest.approx(0.03), True)
    trimmed = {"y1": entries[0]["rungs"]["L5"]["y1"], "y6": {"metrics": {}}}     # a sweep cell keeps no trees
    assert criteria_rows(trimmed, crit, "L5", "apple") == [ref]
    lone = [_entry("c", 1002, [(10.0, 0.5)])]
    (row,) = ladder_verdicts(lone, ladder_pooled(lone, crop="apple"), crit, "apple")["L5"]
    assert row["value"] is None and row["ok"] is None and "y1_judged" not in ladder_pooled(lone)["L5"]


def test_detection_rows_judge_each_pitch_on_the_pooled_operating_point():
    def at(p, r):
        return {"at": {"operating": {"precision": p, "recall": r}}}

    det = {"runs": [{"meta": {"run_id": "a", "seed": 1000}, "groups": {"pitch_45": at(0.95, 0.91)}},
                    {"meta": {"run_id": "b", "seed": 1001}, "groups": {"pitch_45": at(0.85, 0.93),
                                                                         "pitch_70": at(0.97, 0.96)}}],
           "pooled": {"groups": {"pitch_45": at(0.89, 0.92), "pitch_70": at(0.97, 0.96)}}}
    rows = detection_rows(det, CRIT, "pear")
    assert [(r["group"], r["metric"], r["value"], r["ok"]) for r in rows] == [
        ("pitch_45", "detection.recall", 0.92, True), ("pitch_45", "detection.precision", 0.89, False),
        ("pitch_70", "detection.recall", 0.96, True), ("pitch_70", "detection.precision", 0.97, True)]
    assert [p["value"] for p in rows[1]["per_run"]] == [0.95, 0.85] and rows[2]["per_run"][0]["value"] is None
    assert all(r["unit"] == "pooled" and r["limit"] == 0.90 for r in rows) and detection_rows(det, None, "pear") == []


def test_report_head_names_the_detector_source(l5_ladder):
    ctx, _, out, _ = l5_ladder
    assert render_report(_results(ctx, out), criteria=None).splitlines()[2] == \
        "> 검출: 학습 검출기 캐시 · 가중치 - · 사전학습 정보 없음"
    coco = copy.deepcopy(out)
    coco["rungs"]["L5"]["cache"].update(weights_sha256="ab" * 32,
                                        pretrained={"weights": "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1"})
    coco_line = ("> 검출: 학습 검출기 캐시 · 가중치 abababababababab · 사전학습 FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1"
                 " · COCO 사전학습 가중치에서 시작한 결과다(연구·비상업 용도)")
    assert render_report(_results(ctx, coco), criteria=None).splitlines()[2] == coco_line
    oracle_only = {**out, "chain": ["L3"], "rungs": {"L3": out["rungs"]["L3"]}}
    assert "> 검출" not in render_report(_results(ctx, oracle_only), criteria=None)
    cell = {**coco["rungs"]["L5"], "point": "baseline", "base": "L5", "overridden": []}     # sweeps read their cells
    point = {"name": "baseline", "patch": {}, "factor": None, "value": None}
    sweep = _results(ctx, {"cells": [cell]}, kind="sweep", sweep={"grid": "g.yaml", "bases": ["L5"], "points": [point]})
    assert render_report(sweep, criteria=None).splitlines()[2] == coco_line


def test_contrib_pairs_branch_L4_base_and_L5_from_L3():
    full = ["L0", "L1", "L2a", "L2b", "L2c", "L3", "L4_base", "L5"]
    assert contrib_pairs(full, dict.fromkeys([*full, "L3_dd", "L5_L4base"])) == [
        ("L0", "L1"), ("L1", "L2a"), ("L2a", "L2b"), ("L2b", "L2c"), ("L2c", "L3"), ("L3", "L4_base"), ("L3", "L5"),
        ("L3_dd", "L3"), ("L5", "L5_L4base")]
    assert contrib_pairs(["L0", "L1", "L3"], dict.fromkeys(["L0", "L1", "L3"])) == [("L0", "L1"), ("L1", "L3")]
    assert contrib_pairs(["L1", "L4_base"], dict.fromkeys(["L1", "L4_base"])) == [("L1", "L4_base")]
    assert contrib_pairs(["L5"], dict.fromkeys(["L5"])) == []


def test_budget_rows_add_up_and_telescope(l5_ladder, apple_ladder, apple_ladder_ctx):
    ctx, _, out, _ = l5_ladder
    entry = [{"meta": describe_run(ctx), **out}]
    rows = budget_rows(entry, contrib_pairs(out["chain"], out["rungs"]), kind="y6")
    check_budget(rows)
    by = {r["pair"]: r for r in rows if r["run_id"] == "pooled"}
    assert by["L2a→L2b"]["dup_cross"] == pytest.approx(0.25) and by["L2b→L2c"]["dup_track"] == pytest.approx(0.25)
    assert by["L3→L5"]["d_e"] == pytest.approx(10 / 12) and by["L3→L5"]["group"] == "row"
    assert (by["L3→L5"]["dup_track"], by["L3→L5"]["dup_cross"]) == (pytest.approx(1 / 12), pytest.approx(9 / 12))
    assert sum(r["d_e"] for r in by.values()) == pytest.approx(out["rungs"]["L5"]["y6"]["row"]["e_pipe"])
    full = ladder_budget(entry, "apple")
    assert {(r["kind"], r["group"]) for r in full} == {("y6", "row"), ("y1", "pitch_30"), ("y1", "judged"),
                                                       ("y1", "all")}
    y1 = {r["pair"]: r for r in full if r["group"] == "judged" and r["run_id"] == "pooled"}
    assert (y1["L3→L5"]["dup_track"], y1["L3→L5"]["d_e1_median"]) == (pytest.approx(1 / 24), pytest.approx(1 / 24))
    oracle = [{"meta": describe_run(apple_ladder_ctx), **apple_ladder}]       # L4_base carries seed means
    check_budget(budget_rows(oracle, contrib_pairs(apple_ladder["chain"], apple_ladder["rungs"]), kind="y1"))
    broken = copy.deepcopy(rows[0])
    broken["fp_bg"] += 0.1
    with pytest.raises(ValueError, match="ledger terms do not add up to the error change"):
        check_budget([broken])


def _bud_entry(run_id, seed, rungs):
    """A run entry with one pass, one tree and the row ledger per rung: `rungs` is {name: (U, G, {term: count})}."""
    out = {}
    for name, (U, G, terms) in rungs.items():
        L = {**ZERO_LEDGER, "n_pred": U, "n_truth": G, **terms}
        out[name] = {"y1": {"passes": [{"pitch": 30.0, "U1": U, "G1": G, "e1": (U - G) / G, "ledger": L}]},
                     "y6": {"row": {"U": U, "G6": G, "e_pipe": (U - G) / G}, "ledger": L,
                            "trees": [{"tree": 1, "U": U, "G6": G, "U_truth_attr": U}],
                            "metrics": {"tree_mae_over_mean": abs(U - G) / G}}}
    return {"meta": {"run_id": run_id, "seed": seed}, "chain": list(rungs), "rungs": out, "interaction_2x2": None}


def test_budget_pooled_rows_sum_counts_over_bundles():
    entries = [_bud_entry("a", 1000, {"L3": (105, 100, {"dup_track": 7, "miss_det": 2}),
                                      "L5": (96, 100, {"fp_bg": 1, "miss_drop": 5})}),
               _bud_entry("b", 1001, {"L3": (52, 50, {"dup_cross": 2}), "L5": (58, 50, {"fp_bg": 10, "miss_det": 2})})]
    rows = budget_rows(entries, [("L3", "L5")], kind="y6")
    check_budget(rows)
    (p,) = [r for r in rows if r["run_id"] == "pooled"]
    assert (p["n_truth"], p["d_e"], p["fp_bg"], p["dup_track"], p["dup_cross"], p["miss_det"], p["miss_drop"]) == (
        150, pytest.approx(-3 / 150), pytest.approx(11 / 150), pytest.approx(-7 / 150), pytest.approx(-2 / 150), 0,
        pytest.approx(-5 / 150))
    assert p["d_tree_mae"] == pytest.approx(5 / 150) and [r["d_e"] for r in rows[:2]] == pytest.approx([-0.09, 0.12])


def test_ladder_budget_refuses_a_truth_change_and_terms_that_do_not_add_up():
    def ladder(l5):
        return [_bud_entry("a", 1000, {"L3": (105, 100, {"dup_track": 7, "miss_det": 2}), "L5": l5})]

    assert len(ladder_budget(ladder((96, 100, {"fp_bg": 1, "miss_drop": 5})), "apple")) == 8
    with pytest.raises(ValueError, match=re.escape("a: rungs L3 and L5 score different truth (100 and 90)")):
        ladder_budget(ladder((86, 90, {"fp_bg": 1, "miss_drop": 5})), "apple")
    with pytest.raises(ValueError, match="budget row a y6/row L3→L5: ledger terms do not add up to the error change"):
        ladder_budget(ladder((96, 100, {"fp_bg": 2, "miss_drop": 5})), "apple")       # fp_bg off by one


def test_report_draws_the_budget_for_L5_ladders_only(l5_ladder, apple_ladder, apple_ladder_ctx):
    ctx, _, out, _ = l5_ladder

    def with_budget(o):
        return _results(ctx, o, budget=ladder_budget([{"meta": describe_run(ctx), **o}], "apple"))

    lines = render_report(with_budget(out), criteria=None).splitlines()
    assert "**오차 예산 (Y6 줄 장부, 장면 합)**" in lines and "**Y1 오차 예산 (판정 기울기 30·50° 패스 장부 합)**" in lines
    assert any(line.startswith("| L3→L5 (검출) | +83.3% |") for line in lines)
    assert any(line.startswith("| 합 (L0→L5) | +133.3% |") for line in lines)
    assert any(line.startswith("- L5의 −Miss_det는 score_lo 이상 상자가 하나도 맞지 않은 과실이다.") for line in lines)
    l3, l5 = (_entry(n, 1000, e1s)["rungs"]["L5"] for n, e1s in (("a", [(10.0, 0.5), (30.0, 0.02)]),
                                                                  ("b", [(10.0, 0.1), (30.0, 0.06)])))
    judged = _contrib_table(["L3", "L5"], {"L3": l3, "L5": l5}, crop="apple")
    assert judged[0] == ("| 구간 | Δ Y1 e1 중앙값 (판정 기울기 30·50°) | Δ Y1 e1 최대 (판정 기울기 30·50°) "
                         "| Δ 줄 e_pipe | Δ 그루 MAE÷평균 |")
    assert judged[2] == "| L3→L5 (검출) | +4.0% | +4.0% | - | - |"     # every pass would give -18.0% and -40.0%
    y1_only = copy.deepcopy(out)                         # stop_after=y1: no Y6 budget, the Y1 one stays
    for rung in y1_only["rungs"].values():
        rung["y6"] = None
        rung.pop("subsets", None)
    text = render_report(with_budget(y1_only), criteria=None)
    assert "오차 예산 (Y6" not in text and "**Y1 오차 예산" in text
    lone = {"chain": ["L5"], "rungs": {"L5": out["rungs"]["L5"]}, "interaction_2x2": None}
    assert ladder_budget([{"meta": describe_run(ctx), **lone}], "apple") is None
    oracle = render_report(_results(apple_ladder_ctx, apple_ladder), criteria=None)
    assert "오차 예산" not in oracle and "| L3→L4_base (자세·깊이 잡음(기준 조합)) |" in oracle
    assert "| L4_base→" not in oracle
