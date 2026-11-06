import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

import counting.evaluation.runner as runner_module
import counting.evaluation.sweep as sweep_module
from counting.config.eval_schema import LiftConfig
from counting.data.formats.orchard_synth import PassKey, SynthRun
from counting.evaluation.ledger import Ledger
from counting.evaluation.oracle_stages import oracle_dedup
from counting.evaluation.report import SCHEMA_VERSION, criteria_rows, render_report
from counting.evaluation.runner import (
    CheckFailed, ProtectionError, _prepared_settings, depth_error_stats, describe_run, evaluate, evaluate_seeds,
    load_ladder, prepare, rung_config, run_config, run_ladder,
)
from counting.evaluation.sweep import SweepCell, SweepPoint, run_sweep
from counting.tracking.landmark import PassTracks
from tests.unit.synthfake import DET_NOISE, ctx_with, fake_cfg, load_splits, write_run

ROOT = Path(__file__).resolve().parents[2]
LADDER = load_ladder(ROOT / "configs" / "eval" / "sweeps" / "ladder.yaml")
CRIT = yaml.safe_load((ROOT / "configs" / "eval" / "criteria.yaml").read_text(encoding="utf-8"))
# Every Ledger error term: all fields except the two counts, so a new term cannot be left unchecked.
ZERO = tuple(f.name for f in dataclasses.fields(Ledger) if f.name not in ("n_pred", "n_truth"))
EXACT = ("lift.center_offset_m.fruit=0.0", "lift.center_offset_m.bag=0.0", "lift.depth=det_depth")


def _ctx(root, crop, *sets):
    cfg = fake_cfg(root, crop, *sets)
    return prepare(SynthRun.open(root), cfg, load_splits()), cfg


def assert_exact(res):
    """U == G and every ledger error term 0 for each Y1 pass, each Y6 tree and the Y6 row."""
    for p in res["y1"]["passes"]:
        assert p["U1"] == p["G1"] and all(p["ledger"][k] == 0 for k in ZERO), p
    for t in res["y6"]["trees"]:
        assert t["U"] == t["G6"] and all(t["ledger"][k] == 0 for k in ZERO), t
    assert all(res["y6"]["ledger"][k] == 0 for k in ZERO), res["y6"]["ledger"]


def test_exact_check_covers_every_ledger_field():
    """The L0 exact check flags a nonzero error term anywhere: a Y1 pass, a Y6 tree or the row ledger."""
    def clean():
        L = {f.name: 0 for f in dataclasses.fields(Ledger)} | {"n_pred": 3, "n_truth": 3}
        return {"y1": {"passes": [{"U1": 3, "G1": 3, "ledger": dict(L)}]},
                "y6": {"trees": [{"U": 3, "G6": 3, "ledger": dict(L)}], "ledger": dict(L)}}

    assert_exact(clean())
    where = {"pass": lambda r: r["y1"]["passes"][0]["ledger"], "tree": lambda r: r["y6"]["trees"][0]["ledger"],
             "row": lambda r: r["y6"]["ledger"]}
    missed = []
    for place, get in where.items():
        for term in ZERO:
            res = clean()
            get(res)[term] = 1
            try:
                assert_exact(res)
            except AssertionError:
                continue
            missed.append((place, term))
    assert missed == []
    res = clean()
    res["y6"]["trees"][0]["U"] = 4
    with pytest.raises(AssertionError):
        assert_exact(res)


def test_L0_is_exact(apple_run, pear_run, ctx_of):
    for root, crop in ((apple_run, "apple"), (pear_run, "pear")):
        ctx, cfg = ctx_of[crop], fake_cfg(root, crop)
        assert_exact(evaluate(ctx, rung_config(cfg, LADDER, "L0")))


def test_clean_fake_full_chain_is_exact(apple_run, pear_run):
    for root, crop, mode in ((apple_run, "apple", "merge"), (pear_run, "pear", "self_side")):
        ctx, cfg = _ctx(root, crop, *EXACT)
        res = evaluate(ctx, cfg)
        assert_exact(res)
        assert res["y6"]["mode"] == mode and res["y6"]["metrics"]["attribution_error"] == 0.0


def test_ledger_identity_every_rung_and_2x2(apple_ladder):
    out = apple_ladder
    assert out["chain"] == ["L0", "L1", "L2a", "L2b", "L2c", "L3", "L4_base"]
    for name, res in out["rungs"].items():
        ledgers = [p["ledger"] for p in res["y1"]["passes"]] + [res["y6"]["ledger"]]
        ledgers += [t["ledger"] for t in res["y6"]["trees"]]
        for L in ledgers:
            assert L["n_pred"] - L["n_truth"] == (L["fp_bg"] + L["fp_off"] + L["boundary"] + L["dup_track"]
                                                  + L["dup_cross"] - L["miss_det"] - L["miss_drop"]
                                                  - L["miss_merge"]), name
    cells = {(c["track"], c["dedup"]): c for c in out["interaction_2x2"]["cells"]}
    assert set(cells) == {("oracle", "oracle"), ("oracle", "merge"), ("landmark", "oracle"), ("landmark", "merge")}
    assert cells[("oracle", "oracle")]["e_pipe"] == 0.0
    for name in ("L4_base", "L4_weak", "L4_strong"):        # dev seeds [0]: the mean is that seed's result
        res = out["rungs"][name]
        assert res["noisy"] and res["seeds"] == [0] and [p["noise_seed"] for p in res["per_seed"]] == [0]
        assert {k: res[k] for k in ("y1", "y6")} == {k: res["per_seed"][0][k] for k in ("y1", "y6")}
    assert "noisy" not in out["rungs"]["L3"]


def test_oracle_dedup_leaves_no_cross_pass_dup(apple_run, pear_run, ctx_of, apple_ladder):
    """dedup=oracle fuses each id across passes, so every duplicate left is a split inside one pass."""
    dups = 0
    for root, crop in ((apple_run, "apple"), (pear_run, "pear")):
        ctx, cfg = ctx_of[crop], fake_cfg(root, crop)
        names = [n for n in (*LADDER["rungs"], *LADDER["baselines"])
                 if rung_config(cfg, LADDER, n).stages.dedup == "oracle"]
        assert names == ["L0", "L1", "last_box"]
        for name in names:   # apple: the session ladder evaluated these rungs on apple_ladder_ctx
            res = apple_ladder["rungs"][name] if crop == "apple" else evaluate(ctx, rung_config(cfg, LADDER, name))
            y6 = res["y6"]
            for L in [y6["ledger"], y6["ledger_any"]] + [t["ledger"] for t in y6["trees"]]:
                assert L["dup_cross"] == 0, (crop, name, L)
            dups += y6["ledger"]["dup_track"]
    assert dups > 0   # the fakes do leave in-pass splits, so the check above is not vacuous


def test_ladder_box_rungs_inherit_base_depth(apple_run):
    cfg = fake_cfg(apple_run, "apple", "lift.depth=box_far")
    assert all("det" not in (patch.get("stages") or {})
               for section in ("rungs", "baselines") for patch in LADDER[section].values())
    for name in ("L2c", "L3"):
        assert rung_config(cfg, LADDER, name).lift == cfg.lift
    assert rung_config(cfg, LADDER, "L2b").lift.depth == "det_depth"
    dd = rung_config(cfg, LADDER, "L3_dd")
    assert dd.lift.depth == "det_depth" and dd.stages.model_dump() == {
        "det": "oracle", "track": "landmark", "center": "measured", "dedup": "merge", "assign": "grid"}


def test_L4_rungs_add_the_noise_combos_to_L3(apple_run):
    cfg = fake_cfg(apple_run, "apple", "lift.depth=box_far", "noise.seeds=[0,1,2]")
    l3 = rung_config(cfg, LADDER, "L3")
    combos = {"L4_base": (0.02, 0.5, 0.02, 0.02, 0.05), "L4_weak": (0.01, 0.25, 0.01, 0.01, 0.0),
              "L4_strong": (0.05, 1.0, 0.10, 0.04, 0.20)}
    for name, values in combos.items():
        c = rung_config(cfg, LADDER, name)
        assert c.model_dump(exclude={"noise"}) == l3.model_dump(exclude={"noise"}), name   # det, depth from the base
        n = c.noise
        assert (n.pose_trans_sigma_m, n.pose_rot_sigma_deg, n.pass_bias_m, n.depth_rel_sigma, n.depth_dropout) == values
        assert n.seeds == [0, 1, 2] and c.is_noisy()
    assert list(LADDER["rungs"])[-2:] == ["L3", "L4_base"]
    assert list(LADDER["baselines"]) == ["L3_dd", "last_box", "L4_weak", "L4_strong"]


def test_ladder_noise_rung_runs_once_per_seed(leaf_ctx, leaf_runs):
    cfg = fake_cfg(leaf_runs[0], "apple", "noise.seeds=[0,1]")
    out = run_ladder(leaf_ctx[0], cfg, LADDER, ["L3", "L4_base"])
    l4, c = out["rungs"]["L4_base"], rung_config(cfg, LADDER, "L4_base")
    assert out["chain"] == ["L3", "L4_base"] and l4["seeds"] == [0, 1]
    for p, s in zip(l4["per_seed"], (0, 1)):
        alone = evaluate(leaf_ctx[0], c, noise_seed=s)
        assert (p["noise_seed"], p["y1"], p["y6"]) == (s, alone["y1"], alone["y6"])
    e = [p["y6"]["row"]["e_pipe"] for p in l4["per_seed"]]
    assert l4["y6"]["row"]["e_pipe"] == pytest.approx(sum(e) / 2)
    assert l4["y6"]["metrics"]["row_abs_e_pipe"] == pytest.approx(sum(abs(x) for x in e) / 2)
    assert l4["spread"]["y6"]["row"]["e_pipe"] == [min(e), max(e)]
    crit = {r["metric"]: r for r in criteria_rows(l4, CRIT, "L4_base", "apple")}
    assert crit["y6.row_abs_e_pipe"]["value"] == l4["y6"]["metrics"]["row_abs_e_pipe"]
    assert crit["y6.row_abs_e_pipe"]["limit"] == 0.10
    with pytest.raises(ValueError, match="--subsets needs a stage setting without noise"):
        evaluate_seeds(leaf_ctx[0], c, subsets=True)


def test_ladder_refuses_bad_base(leaf_ctx, leaf_runs):
    for sets, phrase in ((["lift.depth=det_depth"], "lift.depth=det_depth"),
                         (["stages.det=oracle_noisy", "detector.noise.miss_p=0.1"], "stages.det=oracle_noisy"),
                         (["noise.depth_dropout=0.05"], "noise set")):
        with pytest.raises(ValueError, match=f"ladder base config has {re.escape(phrase)}"):
            run_ladder(leaf_ctx[0], fake_cfg(leaf_runs[0], "apple", *sets), LADDER, ["L3"])


def test_stop_after_y1_and_naive(apple_run, ctx_of):
    ctx, cfg = ctx_of["apple"], fake_cfg(apple_run, "apple", "stop_after=y1")
    res = evaluate(ctx, cfg)
    assert res["y6"] is None
    assert res["y1"]["summary"]["naive_U"] == sum(p["U1"] for p in res["y1"]["passes"])


def test_cached_needs_a_cache_folder(apple_run):
    with pytest.raises(ValueError, match=re.escape("stages.det=cached needs detector.cache")):
        fake_cfg(apple_run, "apple", "stages.det=cached")
    # an oracle config may carry the folder for rung L5; its own context ignores it
    assert _prepared_settings(fake_cfg(apple_run, "apple", "detector.cache=x")) == \
        _prepared_settings(fake_cfg(apple_run, "apple"))


def test_patched_ladder_is_checked_against_prepare(apple_run, ctx_of, monkeypatch):
    """A hand-edited rung may not break the config checks or change what prepare fixed (frames, detections)."""
    ctx, cfg = ctx_of["apple"], fake_cfg(apple_run, "apple")
    differs = {"stages.det": {"stages": {"det": "oracle_noisy"}, "detector": {"noise": {"miss_p": 0.1}}},
               "detector.positives": {"detector": {"positives": "any"}},
               "passes": {"passes": {"pitches": [30.0]}},
               "crop": {"crop": "pear"},
               "split": {"split": "val"}}
    calls = []
    monkeypatch.setattr(runner_module, "evaluate", lambda *a, **k: calls.append(a[1]) or evaluate(*a, **k))
    cases = [(ValueError, f"{k}=.* differs from the prepared {k}=", p) for k, p in differs.items()]
    cases.append((ValueError, "stages.det=cached needs detector.cache", {"stages": {"det": "cached"}}))
    for exc, phrase, patch in cases:
        ladder = {"rungs": {"L0": LADDER["rungs"]["L0"], "bad": patch}}
        with pytest.raises(exc, match=phrase):
            evaluate(ctx, rung_config(cfg, ladder, "bad"))
        with pytest.raises(exc, match=phrase):
            run_ladder(ctx, cfg, ladder, ["L0", "bad"])
        assert calls == [], phrase   # checked before any rung runs
    ladder = {"rungs": {**LADDER["rungs"], "bad": {"stages": {"det": "cached"}}},
              "interaction_2x2": {"base": "bad", "track": ["oracle"], "dedup": ["oracle"]}}
    with pytest.raises(ValueError, match="stages.det=cached needs detector.cache"):
        run_ladder(ctx, cfg, ladder, ["L0"], with_2x2=True)
    assert calls == []


def test_2x2_without_ladder_section_names_the_file(apple_run, tmp_path):
    p = tmp_path / "ladder.yaml"
    p.write_text("rungs: {L0: {stages: {track: oracle}}}\n", encoding="utf-8")
    ctx, cfg = _ctx(apple_run, "apple")
    with pytest.raises(ValueError, match=r"ladder\.yaml: with_2x2 needs an 'interaction_2x2' mapping"):
        run_ladder(ctx, cfg, load_ladder(p), ["L0"], with_2x2=True)


def test_wrong_crop_refused(apple_run):
    with pytest.raises(ValueError, match="crop mismatch"):
        prepare(SynthRun.open(apple_run), fake_cfg(apple_run, "pear"), load_splits())


def test_split_seed_checks(apple_run, heldout_run):
    with pytest.raises(ValueError, match="belongs to split"):
        _ctx(apple_run, "apple", "split=val")
    with pytest.raises(ProtectionError, match=re.escape("split=test needs prepare(..., allow_test=True)")):
        _ctx(heldout_run, "apple", "split=test", "tuning=false")
    with pytest.raises(CheckFailed, match="seed 1000 belongs to split test, config split is dev"):
        _ctx(heldout_run, "apple")


def test_library_calls_refuse_split_test_unless_allowed(apple_run, heldout_run, monkeypatch):
    cfg = fake_cfg(heldout_run, "apple", "split=test", "tuning=false", "stop_after=y1")
    cell = SweepCell(point=SweepPoint("baseline", {}, None, None), base=None, cfg=cfg, overridden=())
    for call in (lambda: prepare(SynthRun.open(heldout_run), cfg, load_splits()),
                 lambda: run_config(SynthRun.open(heldout_run), cfg, load_splits()),
                 lambda: run_sweep(SynthRun.open(heldout_run), [cell], load_splits())):
        with pytest.raises(ProtectionError, match=r"s1000: split=test needs prepare\(\.\.\., allow_test=True\)"):
            call()
    assert prepare(SynthRun.open(heldout_run), cfg, load_splits(), allow_test=True).prepared["split"] == "test"
    with pytest.raises(CheckFailed, match="seed 5 belongs to split dev, config split is test"):
        prepare(SynthRun.open(apple_run), fake_cfg(apple_run, "apple", "split=test", "tuning=false"), load_splits(),
                allow_test=True)

    seen = []

    def gate(run, cfg, splits, *, allow_test=False, **kw):     # stops before any truth table is built
        seen.append((cfg.stages.det, allow_test))
        raise CheckFailed("prepare reached")

    monkeypatch.setattr(runner_module, "prepare", gate)
    monkeypatch.setattr(sweep_module, "prepare", gate)
    noisy = fake_cfg(heldout_run, "apple", "split=test", "tuning=false", "stop_after=y1", *DET_NOISE)
    for call in (lambda: run_config(SynthRun.open(heldout_run), cfg, load_splits(), allow_test=True),
                 lambda: run_config(SynthRun.open(heldout_run), noisy, load_splits(), allow_test=True),
                 lambda: run_sweep(SynthRun.open(heldout_run), [cell], load_splits(), allow_test=True)):
        with pytest.raises(CheckFailed, match="prepare reached"):
            call()
    assert seen == [("oracle", True), ("oracle_noisy", True), ("oracle", True)]


def test_non_sequence_refused(tmp_path):
    with pytest.raises(ValueError, match="not a sequence bundle"):
        _ctx(write_run(tmp_path, crop="apple", max_frames=10), "apple")


def test_missing_summary_refused(tmp_path):
    run = write_run(tmp_path, crop="apple", pitches=(30.0,))
    (run / "summary.json").unlink()
    with pytest.raises(ValueError, match="summary.json is missing"):
        _ctx(run, "apple")


def test_empty_truth_pass_gives_null(pitch80_ctx):
    res = evaluate(pitch80_ctx, fake_cfg(pitch80_ctx.run.root, "apple"))
    p80 = [p for p in res["y1"]["passes"] if p["pitch"] == 80.0]
    assert len(p80) == 2 and all(p["G1"] == 0 and p["U1"] == 0 and p["e1"] is None for p in p80)
    assert res["y1"]["summary"]["e1_median"] is not None


def test_no_depth_tracks_counted(no_depth_run):
    ctx, cfg = _ctx(no_depth_run, "apple")
    res = evaluate(ctx, cfg)
    for p in res["y1"]["passes"]:
        assert p["U1"] == 0 and p["n_no3d"] == p["n_tracks"] > 0 and p["ledger"]["miss_drop"] == p["G1"]


def test_dim_fruit_is_ignored_and_counted_as_boundary(tmp_path):
    root = write_run(tmp_path, crop="apple", pitches=(30.0,), dim={801: 0.2})   # visible, never count1
    ctx, cfg = _ctx(root, "apple")
    assert 801 not in ctx.tables["count1"].g6 and 801 in ctx.tables["any"].g6
    assert_exact(evaluate(ctx, rung_config(cfg, LADDER, "L0")))
    ctx2, cfg2 = _ctx(root, "apple", "detector.positives=any")
    assert evaluate(ctx2, rung_config(cfg2, LADDER, "L0"))["y6"]["ledger"]["boundary"] == 1


def _leaf(leaf_ctx, leaf_runs, *sets):
    """The apple base config (an L3 setting) with `sets`, evaluated on [clean, occluded]."""
    return [evaluate(ctx, fake_cfg(root, "apple", *sets)) for ctx, root in zip(leaf_ctx, leaf_runs)]


def _leaf_pred(res):
    return ([(p["U1"], p["n_tracks"], p["n_no3d"]) for p in res["y1"]["passes"]],
            [t["U"] for t in res["y6"]["trees"]], res["y6"]["n_landmarks"])


def _dup(res):
    return sum(p["ledger"]["dup_track"] for p in res["y1"]["passes"])


def test_default_box_depth_keeps_baseline_numbers(leaf_ctx, leaf_runs):
    """box_median with the hard depth gate (the defaults) gives the numbers measured at bf72709."""
    clean = _leaf(leaf_ctx, leaf_runs)[0]
    assert _leaf_pred(clean) == ([(18, 18, 0), (12, 12, 0)], [7, 7, 7], 21) and _dup(clean) == 6


def test_front_leaf_patch_splits_tracks_under_box_median(leaf_ctx, leaf_runs):
    clean, occl = _leaf(leaf_ctx, leaf_runs)
    assert _dup(occl) > 5 * _dup(clean) > 0


def test_box_depth_cache_keyed_by_method_and_readonly(leaf_runs, monkeypatch):
    ctx, cfg = _ctx(leaf_runs[0], "apple")
    idx = next(i for i, d in ctx.dets.items() if len(d.boxes))
    calls = []
    real = runner_module.box_far_depth
    monkeypatch.setattr(runner_module, "box_far_depth", lambda *a, **k: calls.append(1) or real(*a, **k))
    far = LiftConfig(depth="box_far")
    med = ctx.box_depth(idx, cfg.lift)
    a = ctx.box_depth(idx, far)
    assert ctx.box_depth(idx, far) is a and len(calls) == 1
    assert ctx.box_depth(idx, far.model_copy(update={"box_frac": 0.7})) is a       # not a box_far parameter
    for n, upd in enumerate(({"far_scale": 0.7}, {"far_gap_m": 0.06}, {"far_min_frac": 0.3}), start=2):
        ctx.box_depth(idx, far.model_copy(update=upd))
        assert len(calls) == n, upd
    assert ctx.box_depth(idx, cfg.lift) is med
    assert ctx.box_depth(idx, cfg.lift.model_copy(update={"far_min_frac": 0.3})) is med   # not a box_median one
    assert ctx.box_depth(idx, cfg.lift.model_copy(update={"box_frac": 0.7})) is not med
    for arr in (med, a):
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 0.0
    with pytest.raises(ValueError, match="box depth method"):
        ctx.box_depth(idx, LiftConfig(depth="det_depth"))


def test_box_far_ignores_front_leaf(leaf_ctx, leaf_runs):
    clean, occl = _leaf(leaf_ctx, leaf_runs, "lift.depth=box_far")
    assert _leaf_pred(occl) == _leaf_pred(clean) == ([(16, 16, 0), (12, 12, 0)], [6, 7, 6], 19)
    assert _dup(occl) == _dup(clean) == 4        # the far cluster is the fruit behind the patch


def test_soft_gate_absorbs_front_leaf_tail(leaf_ctx, leaf_runs):
    hard_clean, hard = _leaf(leaf_ctx, leaf_runs)
    soft = _leaf(leaf_ctx, leaf_runs, "tracker.depth_gate=soft")[1]
    assert 5 * _dup(soft) < _dup(hard)
    assert _leaf_pred(soft) == _leaf_pred(hard_clean) == ([(18, 18, 0), (12, 12, 0)], [7, 7, 7], 21)   # no over-merge
    assert _dup(soft) == _dup(hard_clean) == 6 and soft["y1"]["summary"]["merge_rate"] == 0.0


def test_depth_error_stats_values():
    s = depth_error_stats(np.array([-0.3, -0.1, 0.0, 0.02, 0.2]))
    assert s["n"] == 5 and (s["near_10cm"], s["far_10cm"]) == (0.2, 0.2)      # strictly beyond 10 cm
    assert s["p50_m"] == pytest.approx(0.1)
    assert (s["p90_m"], s["p95_m"]) == pytest.approx((0.26, 0.28))
    assert depth_error_stats(np.array([])) is None


def test_depth_err_in_results(leaf_ctx, leaf_runs):
    clean, occl = _leaf(leaf_ctx, leaf_runs)
    far = _leaf(leaf_ctx, leaf_runs, "lift.depth=box_far")[1]
    near = {}
    for name, res in (("clean", clean), ("occl", occl), ("far", far)):
        s = res["y1"]["summary"]["depth_err"]
        assert s["n"] == sum(p["depth_err"]["n"] for p in res["y1"]["passes"]) > 0
        near[name] = s["near_10cm"]
    assert near["occl"] > 0.3 > near["clean"] and near["far"] < 0.05
    dd = evaluate(leaf_ctx[0], rung_config(fake_cfg(leaf_runs[0], "apple"), LADDER, "L1"))
    assert dd["y1"]["summary"]["depth_err"] is None and all(p["depth_err"] is None for p in dd["y1"]["passes"])


def test_depth_err_skips_nan_box_depth(leaf_ctx, leaf_runs):
    """An extra off-image box has a NaN box depth but a finite detection depth; depth_err leaves it out."""
    ctx, cfg = leaf_ctx[0], fake_cfg(leaf_runs[0], "apple", "stop_after=y1")
    idx = next(i for i, d in ctx.dets.items() if len(d.boxes))
    d = ctx.dets[idx]
    extra = dataclasses.replace(d, boxes=np.r_[d.boxes, [[-20, -20, -10, -10]]].astype(np.float32),
                                scores=np.r_[d.scores, 1.0].astype(np.float32),
                                labels=np.r_[d.labels, 0].astype(np.int8),
                                depth_m=np.r_[d.depth_m, 2.0].astype(np.float32))
    patched = dataclasses.replace(ctx, dets={**ctx.dets, idx: extra}, src={**ctx.src, idx: np.r_[ctx.src[idx], -1]})
    assert np.isnan(patched.box_depth(idx, cfg.lift)[-1])
    s0, s = (evaluate(c, cfg)["y1"]["summary"]["depth_err"] for c in (ctx, patched))
    assert s == s0 and np.isfinite(s["p50_m"])


def _ladder_results(ctx, out):
    return {"schema_version": SCHEMA_VERSION, "kind": "ladder", "run_name": "t", "experiment_id": "abc123abc123",
            "split": "dev", "crop": "apple", "runs": [{"meta": describe_run(ctx), **out}]}


def test_L3_dd_is_L2b_with_grid(leaf_ctx, leaf_runs):
    out = run_ladder(leaf_ctx[0], fake_cfg(leaf_runs[0], "apple"), LADDER, ["L2b", "L3", "L3_dd"])
    assert out["chain"] == ["L2b", "L3"]
    a, b = out["rungs"]["L2b"]["y6"], out["rungs"]["L3_dd"]["y6"]
    assert b["row"]["U"] == a["row"]["U"] == 15 and b["ledger"]["fp_off"] == 0
    row = next(line for line in render_report(_ladder_results(leaf_ctx[0], out), criteria=None).splitlines()
               if line.startswith("| L3_dd→L3 (상자 깊이) |"))
    assert row.split("|")[4].strip() == f"{out['rungs']['L3']['y6']['row']['e_pipe'] - b['row']['e_pipe']:+.1%}"


def test_L3_dd_row_drawn_without_a_chain(leaf_ctx, leaf_runs):
    out = run_ladder(leaf_ctx[0], fake_cfg(leaf_runs[0], "apple"), LADDER, ["L3", "L3_dd"])
    assert out["chain"] == ["L3"]
    lines = render_report(_ladder_results(leaf_ctx[0], out), criteria=None).splitlines()
    assert "**기여**" in lines and any(line.startswith("| L3_dd→L3 (상자 깊이) |") for line in lines)


def _without_mode(y6):
    return {k: v for k, v in y6.items() if k != "mode"}


def test_hybrid_dedup_runs_and_meets_its_band_limits(leaf_ctx, leaf_runs):
    root = leaf_runs[0]
    y6 = {name: evaluate(leaf_ctx[0], fake_cfg(root, "apple", *sets))["y6"] for name, sets in (
        ("self_side", ["stages.dedup=self_side"]), ("merge", []),
        ("band0", ["stages.dedup=hybrid", "dedup.band_m=0.0"]), ("band_inf", ["stages.dedup=hybrid", "dedup.band_m=1e9"]))}
    assert (y6["self_side"]["row"]["U"], y6["merge"]["row"]["U"]) == (18, 21)
    assert y6["band0"]["mode"] == y6["band_inf"]["mode"] == "hybrid"
    assert _without_mode(y6["band0"]) == _without_mode(y6["self_side"])
    assert _without_mode(y6["band_inf"]) == _without_mode(y6["merge"])


def test_subsets_exact_at_L0(apple_run, pear_run, ctx_of):
    for root, crop, pitches in ((apple_run, "apple", (10, 30)), (pear_run, "pear", (45, 70))):
        ctx, cfg = ctx_of[crop], fake_cfg(root, crop)
        res = evaluate(ctx, rung_config(cfg, LADDER, "L0"), subsets=True)
        assert [r["S"] for r in res["subsets"]] == ["all", "A", "B", *(f"{w} {p}" for p in pitches for w in ("only", "drop"))]
        for r in res["subsets"]:
            assert r["n_passes"] > 0 and r["U"] == r["G6"] and r["e_pipe"] == 0.0, r
            assert all(r["ledger"][k] == 0 for k in ZERO), r
        first, y6 = res["subsets"][0], res["y6"]
        assert (first["U"], first["G6"], first["ledger"]) == (y6["row"]["U"], y6["row"]["G6"], y6["ledger"])
        if crop == "pear":                        # one side sees part of the row: coverage against all
            assert [r["e_all"] for r in res["subsets"][1:3]] == pytest.approx([-0.4, -0.6])
            assert [r["tree_mae_over_mean_all"] for r in res["subsets"][1:3]] == pytest.approx([0.4, 0.6])


def test_subsets_match_runs_restricted_to_those_passes(leaf_ctx, leaf_runs):
    root = leaf_runs[0]
    res = evaluate(leaf_ctx[0], fake_cfg(root, "apple"), subsets=True)
    by = {r["S"]: r for r in res["subsets"]}
    assert list(by) == ["all", "A", "B", "only 30", "drop 30"]
    assert by["drop 30"] == {"S": "drop 30", "n_passes": 0, "U": None, "G6": None, "e_pipe": None,
                             "e_all": None, "tree_mae_over_mean_all": None, "tree_mae_over_mean": None,
                             "ledger": None}
    y6 = res["y6"]
    assert by["all"] == {"S": "all", "n_passes": 2, "U": y6["row"]["U"], "G6": y6["row"]["G6"],
                         "e_pipe": y6["row"]["e_pipe"], "e_all": y6["row"]["e_pipe"],
                         "tree_mae_over_mean_all": y6["metrics"]["tree_mae_over_mean"],
                         "tree_mae_over_mean": y6["metrics"]["tree_mae_over_mean"], "ledger": y6["ledger"]}
    for name in ("A", "B"):
        ctx_s, cfg_s = _ctx(root, "apple", f"passes.sides=[{name}]")
        alone = evaluate(ctx_s, cfg_s)["y6"]
        assert (by[name]["U"], by[name]["G6"], by[name]["e_pipe"], by[name]["ledger"]) == \
            (alone["row"]["U"], alone["row"]["G6"], alone["row"]["e_pipe"], alone["ledger"])
    assert [(by[n]["U"], by[n]["G6"]) for n in ("all", "A", "B")] == [(21, 12), (18, 12), (12, 12)]
    assert [by[n]["e_all"] for n in ("A", "B")] == pytest.approx([0.5, 0.0])
    assert [by[n]["tree_mae_over_mean_all"] for n in ("A", "B")] == pytest.approx([0.5, 0.0])
    with pytest.raises(ValueError, match="--subsets needs stop_after=y6"):
        evaluate(leaf_ctx[0], fake_cfg(root, "apple", "stop_after=y1"), subsets=True)


def test_ladder_subsets_on_L3_only_and_in_report(leaf_ctx, leaf_runs):
    cfg = fake_cfg(leaf_runs[0], "apple")
    with pytest.raises(ValueError, match="--subsets needs rung L3"):
        run_ladder(leaf_ctx[0], cfg, LADDER, ["L1"], subsets=True)
    out = run_ladder(leaf_ctx[0], cfg, LADDER, ["L1", "L3"], subsets=True)
    assert "subsets" not in out["rungs"]["L1"] and len(out["rungs"]["L3"]["subsets"]) == 5
    lines = render_report(_ladder_results(leaf_ctx[0], out), criteria=None).splitlines()
    a = next(r for r in out["rungs"]["L3"]["subsets"] if r["S"] == "A")
    L = a["ledger"]
    assert "**패스 묶음 (L3)**" in lines
    assert "| 묶음 | 패스 | U | G6 | e_pipe | e(전부) | 그루 MAE÷평균(전부) | Dup | Miss | FP_off | FP_bg |" in lines
    assert (f"| A | 1 | 18 | 12 | {a['e_pipe']:+.1%} | {a['e_all']:+.1%} | {a['tree_mae_over_mean_all']:+.1%} | "
            f"{L['dup_track'] + L['dup_cross']} | {L['miss_det'] + L['miss_drop'] + L['miss_merge']} | "
            f"{L['fp_off']} | {L['fp_bg']} |") in lines
    assert "| drop 30 | 0 | - | - | - | - | - | - | - | - | - |" in lines


def test_depth_err_null_without_detector_depth(leaf_ctx, leaf_runs):
    cfg, shared = fake_cfg(leaf_runs[0], "apple", "stop_after=y1"), leaf_ctx[0]
    ctx = ctx_with(shared, dets={k: dataclasses.replace(d, depth_m=None) for k, d in shared.dets.items()})
    res = evaluate(ctx, cfg)
    assert res["y1"]["summary"]["depth_err"] is None and all(p["depth_err"] is None for p in res["y1"]["passes"])


def test_oracle_dedup_keeps_background_apart():
    def pt(key, n):
        return PassTracks(pass_key=key, det_track={}, X=np.zeros((n, 3)), labels=np.zeros(n, np.int8),
                          n_obs=np.ones(n, int), spread=np.zeros(n), frames=[frozenset({i}) for i in range(n)])
    a, b = pt(PassKey(30.0, "A", 2), 2), pt(PassKey(30.0, "B", 1), 2)
    lm = oracle_dedup([a, b], [np.ones(2, bool)] * 2, [np.array([7, -1]), np.array([7, -1])])
    assert sorted(len(m) for m in lm.members) == [1, 1, 2]
