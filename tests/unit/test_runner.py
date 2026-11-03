import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import counting.evaluation.runner as runner_module
from counting.config.eval_schema import LiftConfig
from counting.data.formats.orchard_synth import PassKey, SynthRun
from counting.evaluation.ledger import Ledger
from counting.evaluation.oracle_stages import oracle_dedup
from counting.evaluation.report import SCHEMA_VERSION, render_report
from counting.evaluation.runner import (
    depth_error_stats, describe_run, evaluate, load_ladder, prepare, rung_config, run_ladder,
)
from counting.tracking.landmark import PassTracks
from tests.unit.synthfake import fake_cfg, load_splits, write_run

ROOT = Path(__file__).resolve().parents[2]
LADDER = load_ladder(ROOT / "configs" / "eval" / "sweeps" / "ladder.yaml")
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


def test_L0_is_exact(apple_run, pear_run):
    for root, crop in ((apple_run, "apple"), (pear_run, "pear")):
        ctx, cfg = _ctx(root, crop)
        assert_exact(evaluate(ctx, rung_config(cfg, LADDER, "L0")))


def test_clean_fake_full_chain_is_exact(apple_run, pear_run):
    for root, crop, mode in ((apple_run, "apple", "merge"), (pear_run, "pear", "self_side")):
        ctx, cfg = _ctx(root, crop, *EXACT)
        res = evaluate(ctx, cfg)
        assert_exact(res)
        assert res["y6"]["mode"] == mode and res["y6"]["metrics"]["attribution_error"] == 0.0


def test_ledger_identity_every_rung_and_2x2(apple_run):
    ctx, cfg = _ctx(apple_run, "apple")
    out = run_ladder(ctx, cfg, LADDER, list(LADDER["rungs"]) + list(LADDER["baselines"]), with_2x2=True)
    assert out["chain"] == ["L0", "L1", "L2a", "L2b", "L2c", "L3"]
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


def test_oracle_dedup_leaves_no_cross_pass_dup(apple_run, pear_run):
    """dedup=oracle fuses each id across passes, so every duplicate left is a split inside one pass."""
    dups = 0
    for root, crop in ((apple_run, "apple"), (pear_run, "pear")):
        ctx, cfg = _ctx(root, crop)
        names = [n for n in (*LADDER["rungs"], *LADDER["baselines"])
                 if rung_config(cfg, LADDER, n).stages.dedup == "oracle"]
        assert names == ["L0", "L1", "last_box"]
        for name in names:
            y6 = evaluate(ctx, rung_config(cfg, LADDER, name))["y6"]
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


def test_ladder_refuses_bad_base(leaf_ctx, leaf_runs):
    for s in ("lift.depth=det_depth", "stages.det=oracle_noisy"):
        with pytest.raises(ValueError, match=f"ladder base config has {re.escape(s)}"):
            run_ladder(leaf_ctx[0], fake_cfg(leaf_runs[0], "apple", s), LADDER, ["L3"])


def test_stop_after_y1_and_naive(apple_run):
    ctx, cfg = _ctx(apple_run, "apple", "stop_after=y1")
    res = evaluate(ctx, cfg)
    assert res["y6"] is None
    assert res["y1"]["summary"]["naive_U"] == sum(p["U1"] for p in res["y1"]["passes"])


def test_unsupported_options_refused(apple_run):
    for s in ("stages.det=oracle_noisy", "stages.dedup=hybrid", "noise.pose_trans_sigma_m=0.01"):
        with pytest.raises(NotImplementedError, match="is not supported in this version"):
            _ctx(apple_run, "apple", s)


def test_patched_ladder_is_checked_against_prepare(apple_run, monkeypatch):
    """A hand-edited rung may not use unsupported settings or change what prepare fixed (frames, detections)."""
    ctx, cfg = _ctx(apple_run, "apple")
    refused = {"stages.det=oracle_noisy": {"stages": {"det": "oracle_noisy"}},
               "stages.dedup=hybrid": {"stages": {"dedup": "hybrid"}},
               "detector.cache": {"detector": {"cache": "x"}},
               "detector.noise": {"detector": {"noise": {"p": 0.1}}},
               "noise": {"noise": {"pose_trans_sigma_m": 0.01}}}
    differs = {"detector.positives": {"detector": {"positives": "any"}},
               "passes": {"passes": {"pitches": [30.0]}},
               "crop": {"crop": "pear"},
               "split": {"split": "val"}}
    calls = []
    monkeypatch.setattr(runner_module, "evaluate", lambda c, k: calls.append(k) or evaluate(c, k))
    cases = [(NotImplementedError, f"{k} is not supported in this version", p) for k, p in refused.items()]
    cases += [(ValueError, f"{k}=.* differs from the prepared {k}=", p) for k, p in differs.items()]
    for exc, phrase, patch in cases:
        ladder = {"rungs": {"L0": LADDER["rungs"]["L0"], "bad": patch}}
        with pytest.raises(exc, match=phrase):
            evaluate(ctx, rung_config(cfg, ladder, "bad"))
        with pytest.raises(exc, match=phrase):
            run_ladder(ctx, cfg, ladder, ["L0", "bad"])
        assert calls == [], phrase   # checked before any rung runs
    ladder = {"rungs": LADDER["rungs"], "interaction_2x2": {"base": "L2a", "track": ["oracle"], "dedup": ["hybrid"]}}
    with pytest.raises(NotImplementedError, match="stages.dedup=hybrid is not supported in this version"):
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


def test_split_seed_checks(apple_run, tmp_path):
    with pytest.raises(ValueError, match="belongs to split"):
        _ctx(apple_run, "apple", "split=val")
    with pytest.raises(ValueError, match="blocked until test protection"):
        _ctx(apple_run, "apple", "split=test", "tuning=false")
    with pytest.raises(ValueError, match="belongs to split"):
        _ctx(write_run(tmp_path, crop="apple", pitches=(30.0,), seed=1000), "apple")


def test_non_sequence_refused(tmp_path):
    with pytest.raises(ValueError, match="not a sequence bundle"):
        _ctx(write_run(tmp_path, crop="apple", max_frames=10), "apple")


def test_missing_summary_refused(tmp_path):
    run = write_run(tmp_path, crop="apple", pitches=(30.0,))
    (run / "summary.json").unlink()
    with pytest.raises(ValueError, match="summary.json is missing"):
        _ctx(run, "apple")


def test_empty_truth_pass_gives_null(tmp_path):
    ctx, cfg = _ctx(write_run(tmp_path, crop="apple", pitches=(30.0, 80.0)), "apple")
    res = evaluate(ctx, cfg)
    p80 = [p for p in res["y1"]["passes"] if p["pitch"] == 80.0]
    assert len(p80) == 2 and all(p["G1"] == 0 and p["U1"] == 0 and p["e1"] is None for p in p80)
    assert res["y1"]["summary"]["e1_median"] is not None


def test_no_depth_tracks_counted(tmp_path):
    run = write_run(tmp_path, crop="apple", pitches=(30.0,))
    for p in run.glob("*_depth16.png"):
        Image.fromarray(np.zeros((360, 640), np.uint16)).save(p)
    ctx, cfg = _ctx(run, "apple")
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


@pytest.fixture(scope="module")
def leaf_ctx(leaf_runs):
    """Prepared contexts of the (clean, occluded) leaf bundles. Shared: evaluate them, never edit them."""
    return [prepare(SynthRun.open(root), fake_cfg(root, "apple"), load_splits()) for root in leaf_runs]


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
    assert _leaf_pred(occl) == _leaf_pred(clean)        # the far cluster is the fruit behind the patch


def test_soft_gate_absorbs_front_leaf_tail(leaf_ctx, leaf_runs):
    hard = _leaf(leaf_ctx, leaf_runs)[1]
    soft = _leaf(leaf_ctx, leaf_runs, "tracker.depth_gate=soft")[1]
    assert 5 * _dup(soft) < _dup(hard)


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


def test_depth_err_null_without_detector_depth(leaf_runs):
    ctx, cfg = _ctx(leaf_runs[0], "apple", "stop_after=y1")
    ctx.dets = {k: dataclasses.replace(d, depth_m=None) for k, d in ctx.dets.items()}
    res = evaluate(ctx, cfg)
    assert res["y1"]["summary"]["depth_err"] is None and all(p["depth_err"] is None for p in res["y1"]["passes"])


def test_oracle_dedup_keeps_background_apart():
    def pt(key, n):
        return PassTracks(pass_key=key, det_track={}, X=np.zeros((n, 3)), labels=np.zeros(n, np.int8),
                          n_obs=np.ones(n, int), spread=np.zeros(n), frames=[frozenset({i}) for i in range(n)])
    a, b = pt(PassKey(30.0, "A", 2), 2), pt(PassKey(30.0, "B", 1), 2)
    lm = oracle_dedup([a, b], [np.ones(2, bool)] * 2, [np.array([7, -1]), np.array([7, -1])])
    assert sorted(len(m) for m in lm.members) == [1, 1, 2]
