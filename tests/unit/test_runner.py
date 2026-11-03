import dataclasses
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import counting.evaluation.runner as runner_module
from counting.data.formats.orchard_synth import PassKey, SynthRun
from counting.evaluation.ledger import Ledger
from counting.evaluation.oracle_stages import oracle_dedup
from counting.evaluation.runner import evaluate, load_ladder, prepare, rung_config, run_ladder
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


def test_oracle_dedup_keeps_background_apart():
    def pt(key, n):
        return PassTracks(pass_key=key, det_track={}, X=np.zeros((n, 3)), labels=np.zeros(n, np.int8),
                          n_obs=np.ones(n, int), spread=np.zeros(n), frames=[frozenset({i}) for i in range(n)])
    a, b = pt(PassKey(30.0, "A", 2), 2), pt(PassKey(30.0, "B", 1), 2)
    lm = oracle_dedup([a, b], [np.ones(2, bool)] * 2, [np.array([7, -1]), np.array([7, -1])])
    assert sorted(len(m) for m in lm.members) == [1, 1, 2]
