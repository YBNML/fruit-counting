"""The cross-side join rule of the hybrid dedup (`dedup.band_join`, `dedup.band_join_cached`): unset, the two keys
stay out of the dump and the config hash; a run takes the key of its detections, learned-detector runs the cached
key alone; and the config check takes a rule only under a dedup mode that can hold it, with a band wider than half
the radius under `hybrid`. Under the rule the hybrid dedup also joins two tracks of different sides within the radius,
not both inside the band, when each is the other's nearest track across; the frame rule still holds, the other modes
ignore the rule, and without it the dedup gives the band rule's landmarks byte for byte. A run's dedup takes the rule of
its detections; the pose settings name it and the mechanism counts the same-fruit pairs it leaves apart only where the
hybrid dedup took a rule, and the rule's predictions ignore the truth ids."""

import hashlib
import json
import re
from typing import get_args

import numpy as np
import pytest
from pydantic import ValidationError

import counting.evaluation.runner as runner_module
import counting.multiview.dedup as dedup_module
from counting.config.eval_schema import DedupConfig, EvalAppConfig, StagesConfig
from counting.config.hashing import config_hash
from counting.config.loader import load_yaml_model
from counting.data.formats.orchard_synth import OrchardLayout, PassKey, SynthRun
from counting.evaluation.mechanism import band_blocked_pairs, mechanism_entry
from counting.evaluation.posesource import pose_settings
from counting.evaluation.runner import evaluate, prepare
from counting.tracking.landmark import PassTracks
from tests.unit.cachefake import write_truth_cache
from tests.unit.synthfake import ROOT, ctx_with, fake_cfg, load_splits
from tests.unit.test_depth_mechanism import LAYOUT as ROW_LAYOUT, ROW_Y, _pass
from tests.unit.test_leak import _permute, _pred, _random_src
from tests.unit.test_lidar_runner import DRIFT, _exact, lidar_case

pytestmark = pytest.mark.pose

EVAL = ROOT / "configs" / "eval"
KEYS = ("band_join", "band_join_cached")

LAYOUT = OrchardLayout("apple", (0.0, 3.75, 7.5), 1, 3.75, (3, 4, 5), (0.0, 1.25, 2.5))
C = 3.75   # row_y of LAYOUT
PASS_A30, PASS_B30, PASS_A50, PASS_B50 = (
    PassKey(30.0, "A", 2), PassKey(30.0, "B", 1), PassKey(50.0, "A", 2), PassKey(50.0, "B", 1))
BAND = DedupConfig(radius_m=0.09, band_m=0.05)
# the band rule's landmarks of these 300 draws (members, then points), as the code gave them before the join rule
KEYLESS_SHA256 = "e01687f25297dd4a3c9d9e2606efa9d2236ef58f6deec99c57cdffb0fa6756cc"
DIFFERING_DRAWS = 160   # the count this rule gives on these 300 draws


def _cfg(*sets: str) -> EvalAppConfig:
    """apple_dev_pose_c2.yaml (stages.dedup=hybrid, dedup radius_m 0.09 and band_m 0.05) with the overrides `sets`."""
    return load_yaml_model(EVAL / "apple_dev_pose_c2.yaml", EvalAppConfig, overrides=sets)


def _pt(key: PassKey, X, frames=None) -> PassTracks:
    """Tracks of pass `key` at points `X`: label 0, one observation each, track i seen in frame i unless `frames`."""
    X = np.asarray(X, float)
    n = len(X)
    return PassTracks(pass_key=key, det_track={}, X=X, labels=np.zeros(n, np.int8), n_obs=np.ones(n, int),
                      spread=np.zeros(n), frames=frames or [frozenset({i}) for i in range(n)])


def _all(*pts: PassTracks) -> list[np.ndarray]:
    return [np.ones(len(p.X), bool) for p in pts]


def _sets(lm) -> set[frozenset]:
    return {frozenset(m) for m in lm.members}


def _bytes(lm) -> tuple[bytes, bytes]:
    """The landmarks as bytes: the members' repr, then the points."""
    return repr(lm.members).encode(), lm.X.tobytes()


def _rule(pts: list[PassTracks], join: str | None, cfg: DedupConfig = BAND) -> set[frozenset]:
    """Landmark member sets of the hybrid dedup of all tracks of `pts` under the cross-side rule `join`."""
    return _sets(dedup_module.dedup(pts, _all(*pts), LAYOUT, cfg, mode="hybrid", join=join))


def _draws(n: int):
    """`n` draws of twelve tracks in each of four passes, spread across the centre plane (seed 7, draw by draw)."""
    rng = np.random.default_rng(7)
    for _ in range(n):
        yield [_pt(k, np.c_[rng.uniform(0, 1.0, 12), rng.normal(C, 0.06, 12), rng.uniform(1, 2, 12)])
               for k in (PASS_A30, PASS_B30, PASS_A50, PASS_B50)]


def test_keyless_dedup_dumps_two_keys():
    assert DedupConfig().model_dump() == {"radius_m": 0.04, "band_m": 0.10}
    assert DedupConfig(band_join=None).model_dump() == {"radius_m": 0.04, "band_m": 0.10}
    assert DedupConfig(band_join_cached=None).model_dump() == {"radius_m": 0.04, "band_m": 0.10}
    assert DedupConfig(radius_m=0.09, band_m=0.05, band_join="either_mutual").model_dump() == {
        "radius_m": 0.09, "band_m": 0.05, "band_join": "either_mutual"}
    for key in KEYS:
        for bad in ("both", "always"):
            with pytest.raises(ValidationError):
                DedupConfig(**{key: bad})
    assert config_hash(_cfg()) == "e5f1d600adb6ec4b"
    for key in KEYS:
        assert config_hash(_cfg(f"dedup.{key}=either_mutual")) != "e5f1d600adb6ec4b"


def test_band_join_for_reads_the_cached_key_alone():
    dets = get_args(StagesConfig.model_fields["det"].annotation)
    assert dets == ("oracle", "oracle_noisy", "cached")
    plain = DedupConfig(band_join="either_mutual")
    cached = DedupConfig(band_join_cached="either_mutual")
    both = DedupConfig(band_join="either_mutual", band_join_cached="either_mutual")
    assert {det: plain.band_join_for(det) for det in dets} == {
        "oracle": "either_mutual", "oracle_noisy": "either_mutual", "cached": None}
    assert {det: cached.band_join_for(det) for det in dets} == {
        "oracle": None, "oracle_noisy": None, "cached": "either_mutual"}
    assert {det: both.band_join_for(det) for det in dets} == {det: "either_mutual" for det in dets}
    assert {det: DedupConfig().band_join_for(det) for det in dets} == {det: None for det in dets}


def test_join_rule_needs_hybrid_or_oracle():
    with pytest.raises(ValueError, match=re.escape(
            "dedup.band_join=either_mutual needs stages.dedup=hybrid or oracle, got stages.dedup=self_side")):
        _cfg("stages.dedup=self_side", "dedup.band_join=either_mutual")
    for key in KEYS:
        for mode in ("self_side", "merge"):
            with pytest.raises(ValueError, match=re.escape(
                    f"dedup.{key}=either_mutual needs stages.dedup=hybrid or oracle, got stages.dedup={mode}")):
                _cfg(f"stages.dedup={mode}", f"dedup.{key}=either_mutual")
        assert getattr(_cfg("stages.dedup=oracle", f"dedup.{key}=either_mutual").dedup, key) == "either_mutual"
    both = _cfg("stages.dedup=oracle", "dedup.band_join=either_mutual", "dedup.band_join_cached=either_mutual")
    assert (both.dedup.band_join, both.dedup.band_join_cached) == ("either_mutual", "either_mutual")
    for mode in ("self_side", "merge"):
        assert _cfg(f"stages.dedup={mode}").stages.dedup == mode


def test_join_rule_needs_a_band_wider_than_half_the_radius():
    narrow = "needs 2 * dedup.band_m > dedup.radius_m, got band_m 0.045 and radius_m 0.09"
    with pytest.raises(ValueError, match=re.escape(f"dedup.band_join=either_mutual {narrow}")):
        _cfg("dedup.band_m=0.045", "dedup.band_join=either_mutual")
    for key in KEYS:
        with pytest.raises(ValueError, match=re.escape(f"dedup.{key}=either_mutual {narrow}")):
            _cfg("dedup.band_m=0.045", f"dedup.{key}=either_mutual")
        held = _cfg("stages.dedup=oracle", "dedup.band_m=0.045", f"dedup.{key}=either_mutual")
        assert getattr(held.dedup, key) == "either_mutual"
        assert getattr(_cfg("dedup.band_m=0.05", f"dedup.{key}=either_mutual").dedup, key) == "either_mutual"
    assert _cfg("dedup.band_m=0.045").dedup.band_m == 0.045


def test_one_band_pair_joins_under_the_join_rule():
    """A cross-side pair within the radius with one track outside the band stays apart under the band rule; the join
    rule joins it, each track being the other's nearest across."""
    pts = [_pt(PASS_A30, [[2.0, C + 0.07, 1.5]]), _pt(PASS_B30, [[2.0, C + 0.01, 1.5]])]
    # landmarks of the rule as written
    assert _rule(pts, None) == {frozenset({(PASS_A30, 0)}), frozenset({(PASS_B30, 0)})}
    assert _rule(pts, "either_mutual") == {frozenset({(PASS_A30, 0), (PASS_B30, 0)})}


def test_join_rule_needs_mutual_nearest_tracks():
    """The aisle-B track's nearest aisle-A track is the 50-degree one, so of its two cross-side pairs within the radius
    only that one joins; the 30-degree aisle-A track, whose nearest track across is the aisle-B one, stays alone."""
    pts = [_pt(PASS_A30, [[2.06, C + 0.07, 1.5]]), _pt(PASS_A50, [[1.96, C + 0.06, 1.5]]),
           _pt(PASS_B30, [[2.0, C + 0.01, 1.5]])]
    # landmarks of the rule as written
    assert _rule(pts, None) == {frozenset({(PASS_A30, 0)}), frozenset({(PASS_A50, 0)}), frozenset({(PASS_B30, 0)})}
    assert _rule(pts, "either_mutual") == {frozenset({(PASS_A30, 0)}), frozenset({(PASS_A50, 0), (PASS_B30, 0)})}


def test_join_rule_keeps_the_frame_rule():
    """Two tracks of one pass seen in a common frame never share a landmark: under the join rule the closer pair
    joins first and then blocks the band join of the two aisle-B tracks."""
    pts = [_pt(PASS_A30, [[2.0, C + 0.07, 1.5], [2.15, C + 0.02, 1.5]], frames=[frozenset({0}), frozenset({0})]),
           _pt(PASS_B30, [[2.0, C + 0.01, 1.5]]), _pt(PASS_B50, [[2.08, C + 0.01, 1.5]])]
    # landmarks of the rule as written
    assert _rule(pts, None) == {frozenset({(PASS_A30, 0)}),
                                frozenset({(PASS_A30, 1), (PASS_B30, 0), (PASS_B50, 0)})}
    assert _rule(pts, "either_mutual") == {frozenset({(PASS_A30, 0), (PASS_B30, 0)}),
                                           frozenset({(PASS_A30, 1), (PASS_B50, 0)})}


def test_band_edge_pair_follows_the_mutual_rule():
    """An aisle-A track on its bound is kept by the side filter and outside the band; the mutual rule still joins it
    to its nearest track across."""
    edge = DedupConfig(radius_m=0.09, band_m=0.25)
    pts = [_pt(PASS_A30, [[1.0, 3.5, 1.6]]), _pt(PASS_B30, [[1.0, 3.45, 1.6]])]
    assert LAYOUT.row_y - edge.band_m == 3.5 and not abs(3.5 - LAYOUT.row_y) < edge.band_m
    assert dedup_module.side_filter(pts[0], LAYOUT, "hybrid", edge.band_m).tolist() == [True]
    # landmarks of the rule as written
    assert _rule(pts, None, edge) == {frozenset({(PASS_A30, 0)}), frozenset({(PASS_B30, 0)})}
    assert _rule(pts, "either_mutual", edge) == {frozenset({(PASS_A30, 0), (PASS_B30, 0)})}


def test_wide_band_leaves_no_cross_pair_outside_it():
    """With 2 * band_m > radius_m no two entered tracks of different sides within the radius lie both outside the
    band; without a rule (no argument or None) the dedup gives the band rule's landmarks byte for byte; the join rule
    changes the landmarks of DIFFERING_DRAWS of the 300 draws and gives the same bytes when called again."""
    outside = differing = 0
    plain_sha, none_sha = hashlib.sha256(), hashlib.sha256()
    for pts in _draws(300):
        keep = _all(*pts)
        entered = [p.X[dedup_module.side_filter(p, LAYOUT, "hybrid", BAND.band_m)] for p in pts]
        X = np.concatenate(entered)
        side = np.array([p.pass_key.side for p, x in zip(pts, entered) for _ in x])
        out = ~(np.abs(X[:, 1] - LAYOUT.row_y) < BAND.band_m)
        close = np.linalg.norm(X[:, None] - X[None], axis=2) <= BAND.radius_m
        outside += int(np.triu(close & (side[:, None] != side[None]) & out[:, None] & out[None], k=1).sum())
        plain = dedup_module.dedup(pts, keep, LAYOUT, BAND, mode="hybrid")
        none = dedup_module.dedup(pts, keep, LAYOUT, BAND, mode="hybrid", join=None)
        for sha, lm in ((plain_sha, plain), (none_sha, none)):
            for part in _bytes(lm):
                sha.update(part)
        joined = dedup_module.dedup(pts, keep, LAYOUT, BAND, mode="hybrid", join="either_mutual")
        again = dedup_module.dedup(pts, keep, LAYOUT, BAND, mode="hybrid", join="either_mutual")
        assert _bytes(again) == _bytes(joined)
        differing += joined.members != none.members
    assert outside == 0
    assert (plain_sha.hexdigest(), none_sha.hexdigest()) == (KEYLESS_SHA256, KEYLESS_SHA256)
    assert differing == DIFFERING_DRAWS


def test_other_modes_ignore_the_join_rule():
    """`self_side` and `merge` give the same bytes with the join rule as without it; a rule name outside the known
    ones is refused under every mode."""
    for pts in _draws(20):
        keep = _all(*pts)
        for mode in ("self_side", "merge"):
            plain = dedup_module.dedup(pts, keep, LAYOUT, BAND, mode=mode)
            assert _bytes(dedup_module.dedup(pts, keep, LAYOUT, BAND, mode=mode, join="either_mutual")) == _bytes(plain)
    pts = next(_draws(1))
    for mode in ("self_side", "merge", "hybrid"):
        for bad in ("both", "always"):
            with pytest.raises(ValueError, match="unknown dedup join rule"):
                dedup_module.dedup(pts, _all(*pts), LAYOUT, BAND, mode=mode, join=bad)


POSE_KEYS = {"config", "source", "drift", "anchor", "start", "rig", "rig_ref", "fuse", "align", "predict",
             "predict_window", "dedup", "dedup_radius_m", "dedup_band_m", "gate_px_frac", "cache", "imu", "fusion",
             "flow"}
HYBRID = {"radius_m": 0.09, "band_m": 0.05}


def _spy(monkeypatch, name: str) -> list:
    """Wrap the runner's `name` (`dedup`, which `_y6` calls, or `band_blocked_pairs`, which the mechanism counts call)
    so that it records the `join` of each call; returns the record."""
    seen, real = [], getattr(runner_module, name)

    def spy(*args, join=None, **kw):
        seen.append(join)
        return real(*args, join=join, **kw)

    monkeypatch.setattr(runner_module, name, spy)
    return seen


def test_pose_settings_name_the_join_rule_only_when_set():
    """The pose settings name the rule the hybrid dedup took: a cached run takes the cached key alone, and an oracle
    dedup takes none."""
    assert set(pose_settings(_cfg(), 0)) == POSE_KEYS
    on = pose_settings(_cfg("dedup.band_join=either_mutual"), 0)
    assert set(on) == POSE_KEYS | {"dedup_band_join"} and on["dedup_band_join"] == "either_mutual"
    cached = ("stages.det=cached", "detector.cache=runs/none", "dedup.band_join=either_mutual")
    assert set(pose_settings(_cfg(*cached), 0)) == POSE_KEYS
    both = pose_settings(_cfg(*cached, "dedup.band_join_cached=either_mutual"), 0)
    assert set(both) == POSE_KEYS | {"dedup_band_join"} and both["dedup_band_join"] == "either_mutual"
    assert set(pose_settings(_cfg("stages.dedup=oracle", "dedup.band_join=either_mutual"), 0)) == POSE_KEYS


def test_mechanism_counts_cross_pairs_apart_only_with_the_rule():
    """Under the rule the pairs the band keeps apart are counted again in the rule's landmarks: the one-sided pair of
    one fruit joins, the pair whose aisle-B track is nearer another aisle-A track stays apart. The mechanism entry
    gets `cross_pairs_apart` only when given one."""
    a, src_a = _pass(PASS_A30, 0, [[2.0, ROW_Y + 0.07, 1.5]], [[7]])
    b, src_b = _pass(PASS_B30, 100, [[2.0, ROW_Y + 0.01, 1.5]], [[7]])
    assert band_blocked_pairs([a, b], {**src_a, **src_b}, ROW_LAYOUT, BAND) == 1
    assert band_blocked_pairs([a, b], {**src_a, **src_b}, ROW_LAYOUT, BAND, join="either_mutual") == 0

    a, src_a = _pass(PASS_A30, 0, [[2.06, ROW_Y + 0.07, 1.5]], [[7]])
    m, src_m = _pass(PASS_A50, 200, [[1.96, ROW_Y + 0.06, 1.5]], [[8]])
    b, src_b = _pass(PASS_B30, 100, [[2.0, ROW_Y + 0.01, 1.5]], [[7]])
    passes, src = [a, m, b], {**src_a, **src_m, **src_b}
    assert band_blocked_pairs(passes, src, ROW_LAYOUT, BAND) == 1
    assert band_blocked_pairs(passes, src, ROW_LAYOUT, BAND, join="either_mutual") == 1

    keyless = mechanism_entry([], np.array([]), 1)
    assert list(keyless) == ["soft_depth_pairs", "births", "fragments", "band_blocked_pairs"]
    entry = mechanism_entry([], np.array([]), 1, cross_pairs_apart=0)
    assert list(entry) == [*keyless, "cross_pairs_apart"] and entry["cross_pairs_apart"] == 0
    assert {k: entry[k] for k in keyless} == keyless


def test_evaluate_passes_the_rule_of_the_detector(leaf_runs, tmp_path, monkeypatch):
    """`evaluate` hands the dedup and the mechanism counts the rule of the run's detections: oracle detections take
    `band_join`, cached ones `band_join_cached` alone, and an oracle dedup no rule at all. Cached detections run
    without the LiDAR files (the dedup) and with them (the dedup, the mechanism counts and the pose settings)."""
    case = lidar_case()
    base = case.config(DRIFT, {"stages": {"dedup": "hybrid"}, "dedup": HYBRID})
    join = case.config(DRIFT, {"stages": {"dedup": "hybrid"}, "dedup": {**HYBRID, "band_join": "either_mutual"}})
    cached_only = case.config(DRIFT, {"stages": {"dedup": "hybrid"},
                                      "dedup": {**HYBRID, "band_join_cached": "either_mutual"}})
    o0 = case.config(DRIFT, {"stages": {"dedup": "oracle"}, "dedup": HYBRID})
    o1 = case.config(DRIFT, {"stages": {"dedup": "oracle"}, "dedup": {**HYBRID, "band_join": "either_mutual"}})
    with monkeypatch.context() as patch:
        dedups, blocks = _spy(patch, "dedup"), _spy(patch, "band_blocked_pairs")
        for cfg, taken, counted in ((base, [None], [None]), (join, ["either_mutual"], [None, "either_mutual"]),
                                    (cached_only, [None], [None]), (o1, [], [])):
            evaluate(case.ctx, cfg, noise_seed=0)
            assert (dedups, blocks) == (taken, counted), cfg.dedup
            dedups.clear()
            blocks.clear()

    a, b = case.evaluate(base, 0), case.evaluate(join, 0)
    assert _exact(a, ("y1",)) == _exact(b, ("y1",))
    assert a["mechanism"]["band_blocked_pairs"] == b["mechanism"]["band_blocked_pairs"]
    assert "cross_pairs_apart" not in a["mechanism"] and isinstance(b["mechanism"]["cross_pairs_apart"], int)
    assert "dedup_band_join" not in a["pose"] and b["pose"]["dedup_band_join"] == "either_mutual"
    c = case.evaluate(cached_only, 0)
    assert json.dumps([c["mechanism"], c["pose"]]) == json.dumps([a["mechanism"], a["pose"]])

    held = case.evaluate(o1, 0)
    assert _exact(case.evaluate(o0, 0), ("y6",)) == _exact(held, ("y6",))
    assert "dedup_band_join" not in held["pose"] and "cross_pairs_apart" not in held["mechanism"]

    root = leaf_runs[0]
    write_truth_cache(root, tmp_path, shift=-0.2, low_score=0.07)
    common = ("passes.step_m=0.45", f"detector.cache={tmp_path}", "stages.det=cached", "tracker.score_lo=0.05",
              "stages.dedup=hybrid", "dedup.radius_m=0.09", "dedup.band_m=0.05")
    ctx = prepare(SynthRun.open(root), fake_cfg(root, "apple", *common), load_splits())
    with monkeypatch.context() as patch:
        dedups = _spy(patch, "dedup")
        evaluate(ctx, fake_cfg(root, "apple", *common, "dedup.band_join=either_mutual"))
        assert dedups == [None]           # a learned-detector run never takes band_join
        dedups.clear()
        evaluate(ctx, fake_cfg(root, "apple", *common, "dedup.band_join_cached=either_mutual"))
        assert dedups == ["either_mutual"]

    lidar_cache = tmp_path / "lidar"
    write_truth_cache(case.shared.bundle, lidar_cache, shift=-0.2, low_score=0.07)
    learned = {"stages": {"det": "cached", "dedup": "hybrid"}, "detector": {"cache": str(lidar_cache)},
               "tracker": {"score_lo": 0.05}, "pose": {}}
    ctx = prepare(case.run, case.config(learned, {"dedup": HYBRID}), load_splits())
    with monkeypatch.context() as patch:
        dedups, blocks = _spy(patch, "dedup"), _spy(patch, "band_blocked_pairs")
        for key, taken, counted in (("band_join", [None], [None]),
                                    ("band_join_cached", ["either_mutual"], [None, "either_mutual"])):
            res = evaluate(ctx, case.config(learned, {"dedup": {**HYBRID, key: "either_mutual"}}))
            assert (dedups, blocks) == (taken, counted), key
            named = key == "band_join_cached"
            assert ("cross_pairs_apart" in res["mechanism"], "dedup_band_join" in res["pose"]) == (named, named), key
            dedups.clear()
            blocks.clear()


def test_join_rule_predictions_ignore_truth_ids(apple_run, monkeypatch):
    """Under the rule, relabelled truth or random source ids per frame change no prediction."""
    run, splits = SynthRun.open(apple_run), load_splits()
    cfg = fake_cfg(apple_run, "apple", "stages.dedup=hybrid", "dedup.radius_m=0.09", "dedup.band_m=0.05",
                   "dedup.band_join=either_mutual")
    ctx = prepare(run, cfg, splits)
    dedups = _spy(monkeypatch, "dedup")
    a = evaluate(ctx, cfg)
    b = evaluate(prepare(run, cfg, splits, truth=_permute(run.truth(), np.random.default_rng(1))), cfg)
    c = evaluate(ctx_with(ctx, src=_random_src(ctx, np.random.default_rng(2))), cfg)
    assert _pred(a) == _pred(b) == _pred(c)
    assert dedups == ["either_mutual"] * 3
