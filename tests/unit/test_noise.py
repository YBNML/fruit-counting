from pathlib import Path

import numpy as np
import pytest

import counting.detection.oracle as oracle_module
import counting.evaluation.runner as runner_module
from counting.config.eval_schema import DetNoise
from counting.data.formats.orchard_synth import FrameInput, FruitGt, GtRecord, PassKey, SceneTruth, SynthRun
from counting.detection.base import box_iou
from counting.detection.oracle import OracleDetector
from counting.evaluation.report import results_bytes
from counting.evaluation.runner import evaluate, prepare, run_config
from counting.evaluation.seeds import combine_seeds
from counting.geometry.camera import Camera
from counting.geometry.noise import noise_rng, pass_bias, pass_key_code, perturb_depth
from tests.unit.synthfake import DET_NOISE, fake_cfg, load_splits, robot_matrix
from tests.unit.test_detection_match import FRAME, TRUTH


def _cam():
    return Camera.from_blender(robot_matrix(0.4, 3.75, "A", 30.0, 1.75, 0.9), 640, 360, 69.0)


def test_noise_rng_is_keyed_and_rejects_negative_parts():
    a = noise_rng(0, 0, 2, 5, 17).standard_normal(4)
    assert np.array_equal(a, noise_rng(0, 0, 2, 5, 17).standard_normal(4))
    assert np.array_equal(a, np.random.default_rng([0, 0, 2, 5, 17]).standard_normal(4))
    assert not np.array_equal(a, noise_rng(0, 1, 2, 5, 17).standard_normal(4))
    with pytest.raises(ValueError, match="negative part"):
        noise_rng(0, -1, 2)


def test_pass_key_code():
    assert pass_key_code(PassKey(30.0, "A", 2)) == (30000, 0, 2)
    assert pass_key_code(PassKey(45.5, "B", 1)) == (45500, 1, 1)


def test_perturbed_with_zero_sigma_keeps_matrices():
    cam = _cam()
    p = cam.perturbed(noise_rng(1), 0.0, 0.0)
    assert p is not cam and (p.width, p.height) == (cam.width, cam.height)
    assert np.array_equal(p.K, cam.K) and np.array_equal(p.R_c2w, cam.R_c2w) and np.array_equal(p.C, cam.C)


def test_perturbed_moves_pose_by_the_drawn_normals():
    cam = _cam()
    rng = noise_rng(3)
    t, w = rng.standard_normal(3), rng.standard_normal(3)
    p = cam.perturbed(noise_rng(3), 0.02, 0.5, bias_m=np.array([0.0, 0.0, 0.01]))
    np.testing.assert_allclose(p.C, cam.C + 0.02 * t + [0.0, 0.0, 0.01], atol=1e-12)
    R = cam.R_c2w.T @ p.R_c2w                                   # Rot(omega) in the camera frame
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.degrees(np.arccos((np.trace(R) - 1) / 2)) == pytest.approx(0.5 * np.linalg.norm(w), rel=1e-6)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    np.testing.assert_allclose(axis / np.linalg.norm(axis), w / np.linalg.norm(w), atol=1e-9)
    assert np.array_equal(p.K, cam.K)


def test_perturbed_draws_both_normals_whatever_the_sigmas():
    cam = _cam()
    both = cam.perturbed(noise_rng(3), 0.02, 0.5)
    assert np.array_equal(cam.perturbed(noise_rng(3), 0.0, 0.5).R_c2w, both.R_c2w)
    assert np.array_equal(cam.perturbed(noise_rng(3), 0.02, 0.0).C, both.C)


def test_pass_bias_has_length_b_and_one_direction():
    d = [pass_bias(noise_rng(0, 0, 3, 5, 30000, 0, 2), b) for b in (0.0, 0.02, 0.10)]
    assert np.array_equal(d[0], np.zeros(3))
    assert [float(np.linalg.norm(v)) for v in d] == pytest.approx([0.0, 0.02, 0.10])
    np.testing.assert_allclose(d[2] / 0.10, d[1] / 0.02, atol=1e-12)    # same draw, only the length changes


def test_perturb_depth_scales_and_drops_from_one_draw():
    z = np.array([1.0, 2.0, 3.0, np.nan, 2.5] * 40)
    g = noise_rng(9).standard_normal(len(z))
    assert np.array_equal(perturb_depth(z, noise_rng(9), 0.0, 0.0), z, equal_nan=True)
    out = perturb_depth(z, noise_rng(9), 0.02, 0.0)
    np.testing.assert_allclose(out, z * (1 + 0.02 * g))
    drops = [np.isnan(perturb_depth(z, noise_rng(9), 0.02, f)) for f in (0.05, 0.20)]
    assert drops[0].sum() < drops[1].sum() and not (drops[0] & ~drops[1]).any()   # nested as the fraction grows
    assert out is not z and np.array_equal(z[:5], [1.0, 2.0, 3.0, np.nan, 2.5], equal_nan=True)


KEY = (0, 0, 1, 5)


def _many(n=60, seed=0, bag_even=False):
    """One 640 x 360 frame with n positive square boxes, visib 0.4 or 0.9.

    With `bag_even`, a fruit is bagged exactly when its box width is even.
    """
    rng = np.random.default_rng(seed)
    recs, fruits = [], {}
    for i in range(n):
        x0, y0, w = int(rng.integers(0, 600)), int(rng.integers(0, 330)), int(rng.integers(12, 30))
        recs.append(GtRecord(1000 + i, 3, 400, 400, float(rng.choice([0.4, 0.9])),
                             (x0, y0, x0 + w, min(y0 + w, 360)), False, float(2 + rng.random())))
        fruits[1000 + i] = FruitGt(1000 + i, 3, np.zeros(3), 0.07, bag_even and w % 2 == 0)
    frame = FrameInput("t", 0, PassKey(30.0, "A", 2), 0, Camera.from_blender(np.eye(4), 640, 360, 69.0),
                       0.0, Path("r"), Path("d"))
    return SceneTruth({0: tuple(recs)}, fruits, {3: n}, (0.25, 20), None), frame


def _noisy(truth, frame, key=KEY, **kw):
    return OracleDetector(truth, noise=DetNoise(**kw), rng_key=key).detect(frame)


def _same(a, b):
    return all(np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True)
               for f in ("boxes", "scores", "labels", "depth_m"))


def test_zero_detection_noise_equals_the_oracle():
    truth, frame = _many()
    assert _same(_noisy(truth, frame), OracleDetector(truth).detect(frame))
    assert _same(_noisy(TRUTH, FRAME), OracleDetector(TRUTH).detect(FRAME))
    assert OracleDetector(truth).name == "oracle"
    assert OracleDetector(truth, noise=DetNoise(), rng_key=KEY).name == "oracle_noisy"


def test_detection_noise_is_keyed():
    truth, frame = _many()
    kw = dict(miss_p=0.2, fp_lambda=2.0, jitter_frac=0.05, split_p=0.1, score="uniform")
    a = _noisy(truth, frame, **kw)
    assert _same(a, _noisy(truth, frame, **kw))
    assert not _same(a, _noisy(truth, frame, (0, 1, 1, 5), **kw))          # another noise seed
    assert not _same(a, _noisy(truth, frame, (0, 0, 1, 6), **kw))          # another scene seed


def test_misses_grow_as_nested_sets():
    truth, frame = _many()
    clean = {tuple(b) for b in OracleDetector(truth).detect(frame).boxes.tolist()}
    kept = [{tuple(b) for b in _noisy(truth, frame, miss_p=p).boxes.tolist()} for p in (0.1, 0.3)]
    assert kept[1] <= kept[0] <= clean and len(kept[1]) < len(kept[0]) < len(clean)
    assert len(_noisy(truth, frame, miss_p=1.0)) == 0
    jit = {tuple(b) for b in _noisy(truth, frame, jitter_frac=0.05, split_p=0.1).boxes.tolist()}
    both = {tuple(b) for b in _noisy(truth, frame, jitter_frac=0.05, split_p=0.1, miss_p=0.3).boxes.tolist()}
    assert both < jit                                   # a miss leaves every other draw as it was


def test_false_positives_grow_by_prefix_and_have_no_depth():
    truth, frame = _many()
    clean = OracleDetector(truth).detect(frame)
    sizes = {(round(x1 - x0), round(y1 - y0)) for x0, y0, x1, y1 in clean.boxes.tolist()}
    totals = {1.0: 0, 5.0: 0}
    for ns in range(5):
        fp = {}
        for lam in (1.0, 5.0):
            d = _noisy(truth, frame, (0, ns, 1, 5), fp_lambda=lam)
            extra = np.isnan(d.depth_m)
            assert {tuple(b) for b in d.boxes[~extra].tolist()} == {tuple(b) for b in clean.boxes.tolist()}
            b = d.boxes[extra]
            assert (b[:, :2] >= 0).all() and (b[:, 2] <= 640).all() and (b[:, 3] <= 360).all()
            assert {(round(x1 - x0), round(y1 - y0)) for x0, y0, x1, y1 in b.tolist()} <= sizes
            fp[lam] = {tuple(x) for x in b.tolist()}
            totals[lam] += len(fp[lam])
        assert fp[1.0] <= fp[5.0]                               # the same draws: a larger lambda only adds boxes
    assert totals[5.0] > totals[1.0]


def test_false_positives_do_not_move_with_the_split_or_score_factor():
    truth, frame = _many()

    def fp(**kw):
        d = _noisy(truth, frame, fp_lambda=5.0, **kw)
        return d.boxes[np.isnan(d.depth_m)]

    base = fp()
    assert len(base) > 0
    for kw in (dict(split_p=0.1), dict(score="uniform")):
        assert np.array_equal(fp(**kw), base), kw


def test_false_positive_labels_come_from_the_box_they_copy():
    truth, frame = _many(bag_even=True)
    seen = set()
    for ns in range(5):
        d = _noisy(truth, frame, (0, ns, 1, 5), fp_lambda=5.0)
        extra = np.isnan(d.depth_m)
        width = np.round(d.boxes[extra, 2] - d.boxes[extra, 0]).astype(int)
        assert np.array_equal(d.labels[extra], (width % 2 == 0).astype(np.int8))   # a copied size keeps its label
        seen |= set(d.labels[extra].tolist())
    assert seen == {0, 1}


def test_jitter_scales_with_the_box_and_stays_in_the_image():
    truth, frame = _many()
    clean = OracleDetector(truth).detect(frame)
    small = _noisy(truth, frame, jitter_frac=0.05)
    assert len(small) == len(clean) and not _same(small, clean)
    assert (box_iou(small.boxes, clean.boxes).max(axis=1) >= 0.5).all()
    big = _noisy(truth, frame, jitter_frac=1.0)
    w, h = big.boxes[:, 2] - big.boxes[:, 0], big.boxes[:, 3] - big.boxes[:, 1]
    assert 0 < len(big) <= len(clean) and (w >= 1).all() and (h >= 1).all()
    assert (big.boxes[:, :2] >= 0).all() and (big.boxes[:, 2] <= 640).all() and (big.boxes[:, 3] <= 360).all()


def test_split_halves_low_visibility_boxes_along_the_long_side():
    truth, frame = _many()
    clean = OracleDetector(truth).detect(frame)
    low = sum(r.visib < 0.6 for r in truth.records[0])
    d = _noisy(truth, frame, split_p=1.0)
    assert len(d) == len(clean) + low
    assert len(_noisy(truth, frame, split_p=1.0, split_visib_max=0.3)) == len(clean)   # visib 0.4 stays whole

    def area(b):
        return float(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).sum())

    assert area(d.boxes) == pytest.approx(area(clean.boxes))           # the halves cover their parent
    halves = _noisy(TRUTH, FRAME, split_p=1.0)                          # record 801: visib 0.3, box 6 x 5
    assert [b for b in halves.boxes.tolist() if b[1] == 40] == [[10, 40, 13, 45], [13, 40, 16, 45]]


def test_uniform_scores_lie_in_range_and_fixed_scores_are_one():
    truth, frame = _many()
    u = _noisy(truth, frame, score="uniform", score_min=0.3).scores
    assert u.min() >= 0.3 and u.max() <= 1.0 and u.min() < 0.5         # some cannot start a track
    assert (_noisy(truth, frame, miss_p=0.1).scores == 1.0).all()


def test_noisy_oracle_on_an_empty_frame():
    empty = SceneTruth({0: ()}, {}, {}, (0.25, 20), None)
    d = _noisy(empty, FRAME, fp_lambda=5.0, miss_p=0.5, split_p=1.0)
    assert len(d) == 0 and d.boxes.shape == (0, 4) and d.depth_m.shape == (0,)


DET = DET_NOISE
POSE = ("noise.pose_trans_sigma_m=0.02", "noise.pose_rot_sigma_deg=0.5", "noise.pass_bias_m=0.02",
        "noise.depth_rel_sigma=0.02", "noise.depth_dropout=0.05")


def _prep(root, *sets, det_seed=None):
    cfg = fake_cfg(root, "apple", *sets)
    return prepare(SynthRun.open(root), cfg, load_splits(), det_seed=det_seed), cfg


def _drop_stages(res):
    return {k: v for k, v in res.items() if k != "stages"}


def test_zero_noise_matches_the_noiseless_result(leaf_ctx, leaf_runs):
    root = leaf_runs[0]
    clean = evaluate(leaf_ctx[0], fake_cfg(root, "apple"))
    ctx, cfg = _prep(root, "stages.det=oracle_noisy", "detector.noise={}", det_seed=0)
    noisy = evaluate(ctx, cfg, noise_seed=0)
    assert noisy["stages"]["det"] == "oracle_noisy" and _drop_stages(noisy) == _drop_stages(clean)
    assert all(np.array_equal(ctx.dets[i].boxes, leaf_ctx[0].dets[i].boxes) for i in ctx.dets)


def test_pose_and_depth_noise_need_a_seed_and_repeat_per_seed(leaf_ctx, leaf_runs):
    cfg, plain = fake_cfg(leaf_runs[0], "apple", *POSE), fake_cfg(leaf_runs[0], "apple")
    with pytest.raises(ValueError, match="the config has noise, so evaluate needs a noise_seed"):
        evaluate(leaf_ctx[0], cfg)
    with pytest.raises(ValueError, match="noise_seed=0 given, but the config has no noise"):
        evaluate(leaf_ctx[0], plain, noise_seed=0)
    a, b, c = (evaluate(leaf_ctx[0], cfg, noise_seed=s) for s in (0, 0, 1))
    assert results_bytes(a) == results_bytes(b) != results_bytes(c)
    clean = evaluate(leaf_ctx[0], plain)
    assert a["y6"]["metrics"]["pos_err_p50_m"] != clean["y6"]["metrics"]["pos_err_p50_m"]   # noise reaches the points
    assert a["y1"]["summary"]["depth_err"] == clean["y1"]["summary"]["depth_err"]           # taken before depth noise


def test_pose_noise_reaches_the_tracker_and_the_points(leaf_ctx, leaf_runs, monkeypatch):
    root = leaf_runs[0]
    clean = evaluate(leaf_ctx[0], fake_cfg(root, "apple"))
    got, real = [], runner_module.track_pass
    monkeypatch.setattr(runner_module, "track_pass",
                        lambda frames, *a, **k: got.extend(frames) or real(frames, *a, **k))
    noisy = evaluate(leaf_ctx[0], fake_cfg(root, "apple", "noise.pose_trans_sigma_m=0.05"), noise_seed=0)
    C = {f.idx: f.camera.C for frames in leaf_ctx[0].passes.values() for f in frames}
    assert len(got) == len(C) and all(not np.array_equal(f.camera.C, C[f.idx]) for f in got)   # the tracker's poses
    assert noisy["y6"]["metrics"]["pos_err_p50_m"] != clean["y6"]["metrics"]["pos_err_p50_m"]   # the lifted points


def test_depth_noise_reaches_the_points(leaf_ctx, leaf_runs):
    root = leaf_runs[0]
    gone = evaluate(leaf_ctx[0], fake_cfg(root, "apple", "noise.depth_dropout=1.0"), noise_seed=0)
    assert all(p["U1"] == 0 and p["n_no3d"] == p["n_tracks"] > 0 for p in gone["y1"]["passes"])
    clean = evaluate(leaf_ctx[0], fake_cfg(root, "apple"))
    rel = evaluate(leaf_ctx[0], fake_cfg(root, "apple", "noise.depth_rel_sigma=0.05"), noise_seed=0)
    assert rel["y6"]["metrics"]["pos_err_p50_m"] != clean["y6"]["metrics"]["pos_err_p50_m"]


def test_detection_noise_repeats_per_seed_and_needs_its_prepared_seed(leaf_runs, leaf_det_ctx):
    root = leaf_runs[0]
    (c0b, cfg), c0, c1 = _prep(root, *DET, det_seed=0), leaf_det_ctx(0), leaf_det_ctx(1)   # c0b: a second prepare
    a, b, c = evaluate(c0, cfg, noise_seed=0), evaluate(c0b, cfg, noise_seed=0), evaluate(c1, cfg, noise_seed=1)
    assert results_bytes(a) == results_bytes(b) != results_bytes(c)
    with pytest.raises(ValueError, match="det_seed=1 differs from the prepared det_seed=0"):
        evaluate(c0, cfg, noise_seed=1)
    more_misses = fake_cfg(root, "apple", *(s.replace("miss_p=0.1", "miss_p=0.3") for s in DET))
    with pytest.raises(ValueError, match="detector.noise=.* differs from the prepared detector.noise="):
        evaluate(c0, more_misses, noise_seed=0)
    with pytest.raises(ValueError, match="the config has noise, so evaluate needs a noise_seed"):
        evaluate(c0, cfg)
    with pytest.raises(ValueError, match="stages.det=oracle_noisy needs a detection noise seed"):
        _prep(root, *DET)
    with pytest.raises(ValueError, match="det_seed=0 given, but stages.det=oracle adds no detection noise"):
        _prep(root, det_seed=0)


def _record_tracker_frames(monkeypatch) -> list:
    """The frames (with the cameras) `track_pass` receives, in call order."""
    got, real = [], runner_module.track_pass
    monkeypatch.setattr(runner_module, "track_pass",
                        lambda frames, *a, **k: got.extend(frames) or real(frames, *a, **k))
    return got


def test_noise_of_a_pass_does_not_depend_on_the_other_passes(leaf_runs, leaf_det_ctx, monkeypatch):
    root = leaf_runs[0]
    got = _record_tracker_frames(monkeypatch)
    full, cfg = leaf_det_ctx(0), fake_cfg(root, "apple", *DET, *POSE, "stop_after=y1")
    only_b, cfg_b = _prep(root, *DET, *POSE, "stop_after=y1", "passes.sides=[B]", det_seed=0)
    ((key_b, b_frames),) = only_b.passes.items()
    assert list(full.passes).index(key_b) > 0                      # side B moves to the front of the pass list
    assert all(np.array_equal(full.dets[f.idx].boxes, only_b.dets[f.idx].boxes) for f in b_frames)

    def pass_b(res):
        return [(p["U1"], p["n_tracks"], p["n_no3d"], p["e1"]) for p in res["y1"]["passes"] if p["side"] == "B"]

    assert pass_b(evaluate(full, cfg, noise_seed=0)) == pass_b(evaluate(only_b, cfg_b, noise_seed=0))
    by_idx = {}
    for f in got:
        by_idx.setdefault(f.idx, []).append(f.camera)
    assert {i for i, cams in by_idx.items() if len(cams) == 2} == {f.idx for f in b_frames}
    assert all(np.array_equal(a.C, b.C) and np.array_equal(a.R_c2w, b.R_c2w)
               for a, b in (by_idx[f.idx] for f in b_frames))


def test_rotation_and_pass_bias_reach_the_tracker(leaf_ctx, leaf_runs, monkeypatch):
    root = leaf_runs[0]
    got = _record_tracker_frames(monkeypatch)
    clean = {f.idx: f.camera for frames in leaf_ctx[0].passes.values() for f in frames}
    cfg = fake_cfg(root, "apple", "noise.pose_rot_sigma_deg=0.5", "noise.pass_bias_m=0.02", "stop_after=y1")
    evaluate(leaf_ctx[0], cfg, noise_seed=0)
    assert len(got) == len(clean) and all(not np.array_equal(f.camera.R_c2w, clean[f.idx].R_c2w) for f in got)
    offsets = {}
    for f in got:
        offsets.setdefault(f.pass_key, []).append(f.camera.C - clean[f.idx].C)
    assert offsets.keys() == leaf_ctx[0].passes.keys()
    bias = []
    for d in offsets.values():                                      # without trans noise, C moves by the pass bias
        np.testing.assert_allclose(d, np.broadcast_to(d[0], np.shape(d)), rtol=0, atol=1e-12)
        bias.append(d[0])
    assert [float(np.linalg.norm(b)) for b in bias] == pytest.approx([0.02] * len(bias))
    assert not np.allclose(bias[0], bias[1])                         # one bias per pass


def test_noise_keys_hold_the_seeds_the_step_and_the_frame_or_pass(leaf_runs, monkeypatch):
    root = leaf_runs[0]
    keys, real = [], runner_module.noise_rng

    def record(*key):
        keys.append(tuple(int(k) for k in key))
        return real(*key)

    monkeypatch.setattr(runner_module, "noise_rng", record)
    monkeypatch.setattr(oracle_module, "noise_rng", record)
    ctx, cfg = _prep(root, *DET, *POSE, "seed=7", "stop_after=y1", det_seed=3)
    evaluate(ctx, cfg, noise_seed=3)
    scene = ctx.run.meta.seed
    frames = [f for fr in ctx.passes.values() for f in fr]
    want = [(7, 3, step, scene, f.idx) for f in frames for step in (1, 2, 4)]   # detection, frame pose, depth
    want += [(7, 3, 3, scene, round(k.pitch_deg * 1000), 0 if k.side == "A" else 1, k.aisle)   # pass bias
             for k in ctx.passes]
    assert sorted(keys) == sorted(want)


BUNDLE = Path("bundle")


def _res(e_pipe, U, p90, *, y6=True, depth_err=None):
    """A hand-made evaluate() result with the fields combine_seeds reads."""
    return {"stages": {"det": "oracle"}, "lift_depth": "box_median",
            "y1": {"summary": {"e1_median": e_pipe / 2, "naive_U": U, "depth_err": depth_err},
                   "passes": [{"side": "A", "U1": U, "G1": 10}]},
            "y6": None if not y6 else {"mode": "merge", "row": {"U": U, "G6": 10, "e_pipe": e_pipe},
                                       "metrics": {"row_abs_e_pipe": abs(e_pipe), "pos_err_p90_m": p90},
                                       "trees": [{"tree": 3, "U": U, "G6": 10}]}}


def test_combine_seeds_means_ranges_and_keeps_equal_values():
    cfg = fake_cfg(BUNDLE, "apple", "noise.pose_trans_sigma_m=0.02", "noise.seeds=[0,1]")
    d = {"n": 4, "p50_m": 0.01}
    out = combine_seeds(cfg, [(0, _res(0.1, 11, 0.11, depth_err=d)), (1, _res(-0.1, 9, 0.09, depth_err=dict(d)))])
    assert (out["noisy"], out["seeds"], out["stages"], out["lift_depth"]) == (True, [0, 1], {"det": "oracle"}, "box_median")
    assert out["noise"] == {"pose_trans_sigma_m": 0.02, "pose_rot_sigma_deg": 0.0, "pass_bias_m": 0.0,
                            "depth_rel_sigma": 0.0, "depth_dropout": 0.0, "detector": None}
    assert out["y6"]["row"] == {"U": 10.0, "G6": 10, "e_pipe": 0.0} and isinstance(out["y6"]["row"]["G6"], int)
    assert out["y6"]["metrics"] == {"row_abs_e_pipe": pytest.approx(0.1), "pos_err_p90_m": pytest.approx(0.1)}
    assert out["y1"]["summary"] == {"e1_median": 0.0, "naive_U": 10.0, "depth_err": d}       # mean |e|, not |mean|
    assert out["y1"]["passes"] == [{"side": "A", "U1": 10.0, "G1": 10}]
    assert out["y6"]["trees"] == [{"tree": 3, "U": 10.0, "G6": 10}]
    assert out["spread"] == {
        "y1": {"summary": {"e1_median": [-0.05, 0.05], "naive_U": [9, 11], "depth_err": {"n": [4, 4], "p50_m": [0.01, 0.01]}}},
        "y6": {"row": {"U": [9, 11], "G6": [10, 10], "e_pipe": [-0.1, 0.1]},
               "metrics": {"row_abs_e_pipe": [0.1, 0.1], "pos_err_p90_m": [0.09, 0.11]}}}
    assert [p["noise_seed"] for p in out["per_seed"]] == [0, 1] and out["per_seed"][1]["y6"]["row"]["U"] == 9


def test_combine_seeds_none_disagreement_and_y1_only():
    cfg = fake_cfg(BUNDLE, "apple", "noise.depth_dropout=0.05", "noise.seeds=[0,1]")
    a = combine_seeds(cfg, [(0, _res(0.1, 11, 0.11)), (1, _res(0.1, 11, None))])
    assert a["y6"]["metrics"]["pos_err_p90_m"] is None and a["spread"]["y6"]["metrics"]["pos_err_p90_m"] is None
    assert a["y6"]["row"]["U"] == 11 and isinstance(a["y6"]["row"]["U"], int)           # equal ints stay ints
    b = combine_seeds(cfg, [(0, _res(0.1, 11, 0.11, y6=False)), (1, _res(0.3, 9, 0.09, y6=False))])
    assert b["y6"] is None and b["spread"]["y6"] is None and b["y1"]["summary"]["e1_median"] == pytest.approx(0.1)
    bad = _res(0.1, 11, 0.11)
    bad["y6"]["mode"] = "self_side"
    with pytest.raises(ValueError, match="noise seeds disagree"):
        combine_seeds(cfg, [(0, _res(0.1, 11, 0.11)), (1, bad)])
    longer = _res(0.1, 11, 0.11)
    longer["y1"]["passes"].append({"side": "B", "U1": 4, "G1": 5})
    more_keys = _res(0.1, 11, 0.11)
    more_keys["y1"]["summary"]["n_extra"] = 1
    flags = [_res(0.1, 11, 0.11), _res(0.1, 11, 0.11)]
    flags[0]["y1"]["summary"]["flag"], flags[1]["y1"]["summary"]["flag"] = True, False   # bools are not averaged
    for first, other in ((_res(0.1, 11, 0.11), longer), (_res(0.1, 11, 0.11), more_keys), flags):
        with pytest.raises(ValueError, match="noise seeds disagree"):
            combine_seeds(cfg, [(0, first), (1, other)])


def test_run_config_prepares_once_per_detection_seed(leaf_runs, leaf_det_ctx, monkeypatch):
    root = leaf_runs[0]
    cfg = fake_cfg(root, "apple", *DET, "noise.seeds=[0,1]")
    calls, real = [], runner_module.prepare
    monkeypatch.setattr(runner_module, "prepare", lambda *a, **k: calls.append(k.get("det_seed")) or real(*a, **k))
    _, res, timings = run_config(SynthRun.open(root), cfg, load_splits())
    assert calls == [0, 1] and res["seeds"] == [0, 1] and set(timings) == {"prepare", "evaluate"}
    alone = evaluate(leaf_det_ctx(1), cfg, noise_seed=1)
    assert {k: res["per_seed"][1][k] for k in ("y1", "y6")} == {k: alone[k] for k in ("y1", "y6")}
