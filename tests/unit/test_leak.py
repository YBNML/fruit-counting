import ast
import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pytest

from counting.config.eval_schema import EvalAppConfig
from counting.data.formats.orchard_synth import SceneTruth, SynthRun
from counting.evaluation.runner import evaluate, prepare
from tests.unit.cachefake import write_truth_cache
from tests.unit.synthfake import DET_NOISE, ctx_with, fake_cfg, load_splits, python_c

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "counting"
ALLOWED = {"FrameInput", "OrchardLayout", "PassKey"}
BANNED = ("counting.evaluation", "counting.detection.oracle")


def _permute(truth, rng):
    ids = sorted(truth.fruits)
    perm = dict(zip(ids, (int(x) for x in rng.permutation(ids))))
    recs = {k: tuple(dataclasses.replace(r, id=perm[r.id]) for r in v) for k, v in truth.records.items()}
    fruits = {perm[i]: dataclasses.replace(f, id=perm[i]) for i, f in truth.fruits.items()}
    return SceneTruth(recs, fruits, truth.totals, truth.count1, truth.summary)


def _random_src(ctx, rng):
    return {k: rng.integers(10**6, 2 * 10**6, size=len(v)) for k, v in ctx.src.items()}


def _pred(res):
    return ([(p["U1"], p["n_tracks"], p["n_no3d"]) for p in res["y1"]["passes"]],
            [t["U"] for t in res["y6"]["trees"]], res["y6"]["n_landmarks"])


@pytest.fixture(scope="module")
def plain_pred(apple_run, pear_run, ctx_of):
    """{crop: `_pred` of `ctx_of[crop]` under its crop's dev config}, the reference of the two tests below."""
    return {crop: _pred(evaluate(ctx_of[crop], fake_cfg(root, crop))) for root, crop in ((apple_run, "apple"),
                                                                                         (pear_run, "pear"))}


def test_shuffled_truth_ids_do_not_change_predictions(apple_run, pear_run, plain_pred):
    for root, crop in ((apple_run, "apple"), (pear_run, "pear")):
        run, cfg, splits = SynthRun.open(root), fake_cfg(root, crop), load_splits()
        b = evaluate(prepare(run, cfg, splits, truth=_permute(run.truth(), np.random.default_rng(1))), cfg)
        assert plain_pred[crop] == _pred(b)


def test_per_frame_random_source_ids_do_not_change_predictions(apple_run, pear_run, ctx_of, plain_pred):
    """Catches a runner that links tracks or landmarks by source id (consistent permutation cannot)."""
    for root, crop in ((apple_run, "apple"), (pear_run, "pear")):
        cfg, ctx = fake_cfg(root, crop), ctx_of[crop]   # non-oracle stages
        assert _pred(evaluate(ctx_with(ctx, src=_random_src(ctx, np.random.default_rng(2))), cfg)) == plain_pred[crop]


NOISY = (*DET_NOISE, "noise.pose_trans_sigma_m=0.02", "noise.pose_rot_sigma_deg=0.5", "noise.pass_bias_m=0.02",
         "noise.depth_rel_sigma=0.02", "noise.depth_dropout=0.05")


def _permute_and_reorder(truth, rng):
    """`_permute`, then each frame's records in the new id order, so record order follows the ids too."""
    t = _permute(truth, rng)
    recs = {k: tuple(sorted(v, key=lambda r: r.id)) for k, v in t.records.items()}
    return SceneTruth(recs, t.fruits, t.totals, t.count1, t.summary)


def test_noisy_predictions_ignore_truth_ids(leaf_runs, leaf_det_ctx):
    """With detection, pose and depth noise on, relabelled truth or random source ids change no prediction."""
    root = leaf_runs[0]
    run, cfg, splits = SynthRun.open(root), fake_cfg(root, "apple", *NOISY), load_splits()
    ctx = leaf_det_ctx(0)
    a = evaluate(ctx, cfg, noise_seed=0)
    b = evaluate(prepare(run, cfg, splits, truth=_permute_and_reorder(run.truth(), np.random.default_rng(1)),
                         det_seed=0), cfg, noise_seed=0)
    random_src = ctx_with(ctx, src=_random_src(ctx, np.random.default_rng(2)))
    assert _pred(a) == _pred(b) == _pred(evaluate(random_src, cfg, noise_seed=0))


def test_cached_predictions_ignore_truth_ids(leaf_runs, tmp_path):
    root = leaf_runs[0]
    write_truth_cache(root, tmp_path, shift=-0.2, low_score=0.07)
    cfg = fake_cfg(root, "apple", "passes.step_m=0.45", f"detector.cache={tmp_path}", "stages.det=cached",
                   "tracker.score_lo=0.05")
    run, splits = SynthRun.open(root), load_splits()
    ctx = prepare(run, cfg, splits)
    a = evaluate(ctx, cfg)
    b = evaluate(prepare(run, cfg, splits, truth=_permute_and_reorder(run.truth(), np.random.default_rng(1))), cfg)
    c = evaluate(ctx_with(ctx, src=_random_src(ctx, np.random.default_rng(2))), cfg)
    assert _pred(a) == _pred(b) == _pred(c)


def _prediction_files():
    for d in ("tracking", "multiview", "geometry"):
        yield from (SRC / d).glob("*.py")
    for f in ("detection/base.py", "detection/cached.py", "detection/frcnn.py"):
        if (SRC / f).exists():
            yield SRC / f


def test_prediction_modules_cannot_reach_truth():
    for path in _prediction_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith(BANNED + ("counting.data.formats.orchard_synth",)), path
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert node.level == 0 and not mod.startswith(BANNED), path
                if mod == "counting.data.formats.orchard_synth":
                    assert {a.name for a in node.names} <= ALLOWED, path
                if mod == "counting.data.formats":
                    assert "orchard_synth" not in {a.name for a in node.names}, path
                if mod == "counting.detection":
                    assert "oracle" not in {a.name for a in node.names}, path


def test_eval_modules_do_not_import_torch():
    mods = ["counting.evaluation.cli", "counting.evaluation.runner", "counting.evaluation.report",
            "counting.evaluation.sweep", "counting.tracking.landmark", "counting.multiview.dedup",
            "counting.multiview.lift", "counting.detection.oracle", "counting.data.formats.orchard_synth",
            "counting.geometry.noise", "counting.evaluation.seeds", "counting.evaluation.protect",
            "counting.evaluation.verdict", "counting.config.det_schema", "counting.training.detector",
            "counting.detection.cached", "counting.evaluation.detmetrics", "counting.detection.frcnn",
            "counting.evaluation.budget"]
    code = "import importlib, sys\n" + "".join(f"importlib.import_module({m!r})\n" for m in mods) + \
           "print('torch' in sys.modules)"
    out = python_c(code)
    assert out.strip() == "False"


def test_no_visibility_correction_knobs():
    names = re.findall(r'"([A-Za-z_]+)":', json.dumps(EvalAppConfig.model_json_schema()))
    assert not [n for n in names if re.search(r"correct|calib", n)]
