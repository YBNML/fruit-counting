"""The pose source and the odometry stay light and blind to truth: importing the eval side, the command line
or the odometry loads no cv2, gtsam or torch; the odometry input carries no pose and the odometry modules
import nothing from the truth side; drift-model predictions do not change when the truth ids or the source
ids do."""

import ast
import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pytest

from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.posesource import vo_inputs
from counting.evaluation.runner import evaluate, prepare
from counting.odometry.vo import VoFrame, run_vo
from tests.unit.synthfake import ctx_with, fake_cfg, load_splits, python_c
from tests.unit.test_drift import D_EXT_VO
from tests.unit.test_leak import ALLOWED, BANNED, _permute_and_reorder, _pred, _random_src

pytestmark = pytest.mark.pose


def test_runner_import_loads_no_cv2_gtsam_or_torch():
    mods = ["counting.evaluation.runner", "counting.evaluation.cli", "counting.evaluation.posesource",
            "counting.evaluation.sweep", "counting.evaluation.report", "counting.geometry.rig",
            "counting.geometry.noise", "counting.config.pose_schema", "counting.cli", "counting.geometry.posefile",
            "counting.odometry.cache", "counting.odometry.vo"]
    code = "import importlib, sys\n" + "".join(f"importlib.import_module({m!r})\n" for m in mods) + \
           "print(sorted(m for m in ('cv2', 'gtsam', 'torch') if m in sys.modules))"
    assert python_c(code).strip() == "[]"


def test_drift_predictions_ignore_truth_ids(leaf_runs, leaf_ctx):
    """With the drift model, start errors and depth noise on, relabelled truth or random source ids
    change no prediction."""
    root = leaf_runs[0]
    cfg = fake_cfg(root, "apple", "pose.rig_ref=30", *D_EXT_VO, "noise.pass_side_trans_m=0.05",
                   "noise.pass_side_rot_deg=0.5", "noise.depth_rel_sigma=0.02")
    run, splits, ctx = SynthRun.open(root), load_splits(), leaf_ctx[0]
    a = evaluate(ctx, cfg, noise_seed=0)
    b = evaluate(prepare(run, cfg, splits, truth=_permute_and_reorder(run.truth(), np.random.default_rng(1))), cfg,
                 noise_seed=0)
    c = evaluate(ctx_with(ctx, src=_random_src(ctx, np.random.default_rng(2))), cfg, noise_seed=0)
    assert _pred(a) == _pred(b) == _pred(c)


SRC = Path(__file__).resolve().parents[2] / "src" / "counting"
POSE_FIELDS = {"camera", "R_c2w", "C", "cam_x", "matrix_world", "pose", "plan", "t", "pos_k"}


def test_vo_inputs_carry_no_pose(apple_run):
    """The odometry input has images, intrinsics and the pass name only; renumbered positions give the same
    input, and the odometry takes no step value."""
    names = {f.name for f in dataclasses.fields(VoFrame)}
    assert names == {"idx", "pass_key", "K", "width", "height", "rgb_path", "depth_path"} and not names & POSE_FIELDS
    run = SynthRun.open(apple_run)

    def seen(frames):
        return [(v.idx, v.pass_key, v.K.tobytes(), v.width, v.height, v.rgb_path, v.depth_path)
                for v in vo_inputs(frames)]

    assert seen(run.frames) == seen([dataclasses.replace(f, pos_k=f.pos_k + 1) for f in run.frames])
    assert np.array_equal(vo_inputs(run.frames[:1])[0].K, run.frames[0].camera.K)
    assert list(inspect.signature(run_vo).parameters) == ["frames", "cfg"]


def test_odometry_modules_cannot_reach_truth():
    paths = sorted((SRC / "odometry").glob("*.py"))
    assert {p.name for p in paths} >= {"__init__.py", "cache.py", "vo.py"}
    for path in paths:
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
