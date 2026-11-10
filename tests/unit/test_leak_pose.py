"""The pose source stays light and blind to truth ids: importing the eval side loads no cv2, gtsam or
torch, and drift-model predictions do not change when the truth ids or the source ids do."""

import numpy as np
import pytest

from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.runner import evaluate, prepare
from tests.unit.synthfake import ctx_with, fake_cfg, load_splits, python_c
from tests.unit.test_drift import D_EXT_VO
from tests.unit.test_leak import _permute_and_reorder, _pred, _random_src

pytestmark = pytest.mark.pose


def test_runner_import_loads_no_cv2_gtsam_or_torch():
    mods = ["counting.evaluation.runner", "counting.evaluation.cli", "counting.evaluation.posesource",
            "counting.evaluation.sweep", "counting.evaluation.report", "counting.geometry.rig",
            "counting.geometry.noise", "counting.config.pose_schema"]
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
