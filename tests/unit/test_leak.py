import ast
import dataclasses
import json
import re
from pathlib import Path

import numpy as np

from counting.config.eval_schema import EvalAppConfig
from counting.data.formats.orchard_synth import SceneTruth, SynthRun
from counting.evaluation.runner import evaluate, prepare
from tests.unit.synthfake import fake_cfg, load_splits, python_c

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


def _pred(res):
    return ([(p["U1"], p["n_tracks"], p["n_no3d"]) for p in res["y1"]["passes"]],
            [t["U"] for t in res["y6"]["trees"]], res["y6"]["n_landmarks"])


def test_shuffled_truth_ids_do_not_change_predictions(apple_run, pear_run):
    for root, crop in ((apple_run, "apple"), (pear_run, "pear")):
        run, cfg, splits = SynthRun.open(root), fake_cfg(root, crop), load_splits()
        a = evaluate(prepare(run, cfg, splits), cfg)
        b = evaluate(prepare(run, cfg, splits, truth=_permute(run.truth(), np.random.default_rng(1))), cfg)
        assert _pred(a) == _pred(b)


def test_per_frame_random_source_ids_do_not_change_predictions(apple_run, pear_run):
    """Catches a runner that links tracks or landmarks by source id (consistent permutation cannot)."""
    for root, crop in ((apple_run, "apple"), (pear_run, "pear")):
        run, cfg, splits = SynthRun.open(root), fake_cfg(root, crop), load_splits()   # non-oracle stages
        a = evaluate(prepare(run, cfg, splits), cfg)
        ctx, rng = prepare(run, cfg, splits), np.random.default_rng(2)
        ctx.src = {k: rng.integers(10**6, 2 * 10**6, size=len(v)) for k, v in ctx.src.items()}
        assert _pred(evaluate(ctx, cfg)) == _pred(a)


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
            "counting.multiview.lift", "counting.detection.oracle", "counting.data.formats.orchard_synth"]
    code = "import importlib, sys\n" + "".join(f"importlib.import_module({m!r})\n" for m in mods) + \
           "print('torch' in sys.modules)"
    out = python_c(code)
    assert out.strip() == "False"


def test_no_visibility_correction_knobs():
    names = re.findall(r'"([A-Za-z_]+)":', json.dumps(EvalAppConfig.model_json_schema()))
    assert not [n for n in names if re.search(r"correct|calib", n)]
