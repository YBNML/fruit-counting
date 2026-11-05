"""Session-wide fake orchard_synth bundles, their prepared contexts and the full fake ladder.
`heldout_run` is the one bundle with a test seed, for the split=test checks.

The bundles, contexts and ladder are shared by every test in the session, so treat them as
read-only: evaluate them, never edit them. A test that edits files builds its own bundle in
`tmp_path` with `write_run`; a test that edits a context prepares its own, or varies a copy
(`ctx_with`) that keeps the detection boxes.

With `COUNTING_TEST_BUDGET_S` set to a number of seconds (finite and > 0; read once in
`pytest_configure`; unset or empty: no check), a passing session that runs longer than that
fails (`pytest_sessionfinish`).
"""

import math
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.runner import load_ladder, prepare, run_ladder
from tests.unit.synthfake import DET_NOISE, SIDECAR_COMMIT, fake_cfg, load_splits, write_run

_T0 = time.perf_counter()
_BUDGET_S: float | None = None   # COUNTING_TEST_BUDGET_S, set by pytest_configure
ROOT = Path(__file__).resolve().parents[2]


def budget_seconds(raw: str | None) -> float | None:
    """COUNTING_TEST_BUDGET_S as seconds: None when unset or empty, else a finite number > 0
    (anything else: `pytest.UsageError`)."""
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value <= 0:
        raise pytest.UsageError(f"COUNTING_TEST_BUDGET_S must be a number of seconds, got {raw!r}")
    return value


def pytest_configure(config):
    """Read COUNTING_TEST_BUDGET_S once into `_BUDGET_S` (`budget_seconds`): unset or empty means no time check."""
    global _BUDGET_S
    _BUDGET_S = budget_seconds(os.environ.get("COUNTING_TEST_BUDGET_S"))


@pytest.fixture(scope="session")
def apple_run(tmp_path_factory):
    return write_run(tmp_path_factory.mktemp("apple_run"), crop="apple")


@pytest.fixture(scope="session")
def pear_run(tmp_path_factory):
    return write_run(tmp_path_factory.mktemp("pear_run"), crop="pear")


@pytest.fixture(scope="session")
def heldout_run(tmp_path_factory):
    """An apple bundle with the 30 deg passes only, test seed 1000 and an orchard_synth.commit sidecar."""
    root = write_run(tmp_path_factory.mktemp("heldout") / "s1000", crop="apple", pitches=(30.0,), seed=1000)
    (root / "orchard_synth.commit").write_text(SIDECAR_COMMIT + "\n", encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def no_depth_run(tmp_path_factory):
    """An apple bundle with the 10 deg passes only (they see the ground) and every depth16 pixel 0 (no return)."""
    root = write_run(tmp_path_factory.mktemp("no_depth"), crop="apple", pitches=(10.0,))
    for p in root.glob("*_depth16.png"):
        Image.fromarray(np.zeros((360, 640), np.uint16)).save(p)
    return root


@pytest.fixture(scope="session")
def pitch80_ctx(tmp_path_factory):
    """An apple bundle with 30 and 80 deg passes, prepared under the apple dev config; the 80 deg
    passes have no truth fruit and no detection."""
    root = write_run(tmp_path_factory.mktemp("pitch80"), crop="apple", pitches=(30.0, 80.0))
    return prepare(SynthRun.open(root), fake_cfg(root, "apple"), load_splits())


@pytest.fixture(scope="session")
def leaf_runs(tmp_path_factory):
    """(clean, occluded) apple bundles with the 30 deg passes only.

    The occluded copy paints a front-leaf patch (`write_run(occlude=...)`) on each of the
    12 target-row objects in every third frame; records, manifest and summary are identical.
    """
    root = tmp_path_factory.mktemp("leaf")
    target = [200 * (t + 1) + j for t in (3, 4, 5) for j in range(4)]   # fake ids: block 200, trees 3-5
    clean = write_run(root / "clean", crop="apple", pitches=(30.0,))
    occl = write_run(root / "occl", crop="apple", pitches=(30.0,),
                     occlude={i: range(0, 10**4, 3) for i in target})
    return clean, occl


@pytest.fixture(scope="session")
def leaf_ctx(leaf_runs):
    """Prepared contexts of the (clean, occluded) leaf bundles under the apple dev config."""
    return [prepare(SynthRun.open(root), fake_cfg(root, "apple"), load_splits()) for root in leaf_runs]


@pytest.fixture(scope="session")
def leaf_det_ctx(leaf_runs):
    """`leaf_det_ctx(s)`: the clean leaf bundle prepared under the apple dev config with `DET_NOISE`
    and detection noise seed `s`, built on first use."""
    root, made = leaf_runs[0], {}

    def ctx(det_seed: int):
        if det_seed not in made:
            cfg = fake_cfg(root, "apple", *DET_NOISE)
            made[det_seed] = prepare(SynthRun.open(root), cfg, load_splits(), det_seed=det_seed)
        return made[det_seed]

    return ctx


@pytest.fixture(scope="session")
def ctx_of(apple_run, pear_run):
    """{crop: prepared context} of `apple_run` and `pear_run`, each under its crop's dev config."""
    return {crop: prepare(SynthRun.open(root), fake_cfg(root, crop), load_splits())
            for crop, root in (("apple", apple_run), ("pear", pear_run))}


# Every third frame of each pass: both pitches stay, so the 10 deg passes still give fp_off terms.
LADDER_SETS = ("passes.step_m=0.45",)


@pytest.fixture(scope="session")
def apple_ladder_ctx(apple_run):
    """`apple_run` prepared under the apple dev config with `LADDER_SETS`."""
    return prepare(SynthRun.open(apple_run), fake_cfg(apple_run, "apple", *LADDER_SETS), load_splits())


@pytest.fixture(scope="session")
def apple_ladder(apple_run, apple_ladder_ctx):
    """`run_ladder` on `apple_ladder_ctx` over every rung and baseline of the repo ladder, with the 2 x 2 table."""
    ladder = load_ladder(ROOT / "configs" / "eval" / "sweeps" / "ladder.yaml")
    return run_ladder(apple_ladder_ctx, fake_cfg(apple_run, "apple", *LADDER_SETS), ladder,
                      [*ladder["rungs"], *ladder["baselines"]], with_2x2=True)


def pytest_sessionfinish(session, exitstatus):
    """Fail a passing session that took longer than COUNTING_TEST_BUDGET_S seconds (unset or empty: no check)."""
    if _BUDGET_S is None or exitstatus != 0:
        return
    took = time.perf_counter() - _T0
    if took > _BUDGET_S:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        tr = session.config.pluginmanager.get_plugin("terminalreporter")
        if tr is not None:
            tr.write("\n")
            tr.write_line(f"test time budget exceeded: {took:.2f} s > {_BUDGET_S:g} s")
