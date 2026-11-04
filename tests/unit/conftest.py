"""Session-wide fake orchard_synth bundles, their prepared contexts and the full fake ladder.

The bundles, contexts and ladder are shared by every test in the session, so treat them as
read-only: evaluate them, never edit them. A test that edits files builds its own bundle in
`tmp_path` with `write_run`; a test that edits a context prepares its own, or varies a copy
(`ctx_with`) that keeps the detection boxes.

With `COUNTING_TEST_BUDGET_S` set, a passing session that runs longer than that many
seconds fails (`pytest_sessionfinish`).
"""

import os
import time
from pathlib import Path

import pytest

from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.runner import load_ladder, prepare, run_ladder
from tests.unit.synthfake import DET_NOISE, fake_cfg, load_splits, write_run

_T0 = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def apple_run(tmp_path_factory):
    return write_run(tmp_path_factory.mktemp("apple_run"), crop="apple")


@pytest.fixture(scope="session")
def pear_run(tmp_path_factory):
    return write_run(tmp_path_factory.mktemp("pear_run"), crop="pear")


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


@pytest.fixture(scope="session")
def apple_ladder(apple_run, ctx_of):
    """`run_ladder` on `ctx_of["apple"]` over every rung and baseline of the repo ladder, with the 2 x 2 table."""
    ladder = load_ladder(ROOT / "configs" / "eval" / "sweeps" / "ladder.yaml")
    return run_ladder(ctx_of["apple"], fake_cfg(apple_run, "apple"), ladder,
                      [*ladder["rungs"], *ladder["baselines"]], with_2x2=True)


def pytest_sessionfinish(session, exitstatus):
    """Fail a passing session that took longer than COUNTING_TEST_BUDGET_S seconds (unset: no check)."""
    budget = os.environ.get("COUNTING_TEST_BUDGET_S")
    if not budget or exitstatus != 0:
        return
    took = time.perf_counter() - _T0
    if took > float(budget):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        tr = session.config.pluginmanager.get_plugin("terminalreporter")
        if tr is not None:
            tr.write("\n")
            tr.write_line(f"test time budget exceeded: {took:.2f} s > {float(budget):g} s")
