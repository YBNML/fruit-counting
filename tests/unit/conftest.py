"""Session-wide fake orchard_synth bundles.

The bundles are shared by every test in the session, so treat them as read-only:
a test that edits files builds its own bundle in `tmp_path` with `write_run`.
"""

import pytest

from tests.unit.synthfake import write_run


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
