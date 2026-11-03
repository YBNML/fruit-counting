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
