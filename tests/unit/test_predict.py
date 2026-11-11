"""The tracker's prediction from a track's recent observations (`pose_local`): its settings, that a window over
every observation is the pose prediction, and that it breaks fewer tracks under strong drift."""

import re

import pytest
from pydantic import ValidationError

from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.report import results_bytes
from counting.evaluation.runner import evaluate, prepare
from tests.unit.synthfake import fake_cfg, load_splits, write_run

pytestmark = pytest.mark.pose
STRONG = ("noise.drift_scale_sigma=0.045", "noise.drift_rate_deg_per_m=0.45", "noise.drift_step_trans_mm=21",
          "noise.drift_step_rot_deg=0.39")
RIG = ("pose.rig=true", "pose.rig_ref=30")


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    root = write_run(tmp_path_factory.mktemp("predict"), crop="apple", size=(160, 90))
    return root, prepare(SynthRun.open(root), fake_cfg(root, "apple"), load_splits())


@pytest.fixture(scope="module")
def small_result(small):
    """`small_result(*sets, ns=…)`: evaluate of the `small` bundle under `fake_cfg(root, "apple", *sets)` with noise
    seed `ns`, made on first use and shared by the tests of this module, so treat it as read-only."""
    root, ctx = small
    made = {}

    def result(*sets, ns):
        if (sets, ns) not in made:
            made[sets, ns] = evaluate(ctx, fake_cfg(root, "apple", *sets), noise_seed=ns)
        return made[sets, ns]

    return result


def test_pose_local_needs_a_window_and_keeps_the_dumps(small):
    root, _ = small
    cfg = fake_cfg(root, "apple", "tracker.predict=pose_local", "tracker.predict_window=3")
    assert (cfg.tracker.predict, cfg.tracker.predict_window) == ("pose_local", 3)
    assert "predict_window" not in fake_cfg(root, "apple").tracker.model_dump()
    for sets, message in ((("tracker.predict=pose_local",), "tracker.predict=pose_local needs tracker.predict_window"),
                          (("tracker.predict_window=3",), "tracker.predict_window needs tracker.predict=pose_local"),
                          (("tracker.predict=pose_local", "tracker.predict_window=0"), "greater than or equal to 1")):
        with pytest.raises(ValidationError, match=re.escape(message)):
            fake_cfg(root, "apple", *sets)


def test_pose_local_with_every_observation_is_pose(small_result):
    for sets, ns in (((), None), ((*RIG, *STRONG), 0)):
        a = small_result(*sets, ns=ns)
        b = small_result(*sets, "tracker.predict=pose_local", "tracker.predict_window=100000", ns=ns)
        assert results_bytes({k: a[k] for k in ("y1", "y6")}) == results_bytes({k: b[k] for k in ("y1", "y6")})


def test_pose_local_breaks_fewer_tracks_under_drift(small_result):
    def dup_track(*sets):
        return sum(p["ledger"]["dup_track"] for ns in (0, 1)
                   for p in small_result(*RIG, *STRONG, *sets, ns=ns)["y1"]["passes"])

    pose, recent = dup_track(), dup_track("tracker.predict=pose_local", "tracker.predict_window=1")
    assert recent < 0.9 * pose, (pose, recent)
