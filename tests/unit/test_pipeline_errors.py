import numpy as np
import pytest

from counting.models.base import Stage, StageResult
from counting.pipeline import Pipeline


class _IdentityDeblur(Stage):
    name = "deblur"

    def prepare(self, _cfg): pass
    def process(self, image):
        return StageResult(output=image)
    def cleanup(self): pass


class _ConstCount(Stage):
    name = "pseco"

    def __init__(self, n: int):
        self.n = n

    def prepare(self, _cfg): pass
    def process(self, image):
        return StageResult(output={"count": self.n, "crops": [], "points": [], "boxes": []})
    def cleanup(self): pass


class _FailingStage(Stage):
    name = "sr"

    def prepare(self, _cfg): pass
    def process(self, _x): raise RuntimeError("boom")
    def cleanup(self): pass


def _image():
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_pipeline_runs_and_counts():
    pipe = Pipeline.from_stages([_IdentityDeblur(), _ConstCount(5)], device="cpu", config_hash="h")
    res = pipe.run_numpy(_image(), image_path="x.jpg")
    assert res.raw_count == 5
    assert res.verified_count == 5
    assert res.error is None
    assert [t.stage for t in res.timings_ms] == ["deblur", "pseco"]


def test_pipeline_records_stage_failure_without_crashing():
    pipe = Pipeline.from_stages(
        [_IdentityDeblur(), _ConstCount(2), _FailingStage()],
        device="cpu",
        config_hash="h",
    )
    res = pipe.run_numpy(_image(), image_path="x.jpg")
    assert res.raw_count == 2
    assert res.error and "sr" in res.error


def test_pipeline_requires_counting_stage():
    with pytest.raises(ValueError, match="pseco"):
        Pipeline.from_stages([_IdentityDeblur()], device="cpu", config_hash="h").run_numpy(
            _image(), image_path="x.jpg"
        )
