import dataclasses
from pathlib import Path

import numpy as np
import pytest

from counting.data.formats.orchard_synth import FrameInput, FruitGt, GtRecord, PassKey, SceneTruth
from counting.detection.base import Detections
from counting.detection.oracle import OracleDetector
from counting.evaluation.match import match_frame, unit_majority
from counting.geometry.camera import Camera

REC = (
    GtRecord(800, 3, 100, 100, 1.0, (50, 10, 60, 20), False, 2.0),
    GtRecord(801, 3, 30, 100, 0.3, (10, 40, 16, 45), False, 2.5),
    GtRecord(802, 3, 10, 100, 0.1, (30, 30, 34, 33), False, 3.0),   # visible but not count1 → ignore
    GtRecord(803, 3, 0, 100, 0.0, None, False, None),               # fully hidden
    GtRecord(400, 1, 40, 40, 1.0, (10, 5, 18, 10), True, 6.0),      # other row, positive
)
FR = {i: FruitGt(i, t, np.zeros(3), 0.07, i == 800) for i, t in [(800, 3), (801, 3), (802, 3), (803, 3), (400, 1)]}
TRUTH = SceneTruth({0: REC}, FR, {3: 4, 1: 1}, (0.25, 20), None)
FRAME = FrameInput("t", 0, PassKey(30.0, "A", 2), 0, Camera.from_blender(np.eye(4), 64, 64, 69.0),
                   0.0, Path("r"), Path("d"))


def _dets(boxes):
    n = len(boxes)
    return Detections(np.asarray(boxes, np.float32), np.ones(n, np.float32), np.zeros(n, np.int8))


def test_oracle_emits_positives_sorted_by_box():
    d = OracleDetector(TRUTH).detect(FRAME)
    np.testing.assert_array_equal(d.boxes, [[10, 5, 18, 10], [10, 40, 16, 45], [50, 10, 60, 20]])
    assert d.boxes.dtype == np.float32 and d.labels.dtype == np.int8
    np.testing.assert_array_equal(d.labels, [0, 0, 1])
    np.testing.assert_allclose(d.depth_m, [6.0, 2.5, 2.0])
    np.testing.assert_array_equal(d.scores, [1.0, 1.0, 1.0])


def test_oracle_any_includes_ignored_records():
    assert len(OracleDetector(TRUTH, positives="any").detect(FRAME)) == 4


def test_detections_carry_no_ids():
    assert {f.name for f in dataclasses.fields(Detections)} == {"boxes", "scores", "labels", "depth_m"}


def test_detections_validate_shapes():
    with pytest.raises(ValueError):
        Detections(np.zeros((2, 3), np.float32), np.ones(2, np.float32), np.zeros(2, np.int8))


def test_oracle_noise_not_supported():
    with pytest.raises(NotImplementedError, match="detection noise is not supported yet"):
        OracleDetector(TRUTH, noise={"miss": 0.1})


def test_match_oracle_gets_source_ids():
    assert match_frame(OracleDetector(TRUTH).detect(FRAME), REC).tolist() == [400, 801, 800]


def test_match_iou_threshold():
    recs = (GtRecord(1, 3, 100, 100, 1.0, (0, 0, 10, 10), False, 1.0),)
    assert match_frame(_dets([[0, 0, 10, 5]]), recs).tolist() == [1]
    assert match_frame(_dets([[0, 0, 10, 4.9]]), recs).tolist() == [-1]


def test_match_one_to_one():
    recs = (GtRecord(1, 3, 100, 100, 1.0, (0, 0, 10, 10), False, 1.0),)
    assert match_frame(_dets([[0, 0, 10, 8], [0, 0, 10, 10]]), recs).tolist() == [-1, 1]


def test_match_ignored_record_gives_id():
    assert match_frame(_dets([[30, 30, 34, 33]]), REC).tolist() == [802]


def test_match_skips_hidden_records():
    assert match_frame(Detections.empty(), REC).shape == (0,)
    assert match_frame(_dets([[0, 0, 1, 1]]), REC).tolist() == [-1]


def test_match_tie_prefers_positive_then_visible():
    box = (10, 10, 20, 20)
    ign = GtRecord(5, 3, 90, 1000, 0.09, box, False, 2.0)     # more pixels, but not count1
    pos = GtRecord(9, 3, 40, 100, 0.40, box, False, 2.0)
    d = _dets([box])
    is_pos = lambda r: r.visib >= 0.25 and r.vis_px >= 20       # noqa: E731
    assert match_frame(d, (ign, pos), prefer=is_pos).tolist() == [9]
    assert match_frame(d, (ign, pos)).tolist() == [5]           # no preference: more visible pixels wins


def test_unit_majority_ties_and_purity():
    m, p = unit_majority([[5, 7], [7, 5, 5], [-1, 3], [9]])
    assert m.tolist() == [5, 5, -1, 9]
    np.testing.assert_allclose(p, [0.5, 2 / 3, 0.5, 1.0])


def test_unit_majority_empty_raises():
    with pytest.raises(ValueError):
        unit_majority([[]])
