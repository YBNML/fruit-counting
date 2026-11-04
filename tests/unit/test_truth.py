import copy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from counting.data.formats.orchard_synth import (
    FrameInput, FruitGt, GtRecord, OrchardLayout, PassKey, SceneTruth, SynthRun,
)
from counting.evaluation.truth import (
    SummaryMismatch, check_geometry, check_summary, is_positive, pass_subsets, subset_table, truth_sets,
)
from counting.geometry.camera import Camera
from tests.unit.synthfake import move_unoccluded_box, write_run

A, B = PassKey(30.0, "A", 2), PassKey(30.0, "B", 1)
LAYOUT = OrchardLayout("apple", (0.0, 3.75, 7.5), 1, 3.75, (3, 4), (0.0, 1.25))
CAM = Camera.from_blender(np.eye(4), 64, 36, 69.0)


def _f(idx, key, pos):
    return FrameInput("t", idx, key, pos, CAM, 0.0, Path("rgb"), Path("depth"))


def _r(i, tree, visib, px, trunc=False):
    return GtRecord(id=i, tree=tree, vis_px=px, full_px=100, visib=visib,
                    bbox=(0, 0, 10, 10) if px else None, trunc=trunc, depth_med=2.0 if px else None)


RECS = {
    0: (_r(800, 3, 0.5, 100), _r(801, 3, 0.2, 100), _r(400, 1, 1.0, 50)),
    1: (_r(801, 3, 0.3, 19), _r(1000, 4, 0.25, 20, trunc=True)),
    2: (_r(1001, 4, 0.9, 30), _r(800, 3, 0.0, 0)),
    3: (_r(1000, 4, 0.6, 40),),
}
FRUITS = {i: FruitGt(i, t, np.zeros(3), 0.07, False) for i, t in [(800, 3), (801, 3), (1000, 4), (1001, 4), (400, 1)]}
SUMMARY = {3: {"total": 5, "count1": {"A": 1, "B": 0, "union": 1}, "any": {"A": 2, "B": 0, "union": 2}},
           4: {"total": 6, "count1": {"A": 1, "B": 2, "union": 2}, "any": {"A": 1, "B": 2, "union": 2}}}
TRUTH = SceneTruth(RECS, FRUITS, {1: 3, 3: 5, 4: 6}, (0.25, 20), SUMMARY)
PASSES = {A: [_f(0, A, 0), _f(1, A, 1)], B: [_f(2, B, 1), _f(3, B, 0)]}


def test_is_positive_rules():
    assert is_positive(_r(1, 3, 0.25, 20), "count1", (0.25, 20))
    assert not is_positive(_r(1, 3, 0.25, 19), "count1", (0.25, 20))
    assert is_positive(_r(1, 3, 0.01, 1), "any", (0.25, 20))
    assert not is_positive(_r(1, 3, 0.9, 50, trunc=True), "strict", (0.25, 20))


def test_truth_sets_count1():
    tt = truth_sets(TRUTH, LAYOUT, PASSES)
    assert tt.g1[A] == {800, 1000} and tt.g1[B] == {1000, 1001}
    assert tt.g6 == {800, 1000, 1001} and tt.per_tree(tt.g6) == {3: 1, 4: 2}
    assert dict(tt.g6_total) == {3: 5, 4: 6} and 400 not in tt.target_ids


def test_truth_sets_any_and_strict():
    any_ = truth_sets(TRUTH, LAYOUT, PASSES, "any")
    assert any_.g1[A] == {800, 801, 1000} and any_.g6 == {800, 801, 1000, 1001}
    strict = truth_sets(TRUTH, LAYOUT, PASSES, "strict")
    assert strict.g1[A] == {800} and strict.g6 == {800, 1000, 1001}


def test_truth_recount_on_thinned_frames():
    tt = truth_sets(TRUTH, LAYOUT, {A: [PASSES[A][0]], B: [PASSES[B][0]]})
    assert tt.g1[A] == {800} and tt.g1[B] == {1001}


def test_check_summary_passes_and_detects_mismatch():
    check_summary(TRUTH, LAYOUT, PASSES)
    bad = copy.deepcopy(SUMMARY)
    bad[4]["count1"]["B"] = 3
    with pytest.raises(SummaryMismatch, match="tree 4"):
        check_summary(SceneTruth(RECS, FRUITS, TRUTH.totals, (0.25, 20), bad), LAYOUT, PASSES)
    with pytest.raises(ValueError, match="summary.json is missing"):
        check_summary(SceneTruth(RECS, FRUITS, TRUTH.totals, (0.25, 20), None), LAYOUT, PASSES)


def test_subset_table_rows():
    by = {r["S"]: r for r in subset_table(TRUTH, LAYOUT, PASSES)}
    assert set(by) == {"all", "A", "B", "only 30", "drop 30"}
    assert by["all"]["G6"] == 3 and by["all"]["ratio_total"] == pytest.approx(3 / 11)
    assert by["A"]["G6"] == 2 and by["A"]["ratio_all"] == pytest.approx(2 / 3)
    assert by["drop 30"]["G6"] == 0 and by["drop 30"]["n_passes"] == 0


def test_pass_subsets_names_and_order():
    a10, b10 = PassKey(10.0, "A", 2), PassKey(10.0, "B", 1)
    keys = [A, B, a10, b10]                                   # A and B are the 30 deg keys
    assert pass_subsets(keys) == [("all", keys), ("A", [A, a10]), ("B", [B, b10]),
                                  ("only 10", [a10, b10]), ("drop 10", [A, B]),
                                  ("only 30", [A, B]), ("drop 30", [a10, b10])]


def test_check_geometry_on_fake(apple_run):
    run = SynthRun.open(apple_run)
    res = check_geometry(run, run.truth(), stride=10)
    assert res["passed"] and res["ground"] == "ok" and res["center_in_bbox"] == 1.0 and res["n_unoccluded"] > 0
    assert res["n_ground_px"] > 0 and res["ground_abs_z_median_m"] <= 0.001


def test_check_geometry_catches_depth_offset(tmp_path):
    root = write_run(tmp_path, crop="apple", pitches=(10.0,))   # 10 deg sees the ground
    for p in root.glob("*_depth16.png"):
        d = np.array(Image.open(p), dtype=np.uint16)
        Image.fromarray(np.where(d > 0, np.minimum(d.astype(np.int32) + 50, 65535), 0).astype(np.uint16)).save(p)
    run = SynthRun.open(root)
    res = check_geometry(run, run.truth(), stride=5)
    assert res["ground"] == "fail" and res["passed"] is False


def test_check_geometry_fails_when_ground_has_no_depth(tmp_path):
    """Ground in view but every depth16 pixel 0 (no return): the ground check applies and fails, it is not n/a."""
    root = write_run(tmp_path, crop="apple", pitches=(10.0,))   # 10 deg only; it sees the ground
    for p in root.glob("*_depth16.png"):
        Image.fromarray(np.zeros_like(np.array(Image.open(p), dtype=np.uint16))).save(p)
    run = SynthRun.open(root)
    res = check_geometry(run, run.truth(), stride=10)
    assert (res["ground"], res["passed"], res["n_ground_px"], res["ground_abs_z_median_m"]) == ("fail", False, 0, None)
    assert res["center_in_bbox"] == 1.0 and res["n_unoccluded"] > 0   # the centre check never reads depth


def test_check_geometry_ground_not_applicable(pear_run):
    """45 and 70 deg passes never see the ground: the ground check is n/a and the centre check decides."""
    run = SynthRun.open(pear_run)
    res = check_geometry(run, run.truth(), stride=10)
    assert res["ground"] == "n/a" and (res["n_ground_px"], res["ground_abs_z_median_m"]) == (0, None)
    assert res["center_in_bbox"] == 1.0 and res["n_unoccluded"] == 37 and res["passed"] is True


def test_check_geometry_center_fail_without_ground(tmp_path):
    root = write_run(tmp_path, crop="apple", pitches=(30.0,))       # 30 deg only: no ground in view
    move_unoccluded_box(root / "000000_frame.json")
    run = SynthRun.open(root)
    res = check_geometry(run, run.truth(), stride=10)
    assert res["ground"] == "n/a" and res["center_in_bbox"] == pytest.approx(40 / 41) and res["passed"] is False
