import dataclasses
import json

import numpy as np
import pytest
from PIL import Image

from counting.data.formats.orchard_synth import (
    FrameInput, PassKey, SynthRun, read_depth_m, read_png_u16, select_frames,
)
from tests.unit.synthfake import write_run

A10, B10, A30, B30 = PassKey(10.0, "A", 2), PassKey(10.0, "B", 1), PassKey(30.0, "A", 2), PassKey(30.0, "B", 1)


def test_open_layout_meta_and_passes(apple_run):
    run = SynthRun.open(apple_run)
    m, L = run.meta, run.layout
    assert (m.crop, m.preset, m.seed, m.is_sequence) == ("apple", "apple_robot_side", 5, True)
    assert m.base_step_m == pytest.approx(0.15) and m.synth_commit is None and len(m.fingerprint) == 64
    assert m.run_id == f"apple_s5_{m.fingerprint[:8]}"
    assert L.row_ys == (0.0, 3.75, 7.5) and L.target_row == 1 and L.row_y == 3.75
    assert L.row_spacing == pytest.approx(3.75)
    assert L.tree_ids == (3, 4, 5) and L.tree_x == pytest.approx((0.0, 1.25, 2.5))
    assert len(run.frames) == 124
    assert list(run.passes()) == [A10, B10, A30, B30]
    assert all(len(v) == 31 for v in run.passes().values())


def test_pos_k_follows_travel_direction(apple_run):
    p = SynthRun.open(apple_run).passes()
    assert [f.pos_k for f in p[A10]] == list(range(31))
    assert [f.pos_k for f in p[B10]] == list(range(30, -1, -1))


def test_open_expands_home(tmp_path, monkeypatch):
    write_run(tmp_path / "r", crop="apple", pitches=(30.0,))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert SynthRun.open("~/r").meta.seed == 5


def test_truth_contents(apple_run):
    t = SynthRun.open(apple_run).truth()
    assert t.count1 == (0.25, 20) and t.totals[3] == 4
    assert {800, 801, 802, 803} <= set(t.fruits)
    assert t.fruits[800].bagged and not t.fruits[801].bagged
    m = json.loads((apple_run / "manifest.json").read_text(encoding="utf-8"))
    bag = next(b for b in m["spec"]["trees"][3]["bags"] if b[0] == 800)
    np.testing.assert_allclose(t.fruits[800].center, bag[1])
    assert t.summary is not None and t.summary[3]["count1"]["union"] >= 1


def test_frame_input_has_no_truth_fields():
    assert {f.name for f in dataclasses.fields(FrameInput)} == {
        "run_id", "idx", "pass_key", "pos_k", "camera", "cam_x", "rgb_path", "depth_path"}


def test_read_png_u16_roundtrip(tmp_path):
    arr = np.array([[0, 1, 1234], [40000, 65535, 7]], dtype=np.uint16)
    Image.fromarray(arr).save(tmp_path / "d.png")
    out = read_png_u16(tmp_path / "d.png")
    assert out.dtype == np.uint16 and np.array_equal(out, arr)
    z = read_depth_m(tmp_path / "d.png")
    assert np.isnan(z[0, 0]) and z[0, 2] == pytest.approx(1.234) and z[1, 1] == pytest.approx(65.535)


def _edit_json(path, fn):
    data = json.loads(path.read_text(encoding="utf-8"))
    fn(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_bad_fruit_tuple_raises_with_path(tmp_path):
    run = write_run(tmp_path, crop="apple", pitches=(30.0,))
    _edit_json(run / "manifest.json", lambda m: m["spec"]["trees"][0]["fruits"][0].pop())
    with pytest.raises(ValueError, match="fruit tuple has length") as e:
        SynthRun.open(run)
    assert "manifest.json" in str(e.value)


def test_bad_bag_tuple_raises(tmp_path):
    run = write_run(tmp_path, crop="apple", pitches=(30.0,))
    _edit_json(run / "manifest.json", lambda m: m["spec"]["trees"][0]["bags"][0].append(0))
    with pytest.raises(ValueError, match="bag tuple has length"):
        SynthRun.open(run)


def test_missing_planar_depth_convention_raises(tmp_path):
    run = write_run(tmp_path, crop="apple", pitches=(30.0,))
    _edit_json(run / "manifest.json", lambda m: m["conventions"].update(depth16="mm"))
    with pytest.raises(ValueError, match="conventions.depth16 is not planar depth"):
        SynthRun.open(run)


def test_missing_count1_raises(tmp_path):
    run = write_run(tmp_path, crop="apple", pitches=(30.0,))
    _edit_json(run / "manifest.json", lambda m: m["conventions"].pop("count1"))
    with pytest.raises(ValueError, match="conventions.count1 is missing"):
        SynthRun.open(run)


def test_record_tree_mismatch_raises(tmp_path):
    run = write_run(tmp_path, crop="apple", pitches=(30.0,))
    _edit_json(run / "000000_frame.json", lambda f: f["records"][0].update(tree=f["records"][0]["tree"] + 1))
    with pytest.raises(ValueError, match="record tree != fruit tree"):
        SynthRun.open(run)


def test_max_frames_bundle_is_not_sequence(tmp_path):
    run = SynthRun.open(write_run(tmp_path, crop="apple", max_frames=10))
    assert run.meta.is_sequence is False and run.meta.base_step_m is None and len(run.frames) == 10
    assert all(f.pos_k == -1 for f in run.frames)


def test_sequence_without_step_param(tmp_path):
    run = SynthRun.open(write_run(tmp_path, crop="apple", pitches=(30.0,), step_param=False))
    assert run.meta.is_sequence and run.meta.base_step_m == pytest.approx(0.15)


def test_select_frames_thins_by_position(apple_run):
    run = SynthRun.open(apple_run)
    kept = select_frames(run.frames, base_step_m=run.meta.base_step_m, step_m=0.30)
    assert [f.pos_k for f in kept if f.pass_key == A10] == list(range(0, 31, 2))
    with pytest.raises(ValueError, match="not a multiple of the base step"):
        select_frames(run.frames, base_step_m=0.15, step_m=0.20)
    with pytest.raises(ValueError, match="not a sequence bundle"):
        select_frames(run.frames, base_step_m=None, step_m=0.30)


def test_select_frames_pitch_and_side_filters(apple_run):
    run = SynthRun.open(apple_run)
    kept = select_frames(run.frames, base_step_m=0.15, pitches=[30], sides=["B"])
    assert {f.pass_key for f in kept} == {B30} and len(kept) == 31
    with pytest.raises(ValueError, match="available pitches"):
        select_frames(run.frames, base_step_m=0.15, pitches=[45])


def test_fingerprint_stable_ignores_rgb_and_detects_edits(tmp_path):
    run = write_run(tmp_path, crop="apple", pitches=(30.0,))
    f0 = SynthRun.open(run).meta.fingerprint
    (run / "000000_rgb.png").write_bytes(b"x")
    assert SynthRun.open(run).meta.fingerprint == f0
    p = run / "000001_frame.json"
    p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert SynthRun.open(run).meta.fingerprint != f0


def test_sidecar_commit(tmp_path):
    run = write_run(tmp_path, crop="apple", pitches=(30.0,))
    (run / "orchard_synth.commit").write_text("abc123\n", encoding="utf-8")
    assert SynthRun.open(run).meta.synth_commit == "abc123"


def test_fake_scene_all_target_fruits_positive(apple_run, pear_run):
    for root in (apple_run, pear_run):
        run = SynthRun.open(root)
        t = run.truth()
        target = {i for i, f in t.fruits.items() if f.tree in run.layout.tree_ids}
        seen = {r.id for recs in t.records.values() for r in recs if r.visib >= 0.25 and r.vis_px >= 20}
        assert target <= seen


def test_fake_scene_follows_real_record_rule(apple_run, pear_run):
    """Cut fragments are records too, as orchard_synth writes them; full_px is the on-screen area."""
    for root in (apple_run, pear_run):
        run = SynthRun.open(root)
        t, cam0 = run.truth(), run.frames[0].camera
        W, H = cam0.width, cam0.height
        frames = {f.idx: f for f in run.frames}
        target = {i for i, f in t.fruits.items() if f.tree in run.layout.tree_ids}
        recs = [(k, r) for k, v in t.records.items() for r in v if r.id in target and r.bbox is not None]
        assert all(0 < r.bbox[1] and r.bbox[3] < H for _, r in recs)      # no top/bottom cut on the target row
        off = [r for k, r in recs if r.visib >= 0.25 and r.vis_px >= 20 and not frames[k].camera.in_image(
            frames[k].camera.project(t.fruits[r.id].center[None])[0])[0]]
        assert off and all(r.trunc and (r.bbox[0] == 0 or r.bbox[2] == W) for r in off)


# --- malformed bundles: ValueError naming the file, with a phrase per error class ----------------


def _small_run(tmp_path):
    return write_run(tmp_path, crop="apple", pitches=(30.0,), max_frames=4)


def test_invalid_summary_json_raises_with_path(tmp_path):
    run = _small_run(tmp_path)
    (run / "summary.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match=r"summary\.json: invalid JSON"):
        SynthRun.open(run).truth()


def test_summary_without_trees_raises_with_path(tmp_path):
    run = _small_run(tmp_path)
    _edit_json(run / "summary.json", lambda s: s["summary"].pop("trees"))
    with pytest.raises(ValueError, match=r"summary\.json: no 'summary\.trees' mapping"):
        SynthRun.open(run).truth()


@pytest.mark.parametrize("row_ys, phrase", [
    ([0.0], "needs at least 2 rows"),
    ([7.5, 3.75, 0.0], "is not strictly ascending"),
    ([0.0, 3.75, 7.6], "is not evenly spaced"),
])
def test_bad_row_ys_raises_with_path(tmp_path, row_ys, phrase):
    run = _small_run(tmp_path)
    _edit_json(run / "manifest.json", lambda m: m["spec"].update(row_ys=row_ys))
    with pytest.raises(ValueError, match=rf"manifest\.json: spec\.row_ys {phrase}"):
        SynthRun.open(run)


def test_missing_manifest_field_raises_with_path(tmp_path):
    run = _small_run(tmp_path)
    _edit_json(run / "manifest.json", lambda m: m.pop("trees"))
    with pytest.raises(ValueError, match=r"manifest\.json: missing field 'trees'"):
        SynthRun.open(run)


def test_missing_frame_field_raises_with_path(tmp_path):
    run = _small_run(tmp_path)
    _edit_json(run / "000001_frame.json", lambda f: f["pose"].pop("hfov_deg"))
    with pytest.raises(ValueError, match=r"000001_frame\.json: missing field 'hfov_deg'"):
        SynthRun.open(run)


def test_short_frame_matrix_raises_with_path(tmp_path):
    run = _small_run(tmp_path)
    _edit_json(run / "000001_frame.json", lambda f: f["pose"].update(matrix_world=f["pose"]["matrix_world"][:2]))
    with pytest.raises(ValueError, match=r"000001_frame\.json: malformed field \(IndexError"):
        SynthRun.open(run)


@pytest.mark.parametrize("cut", ["garbage", "truncated"])
def test_unreadable_png_raises_with_path(tmp_path, cut):
    p = tmp_path / "000000_depth16.png"
    Image.fromarray((np.arange(360 * 640) % 60000).astype(np.uint16).reshape(360, 640)).save(p)
    data = p.read_bytes()
    p.write_bytes(data[:8] + b"garbage" if cut == "garbage" else data[: len(data) // 2])
    with pytest.raises(ValueError, match=r"000000_depth16\.png: unreadable PNG"):
        read_depth_m(p)
