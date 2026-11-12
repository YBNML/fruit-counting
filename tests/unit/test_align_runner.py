"""Alignment in evaluate: the tracked passes moved by the transforms of their segments, pose quality after the
alignment, the truth and oracle alignment rungs, and the alignment entry of the results through seed means,
sweeps and the report."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

import counting.evaluation.runner as runner_module
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.alignment import oracle_alignment, structure_points
from counting.evaluation.cli import eval_app
from counting.evaluation.posequality import pose_quality
from counting.evaluation.posesource import check_pose_supported
from counting.evaluation.report import results_bytes
from counting.evaluation.runner import evaluate, evaluate_seeds, prepare
from counting.evaluation.sweep import SweepCell, SweepPoint, _kept
from counting.geometry.rig import pose_matrix, rigid, with_pose
from counting.multiview.align import PassAlignment, Segments, track_x
from tests.unit.posefake import write_truth_pose_cache
from tests.unit.synthfake import EVAL_CONFIGS, ROOT, ctx_with, fake_cfg, load_splits, write_run
from tests.unit.test_leak import ALLOWED, _permute_and_reorder, _pred, _random_src

pytestmark = pytest.mark.pose
START = ("noise.pass_side_trans_m=0.05", "noise.pass_side_rot_deg=0.5")
RIG = ("pose.rig=true", "pose.rig_ref=30")
SPLITS = str(ROOT / "configs" / "eval" / "splits.yaml")


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """A 160 x 90 fake apple bundle of three trees and its prepared context."""
    root = write_run(tmp_path_factory.mktemp("small"), crop="apple", size=(160, 90))
    return root, prepare(SynthRun.open(root), fake_cfg(root, "apple"), load_splits())


@pytest.fixture(scope="module")
def small_result(small):
    """`small_result(*sets)`: evaluate of the `small` bundle under `fake_cfg(root, "apple", *sets)` with noise seed 0,
    made on first use and shared by the tests of this module, so treat it as read-only."""
    root, ctx = small
    made = {}

    def result(*sets):
        if sets not in made:
            made[sets] = evaluate(ctx, fake_cfg(root, "apple", *sets), noise_seed=0)
        return made[sets]

    return result


ALIGN_BASE = (*RIG, *START, "align.min_pairs=4")              # what the alignment modes run under on `small`


def test_alignment_takes_out_side_start_errors(small):
    root, ctx = small
    plain = fake_cfg(root, "apple", *RIG, *START)
    aligned = fake_cfg(root, "apple", *RIG, *START, "align.mode=[tracks]", "align.min_pairs=4")
    for ns in (0, 1):
        a, b = evaluate(ctx, plain, noise_seed=ns), evaluate(ctx, aligned, noise_seed=ns)
        before, after = (r["pose_quality"]["interpass"]["both_sides"]["p90_m"] for r in (a, b))
        assert a["alignment"] is None and after < min(0.06, before), ns
        assert b["y6"]["ledger"]["dup_cross"] < a["y6"]["ledger"]["dup_cross"]
        assert [p["n_tracks"] for p in b["y1"]["passes"]] == [p["n_tracks"] for p in a["y1"]["passes"]]
        (stage,) = b["alignment"]["stages"]
        assert (stage["stage"], stage["group"], [s["use"] for s in stage["segments"]]) == ("both_sides", "B",
                                                                                          ["own", "own"])
        assert b["alignment"]["settings"] == {"segment_m": 3.0, "pair_radius_m": 0.5, "roll_pitch_sigma_deg": None,
                                              "min_pairs": 4, "stress_row_bias_m": None, "stress_pair_keep": None,
                                              "oracle_sigma_m": None}


def test_tracks_move_with_the_segment_of_their_frames(small):
    root, ctx = small
    cfg = fake_cfg(root, "apple", *RIG, *START, "align.mode=[tracks,row_plane]", "align.min_pairs=4")
    staged = runner_module._track_passes(ctx, cfg, 0)
    aligned, transforms, entry = runner_module._align_passes(ctx, cfg, staged, noise_seed=0)
    for s, a in zip(staged, aligned):
        x = track_x(s.tracks.frames, {i: float(c.C[0]) for i, c in s.cameras.items()})
        np.testing.assert_array_equal(a.tracks.X, transforms[s.key].move(s.tracks.X, x))
        assert a.tracks.n_obs is s.tracks.n_obs and a.frames is s.frames and a.key == s.key
    assert [st["stage"] for st in entry["stages"]] == ["both_sides", "row_plane"]
    summary = entry["summary"]
    assert summary["segments"] == 4 and summary["own"] + summary["nearest"] + summary["line"] == 4
    assert summary["row_shift_p50_m"] is not None and summary["pairs_p50"] >= 4


def test_pose_quality_moves_by_segment_and_reports_the_aligned_world_end(small):
    _, ctx = small
    shift = rigid([0.0, 0.0, 0.0], [0.0, 0.02, 0.0])
    items = [(k, [f.camera for f in fr],
              [with_pose(f.camera, shift @ pose_matrix(f.camera)) if k.side == "B" else f.camera for f in fr])
             for k, fr in ctx.passes.items()]
    seg = Segments.along(ctx.layout.tree_x, 3.0)                 # centres 0 and 1.5; side B ends near x = -1
    assert len(seg.centres) == 2
    undo, keep = np.linalg.inv(shift), np.eye(4)

    def side_b(segments_undone):
        return {k: PassAlignment(seg, np.stack([undo if k.side == "B" and j in segments_undone else keep
                                                for j in range(2)])) for k, _, _ in items}

    plain, fixed, part = (pose_quality(items, ctx.layout, t) for t in (None, side_b({0, 1}), side_b({0})))
    assert plain["interpass"]["both_sides"]["p90_m"] == pytest.approx(0.02, abs=1e-12)
    assert fixed["interpass"]["both_sides"]["p90_m"] == pytest.approx(0.0, abs=1e-12)
    assert part["interpass"]["both_sides"]["p90_m"] == pytest.approx(0.02, abs=1e-12)   # frames of the other segment
    for p in part["passes"]:
        want = 0.02 if p["side"] == "B" else 0.0
        assert p["world_end"]["pos_m"] == pytest.approx(want, abs=1e-12)
        assert p["world_end_aligned"]["pos_m"] == pytest.approx(0.0, abs=1e-12)          # its last frame is in segment 0
    assert all(v is None for p in plain["passes"] for v in p["world_end_aligned"].values())
    assert fixed["summary"]["world_end_aligned"]["pos_m"] == pytest.approx(0.0, abs=1e-12)


def test_alignment_ignores_truth_ids_and_true_poses(small, small_result, tmp_path):
    root, ctx = small
    aligned = (*ALIGN_BASE, "align.mode=[tracks,row_plane]")
    cfg = fake_cfg(root, "apple", *aligned)
    run, splits = SynthRun.open(root), load_splits()
    a = small_result(*aligned)
    b = evaluate(prepare(run, cfg, splits, truth=_permute_and_reorder(run.truth(), np.random.default_rng(1))), cfg,
                 noise_seed=0)
    c = evaluate(ctx_with(ctx, src=_random_src(ctx, np.random.default_rng(2))), cfg, noise_seed=0)
    assert a["alignment"] == b["alignment"] == c["alignment"]
    assert _pred(a) == _pred(b) == _pred(c)
    write_truth_pose_cache(run, tmp_path)               # a file source: the used cameras come from the cache
    filed = fake_cfg(root, "apple", *aligned, "pose.source=file", "pose.cache=truth",
                     f"pose.caches.truth.dir={tmp_path}")
    jolt = rigid([0.0, 0.01, 0.02], [0.0, 0.03, -0.02])
    shaken = {k: [fr[0], *(replace(f, camera=with_pose(f.camera, jolt @ pose_matrix(f.camera))) for f in fr[1:])]
              for k, fr in ctx.passes.items()}                   # the first frame anchors the nominal start
    d, e = evaluate(ctx, filed, noise_seed=0), evaluate(ctx_with(ctx, passes=shaken), filed, noise_seed=0)
    assert d["alignment"]["stages"] and d["alignment"] == e["alignment"]
    assert _pred(d) == _pred(e)


STRONG = ("noise.drift_scale_sigma=0.045", "noise.drift_rate_deg_per_m=0.45", "noise.drift_step_trans_mm=21",
          "noise.drift_step_rot_deg=0.39")


def _with_structures(root):
    """`root` with a straight 2.4 m trunk on each tree of the target row and one 3.5 m post at x = 0.625 (posts
    10 m apart leave one in a row of 8.75 m) written into its manifest."""
    path = root / "manifest.json"
    man = json.loads(path.read_text(encoding="utf-8"))
    row_y = man["spec"]["row_ys"][man["args"]["target_row"]]
    for t in man["spec"]["trees"]:
        if t["id"] in man["target_row_trees"]:
            x, y = t["base"][0], t["base"][1]
            t["segments"] = [[[x, y, 0.4 * k], [x, y, 0.4 * (k + 1)], 0.05, 0.045, "trunk"] for k in range(6)]
    man["spec"]["structures"] = [{"kind": "post", "params": {"height": 3.5, "radius": 0.04, "x": 0.625, "y": row_y}}]
    path.write_text(json.dumps(man), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def structures(tmp_path_factory):
    """A 160 x 90 fake apple bundle of eight trees with trunks and one post, and its prepared context."""
    root = _with_structures(write_run(tmp_path_factory.mktemp("structures"), crop="apple", size=(160, 90),
                                      trees_per_row=8))
    return root, prepare(SynthRun.open(root), fake_cfg(root, "apple"), load_splits())


def test_structures_stay_on_the_truth_side(tmp_path):
    root = write_run(tmp_path / "s", crop="apple", size=(64, 36))
    assert SynthRun.open(root).structures() == ()
    path = root / "manifest.json"
    man = json.loads(path.read_text(encoding="utf-8"))
    row_y = man["spec"]["row_ys"][man["args"]["target_row"]]
    man["spec"]["structures"] = [
        {"kind": "post", "params": {"height": 1.0, "radius": 0.02, "x": 0.5, "y": row_y}},
        {"kind": "arch", "params": {"points": [[1.5, row_y - 0.5, 1.8], [1.5, row_y - 3.0, 1.0]], "radius": 0.0125}},
        {"kind": "purlin", "params": {"radius": 0.0125, "x0": 0.0, "x1": 1.0, "y": row_y + 1.875, "z": 2.1}},
        {"kind": "wire", "params": {"radius": 0.0015, "x0": 0.0, "x1": 2.0, "y": row_y, "z": 1.0}}]
    tree = next(t for t in man["spec"]["trees"] if t["id"] in man["target_row_trees"])
    tree["segments"] = [[[0.0, row_y, 0.0], [0.0, row_y, 0.6], 0.05, 0.045, "trunk"],
                        [[0.0, row_y, 0.6], [0.1, row_y, 1.2], 0.045, 0.04, "trunk"],
                        [[0.0, row_y, 1.0], [0.5, row_y, 1.4], 0.02, 0.01, "lateral"]]
    path.write_text(json.dumps(man), encoding="utf-8")
    run = SynthRun.open(root)
    assert [s.kind for s in run.structures()] == ["post", "arch", "purlin", "wire", "trunk"]
    assert run.structures()[0].params["height"] == 1.0
    assert run.structures()[-1].params == {"tree": tree["id"],
                                           "points": [[0.0, row_y, 0.0], [0.0, row_y, 0.6], [0.1, row_y, 1.2]]}
    np.testing.assert_allclose(structure_points(run.structures(), run.layout),
                               [[0.5, row_y, 0.25], [0.5, row_y, 0.75], [1.5, row_y - 0.5, 1.8],
                                [0.0, row_y + 1.875, 2.1], [0.5, row_y + 1.875, 2.1], [1.0, row_y + 1.875, 2.1],
                                [0.0, row_y, 0.25], [0.025, row_y, 0.75]])        # the trunk axis up to its top
    assert "Structure" not in ALLOWED                            # not a name the prediction side may import


def test_truth_fit_by_segment_beats_the_fit_by_pass_under_drift(small):
    root, ctx = small
    for ns in (0, 1):
        by_pass, by_segment = (evaluate(ctx, fake_cfg(root, "apple", *RIG, *START, *STRONG, f"align.mode={mode}"),
                                        noise_seed=ns) for mode in ("fit_truth_pass", "fit_truth_segment"))
        p90 = [r["pose_quality"]["interpass"]["both_sides"]["p90_m"] for r in (by_pass, by_segment)]
        assert p90[1] < 0.85 * p90[0], (ns, p90)
        assert [(s["stage"], len(s["segments"]), s["segments"][0]["centre_x_m"])
                for s in by_pass["alignment"]["stages"]] == [("truth_pass", 1, None)] * 4
        assert [(s["stage"], len(s["segments"])) for s in by_segment["alignment"]["stages"]] == [("truth_segment", 2)] * 4
        assert (by_pass["alignment"]["settings"]["segment_m"], by_segment["alignment"]["settings"]["segment_m"]) == (None, 3.0)


def test_oracle_structures_align_the_passes(structures, small_result):
    root, ctx = structures
    plain = fake_cfg(root, "apple", *RIG, *START)
    oracle = fake_cfg(root, "apple", *RIG, *START, "align.mode=oracle_struct")
    for ns in (0, 1):
        a, b = evaluate(ctx, plain, noise_seed=ns), evaluate(ctx, oracle, noise_seed=ns)
        before, after = (r["pose_quality"]["interpass"]["both_sides"]["p90_m"] for r in (a, b))
        assert after < min(0.05, before / 3), (ns, before, after)
        assert [(s["stage"], s["group"]) for s in b["alignment"]["stages"]] == [("oracle_struct", "A"),
                                                                              ("oracle_struct", "B")]
        assert {s["use"] for st in b["alignment"]["stages"] for s in st["segments"]} == {"own"}   # a trunk every 1.25 m
        assert b["alignment"]["settings"]["oracle_sigma_m"] == 0.02
    items = [(k, [f.camera for f in fr], [f.camera for f in fr]) for k, fr in ctx.passes.items()]
    seg, found = Segments.along(ctx.layout.tree_x, 3.0), SynthRun.open(root).structures()
    post = structure_points([s for s in found if s.kind == "post"], ctx.layout)
    alone = oracle_alignment(items, post, seg, rig=True, key=(0, 0, 5), sigma_m=0.02)
    assert {f.use for st in alone.stages for f in st.fits} == {"identity"}         # one post is a line: no rigid fit
    walled = oracle_alignment(items, structure_points(found, ctx.layout), seg, rig=True, key=(0, 0, 5), sigma_m=0.02,
                              depth=lambda key, k: np.full((90, 160), 0.5, np.float32))   # a wall 0.5 m in front
    assert {(f.use, f.n) for st in walled.stages for f in st.fits} == {("identity", 0)}
    bare = small_result(*ALIGN_BASE, "align.mode=oracle_struct")   # no structures: every segment keeps the identity
    assert {s["use"] for st in bare["alignment"]["stages"] for s in st["segments"]} == {"identity"}


def test_every_alignment_mode_runs(small, small_result):
    root, ctx = small
    stages = {"[tracks,row_plane]": ["both_sides", "row_plane"], "[row_plane,tracks]": ["row_plane", "both_sides"],
              "[row_plane]": ["row_plane"], "fit_truth_pass": ["truth_pass"] * 4,
              "fit_truth_segment": ["truth_segment"] * 4, "oracle_struct": ["oracle_struct"] * 2}
    for mode, want in stages.items():
        cfg = fake_cfg(root, "apple", *ALIGN_BASE, f"align.mode={mode}")
        check_pose_supported(cfg)
        first = small_result(*ALIGN_BASE, f"align.mode={mode}")      # may come from another test of the module
        again = evaluate(ctx, cfg, noise_seed=0)                       # always a new evaluation
        assert results_bytes(first) == results_bytes(again), mode
        assert [s["stage"] for s in first["alignment"]["stages"]] == want, mode


def test_alignment_entry_through_seeds_and_sweeps(small):
    root, ctx = small
    align = ("align.mode=[tracks,row_plane]", "align.min_pairs=4")
    cfg = fake_cfg(root, "apple", *RIG, *START, *align, "noise.seeds=[0,1]")
    combined = evaluate_seeds(ctx, cfg)
    entry, per = combined["alignment"], [p["alignment"] for p in combined["per_seed"]]
    assert set(entry) == {"mode", "settings", "summary"} and entry["mode"] == ["tracks", "row_plane"]
    assert [[s["stage"] for s in e["stages"]] for e in per] == [["both_sides", "row_plane"]] * 2
    assert entry["summary"]["segments"] == 4
    assert entry["summary"]["pairs_p50"] == pytest.approx((per[0]["summary"]["pairs_p50"] + per[1]["summary"]["pairs_p50"]) / 2)
    assert set(combined["spread"]["alignment"]) == {"summary"}
    point = SweepPoint("p", {}, None, None)
    assert _kept(SweepCell(point, None, cfg, ()), combined)["alignment"] == entry
    single_cfg = fake_cfg(root, "apple", *RIG, *align)
    single = _kept(SweepCell(point, None, single_cfg, ()), evaluate(ctx, single_cfg))
    assert set(single["alignment"]) == {"mode", "settings", "summary"}
    plain_cfg = fake_cfg(root, "apple", *RIG, *START, "noise.seeds=[0,1]")
    assert evaluate_seeds(ctx, plain_cfg)["alignment"] is None


def test_report_names_the_alignment_and_the_aligned_world_end(small, tmp_path):
    root, _ = small
    raw = yaml.safe_load((EVAL_CONFIGS / "apple_dev.yaml").read_text(encoding="utf-8"))
    raw.update(runs=[str(root)], output_dir=str(tmp_path / "out"), pose={"rig": True, "rig_ref": 30},
               align={"mode": ["tracks", "row_plane"], "min_pairs": 4})
    config = tmp_path / "aligned.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    r = CliRunner().invoke(eval_app, ["run", "--config", str(config), "--splits", SPLITS])
    assert r.exit_code == 0, r.output
    folder = Path(r.output.splitlines()[0])
    results = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    rung = results["runs"][0]["rungs"]["run"]
    assert results["schema_version"] == 11
    assert (rung["pose"]["predict_window"], rung["pose"]["flow"]) == (None, None)
    assert [s["stage"] for s in rung["alignment"]["stages"]] == ["both_sides", "row_plane"]
    report = (folder / "report.md").read_text(encoding="utf-8")
    assert "정합 tracks+row_plane" in report and "**정합**" in report
    assert "> 정합 설정: 구간 3 m · 짝 반경 0.5 m · σ_rp - · 짝 하한 4" in report
    assert "| 월드 끝 오차(정합 뒤) |" in report and "| 양면 | B |" in report and "| 줄 중심면 | AB |" in report
