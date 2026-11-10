"""The pass loop of `evaluate` in three stages: track every pass, align the passes, score each pass;
the cameras of the truth pose source in that loop, and the pose settings the runner refuses."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

import counting.evaluation.runner as runner_module
import counting.geometry.noise as noise_module
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.cli import eval_app
from counting.evaluation.posequality import apply_rigid
from counting.evaluation.runner import check_ladder_request, evaluate, run_ladder
from tests.unit.synthfake import DET_NOISE, EVAL_CONFIGS, fake_cfg
from tests.unit.test_drift import D_EXT_VO, ZERO_DRIFT, _core

pytestmark = pytest.mark.pose

SPLITS = str(Path(__file__).resolve().parents[2] / "configs" / "eval" / "splits.yaml")
POSE_LADDER = {"rungs": {"L3": {"stages": {"track": "landmark", "center": "measured", "assign": "grid"}},
                         "L4_drift": {"pose": {"rig": True, "fuse": "imu"}}}}

POSE_DEPTH = ("noise.pose_trans_sigma_m=0.02", "noise.pose_rot_sigma_deg=0.5", "noise.pass_bias_m=0.02",
              "noise.depth_rel_sigma=0.02", "noise.depth_dropout=0.05")
STAGE_ONE_COUNTS = {   # the counts of the loop before it was split into stages
    "pose_depth_seed1": (
        (
            (65, 66, 1, (("boundary", 0), ("dup_cross", 0), ("dup_track", 53), ("fp_bg", 0),
                         ("fp_off", 0), ("impure", 3), ("miss_det", 0), ("miss_drop", 0),
                         ("miss_merge", 0), ("n_pred", 65), ("n_truth", 12))),
            (70, 77, 7, (("boundary", 0), ("dup_cross", 0), ("dup_track", 58), ("fp_bg", 0),
                         ("fp_off", 0), ("impure", 0), ("miss_det", 0), ("miss_drop", 0),
                         ("miss_merge", 0), ("n_pred", 70), ("n_truth", 12))),
        ),
        ((45, 34, 37), (("boundary", 0), ("dup_cross", 38), ("dup_track", 66), ("fp_bg", 0),
                        ("fp_off", 0), ("impure", 3), ("miss_det", 0), ("miss_drop", 0),
                        ("miss_merge", 0), ("n_pred", 116), ("n_truth", 12))),
    ),
    "det_pose_depth_seed0": (
        (
            (57, 79, 22, (("boundary", 0), ("dup_cross", 0), ("dup_track", 44), ("fp_bg", 1),
                          ("fp_off", 0), ("impure", 2), ("miss_det", 0), ("miss_drop", 0),
                          ("miss_merge", 0), ("n_pred", 57), ("n_truth", 12))),
            (45, 65, 20, (("boundary", 0), ("dup_cross", 0), ("dup_track", 33), ("fp_bg", 0),
                          ("fp_off", 0), ("impure", 0), ("miss_det", 0), ("miss_drop", 0),
                          ("miss_merge", 0), ("n_pred", 45), ("n_truth", 12))),
        ),
        ((27, 26, 28), (("boundary", 0), ("dup_cross", 16), ("dup_track", 52), ("fp_bg", 1),
                        ("fp_off", 0), ("impure", 3), ("miss_det", 0), ("miss_drop", 0),
                        ("miss_merge", 0), ("n_pred", 81), ("n_truth", 12))),
    ),
}


def _counts(res: dict) -> tuple:
    """The integers of an evaluate result: per pass (U1, n_tracks, n_no3d, ledger counts), Y6 tree U and ledger counts."""
    def led(d):
        return tuple(sorted((k, v) for k, v in d.items() if isinstance(v, int) and not isinstance(v, bool)))
    passes = tuple((p["U1"], p["n_tracks"], p["n_no3d"], led(p["ledger"])) for p in res["y1"]["passes"])
    return passes, (None if res["y6"] is None else (tuple(t["U"] for t in res["y6"]["trees"]), led(res["y6"]["ledger"])))


def test_split_loop_keeps_stage_one_counts(leaf_runs, leaf_ctx, leaf_det_ctx):
    root = leaf_runs[0]
    got = {"pose_depth_seed1": _counts(evaluate(leaf_ctx[0], fake_cfg(root, "apple", *POSE_DEPTH), noise_seed=1)),
           "det_pose_depth_seed0": _counts(evaluate(leaf_det_ctx(0), fake_cfg(root, "apple", *DET_NOISE, *POSE_DEPTH),
                                                    noise_seed=0))}
    assert got == STAGE_ONE_COUNTS


def test_pass_stages_follow_ctx_passes(apple_run, ctx_of):
    ctx, cfg = ctx_of["apple"], fake_cfg(apple_run, "apple")
    staged = runner_module._track_passes(ctx, cfg, None)
    assert [p.key for p in staged] == list(ctx.passes)
    assert all(p.cameras[f.idx] is f.camera for p in staged for f in p.frames)
    aligned, transforms = runner_module._align_passes(ctx, cfg, staged)
    assert aligned is staged and transforms is None


def test_zero_drift_keys_match_l3_bytes(apple_run, ctx_of):
    cfg = fake_cfg(apple_run, "apple", "pose.rig_ref=30", *ZERO_DRIFT)
    assert not cfg.is_noisy()
    assert _core(evaluate(ctx_of["apple"], cfg)) == _core(evaluate(ctx_of["apple"], fake_cfg(apple_run, "apple")))


def test_zero_drift_keys_with_depth_noise_match_stage_one_bytes(apple_run, ctx_of):
    depth = "noise.depth_rel_sigma=0.02"
    cfg = fake_cfg(apple_run, "apple", "pose.rig_ref=30", *ZERO_DRIFT, depth)
    stage_one = fake_cfg(apple_run, "apple", depth)
    assert _core(evaluate(ctx_of["apple"], cfg, noise_seed=0)) == _core(evaluate(ctx_of["apple"], stage_one,
                                                                                 noise_seed=0))


def test_drift_changes_the_result(apple_run, ctx_of):
    cfg = fake_cfg(apple_run, "apple", "pose.rig_ref=30", *D_EXT_VO)
    plain = evaluate(ctx_of["apple"], fake_cfg(apple_run, "apple"))
    assert _core(evaluate(ctx_of["apple"], cfg, noise_seed=0)) != _core(plain)


def test_new_noise_steps_only_with_pose_noise_keys(apple_run, ctx_of, monkeypatch):
    steps: set[int] = set()

    def recording(original):
        def rng(*key):
            steps.add(key[2])
            return original(*key)
        return rng

    monkeypatch.setattr(runner_module, "noise_rng", recording(runner_module.noise_rng))
    monkeypatch.setattr(noise_module, "noise_rng", recording(noise_module.noise_rng))

    def steps_of(*sets: str) -> set[int]:
        steps.clear()
        evaluate(ctx_of["apple"], fake_cfg(apple_run, "apple", *sets), noise_seed=0)
        return set(steps)

    assert steps_of("pose.rig_ref=30", *POSE_DEPTH) == {2, 3, 4}
    assert steps_of("pose.rig=true", "pose.rig_ref=30", *D_EXT_VO, "noise.pass_side_trans_m=0.05",
                    "noise.pass_side_rot_deg=0.5", "noise.rig_extr_trans_mm=2", "noise.rig_extr_rot_deg=0.1",
                    "noise.depth_k=0.0036") == {2, 3, 4, 5, 6, 7, 9, 12}
    zero = steps_of("pose.rig_ref=30", *ZERO_DRIFT, "noise.depth_rel_sigma=0.02")
    assert {5, 6, 9, 12, 13} <= zero, zero   # drawn at 0 too: 13 for pass_pitch_* at 0, 9 for depth_k at 0
    assert 7 not in zero and 8 not in zero, zero


def test_pose_section_adds_pose_entries(apple_run, ctx_of):
    ctx = ctx_of["apple"]
    assert set(evaluate(ctx, fake_cfg(apple_run, "apple"))) == {"stages", "lift_depth", "y1", "y6"}
    res = evaluate(ctx, fake_cfg(apple_run, "apple", "pose.rig_ref=30"))
    assert res["pose"] == {"config": None, "source": "truth", "drift": False, "anchor": "nominal", "start": None,
                           "rig": False, "rig_ref": 30.0, "fuse": "none", "align": "none", "predict": "pose",
                           "dedup": "merge", "dedup_radius_m": 0.04, "dedup_band_m": 0.1, "gate_px_frac": 0.5}
    assert set(res["pose_quality"]) == {"rpe", "passes", "summary", "interpass"}
    drift = evaluate(ctx, fake_cfg(apple_run, "apple", "pose.rig_ref=30", *D_EXT_VO), noise_seed=0)["pose"]
    assert drift["drift"] is True and drift["start"] == "nominal_from_first_frame"
    rig = evaluate(ctx, fake_cfg(apple_run, "apple", "pose.rig=true", "pose.rig_ref=30", "noise.rig_extr_trans_mm=5",
                                 "noise.rig_extr_rot_deg=0.5"), noise_seed=0)
    assert rig["pose"]["drift"] is True
    quality = rig["pose_quality"]
    assert {p["pitch"] for p in quality["passes"]} == {10.0, 30.0}
    assert all(abs(p["in_pass"]["pos_m"]) <= 1e-9 for p in quality["passes"])
    assert abs(quality["rpe"]["trans_m"]["max"]) <= 1e-9
    assert all(p["in_pass_cam"]["pos_m"] > 1e-4 for p in quality["passes"] if p["pitch"] == 10.0)


def test_fit_truth_pass_takes_out_start_errors(apple_run, ctx_of):
    ctx = ctx_of["apple"]
    sets = ("pose.rig_ref=30", "noise.pass_side_trans_m=0.05", "noise.pass_side_rot_deg=0.5")
    fit_cfg = fake_cfg(apple_run, "apple", *sets, "align.mode=fit_truth_pass")
    plain = evaluate(ctx, fake_cfg(apple_run, "apple", *sets), noise_seed=0)
    fitted = evaluate(ctx, fit_cfg, noise_seed=0)
    assert plain["pose_quality"]["interpass"]["both_sides"]["p90_m"] > 0.01
    assert fitted["pose_quality"]["interpass"]["both_sides"]["p90_m"] <= 1e-9
    assert fitted["pose"]["align"] == "fit_truth_pass"
    # the fit moves the track centres, so the start errors no longer split fruits across passes
    assert fitted["y6"]["ledger"]["dup_cross"] < plain["y6"]["ledger"]["dup_cross"]

    def track_counts(res):
        return [(p["n_tracks"], p["n_no3d"]) for p in res["y1"]["passes"]]

    assert track_counts(fitted) == track_counts(plain)

    # each pass's centres move by the transform reported for that pass; a centre without 3D stays NaN
    staged = runner_module._track_passes(ctx, fit_cfg, 0)
    no3d = staged[0].tracks.X.copy()
    no3d[0] = np.nan
    staged = [replace(staged[0], tracks=replace(staged[0].tracks, X=no3d)), *staged[1:]]
    aligned, transforms = runner_module._align_passes(ctx, fit_cfg, staged)
    assert [p.key for p in aligned] == list(transforms) == [p.key for p in staged]
    for s, a in zip(staged, aligned):
        assert not np.allclose(transforms[s.key], np.eye(4))
        np.testing.assert_array_equal(a.tracks.X, apply_rigid(transforms[s.key], s.tracks.X))
        np.testing.assert_array_equal(np.isnan(a.tracks.X), np.isnan(s.tracks.X))
    assert np.isnan(aligned[0].tracks.X[0]).all()


def test_unsupported_pose_settings_stop_before_any_rung_runs(apple_run, ctx_of, monkeypatch):
    with pytest.raises(NotImplementedError, match="align.mode=tracks,row_plane is not supported"):
        evaluate(ctx_of["apple"], fake_cfg(apple_run, "apple", "pose.rig_ref=30", "align.mode=[tracks,row_plane]"))
    with pytest.raises(NotImplementedError, match="pose.fuse=imu is not supported"):
        evaluate(ctx_of["apple"], fake_cfg(apple_run, "apple", "pose.rig_ref=30", "pose.fuse=imu"))
    calls = []
    monkeypatch.setattr(runner_module, "evaluate", lambda *args, **kwargs: calls.append(args))
    cfg = fake_cfg(apple_run, "apple", "pose.rig_ref=30")
    with pytest.raises(NotImplementedError, match="pose.fuse=imu is not supported"):
        run_ladder(ctx_of["apple"], cfg, POSE_LADDER, ["L3", "L4_drift"])
    assert len(calls) == 0


def test_ladder_base_config_takes_truth_without_rig(apple_run):
    with pytest.raises(ValueError, match="ladder base config has pose.source, pose.rig or pose.fuse set"):
        check_ladder_request(fake_cfg(apple_run, "apple", "pose.rig=true", "pose.rig_ref=30"), POSE_LADDER, ["L3"])
    aligned = fake_cfg(apple_run, "apple", "pose.rig_ref=30", "align.mode=fit_truth_pass")
    with pytest.raises(ValueError, match="ladder base config has align.mode set"):
        check_ladder_request(aligned, POSE_LADDER, ["L3"])


def test_eval_refuses_unsupported_pose_rungs_before_opening_bundles(apple_run, tmp_path, monkeypatch):
    ladder = tmp_path / "ladder.yaml"
    ladder.write_text(yaml.safe_dump(POSE_LADDER), encoding="utf-8")
    raw = yaml.safe_load((EVAL_CONFIGS / "apple_dev.yaml").read_text(encoding="utf-8"))
    raw.update(runs=[str(apple_run)], output_dir=str(tmp_path / "out"), pose={"rig_ref": 30})
    cfg = tmp_path / "apple_pose.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    opened = []
    monkeypatch.setattr(SynthRun, "open", lambda root: opened.append(root))
    res = CliRunner().invoke(eval_app, ["ladder", "--config", str(cfg), "--ladder", str(ladder),
                                        "--rungs", "L3,L4_drift", "--splits", SPLITS])
    assert res.exit_code == 2, res.output
    assert "error: pose.fuse=imu is not supported" in res.output
    assert opened == [] and not (tmp_path / "out").exists()
