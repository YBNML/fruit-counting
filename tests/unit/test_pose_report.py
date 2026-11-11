"""Pose entries through the seed means, the sweep cells, the budget pairs and the report; results schema 7."""

from pathlib import Path

import pytest
import yaml

from counting.config.eval_schema import EvalAppConfig
from counting.config.hashing import config_hash
from counting.config.loader import load_yaml_model
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.budget import contrib_pairs, ladder_pairs, pose_pairs
from counting.evaluation.report import SCHEMA_VERSION, render_report
from counting.evaluation.runner import _patched, describe_run, evaluate, load_ladder, run_ladder, rung_config
from counting.evaluation.seeds import combine_seeds
from counting.evaluation.sweep import SweepCell, SweepPoint, _kept, grid_points, load_grid, run_sweep, sweep_cells
from tests.unit.synthfake import fake_cfg, load_splits
from tests.unit.test_sweep import GRIDS

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]
SWEEPS = ROOT / "configs" / "eval" / "sweeps"
BUNDLE = Path("bundle")        # sweep_cells never opens it
L3_STAGES = {"track": "landmark", "center": "measured", "assign": "grid"}   # the stages of ladder rung L3
PLAIN_RUNGS = ["L0", "L1", "L2a", "L2b", "L2c", "L3", "L4_base", "L5", "L3_dd", "L5_L4base"]
RIG_NOTE = "> 패스 안 표류와 RPE는 기준 카메라 틀이다(리그 보정을 믿는 값으로 옮김)."


def _grid(tmp_path: Path, grid: dict) -> Path:
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(grid), encoding="utf-8")
    return p


def _quality(rpe_p50: float, both_p50: float | None) -> dict:
    """A hand-made `pose_quality` entry with the RPE translation p50 and the both-sides p50 given."""
    errors = {"pos_m": 0.01, "rot_deg": 0.1, "along_m": 0.008, "side_m": 0.004, "height_m": 0.002,
              "yaw_deg": 0.08, "tilt_deg": 0.05}
    return {"rpe": {"n": 10, "trans_m": {"p50": rpe_p50, "p90": 0.004, "max": 0.005},
                    "rot_deg": {"p50": 0.01, "p90": 0.02, "max": 0.03}},
            "passes": [],
            "summary": {"in_pass": {**errors, "pos_frac": 0.002}, "in_pass_cam": {**errors, "pos_frac": 0.003},
                        "start": dict(errors), "world_end": dict(errors)},
            "interpass": {"same_side": {"n": 3, "p50_m": 0.01, "p90_m": 0.02},
                          "both_sides": {"n": 2, "p50_m": both_p50, "p90_m": 0.03}}}


def _without_pose(res: dict) -> dict:
    return {k: v for k, v in res.items() if k not in ("pose", "pose_quality", "alignment")}


@pytest.fixture(scope="module")
def pose_seeds(apple_run, ctx_of):
    """The config with a pose section, and two seed results of its `evaluate` with hand-made pose quality."""
    cfg = fake_cfg(apple_run, "apple", "pose.rig_ref=30")
    res = evaluate(ctx_of["apple"], cfg)
    return cfg, [{**res, "pose_quality": _quality(0.001, 0.01)}, {**res, "pose_quality": _quality(0.003, None)}]


def test_seeds_combine_pose_quality(pose_seeds):
    cfg, (r0, r1) = pose_seeds
    out = combine_seeds(cfg, [(0, r0), (1, r1)])
    assert out["pose"] == r0["pose"]
    assert out["pose_quality"]["rpe"]["trans_m"]["p50"] == pytest.approx(0.002)
    assert out["pose_quality"]["rpe"]["n"] == 10
    assert set(out["spread"]["pose_quality"]) == {"rpe", "summary", "interpass"}
    assert out["spread"]["pose_quality"]["rpe"]["trans_m"]["p50"] == [0.001, 0.003]
    assert out["pose_quality"]["interpass"]["both_sides"]["p50_m"] is None     # one seed has none
    assert out["spread"]["pose_quality"]["interpass"]["both_sides"]["p50_m"] is None
    assert [p["pose_quality"] for p in out["per_seed"]] == [r0["pose_quality"], r1["pose_quality"]]
    plain = combine_seeds(cfg, [(0, _without_pose(r0)), (1, _without_pose(r1))])
    assert "pose" not in plain and "pose_quality" not in plain and "pose_quality" not in plain["spread"]
    assert all("pose_quality" not in p for p in plain["per_seed"])


def test_sweep_keeps_pose_entries(pose_seeds):
    cfg, (r0, r1) = pose_seeds
    cell = SweepCell(point=SweepPoint("baseline", {}, None, None), base=None, cfg=cfg, overridden=())
    combined = combine_seeds(cfg, [(0, r0), (1, r1)])
    kept = _kept(cell, combined)
    assert kept["pose"] == combined["pose"] and kept["pose_quality"] == combined["pose_quality"]
    assert kept["spread"]["pose_quality"] == combined["spread"]["pose_quality"]
    assert [p["pose_quality"] for p in kept["per_seed"]] == [r0["pose_quality"], r1["pose_quality"]]
    one = _kept(cell, r0)
    assert one["pose"] == r0["pose"] and one["pose_quality"] == r0["pose_quality"]
    plain = _kept(cell, combine_seeds(cfg, [(0, _without_pose(r0)), (1, _without_pose(r1))]))
    assert set(plain) == {"point", "base", "overridden", "stages", "lift_depth", "y1", "y6", "noisy", "seeds",
                          "noise", "spread", "per_seed"}
    assert all(set(p) == {"noise_seed", "y1", "y6"} for p in plain["per_seed"])


def test_pose_pairs_and_ladder_pairs():
    rungs = {n: {"pose": {}} for n in ["L3", "L4_drift", "LVI", "L5", "L5VI", "L4_drift_noimu", "LV", "L3_al"]}
    expected = [("L3", "L4_drift"), ("L3", "LVI"), ("L5", "L5VI"), ("LVI", "L5VI"), ("L4_drift_noimu", "L4_drift"),
                ("LV", "LVI"), ("L3", "L3_al")]
    assert pose_pairs(rungs) == expected
    assert ladder_pairs(list(rungs), rungs) == expected
    chain = PLAIN_RUNGS[:8]
    for rungs in (dict.fromkeys(PLAIN_RUNGS), {n: {"pose": {"source": "truth"}} for n in PLAIN_RUNGS}):
        assert ladder_pairs(chain, rungs) == contrib_pairs(chain, rungs)


def test_report_names_the_pose_source(leaf_runs, leaf_ctx):
    root, ctx = leaf_runs[0], leaf_ctx[0]
    cfg = fake_cfg(root, "apple", "pose.rig_ref=30", "pose.config_label=c2")
    ladder = {"rungs": {"L4_drift_noimu": {
        "stages": L3_STAGES, "pose": {"rig": True},
        "noise": {"drift_scale_sigma": 0.0045, "drift_rate_deg_per_m": 0.045, "drift_step_trans_mm": 2.1,
                  "drift_step_rot_deg": 0.039}}}}
    out = run_ladder(ctx, cfg, ladder, ["L4_drift_noimu"])
    entry = {"meta": describe_run(ctx), **out}
    results = {"schema_version": SCHEMA_VERSION, "kind": "ladder", "run_name": "t", "experiment_id": "abc123abc123",
               "split": "dev", "crop": "apple", "runs": [entry], "pooled": None, "verdicts": None}
    lines = render_report(results, criteria=None).splitlines()
    assert lines[1] == ("> 합성 렌더 · 구성 c2 · 자세 정답 + 표류 모형 · 리그 켬(기준 30°) · 결합 none · 시작점 nominal · "
                        "정합 없음 · 예측 pose · 보정 전(U). 실제 정확도가 아니다.")
    assert lines[2] == "> 직선 묶음: 명목 시작점은 첫 프레임 정답 자세에 가정한 시작 오차를 더한 것이다. 판정에 쓰지 않는다."
    for line in ("**자세 품질**", "| RPE | p50 | p90 | 최대 |", "| 패스 사이 어긋남 | 짝 수 | p50 cm | p90 cm |",
                 "- 자세 모형 잡음: drift_step_trans_mm 2.1, drift_step_rot_deg 0.039, drift_scale_sigma 0.0045, "
                 "drift_rate_deg_per_m 0.045", RIG_NOTE):
        assert line in lines, line
    assert any(line.startswith("| 패스 안 표류(카메라 틀, 보조) |") for line in lines)


def test_sweep_report_has_a_pose_table(leaf_runs, tmp_path):
    root = leaf_runs[0]
    cfg = fake_cfg(root, "apple", "pose.rig_ref=30")   # a point's drift key needs the pose section
    grid = load_grid(_grid(tmp_path, {"bases": ["B"], "factors": {"noise.drift_step_trans_mm": [2.1]}}))
    cells = sweep_cells(cfg, grid, {"rungs": {"B": {"stages": L3_STAGES, "pose": {"rig": True}}}})
    entry, _ = run_sweep(SynthRun.open(root), cells, load_splits())
    results = {"schema_version": SCHEMA_VERSION, "kind": "sweep", "run_name": "t", "experiment_id": "abc123abc123",
               "split": "dev", "crop": "apple", "runs": [entry], "pooled": None, "verdicts": None,
               "sweep": {"grid": grid.path.name, "bases": list(grid.bases),
                         "points": [{"name": p.name, "patch": p.patch, "factor": p.factor, "value": p.value}
                                    for p in grid_points(grid)]}}
    lines = render_report(results, criteria=None).splitlines()
    assert "**자세 품질(점마다)**" in lines
    assert ("| 점 | 바탕 | RPE 이동 p90 cm | RPE 회전 p90 ° | 패스 안 끝 위치 % | 시작 위치 cm | 같은 면 p90 cm "
            "| 양면 p90 cm |") in lines


def test_sweep_cells_check_pose_settings_after_the_rung(tmp_path):
    ladder = {"rungs": {"B": {"stages": L3_STAGES, "pose": {"rig": True, "fuse": "imu"}}}}
    grid = load_grid(_grid(tmp_path, {"bases": ["B"],
                                      "common": {"noise": {"rig_extr_trans_mm": 2.0, "imu_gyro_n": 1.16e-3}},
                                      "factors": {"noise.pass_side_trans_m": [0.05]}}))
    cells = sweep_cells(fake_cfg(BUNDLE, "apple", "pose.rig_ref=30"), grid, ladder)
    assert [(c.point.name, c.base) for c in cells] == [("baseline", "B"), ("noise.pass_side_trans_m=0.05", "B")]
    assert all(c.cfg.pose.rig is True and c.cfg.pose.fuse == "imu" for c in cells)
    base, point = cells[0].cfg.noise, cells[1].cfg.noise
    assert base.imu_gyro_n is None and base.rig_extr_trans_mm is None      # the baseline point has no common
    assert point.imu_gyro_n == 1.16e-3 and point.rig_extr_trans_mm == 2.0 and point.pass_side_trans_m == 0.05

    # the committed grids without a pose section keep their cells: one validation gives what two gave
    ladder = load_ladder(SWEEPS / "ladder_l5.yaml")
    for name, (_, _, crops) in GRIDS.items():
        grid = load_grid(SWEEPS / name)
        for crop in crops:
            cfg = load_yaml_model(ROOT / "configs" / "eval" / f"{crop}_dev.yaml", EvalAppConfig)
            cfg = _patched(cfg, {"detector": {"cache": "runs/detect/x"}})     # for rung L5
            for cell in sweep_cells(cfg, grid, ladder):
                twice = _patched(cfg, cell.point.patch)
                if cell.base is not None:
                    twice = rung_config(twice, ladder, cell.base)
                assert cell.cfg == twice, (name, crop, cell.point.name, cell.base)
                assert config_hash(cell.cfg) == config_hash(twice), (name, crop, cell.point.name, cell.base)


def test_schema_version_is_10():
    assert SCHEMA_VERSION == 10
