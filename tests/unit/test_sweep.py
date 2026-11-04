import csv
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import counting.evaluation.sweep as sweep_module
from counting.config.eval_schema import EvalAppConfig
from counting.config.loader import load_yaml_model
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.cli import eval_app
from counting.evaluation.record import experiment_id
from counting.evaluation.report import _sweep_collapse_lines
from counting.evaluation.runner import _patched, evaluate, load_ladder, prepare, rung_config
from counting.evaluation.sweep import check_set_overlap, grid_points, load_grid, run_sweep, sweep_cells
from tests.unit.synthfake import fake_cfg, load_splits, python_c, write_run
from tests.unit.test_eval_cli import CRIT, SPLITS, _cfg_file

ROOT = Path(__file__).resolve().parents[2]
LADDER = load_ladder(ROOT / "configs" / "eval" / "sweeps" / "ladder.yaml")
PASS_KEYS = ("pitch", "side", "aisle", "n_frames", "U1", "G1", "e1", "ledger")
Y6_KEYS = ("mode", "n_landmarks", "row", "ledger", "metrics")
BUNDLE = Path("bundle")        # sweep_cells never opens it
runner = CliRunner()

SWEEPS = ROOT / "configs" / "eval" / "sweeps"
# grid file: (bases, points including the baseline, crops it is written for)
GRIDS = {
    "dedup_modes.yaml": (("L3", "L4_base"), 6, ("apple", "pear")),
    "det_noise.yaml": (("L3",), 19, ("apple", "pear")),
    "pose_noise.yaml": (("L3", "L3_dd"), 17, ("apple", "pear")),
    "spacing.yaml": (("L1", "L3", "last_box"), 5, ("apple", "pear")),
    "val/depth.yaml": (("L1", "L3"), 5, ("apple", "pear")),
    "val/depth_far.yaml": (("L1", "L3"), 8, ("apple", "pear")),
    "val/gate_apple.yaml": (("L1", "L3"), 10, ("apple",)),
    "val/gate_pear.yaml": (("L1", "L3"), 10, ("pear",)),
    "val/offset_apple.yaml": (("L1", "L2b", "L2c", "L3"), 3, ("apple",)),
    "val/offset_pear.yaml": (("L1", "L2b", "L2c", "L3"), 3, ("pear",)),
    "val/radius_apple.yaml": (("L1", "L3"), 6, ("apple",)),
    "val/radius_pear.yaml": (("L1", "L3"), 6, ("pear",)),
    "val/repair_apple.yaml": (("L1", "L3"), 6, ("apple",)),
    "val/repair_pear.yaml": (("L1", "L3"), 6, ("pear",)),
}


def _crop_cfg(crop, split):
    return load_yaml_model(ROOT / "configs" / "eval" / f"{crop}_{split}.yaml", EvalAppConfig)


def test_committed_grids_build_for_every_config():
    assert sorted(str(p.relative_to(SWEEPS)) for p in SWEEPS.rglob("*.yaml") if p.name != "ladder.yaml") == sorted(GRIDS)
    for name, (bases, n_points, crops) in GRIDS.items():
        grid = load_grid(SWEEPS / name)
        assert grid.bases == bases and len(grid_points(grid)) == n_points, name
        for crop in crops:
            for split in ("dev", "val", "exp"):
                assert len(sweep_cells(_crop_cfg(crop, split), grid, LADDER)) == n_points * len(bases), (name, crop)


def _leaves(patch, prefix=()):
    for k, v in patch.items():
        if isinstance(v, dict) and v:
            yield from _leaves(v, (*prefix, k))
        else:
            yield (*prefix, k), v


def test_committed_grid_values_survive_validation():
    for name, (_, _, crops) in GRIDS.items():
        for p in grid_points(load_grid(SWEEPS / name))[1:]:
            for crop in crops:
                dumped = _patched(_crop_cfg(crop, "val"), p.patch).model_dump()
                for path, value in _leaves(p.patch):
                    node = dumped
                    for part in path:
                        node = node[part]
                    # a YAML "1e-3" would stay a string here; an empty mapping is filled with defaults
                    assert isinstance(node, dict) if value == {} else node == value, (name, p.name, path)


def test_noise_grids_follow_the_noise_tables():
    det = load_grid(SWEEPS / "det_noise.yaml")
    assert det.common == {"stages": {"det": "oracle_noisy"}, "detector": {"noise": {}}}
    assert det.factors == {"detector.noise.miss_p": [0.1, 0.2, 0.3], "detector.noise.fp_lambda": [1.0, 2.0, 5.0],
                           "detector.noise.jitter_frac": [0.05, 0.10], "detector.noise.split_p": [0.05, 0.10],
                           "detector.noise.score": ["uniform"]}
    assert list(det.combos) == ["fp_0_min_obs_2", "fp_1_min_obs_2", "fp_2_min_obs_2", "fp_5_min_obs_2",
                                "jitter_0_box_median", "jitter_0.05_box_median", "jitter_0.10_box_median"]
    pose = load_grid(SWEEPS / "pose_noise.yaml")
    assert pose.factors == {"noise.pose_trans_sigma_m": [0.01, 0.02, 0.05], "noise.pose_rot_sigma_deg": [0.5, 1.0],
                            "noise.pass_bias_m": [0.02, 0.05, 0.10], "noise.depth_rel_sigma": [0.01, 0.02, 0.04],
                            "noise.depth_dropout": [0.05, 0.20]}
    rungs = {"weak": LADDER["baselines"]["L4_weak"], "base": LADDER["rungs"]["L4_base"],
             "strong": LADDER["baselines"]["L4_strong"]}
    assert pose.combos == {name: {"noise": patch["noise"]} for name, patch in rungs.items()}   # the L4 rung values


def _grid(tmp_path, text, name="g.yaml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_load_grid_and_points(tmp_path):
    g = load_grid(_grid(tmp_path, "bases: [L1, L3]\nfactors:\n  lift.far_min_frac: [0.2, 0.3]\n"
                                  "  tracker.max_gap: [null]\ncombos:\n  tight: {dedup: {radius_m: 0.03}}\n"))
    assert g.bases == ("L1", "L3")
    assert [(p.name, p.patch, p.factor, p.value) for p in grid_points(g)] == [
        ("baseline", {}, None, None),
        ("lift.far_min_frac=0.2", {"lift": {"far_min_frac": 0.2}}, "lift.far_min_frac", 0.2),
        ("lift.far_min_frac=0.3", {"lift": {"far_min_frac": 0.3}}, "lift.far_min_frac", 0.3),
        ("tracker.max_gap=null", {"tracker": {"max_gap": None}}, "tracker.max_gap", None),
        ("tight", {"dedup": {"radius_m": 0.03}}, None, None)]
    p = load_grid(_grid(tmp_path, "product:\n  lift.depth: [box_median, box_far]\n  tracker.depth_gate: [hard, soft]\n",
                        "p.yaml"))
    points = grid_points(p)
    assert p.bases == () and [q.name for q in points[1:]] == [
        "lift.depth=box_median,tracker.depth_gate=hard", "lift.depth=box_median,tracker.depth_gate=soft",
        "lift.depth=box_far,tracker.depth_gate=hard", "lift.depth=box_far,tracker.depth_gate=soft"]
    assert points[3].patch == {"lift": {"depth": "box_far"}, "tracker": {"depth_gate": "hard"}}
    same = load_grid(_grid(tmp_path, "product:\n  lift.depth: [box_far]\n  lift.far_scale: [0.7]\n", "s.yaml"))
    assert grid_points(same)[1].patch == {"lift": {"depth": "box_far", "far_scale": 0.7}}   # one section, both keys


BAD_GRIDS = [
    ("[1, 2]\n", "a grid file needs a mapping"),
    ("points: {}\n", "unknown grid key 'points'"),
    ("factors: {tracker.repair_m: [0.06]}\nproduct: {tracker.gate_m: [0.1]}\n",
     "factors and product cannot be used together"),
    ("bases: [L3]\n", "a grid needs factors, product or combos"),
    ("bases: [L3, L3]\nfactors: {tracker.repair_m: [0.06]}\n", "duplicate base 'L3'"),
    ("factors: {tracker.repair_m: []}\n", "tracker.repair_m needs a non-empty list of values"),
    ("factors: {seed: [1]}\n", "seed is not a stage setting"),
    ("combos: {x: 3}\n", "combo 'x' needs a mapping"),
    ("combos: {0.05: {dedup: {band_m: 0.05}}}\n", "combo names must be strings; quote the name 0.05"),
    ("combos: {7: {tracker: {max_gap: 7}}}\n", "combo names must be strings; quote the name 7"),
    ("combos: {true: {stages: {dedup: hybrid}}}\n", "combo names must be strings; quote the name True"),
    ("combos: {baseline: {dedup: {radius_m: 0.05}}}\n", "duplicate point name 'baseline'"),
    ("factors: [lift.depth]\n", "factors needs a mapping"),
    ("factors: {tracker.repair_m: [0.06]\n", "not valid YAML"),
    ("bases: L3\nfactors: {tracker.repair_m: [0.06]}\n", "bases needs a list of rung names"),
    ("product: [lift.depth]\n", "product needs a mapping"),
    ("combos: [x]\n", "combos needs a mapping"),
    ("", "a grid file needs a mapping"),
    ("common: [x]\nfactors: {tracker.repair_m: [0.06]}\n", "common needs a mapping"),
    ("common: {stages: {det: oracle_noisy}}\n", "a grid needs factors, product or combos"),
    ("common: {lift.depth: box_far}\nfactors: {tracker.repair_m: [0.06]}\n", "common key 'lift.depth' must be one of"),
    ("combos: {c: {lift.depth: box_far}}\n", "combo 'c' key 'lift.depth' must be one of"),
]


@pytest.mark.parametrize("text, phrase", BAD_GRIDS)
def test_load_grid_refuses_bad_files(tmp_path, text, phrase):
    p = _grid(tmp_path, text)
    with pytest.raises(ValueError, match=re.escape(phrase)) as e:
        grid_points(load_grid(p))
    assert str(e.value).startswith(f"{p}: ")


def test_sweep_cells_merge_order_and_overridden(tmp_path):
    cfg = fake_cfg(BUNDLE, "apple")
    g = load_grid(_grid(tmp_path, "bases: [L1, L3]\nfactors:\n  lift.depth: [box_far]\n  tracker.depth_gate: [soft]\n"
                                  "  stages.dedup: [self_side, oracle]\n  lift.center_offset_m: [{fruit: 0.015}]\n"))
    cells = sweep_cells(cfg, g, LADDER)
    offset = 'lift.center_offset_m={"fruit":0.015}'
    assert [(c.point.name, c.base, c.overridden) for c in cells] == [
        ("baseline", "L1", ()), ("baseline", "L3", ()),
        ("lift.depth=box_far", "L1", ("lift.depth",)), ("lift.depth=box_far", "L3", ()),
        ("tracker.depth_gate=soft", "L1", ()), ("tracker.depth_gate=soft", "L3", ()),
        ("stages.dedup=self_side", "L1", ("stages.dedup",)), ("stages.dedup=self_side", "L3", ()),
        ("stages.dedup=oracle", "L1", ("stages.dedup",)), ("stages.dedup=oracle", "L3", ()),   # same value: still listed
        (offset, "L1", ()), (offset, "L3", ())]
    by = {(c.point.name, c.base): c.cfg for c in cells}
    assert by[("lift.depth=box_far", "L1")].lift.depth == "det_depth"             # the rung wins
    assert by[("lift.depth=box_far", "L3")] == rung_config(fake_cfg(BUNDLE, "apple", "lift.depth=box_far"), LADDER, "L3")
    assert by[("tracker.depth_gate=soft", "L1")].tracker.depth_gate == "soft"     # a key the rung leaves alone stays
    assert by[("stages.dedup=self_side", "L1")].stages.dedup == "oracle"
    assert by[(offset, "L3")].lift.center_offset_m == {"fruit": 0.015, "bag": 0.03}   # a mapping value merges
    last = sweep_cells(cfg, load_grid(_grid(tmp_path, "bases: [last_box]\nfactors: {tracker.repair_m: [0.06]}\n",
                                            "last.yaml")), LADDER)
    assert [c.overridden for c in last] == [(), ("tracker.repair_m",)] and last[1].cfg.tracker.repair_m == 0
    flat = sweep_cells(cfg, load_grid(_grid(tmp_path, "factors: {tracker.repair_m: [0.06]}\n", "flat.yaml")), None)
    assert [(c.base, c.cfg) for c in flat] == [(None, cfg), (None, fake_cfg(BUNDLE, "apple", "tracker.repair_m=0.06"))]


def test_sweep_cells_overridden_includes_prefix_paths(tmp_path):
    """A point leaf under a rung leaf, or above one, is overridden; a sibling key is not."""
    ladder = {"rungs": {"R": {"stages": {"det": "oracle"}, "detector": {"noise": None},
                              "lift": {"center_offset_m": {"fruit": 0.0}}}}}
    g = load_grid(_grid(tmp_path, "bases: [R]\ncombos:\n"
                                  "  under: {stages: {det: oracle_noisy}, detector: {noise: {miss_p: 0.1}}}\n"
                                  "  above: {lift: {center_offset_m: {}, depth: box_far}}\n"))
    cells = sweep_cells(fake_cfg(BUNDLE, "apple"), g, ladder)
    assert [(c.point.name, c.overridden) for c in cells] == [
        ("baseline", ()), ("under", ("detector.noise.miss_p", "stages.det")), ("above", ("lift.center_offset_m",))]
    assert cells[1].cfg.detector.noise is None and cells[2].cfg.lift.depth == "box_far"   # the rung wins


def test_sweep_collapse_names_the_first_value_over_10_percent():
    def cell(point, row_abs, mae):
        return {"point": point, "base": "L3", "y6": {"metrics": {"row_abs_e_pipe": row_abs, "tree_mae_over_mean": mae}}}

    factor = "tracker.gate_m"
    points = [{"name": "baseline", "factor": None},
              *({"name": f"{factor}={v}", "factor": factor} for v in (0.05, 0.1, 0.2))]
    cells = [cell("baseline", 0.0, 0.0),
             cell(f"{factor}=0.05", 0.10, 0.10),               # exactly 10 %: not over
             cell(f"{factor}=0.1", 0.0, 0.25),                 # over (tree MAE)
             cell(f"{factor}=0.2", 0.5, 0.0)]                  # over (row e_pipe), but later
    assert _sweep_collapse_lines(points, ["L3"], cells)[1:] == [f"- L3 · {factor}: 0.1"]


def test_sweep_collapse_none_and_dash():
    points = [{"name": "baseline", "factor": None}, {"name": "tracker.gate_m=0.1", "factor": "tracker.gate_m"}]
    under = [{"point": n, "base": "L3", "y6": {"metrics": {"row_abs_e_pipe": 0.05, "tree_mae_over_mean": 0.1}}}
             for n in ("baseline", "tracker.gate_m=0.1")]
    assert _sweep_collapse_lines(points, ["L3"], under)[1:] == ["- L3 · tracker.gate_m: 없음"]
    assert _sweep_collapse_lines(points, ["L3"], [{**c, "y6": None} for c in under])[1:] == ["- L3 · tracker.gate_m: -"]


def test_sweep_cells_refuse_unsupported_and_invalid_points(tmp_path):
    cfg = fake_cfg(BUNDLE, "apple")
    g = _grid(tmp_path, "factors: {stages.det: [oracle_noisy]}\n")
    with pytest.raises(ValueError, match=re.escape(f"{g}: point 'stages.det=oracle_noisy': ")) as e:
        sweep_cells(cfg, load_grid(g), None)
    assert "stages.det=oracle_noisy needs a detector.noise section" in str(e.value)
    c = _grid(tmp_path, "factors: {detector.cache: [x]}\n", "c.yaml")
    with pytest.raises(NotImplementedError, match=re.escape(
            f"{c}: point 'detector.cache=x': detector.cache is not supported in this version")):
        sweep_cells(cfg, load_grid(c), None)
    v = _grid(tmp_path, "factors: {lift.far_min_frac: [1.5]}\n", "v.yaml")
    with pytest.raises(ValueError, match=re.escape(f"{v}: point 'lift.far_min_frac=1.5'")):
        sweep_cells(cfg, load_grid(v), None)
    b = load_grid(_grid(tmp_path, "bases: [L3]\nfactors: {tracker.repair_m: [0.06]}\n", "b.yaml"))
    with pytest.raises(ValueError, match="ladder base config has lift.depth=det_depth"):
        sweep_cells(fake_cfg(BUNDLE, "apple", "lift.depth=det_depth"), b, LADDER)


def test_check_set_overlap(tmp_path):
    points = grid_points(load_grid(_grid(tmp_path, "factors: {tracker.repair_m: [0.06]}\n"
                                                   "combos: {c: {lift: {center_offset_m: {fruit: 0.0}}}}\n")))
    check_set_overlap(["tracker.gate_m", "lift.depth", "run_name", "output_dir"], points)    # siblings are fine
    for key, grid_key in (("tracker.repair_m", "tracker.repair_m"), ("tracker", "tracker.repair_m"),
                          ("lift.center_offset_m", "lift.center_offset_m.fruit"),
                          ("lift.center_offset_m.fruit.x", "lift.center_offset_m.fruit")):
        with pytest.raises(ValueError, match=re.escape(f"--set {key} overlaps the grid key {grid_key}")):
            check_set_overlap([key], points)


def test_run_sweep_prepares_once_per_group_and_matches_separate_runs(leaf_runs, tmp_path, monkeypatch):
    root = leaf_runs[0]
    g = load_grid(_grid(tmp_path, "factors:\n  tracker.repair_m: [0.06, 0.08]\n  passes.step_m: [0.3]\n"))
    cells = sweep_cells(fake_cfg(root, "apple"), g, None)
    calls, real = [], sweep_module.prepare
    monkeypatch.setattr(sweep_module, "prepare", lambda run, c, splits, **k: calls.append(
        (c.passes.step_m, k.get("det_seed"))) or real(run, c, splits, **k))
    entry, timings = run_sweep(SynthRun.open(root), cells, load_splits())
    assert calls == [(None, None), (0.3, None)] and set(timings) == {"prepare", "evaluate"}
    assert [c["point"] for c in entry["cells"]] == ["baseline", "tracker.repair_m=0.06", "tracker.repair_m=0.08",
                                                    "passes.step_m=0.3"]
    assert entry["meta"]["n_frames"] == 62
    step = entry["cells"][3]
    cfg3 = fake_cfg(root, "apple", "passes.step_m=0.3")
    alone = evaluate(prepare(SynthRun.open(root), cfg3, load_splits()), cfg3)
    assert step["y1"]["summary"] == alone["y1"]["summary"]
    assert step["y1"]["passes"] == [{k: p[k] for k in PASS_KEYS} for p in alone["y1"]["passes"]]
    assert all(step["y6"][k] == alone["y6"][k] for k in Y6_KEYS)
    assert [p["n_frames"] for p in step["y1"]["passes"]] == [16, 16]
    assert "trees" in entry["cells"][0]["y6"] and "trees" not in step["y6"]


def test_common_patch_goes_under_every_point_but_the_baseline(tmp_path):
    g = load_grid(_grid(tmp_path, "bases: [L3]\ncommon: {stages: {det: oracle_noisy}, detector: {noise: {}}}\n"
                                  "factors: {detector.noise.miss_p: [0.1]}\ncombos:\n  med: {lift: {depth: box_median}}\n"))
    assert [(p.name, p.patch) for p in grid_points(g)] == [
        ("baseline", {}),
        ("detector.noise.miss_p=0.1", {"stages": {"det": "oracle_noisy"}, "detector": {"noise": {"miss_p": 0.1}}}),
        ("med", {"stages": {"det": "oracle_noisy"}, "detector": {"noise": {}}, "lift": {"depth": "box_median"}})]
    cells = sweep_cells(fake_cfg(BUNDLE, "apple"), g, LADDER)
    assert [c.cfg.stages.det for c in cells] == ["oracle", "oracle_noisy", "oracle_noisy"]   # L3 keeps the point's det
    assert [c.overridden for c in cells] == [(), (), ()]
    with pytest.raises(ValueError, match=re.escape("--set stages.det overlaps the grid key stages.det")):
        check_set_overlap(["stages.det"], grid_points(g))
    with pytest.raises(ValueError, match="bases need a ladder"):
        sweep_cells(fake_cfg(BUNDLE, "apple"), g, None)


def test_run_sweep_prepares_once_per_detection_seed(leaf_runs, tmp_path, monkeypatch):
    root = leaf_runs[0]
    g = load_grid(_grid(tmp_path, "factors: {noise.depth_dropout: [0.05]}\n"
                                  "combos:\n  det: {stages: {det: oracle_noisy}, detector: {noise: {miss_p: 0.2}}}\n"))
    cells = sweep_cells(fake_cfg(root, "apple", "noise.seeds=[0,1]"), g, None)
    calls, real = [], sweep_module.prepare
    monkeypatch.setattr(sweep_module, "prepare", lambda run, c, splits, **k: calls.append(k.get("det_seed"))
                        or real(run, c, splits, **k))
    entry, _ = run_sweep(SynthRun.open(root), cells, load_splits())
    assert calls == [None, 0, 1]                  # the pose and depth cell shares the oracle context
    base, drop, det = entry["cells"]
    assert "noisy" not in base and drop["seeds"] == det["seeds"] == [0, 1] and "trees" in base["y6"]
    alone = evaluate(real(SynthRun.open(root), cells[2].cfg, load_splits(), det_seed=1), cells[2].cfg, noise_seed=1)
    p0, p1 = det["per_seed"]
    assert p1["noise_seed"] == 1 and p1["y1"]["summary"] == alone["y1"]["summary"]
    assert all(p1["y6"][k] == alone["y6"][k] for k in Y6_KEYS) and p0["y1"]["summary"] != p1["y1"]["summary"]


def test_eval_sweep_detection_noise_on_L3_writes_seed_rows(leaf_runs, tmp_path):
    cfg = _cfg_file(tmp_path, leaf_runs[0], "apple")
    g = _grid(tmp_path, "bases: [L3]\ncommon: {stages: {det: oracle_noisy}, detector: {noise: {}}}\n"
                        "factors: {detector.noise.miss_p: [0.2]}\n")
    r = runner.invoke(eval_app, ["sweep", "--config", cfg, "--grid", str(g), "--splits", SPLITS,
                                 "--set", "noise.seeds=[0,1]"])
    assert r.exit_code == 0, r.output
    (d,) = (tmp_path / "out" / "apple_dev").iterdir()
    base, miss = json.loads((d / "results.json").read_text(encoding="utf-8"))["runs"][0]["cells"]
    assert base["stages"]["det"] == "oracle" and "noisy" not in base
    assert (miss["stages"]["det"], miss["base"], miss["overridden"], miss["seeds"]) == ("oracle_noisy", "L3", [], [0, 1])
    rows = list(csv.reader((d / "sweep.csv").read_text(encoding="utf-8").splitlines()))
    assert [(row[2], row[5]) for row in rows[1:]] == [
        ("baseline", ""), ("detector.noise.miss_p=0.2", "0"), ("detector.noise.miss_p=0.2", "1"),
        ("detector.noise.miss_p=0.2", "mean")]
    line = next(line for line in r.output.splitlines() if line.startswith("detector.noise.miss_p=0.2 @ L3: row U="))
    assert line.endswith(" (mean over noise seeds 0,1)")
    report = (d / "report.md").read_text(encoding="utf-8").splitlines()
    assert "**sweep** — g.yaml: 점 2개 × 칸 1개 · 잡음 칸은 시드 평균, 시드별 값은 sweep.csv" in report
    assert report[1] == "> 합성 렌더 · 정답 자세 · 보정 전(U). 실제 정확도가 아니다."            # detection noise only
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    assert rec["detector"]["name"] == "oracle" and rec["extra"]["detectors"] == ["oracle", "oracle_noisy"]


@pytest.mark.slow
def test_noisy_sweep_bytes_stable_across_hash_seeds(leaf_runs, tmp_path, monkeypatch):
    cfg = _cfg_file(tmp_path, leaf_runs[0], "apple")
    g = _grid(tmp_path, "bases: [L3]\nfactors: {noise.pose_trans_sigma_m: [0.02]}\n"
                        "combos:\n  det: {stages: {det: oracle_noisy}, detector: {noise: {miss_p: 0.1, fp_lambda: 1.0}}}\n")
    args = ["sweep", "--config", cfg, "--grid", str(g), "--splits", SPLITS, "--set", "noise.seeds=[0,1]"]
    r = runner.invoke(eval_app, args)
    assert r.exit_code == 0, r.output
    (d,) = (tmp_path / "out" / "apple_dev").iterdir()
    first = {n: (d / n).read_bytes() for n in ("results.json", "sweep.csv")}
    monkeypatch.setenv("PYTHONHASHSEED", "1")                # a fresh interpreter with another hash seed
    out2 = tmp_path / "out2"
    python_c(f"from counting.evaluation.cli import eval_app; eval_app({[*args, '--set', f'output_dir={out2}']!r})")
    assert {n: (out2 / "apple_dev" / d.name / n).read_bytes() for n in first} == first


def test_run_sweep_checks_frame_selection_before_any_prepare(leaf_runs, tmp_path, monkeypatch):
    root = leaf_runs[0]
    cells = sweep_cells(fake_cfg(root, "apple"), load_grid(_grid(tmp_path, "factors: {passes.step_m: [0.2]}\n")), None)
    calls = []
    monkeypatch.setattr(sweep_module, "prepare", lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match="step_m is not a multiple of the base step"):
        run_sweep(SynthRun.open(root), cells, load_splits())
    assert calls == []


def test_eval_sweep_matches_eval_run_and_writes_the_folder(leaf_runs, tmp_path):
    from counting.evaluation.report import _SWEEP_HEADER
    cfg = _cfg_file(tmp_path, leaf_runs[0], "apple")
    g = _grid(tmp_path, "factors: {tracker.repair_m: [0.06]}\n")
    r = runner.invoke(eval_app, ["sweep", "--config", cfg, "--grid", str(g), "--splits", SPLITS])
    assert r.exit_code == 0, r.output
    for extra in ([], ["--set", "tracker.repair_m=0.06"]):
        r = runner.invoke(eval_app, ["run", "--config", cfg, "--splits", SPLITS, *extra])
        assert r.exit_code == 0, r.output
    sweep_dir, sweep, runs = None, None, {}
    for d in (tmp_path / "out" / "apple_dev").iterdir():
        res = json.loads((d / "results.json").read_text(encoding="utf-8"))
        if res["kind"] == "sweep":
            sweep_dir, sweep = d, res
        else:
            repair = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))["tracker"]["repair_m"]
            runs[repair] = res["runs"][0]["rungs"]["run"]
    assert sorted(runs) == [0.04, 0.06]
    assert sorted(p.name for p in sweep_dir.iterdir()) == ["config.yaml", "record.json", "report.md", "results.json",
                                                         "sweep.csv"]
    assert sweep["schema_version"] == 4 and sweep["sweep"] == {"grid": "g.yaml", "bases": [], "points": [
        {"name": "baseline", "patch": {}, "factor": None, "value": None},
        {"name": "tracker.repair_m=0.06", "patch": {"tracker": {"repair_m": 0.06}}, "factor": "tracker.repair_m",
         "value": 0.06}]}
    base_cell, point_cell = sweep["runs"][0]["cells"]
    for cell, run in ((base_cell, runs[0.04]), (point_cell, runs[0.06])):
        assert cell["base"] is None and cell["overridden"] == []
        assert cell["y1"]["summary"] == run["y1"]["summary"]
        assert cell["y1"]["passes"] == [{k: p[k] for k in PASS_KEYS} for p in run["y1"]["passes"]]
        assert all(cell["y6"][k] == run["y6"][k] for k in Y6_KEYS)
    assert base_cell["y6"]["trees"] == runs[0.04]["y6"]["trees"] and "trees" not in point_cell["y6"]
    rows = list(csv.reader((sweep_dir / "sweep.csv").read_text(encoding="utf-8").splitlines()))
    run_id = sweep["runs"][0]["meta"]["run_id"]
    assert rows[0] == _SWEEP_HEADER and [row[:6] for row in rows[1:]] == [
        [run_id, "", "baseline", "{}", "", ""],
        [run_id, "", "tracker.repair_m=0.06", '{"tracker":{"repair_m":0.06}}', "", ""]]
    rec = json.loads((sweep_dir / "record.json").read_text(encoding="utf-8"))
    g_sha = hashlib.sha256(g.read_bytes()).hexdigest()
    assert rec["extra"] == {"grid_sha256": g_sha, "ladder_sha256": None, "detectors": ["oracle"]}
    fps = [d["fingerprint"] for d in rec["data"]] + [d["depth_fingerprint"] for d in rec["data"]]
    assert sweep["experiment_id"] == experiment_id(config_hash=rec["config_hash"], fingerprints=fps, code=rec["code"],
                                                   seed=rec["seeds"]["seed"], extra=f"sweep:g.yaml:{g_sha[:16]}:-")


@pytest.mark.slow
def test_eval_sweep_marks_overridden_keys_and_is_byte_stable(leaf_runs, tmp_path, monkeypatch):
    cfg = _cfg_file(tmp_path, leaf_runs[0], "apple")
    g = _grid(tmp_path, "bases: [L1, L3]\nfactors: {lift.depth: [box_far]}\n"
                        "combos:\n  both: {stages: {dedup: self_side}, lift: {depth: box_far}}\n")
    args = ["sweep", "--config", cfg, "--grid", str(g), "--splits", SPLITS, "--criteria", CRIT]
    r1 = runner.invoke(eval_app, args)
    assert r1.exit_code == 0, r1.output
    assert "baseline @ L3: row U=21 G6=12 e_pipe=+75.0%" in r1.output
    (d,) = (tmp_path / "out" / "apple_dev").iterdir()
    first = {n: (d / n).read_bytes() for n in ("results.json", "sweep.csv")}
    r2 = runner.invoke(eval_app, args)
    assert r2.exit_code == 0 and "reproduced (same results.json)" in r2.output, r2.output
    assert {n: (d / n).read_bytes() for n in first} == first
    monkeypatch.setenv("PYTHONHASHSEED", "1")              # a fresh interpreter with another hash seed
    out2 = tmp_path / "out2"
    cmd = [*args, "--set", f"output_dir={out2}"]
    python_c(f"from counting.evaluation.cli import eval_app; eval_app({cmd!r})")
    assert {n: (out2 / "apple_dev" / d.name / n).read_bytes() for n in first} == first
    rows = [row for row in csv.reader(first["sweep.csv"].decode("utf-8").splitlines()[1:])]
    assert [(row[1], row[2], row[4], row[6], row[7]) for row in rows] == [
        ("L1", "baseline", "", "det_depth", "oracle"), ("L3", "baseline", "", "box_median", "merge"),
        ("L1", "lift.depth=box_far", '["lift.depth"]', "det_depth", "oracle"),
        ("L3", "lift.depth=box_far", "", "box_far", "merge"),
        ("L1", "both", '["lift.depth","stages.dedup"]', "det_depth", "oracle"),     # sorted, not patch order
        ("L3", "both", "", "box_far", "self_side")]
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    g_sha = hashlib.sha256(g.read_bytes()).hexdigest()
    l_sha = hashlib.sha256((ROOT / "configs" / "eval" / "sweeps" / "ladder.yaml").read_bytes()).hexdigest()
    assert rec["extra"] == {"grid_sha256": g_sha, "ladder_sha256": l_sha, "detectors": ["oracle"]}
    fps = [x["fingerprint"] for x in rec["data"]] + [x["depth_fingerprint"] for x in rec["data"]]
    exp_id = json.loads(first["results.json"])["experiment_id"]
    assert exp_id == experiment_id(config_hash=rec["config_hash"], fingerprints=fps, code=rec["code"],
                                   seed=rec["seeds"]["seed"], extra=f"sweep:g.yaml:{g_sha[:16]}:{l_sha[:16]}")
    lines = (d / "report.md").read_text(encoding="utf-8").splitlines()
    assert "**sweep** — g.yaml: 점 3개 × 칸 2개" in lines
    assert "| 점 | 칸 | 덮인 키 | e1 중앙값 | e1 최대 | U | G6 | e_pipe | 그루 MAE÷평균 | 기준 |" in lines
    assert any(line.startswith("| lift.depth=box_far | L1 | lift.depth |") for line in lines)
    assert any(line.startswith("| both | L1 | lift.depth, stages.dedup |") for line in lines)
    assert any(re.fullmatch(r"\| baseline \| L3 \| - \|.*\| \d+/\d+ \|", line) for line in lines)
    assert "- L3 · lift.depth: box_far" in lines           # L3 row e_pipe is far above 10 % on this bundle


def test_eval_sweep_checks_every_bundle_first_and_sorts_bundles(leaf_runs, tmp_path, monkeypatch):
    coarse = write_run(tmp_path / "coarse", crop="apple", pitches=(30.0,), step=0.3)    # base step 0.3, not 0.15
    cfg = _cfg_file(tmp_path, leaf_runs[0], "apple")
    orders = [(leaf_runs[0], coarse), (coarse, leaf_runs[0])]

    def sweep(grid, runs):
        return runner.invoke(eval_app, ["sweep", "--config", cfg, "--grid", str(grid), "--splits", SPLITS,
                                        *(a for p in runs for a in ("--run", str(p)))])

    g = _grid(tmp_path, "factors: {tracker.repair_m: [0.06]}\n")
    r1, r2 = (sweep(g, runs) for runs in orders)
    assert r1.exit_code == 0 and r2.exit_code == 0 and "reproduced (same results.json)" in r2.output, r2.output
    calls = []
    monkeypatch.setattr(sweep_module, "prepare", lambda *a, **k: calls.append(1))
    step = _grid(tmp_path, "factors: {passes.step_m: [0.45]}\n", "step.yaml")       # a multiple of 0.15 only
    for runs in orders:
        r = sweep(step, runs)
        assert r.exit_code == 2 and "step_m is not a multiple of the base step" in r.output and calls == [], r.output


BAD_SWEEPS = [
    ("factors: {tracker.repair_m: [0.06]}\nproduct: {tracker.gate_m: [0.1]}\n", [],
     "factors and product cannot be used together"),
    ("factors: {tracker.repair_m: [0.06]}\n", ["--set", "tracker.repair_m=0.05"],
     "--set tracker.repair_m overlaps the grid key tracker.repair_m"),
    ("factors: {stages.det: [oracle_noisy]}\n", [], "stages.det=oracle_noisy needs a detector.noise section"),
    ("bases: [L9]\nfactors: {tracker.repair_m: [0.06]}\n", [], "unknown rung 'L9'"),
    ("bases: [L3]\nfactors: {tracker.repair_m: [0.06]}\n", ["--set", "lift.depth=det_depth"],
     "ladder base config has lift.depth=det_depth"),
    ("bases: [L3]\nfactors: {tracker.repair_m: [0.06]}\n", ["--set", "noise.depth_dropout=0.05"],
     "ladder base config has noise set"),
    ("factors: {tracker.repair_m: [0.06]}\n", ["--set", "lift.depth.x=1"], "error: --set: "),
    ("combos: {0.05: {stages: {dedup: hybrid}, dedup: {band_m: 0.05}}}\n", [], "combo names must be strings"),
]


@pytest.mark.parametrize("grid, sets, phrase", BAD_SWEEPS)
def test_eval_sweep_input_errors_exit_2_before_any_bundle(tmp_path, grid, sets, phrase):
    cfg = _cfg_file(tmp_path, tmp_path / "missing", "apple")
    r = runner.invoke(eval_app, ["sweep", "--config", cfg, "--grid", str(_grid(tmp_path, grid)), "--splits", SPLITS,
                                 *sets])
    assert r.exit_code == 2 and phrase in r.output and "manifest.json" not in r.output, r.output
