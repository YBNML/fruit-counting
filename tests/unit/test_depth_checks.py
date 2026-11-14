"""The required checks of the LiDAR depth rungs, at the level of a ladder run through `eval ladder`.

One small ladder on the shared fake shake bundle runs twice under a base config that holds the LiDAR files and the
LiDAR section, and once without those rungs under the same base without the LiDAR keys. The checks: a rung that keeps
the camera box depth scores what it scores without the LiDAR keys; a LiDAR rung whose boxes all fall back scores what
its camera-only rung scores; the camera depth draws of a noisy LiDAR rung are those of its camera-only rung, seed by
seed; the same request reproduces its files byte for byte, and the pairing on the truth labels converges on the true
poses without LiDAR noise."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from counting.config.eval_schema import EvalAppConfig
from counting.config.hashing import config_hash
from counting.config.loader import load_yaml_model
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.budget import check_budget
from counting.evaluation.cli import eval_app
from counting.evaluation.runner import _merge, load_ladder, rung_config
from tests.unit.cachefake import write_truth_cache
from tests.unit.lidarfake import shared_lidar_bundle
from tests.unit.synthfake import EVAL_CONFIGS
from tests.unit.test_lidar_runner import DEPTH, LIDAR_EXTR, LIDAR_NOISE, RIG, START, VO, lidar_files

pytestmark = pytest.mark.pose

SPLITS = str(EVAL_CONFIGS / "splits.yaml")
CRITERIA = str(EVAL_CONFIGS / "criteria.yaml")
# The passes of the bundle's lowest pitch alone: the fake LiDAR points are the pixels of those frames, so there a
# fruit's points lie square to the camera, as its truth depth is taken.
PITCH = 10.0
STAGES = {"track": "landmark", "center": "measured", "assign": "grid"}
LD = {"lift": {"depth": "lidar", "lidar_file": "depth"}, "noise": {**LIDAR_NOISE, **LIDAR_EXTR}}
DRIFT_NOIMU = {"stages": STAGES, "pose": {"rig": True, "rig_ref": PITCH}, "align": {"mode": ["tracks", "row_plane"]},
               "noise": {**VO, **START, **RIG, **DEPTH}}
DRIFT_NOIMU_LD = _merge(DRIFT_NOIMU, LD)
# The ladder: the camera-only chain with the cached detections, the true poses with the judged LiDAR noise and
# extrinsic (no camera depth noise), the truth-label pairing with every point taken as it is and without LiDAR noise
# or extrinsic error, and the drift on the rig without fusion: camera only, with the LiDAR depth, and with every box
# falling back under either combine.
RUNGS = {"L3": {"stages": STAGES, "pose": {}}, "L5": {"stages": {"det": "cached", **STAGES}, "pose": {}}}
BASELINES = {
    "L3_ld": _merge({"stages": STAGES, "pose": {}}, LD),
    "L3_ld_oracle0": {"stages": STAGES, "pose": {},
                      "lift": {"depth": "lidar_oracle", "lidar_file": "depth",
                               "lidar": {"combine": "replace", "min_points": 1}},
                      "noise": {**{key: 0.0 for key in LIDAR_NOISE}, "lidar_near_range_m": 1.0,
                                **{key: 0.0 for key in LIDAR_EXTR}}},
    "L4_drift_noimu": DRIFT_NOIMU,
    "L4_drift_noimu_ld": DRIFT_NOIMU_LD,
    **{f"L4_drift_noimu_ld_skip_{combine}": _merge(DRIFT_NOIMU_LD, {"lift": {"lidar": {
        "min_points": 1_000_000, "combine": combine}}}) for combine in ("replace", "ivw")},
}
CAMERA_ONLY = ("L3", "L5", "L4_drift_noimu")
SKIPPED = ("L4_drift_noimu_ld_skip_replace", "L4_drift_noimu_ld_skip_ivw")


@dataclass
class LadderRuns:
    """The ladder under base A (LiDAR files and section) and its camera-only rungs under base B (without them):
    the files of each request, the results of each run, A's experiment folder and the bytes of its depth rows files
    and `budget.csv` as the first run wrote them."""

    root: Path
    config_a: Path
    config_b: Path
    ladder_a: Path
    ladder_b: Path
    results_a: dict
    results_b: dict
    folder_a: Path
    written: dict


def _write_yaml(path: Path, obj: dict) -> Path:
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")
    return path


def _ladder(config: Path, ladder: Path):
    return CliRunner().invoke(eval_app, ["ladder", "--config", str(config), "--ladder", str(ladder),
                                         "--splits", SPLITS, "--criteria", CRITERIA])


def _folder_of(result) -> Path:
    assert result.exit_code == 0, result.output
    return Path(result.output.splitlines()[0])


def _results(folder: Path) -> dict:
    return json.loads((folder / "results.json").read_bytes())


def _snapshot(folder: Path) -> dict:
    """{relative path: bytes} of the depth rows files and `budget.csv` of an experiment folder."""
    files = [*sorted((folder / "depth_rows").iterdir()), folder / "budget.csv"]
    return {str(p.relative_to(folder)): p.read_bytes() for p in files}


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> LadderRuns:
    shared = shared_lidar_bundle()
    run_id = SynthRun.open(shared.bundle).meta.run_id
    root = tmp_path_factory.mktemp("depth_checks")
    write_truth_cache(shared.bundle, root / "cache")
    plain = yaml.safe_load((EVAL_CONFIGS / "apple_dev.yaml").read_text(encoding="utf-8"))
    plain = _merge(plain, {"runs": [str(shared.bundle)], "passes": {"pitches": [PITCH]}, "lift": {"depth": "box_far"},
                           "detector": {"cache": str(root / "cache")}, "noise": {"seeds": [0, 1]},
                           "output_dir": str(root / "out")})
    with_lidar = _merge(plain, {"lift": {"lidar": {}, "lidar_files": lidar_files(shared, run_id)}})
    config_a, config_b = _write_yaml(root / "a.yaml", with_lidar), _write_yaml(root / "b.yaml", plain)
    ladder_a = _write_yaml(root / "ladder_a.yaml", {"rungs": RUNGS, "baselines": BASELINES})
    ladder_b = _write_yaml(root / "ladder_b.yaml", {"rungs": RUNGS,
                                                    "baselines": {"L4_drift_noimu": BASELINES["L4_drift_noimu"]}})
    folder_a, folder_b = _folder_of(_ladder(config_a, ladder_a)), _folder_of(_ladder(config_b, ladder_b))
    return LadderRuns(root=root, config_a=config_a, config_b=config_b, ladder_a=ladder_a, ladder_b=ladder_b,
                      results_a=_results(folder_a), results_b=_results(folder_b), folder_a=folder_a,
                      written=_snapshot(folder_a))


def _text(obj) -> str:
    return json.dumps(obj, sort_keys=True)


def _seed_counts(rung: dict) -> str:
    """Each seed's Y1 and Y6 (counts and ledgers) of a rung with noise, as JSON text; `[]` without noise."""
    return _text([(p["noise_seed"], p["y1"], p["y6"]) for p in rung.get("per_seed", ())])


def _paths_that_differ(a, b, prefix: str = "") -> set[str]:
    """The dotted paths at which the dumps `a` and `b` differ: a key on one side only, or a leaf value."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return set() if a == b else {prefix}
    out = set()
    for key in {*a, *b}:
        path = f"{prefix}.{key}" if prefix else key
        if key not in a or key not in b:
            out.add(path)
        else:
            out |= _paths_that_differ(a[key], b[key], path)
    return out


def test_ladder_lidar_off_rung_equals_camera_only(runs):
    with_lidar, without = (r["runs"][0]["rungs"] for r in (runs.results_a, runs.results_b))
    assert set(without) == set(CAMERA_ONLY)
    for name in CAMERA_ONLY:
        a, b = with_lidar[name], without[name]
        assert a["y6"]["ledger"] and all(p["ledger"] for p in a["y1"]["passes"]), name
        assert _text((a["y1"], a["y6"])) == _text((b["y1"], b["y6"])), name
        assert _seed_counts(a) == _seed_counts(b), name
        assert "camera_depth_sha256" in _text(a) and "camera_depth_sha256" not in _text(b), name
    assert len(with_lidar["L4_drift_noimu"]["per_seed"]) == 2 and "per_seed" not in with_lidar["L3"]

    cfg_a = load_yaml_model(runs.config_a, EvalAppConfig)
    cfg_b = load_yaml_model(runs.config_b, EvalAppConfig)
    rung_a = rung_config(cfg_a, load_ladder(runs.ladder_a), "L4_drift_noimu")
    rung_b = rung_config(cfg_b, load_ladder(runs.ladder_b), "L4_drift_noimu")
    dump_a, dump_b = rung_a.model_dump(mode="json"), rung_b.model_dump(mode="json")
    assert _paths_that_differ(dump_a, dump_b) == {"lift.lidar", "lift.lidar_files"}
    assert config_hash(rung_a) != config_hash(rung_b)
    del dump_a["lift"]["lidar"], dump_a["lift"]["lidar_files"]
    assert config_hash(EvalAppConfig.model_validate(dump_a)) == config_hash(rung_b)


def test_ladder_skip_rungs_equal_camera_only(runs):
    rungs = runs.results_a["runs"][0]["rungs"]
    camera = rungs["L4_drift_noimu"]
    for name in SKIPPED:
        rung = rungs[name]
        assert _text((rung["y1"], rung["y6"])) == _text((camera["y1"], camera["y6"])), name
        assert _seed_counts(rung) == _seed_counts(camera), name
        assert rung["lidar"]["coverage"] == 0.0 and rung["lidar"]["points"]["p90"] > 0, name   # points, none used
        assert [p["lidar"]["coverage"] for p in rung["per_seed"]] == [0.0, 0.0], name
    assert rungs["L4_drift_noimu_ld"]["lidar"]["coverage"] > 0


def test_ladder_camera_draw_digests_pair_up(runs):
    rungs = runs.results_a["runs"][0]["rungs"]

    def digests(name: str) -> list[str]:
        return [p["camera_depth_sha256"] for p in rungs[name]["per_seed"]]

    camera = digests("L4_drift_noimu")
    assert [p["noise_seed"] for p in rungs["L4_drift_noimu"]["per_seed"]] == [0, 1]
    assert all(re.fullmatch(r"[0-9a-f]{64}", sha) for sha in camera) and camera[0] != camera[1]
    for name in ("L4_drift_noimu_ld", *SKIPPED):
        assert digests(name) == camera, name
    assert digests("L3_ld") == ["none", "none"]
    for name in ("L3", "L3_ld_oracle0"):
        assert rungs[name]["camera_depth_sha256"] == "none" and "per_seed" not in rungs[name], name


def test_ladder_reruns_and_true_pose_oracle_converges(runs):
    again = _ladder(runs.config_a, runs.ladder_a)
    assert _folder_of(again) == runs.folder_a
    assert "reproduced (same results.json)" in again.output.splitlines()
    assert {Path(p).name for p in runs.written} == {
        *(f"{name}.csv" for name in [*RUNGS, *BASELINES]), "index.csv", "budget.csv"}
    assert _snapshot(runs.folder_a) == runs.written

    budget = runs.results_a["budget"]
    assert budget and {row["pair"] for row in budget} == {"L3→L3_ld"}
    check_budget(budget)

    oracle = runs.results_a["runs"][0]["rungs"]["L3_ld_oracle0"]["lidar"]
    assert oracle["method"] == "lidar_oracle" and oracle["depth"]["n_with_depth"] > 50
    assert oracle["depth"]["abs_error_p90_m"] <= 0.002
