"""Held-out depth requests: the held-out seeds closed for good, every held-out seed listed under test only, the
protect section a frozen config needs for the new held-out seeds, and the request that section binds (the ladder,
splits and criteria files with their sha256, the output folder, the allowed rung lists, the grids and the judged
rungs, which rung patches may not change). Then what a request of such a config must also pass: it runs once,
wherever it runs from, its rungs keep the frozen lift settings but the depth method and the LiDAR file, its LiDAR
rungs and grid cells have LiDAR noise (except the cells of a grid's baseline point) and its judged rung the depth
unit's LiDAR depth, it opens every held-out bundle, and the LiDAR files it reads are the pinned ones of those
bundles, from one clean simulator commit, with the noise values of the files' contract on its LiDAR depth rungs.

The requests run on a stand-in counting repository with three fake held-out bundles (seeds 1006 to 1008, shaken,
with the LiDAR files of a depth unit and of a navigation unit beside them) and one fake val bundle (seed 200).
Scoring is replaced by a stub that ends the command with exit code 0, so a request that reaches it passed every
check before scoring; one test runs a held-out request through to its results."""

import copy
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

import counting.evaluation.fusion as fusion
import counting.evaluation.protect as protect
import counting.evaluation.record as record
import counting.evaluation.runner as runner_module
import counting.evaluation.sweep as sweep_module
from counting.config.eval_schema import CLOSED_TEST_SEEDS, LEGACY_TEST_SEEDS, EvalAppConfig, ProtectConfig, SplitsConfig
from counting.config.hashing import config_hash
from counting.config.loader import load_yaml_model
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.cli import eval_app
from counting.evaluation.runner import ProtectionError, _merge, rung_config
from tests.unit.lidarfake import sidecar_sha256, write_lidar_files
from tests.unit.synthfake import ROOT, SIDECAR_COMMIT, write_run

pytestmark = pytest.mark.pose
runner = CliRunner()
EVAL = ROOT / "configs" / "eval"
TEST_SEEDS = (1006, 1007, 1008)
SIM_COMMIT = "1" * 40
CLEAN = {"commit": "a" * 40, "dirty": False, "diff_sha256": None}
BUNDLE = {"crop": "apple", "pitches": (10.0,), "size": (160, 90), "trees_per_row": 1}
# LiDAR units beside the held-out bundles: (mount position in m, sha256 of the simulation config).
UNITS = {"depth": ((0.0, 0.0, 0.1), "ab" * 32), "nav": ((0.0, 0.0, 0.5), "cd" * 32)}

# Files of the stand-in repository, by their repository paths.
SPLITS = "configs/eval/splits_depth.yaml"
SPLITS_B = "configs/eval/splits_depth_b.yaml"           # the same bytes at another path
SPLITS_POSE = "configs/eval/splits_pose.yaml"
SPLITS_FIRST = "configs/eval/splits.yaml"
CRITERIA = "configs/eval/criteria_depth.yaml"
CRITERIA_B = "configs/eval/criteria_depth_b.yaml"       # the same bytes at another path
CRITERIA_OTHER = "configs/eval/criteria_other.yaml"     # judges L4_drift instead
LADDER = "configs/eval/sweeps_depth/ladder_depth.yaml"
LADDER_B = "configs/eval/sweeps_depth/ladder_depth_b.yaml"   # the same bytes at another path
LADDER_WIDE = "configs/eval/sweeps_depth/ladder_widened.yaml"  # rung L3 widens protect.ladders
POSE_LADDER = "configs/eval/sweeps_pose/ladder_pose.yaml"
GRID = "configs/eval/sweeps_depth/depth_grid.yaml"
GRID_B = "configs/eval/sweeps_depth/depth_grid_b.yaml"   # not one of protect.grids
GRID_L3 = "configs/eval/sweeps_depth/grid_l3.yaml"       # built on rung L3 of the widened ladder
FUSE_FIXED = "configs/pose/fuse_fixed.yaml"
VAL_CONFIG = "configs/eval/apple_val_depth.yaml"
LADDER_VARIANTS = "configs/eval/sweeps_depth/ladder_variants.yaml"   # the small ladder and `_variant_rungs`
GRID_ZERO = "configs/eval/sweeps_depth/grid_zero.yaml"     # built on rung ld_range0 (no LiDAR range noise)
GRID_BIAS = "configs/eval/sweeps_depth/grid_bias.yaml"     # a LiDAR range bias on rung ld_no_bias
GRID_LIFT = "configs/eval/sweeps_depth/grid_lift.yaml"     # built on rung lift_window (another LiDAR depth window)
GRID_BARE = "configs/eval/sweeps_depth/grid_bare.yaml"     # LiDAR noise axes on rung B_drift_ld (no noise of its own)
HERE = "runs/eval_here"     # an output folder relative to the counting repository, ignored by its git
JUDGED_VARIANTS = ("oracle", "nav", "box", "extr")         # the judged rung of each `_judged_ladder` (`_judged_rungs`)

ALLOWED = [["L3", "L4_drift", "L4_drift_ld"], ["L3", "L4_drift_ldn", "L4_drift_ld_orc"]]
RUN_NAME = "apple_test_depth"
POSE_LADDER_RULE = "split=test with a pose section needs --ladder configs/eval/sweeps_pose/ladder_pose.yaml"
ZERO = "0" * 64
OTHER_COMMIT = "2" * 40

# Noise of the small ladder: the body odometry flow, the side start, the rig calibration, the camera depth, the
# LiDAR points and the extrinsic of each unit (the depth unit's own, the navigation unit's camera-IMU one).
VO = {"drift_scale_sigma": 0.0045, "drift_rate_deg_per_m": 0.045, "drift_step_trans_mm": 2.1,
      "drift_step_rot_deg": 0.039}
START = {"pass_side_trans_m": 0.05, "pass_side_rot_deg": 0.5}
RIG = {"rig_extr_trans_mm": 14.0, "rig_extr_rot_deg": 0.4}
DEPTH = {"depth_rel_sigma": 0.02, "depth_dropout": 0.05}
LIDAR_NOISE = {"lidar_range_sigma_m": 0.02, "lidar_range_sigma_near_m": 0.03, "lidar_near_range_m": 1.0,
               "lidar_ang_sigma_deg": 0.15, "lidar_dropout": 0.05, "lidar_range_bias_m": 0.0}
LIDAR_EXTR = {"lidar_extr_trans_mm": 10.0, "lidar_extr_rot_deg": 0.3}
IMU_EXTR = {"imu_extr_trans_mm": 10.0, "imu_extr_rot_deg": 0.3}
DRIFT = {"pose": {"rig": True}, "noise": {**VO, **START, **RIG, **DEPTH}}


def _small_ladder() -> dict:
    """Rungs L3 (true poses), L4_drift (the drift model on the rig), L4_drift_ld (with the depth-unit LiDAR),
    L4_drift_ldn (the navigation-unit LiDAR) and L4_drift_ld_orc (the truth-label pairing), and the baseline
    L4_drift_ld_val that runs on val only."""
    ld = _merge(DRIFT, {"lift": {"depth": "lidar"}, "noise": {**LIDAR_NOISE, **LIDAR_EXTR}})
    return {"rungs": {"L3": {"pose": {}}, "L4_drift": DRIFT, "L4_drift_ld": ld,
                      "L4_drift_ldn": _merge(DRIFT, {"lift": {"depth": "lidar", "lidar_file": "nav"},
                                                     "noise": {**LIDAR_NOISE, **IMU_EXTR}}),
                      "L4_drift_ld_orc": _merge(ld, {"lift": {"depth": "lidar_oracle"}})},
            "baselines": {"L4_drift_ld_val": ld},
            "val_only": ["L4_drift_ld_val"]}


def _without(rung: dict, key: str) -> dict:
    """`rung` without its noise key `key`."""
    out = copy.deepcopy(rung)
    del out["noise"][key]
    return out


def _variant_rungs() -> dict:
    """More rungs on those of the small ladder. `lift_*` change one lift setting (`lift_det_depth` only the depth
    method); `ld_range0`, `ldn_ang0` and `orc_ang0` take the LiDAR depth (of the depth unit, of the navigation unit,
    of the truth-label pairing) with a LiDAR range or angle sigma of 0; `ld_range03` moves the LiDAR range sigma off
    the files' contract, and `ld_no_bias` and `ld_no_dropout` leave out one LiDAR noise key. `B_drift_ld` is a grid
    base on the LiDAR depth without any noise, whose grid adds the noise in its common block."""
    small = _small_ladder()["rungs"]
    ld, ldn, orc = small["L4_drift_ld"], small["L4_drift_ldn"], small["L4_drift_ld_orc"]
    return {"B_drift_ld": {"pose": {"rig": True}, "lift": {"depth": "lidar"}},
            "lift_far_min": _merge(DRIFT, {"lift": {"far_min_frac": 0.3}}),
            "lift_box_frac": _merge(DRIFT, {"lift": {"box_frac": 0.6}}),
            "lift_offset": _merge(DRIFT, {"lift": {"center_offset_m": {"fruit": 0.025}}}),
            "lift_window": _merge(ld, {"lift": {"lidar": {"window_s": 0.25}}}),
            "lift_files": _merge(DRIFT, {"lift": {"lidar_files": {"nav": {"dir": "elsewhere"}}}}),
            "lift_det_depth": _merge(DRIFT, {"lift": {"depth": "det_depth"}}),
            "ld_range0": _merge(ld, {"noise": {"lidar_range_sigma_m": 0.0}}),
            "ldn_ang0": _merge(ldn, {"noise": {"lidar_ang_sigma_deg": 0.0}}),
            "orc_ang0": _merge(orc, {"noise": {"lidar_ang_sigma_deg": 0.0}}),
            "ld_range03": _merge(ld, {"noise": {"lidar_range_sigma_m": 0.03}}),
            "ld_no_bias": _without(ld, "lidar_range_bias_m"),
            "ld_no_dropout": _without(ld, "lidar_dropout")}


def _judged_rungs() -> dict:
    """Judged rungs L4_drift_ld that a held-out request refuses, each one a valid config: on the truth-label pairing,
    on the navigation unit's LiDAR file (its camera-IMU extrinsic keys in place of the LiDAR extrinsic ones), on the
    camera box depth (without LiDAR keys) and with no LiDAR extrinsic rotation error."""
    small = _small_ladder()["rungs"]
    ld = small["L4_drift_ld"]
    return {"oracle": small["L4_drift_ld_orc"], "nav": small["L4_drift_ldn"],
            "box": _merge(DRIFT, {"lift": {"depth": "box_far"}}),
            "extr": _merge(ld, {"noise": {"lidar_extr_rot_deg": 0.0}})}


def _judged_ladder(variant: str) -> str:
    """The small ladder with judged rung `_judged_rungs()[variant]`, by its repository path."""
    return f"configs/eval/sweeps_depth/ladder_judged_{variant}.yaml"


def _set(*keys: str, value):
    """A sidecar edit that sets the field at `keys` to `value`."""
    def edit(sidecar: dict) -> None:
        node = sidecar
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return edit


# How each LiDAR provenance case breaks the depth unit's files of the second held-out bundle (seed 1007): `digest`
# pins another digest (None: none) for the files as written; otherwise the case copies the files of the three bundles
# (under `folder`, the name of the simulation config folder) and `edit`s the second one's sidecar or `damage`s one of
# its files (`lidar.json` is removed, another file gets one more byte); `sim_commits` replaces the expected commits.
PROVENANCE_CASES = {
    "pin": {"digest": None},
    "sidecar": {"damage": "lidar.json"},
    "sidecar_sha": {"digest": ZERO},
    "format": {"edit": _set("format", value="orchard-lidar-sidecar/0")},
    "variant": {"edit": _set("variant", value="tilt")},
    "device": {"edit": _set("device", value="nav")},
    "crop": {"edit": _set("crop", value="pear")},
    "bundle": {"edit": _set("bundle", value="test_s1009_shake")},
    "seed": {"edit": _set("seed", value=1009)},
    "manifest": {"edit": _set("inputs", "manifest.json", value=ZERO)},
    "trajectory": {"edit": _set("inputs", "trajectory.json", value=ZERO)},
    "render_commit": {"edit": _set("inputs", "render_commit", value=OTHER_COMMIT)},
    "npz": {"damage": "lidar_A.npz"},
    "labels": {"damage": "lidar_B_labels.npz"},
    "config_sha": {"edit": _set("config", "sha256", value="ef" * 32)},
    "folder": {"folder": "ef" * 8},
    "dirty": {"edit": _set("orchard_synth", "dirty", value=True)},
    "commit": {"edit": _set("orchard_synth", "commit", value="3" * 40)},
    "one_commit": {"edit": _set("orchard_synth", "commit", value=OTHER_COMMIT),
                   "sim_commits": [SIM_COMMIT, OTHER_COMMIT]},
}


def _lidar_cases(tmp: Path, held: dict, tests: list[Path], run_ids: list[str]) -> dict:
    """The frozen config of each LiDAR provenance case, by case name: `held` with its depth-unit LiDAR files broken
    as `PROVENANCE_CASES` says (the copies go under `tmp/lidar_cases/<case>/`, pinned by their own sidecar sha256
    where it is there), and `unpinned`: `held` with no digest pinned for the navigation unit."""
    depth = held["lift"]["lidar_files"]["depth"]
    src = Path(depth["dir"])
    out = {}
    for case, spec in PROVENANCE_CASES.items():
        entry = copy.deepcopy(depth)
        if "digest" in spec:
            entry["digests"].pop(run_ids[1])
            if spec["digest"] is not None:
                entry["digests"][run_ids[1]] = spec["digest"]
        else:
            folder = tmp / "lidar_cases" / case / spec.get("folder", src.name)
            for root in tests:
                (folder / root.name).mkdir(parents=True)
                for f in (src / root.name).iterdir():
                    (folder / root.name / f.name).write_bytes(f.read_bytes())
            second = folder / tests[1].name
            if "edit" in spec:
                sidecar = json.loads((second / "lidar.json").read_bytes())
                spec["edit"](sidecar)
                (second / "lidar.json").write_text(json.dumps(sidecar), encoding="utf-8")
            if spec.get("damage") == "lidar.json":
                (second / "lidar.json").unlink()
            elif "damage" in spec:
                damaged = second / spec["damage"]
                damaged.write_bytes(damaged.read_bytes() + b"\0")
            entry["dir"] = str(folder)
            entry["digests"] = {rid: sidecar_sha256(folder / root.name) if (folder / root.name / "lidar.json").is_file()
                                else depth["digests"][rid] for rid, root in zip(run_ids, tests)}
            entry["sim_commits"] = spec.get("sim_commits", depth["sim_commits"])
        conf = copy.deepcopy(held)
        conf["lift"]["lidar_files"]["depth"] = entry
        out[case] = conf
    out["unpinned"] = copy.deepcopy(held)
    out["unpinned"]["lift"]["lidar_files"]["nav"]["digests"] = {}
    return out


def _criteria(judged: str) -> dict:
    spec = {"row_abs_e_pipe": {"le": 0.10}, "miss_rate": {"le": 0.08}}
    return {"judged_config": "c2l", "y6": {judged: {"apple": spec, "pear": spec}}}


def frozen(name: str) -> str:
    return f"configs/eval/frozen/{name}.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _commit_all(repo: Path) -> None:
    g = ["git", "-C", str(repo)]
    subprocess.run([*g, "init", "-q"], check=True)
    subprocess.run([*g, "add", "-A"], check=True)
    subprocess.run([*g, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "commit", "-qm",
                    "c"], check=True)


@pytest.fixture(scope="module")
def depth_held(tmp_path_factory):
    """The stand-in repository `repo`, the held-out bundles `tests` (seeds 1006 to 1008, shaken, with the LiDAR files
    of both units beside them) and the val bundle `val` (seed 200).

    `repo` commits copies of the depth, pose and first splits (and the depth splits again as `SPLITS_B`), of the
    committed pose ladder and of the fixed fusion settings; the small ladder (`_small_ladder`, again as `LADDER_B`)
    and its copy whose rung L3 widens `protect.ladders` (`LADDER_WIDE`); the criteria judging L4_drift_ld (again as
    `CRITERIA_B`) and the criteria judging L4_drift (`CRITERIA_OTHER`); the grids `GRID` (on L4_drift_ld), `GRID_B`
    and `GRID_L3` (on L3); a `.gitignore` that ignores `runs/`; and these configs. Under configs/eval/frozen/:
    `apple_test_depth` (`cfg`): the c2 dev
    pose config on the held-out bundles at 10 deg, one noise seed, `pose.config_label` c2l, both LiDAR file folders
    with their pinned `lidar.json` digests (depth chosen, box depth from the camera) and the protect section that
    pins `LADDER`, `SPLITS`, `CRITERIA`, the output folder, judged rung L4_drift_ld, the `ALLOWED` rung lists (L3 in
    both) and `GRID`; `apple_test_depth_open` (`open_cfg`): the same without the protect section; `..._outdir`: an
    output folder other than `protect.output_dir`; `..._ladder_sha`, `..._splits_sha`, `..._criteria_sha` and
    `..._grid_sha`: one pinned sha256 replaced by zeros; `..._judged`: the protect section pins `CRITERIA_OTHER`;
    `..._widened`: it pins `LADDER_WIDE` and `GRID_L3`; `..._overlap`: L4_drift_ld in two allowed lists; and
    `pose_test` (the shape of the earlier frozen pose configs): no protect section and no LiDAR keys.
    `VAL_CONFIG` is that last shape on the val bundle with split val. The output folder `out` holds a test log
    with one row of another config.

    More frozen configs, each `cfg` with one change: `..._once` writes into its own output folder `once` (with no
    test log at first) and `..._once_v2` there too under run name `..._v2`; `..._one` and `..._two` list the first
    one or two held-out bundles only; `..._variants` pins `LADDER_VARIANTS` (the small ladder with `_variant_rungs`),
    allows each variant rung alone too, and pins `GRID_ZERO`, `GRID_BIAS`, `GRID_LIFT` and `GRID_BARE`; `..._here`
    writes into `HERE`, relative to the repository, and also allows rung L3 alone; `..._judged_<variant>` pins
    `_judged_ladder` of each `JUDGED_VARIANTS`; `..._lidar_<case>` breaks the depth unit's LiDAR files of a
    `PROVENANCE_CASES` case, and `..._lidar_unpinned` pins no navigation unit digest (`_lidar_cases`).
    """
    tmp = tmp_path_factory.mktemp("depth_held")
    repo, out = tmp / "repo", tmp / "eval_depth"
    tests = []
    for seed in TEST_SEEDS:
        root = write_run(tmp / "apple" / f"test_s{seed}_shake", **BUNDLE, seed=seed, shake=True, images=True)
        (root / "orchard_synth.commit").write_text(SIDECAR_COMMIT + "\n", encoding="utf-8")
        tests.append(root)
    run_ids = [SynthRun.open(root).meta.run_id for root in tests]
    lidar_files = {}
    for unit, (mount, sim_sha) in UNITS.items():
        folders = [write_lidar_files(root, unit=unit, mount_xyz=mount, config_sha256=sim_sha, commit=SIM_COMMIT)
                   for root in tests]
        lidar_files[unit] = {"dir": str(folders[0].parent), "unit": unit, "sim_config_sha256": sim_sha,
                             "sim_commits": [SIM_COMMIT],
                             "digests": {rid: sidecar_sha256(f) for rid, f in zip(run_ids, folders)}}
    val = write_run(tmp / "apple" / "val_s200", **BUNDLE, seed=200)

    def put(rel: str, text: str) -> None:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")

    for rel in (SPLITS, SPLITS_POSE, SPLITS_FIRST, POSE_LADDER, FUSE_FIXED):
        put(rel, (ROOT / rel).read_text(encoding="utf-8"))
    put(SPLITS_B, (ROOT / SPLITS).read_text(encoding="utf-8"))
    ladder = _small_ladder()
    for rel in (LADDER, LADDER_B):
        put(rel, yaml.safe_dump(ladder, sort_keys=False))     # rungs in the order written
    widened = copy.deepcopy(ladder)
    widened["rungs"]["L3"] = {"pose": {}, "protect": {"ladders": [*ALLOWED, ["L3", "L4_drift_ld_orc"]]}}
    put(LADDER_WIDE, yaml.safe_dump(widened, sort_keys=False))
    for rel in (CRITERIA, CRITERIA_B):
        put(rel, yaml.safe_dump(_criteria("L4_drift_ld")))
    put(CRITERIA_OTHER, yaml.safe_dump(_criteria("L4_drift")))
    put(GRID, yaml.safe_dump({"bases": ["L4_drift_ld"], "factors": {"noise.lidar_range_sigma_m": [0.03]}}))
    put(GRID_B, yaml.safe_dump({"bases": ["L4_drift_ld"], "factors": {"noise.lidar_dropout": [0.1]}}))
    put(GRID_L3, yaml.safe_dump({"bases": ["L3"], "factors": {"noise.depth_rel_sigma": [0.03]}}))
    put(LADDER_VARIANTS, yaml.safe_dump(_merge(ladder, {"rungs": _variant_rungs()}), sort_keys=False))
    for variant, rung in _judged_rungs().items():
        judged = copy.deepcopy(ladder)
        judged["rungs"]["L4_drift_ld"] = rung
        put(_judged_ladder(variant), yaml.safe_dump(judged, sort_keys=False))
    put(GRID_ZERO, yaml.safe_dump({"bases": ["ld_range0"], "factors": {"noise.depth_rel_sigma": [0.03]}}))
    put(GRID_BIAS, yaml.safe_dump({"bases": ["ld_no_bias"], "factors": {"noise.lidar_range_bias_m": [0.01]}}))
    put(GRID_LIFT, yaml.safe_dump({"bases": ["lift_window"], "factors": {"noise.depth_rel_sigma": [0.03]}}))
    put(GRID_BARE, yaml.safe_dump({"bases": ["B_drift_ld"],
                                   "common": {"noise": {**DRIFT["noise"], **LIDAR_NOISE, **LIDAR_EXTR}},
                                   "factors": {"noise.lidar_range_sigma_m": [0.01, 0.05],
                                               "noise.lidar_ang_sigma_deg": [0.05, 0.3]}}, sort_keys=False))
    put(".gitignore", "runs/\n")
    sha = {rel: _sha256(repo / rel) for rel in (SPLITS, CRITERIA, CRITERIA_OTHER, LADDER, LADDER_WIDE, GRID, GRID_L3,
                                                LADDER_VARIANTS, GRID_ZERO, GRID_BIAS, GRID_LIFT, GRID_BARE,
                                                *map(_judged_ladder, JUDGED_VARIANTS))}

    def pin(rel: str) -> dict:
        return {"path": rel, "sha256": sha[rel]}

    plain = yaml.safe_load((EVAL / "apple_dev_pose_c2.yaml").read_text(encoding="utf-8"))
    plain.update(run_name=RUN_NAME, split="test", tuning=False, runs=[str(root) for root in tests],
                 output_dir=str(out), pose={"rig_ref": 10, "config_label": "c2l"})
    plain["passes"]["pitches"] = [10.0]
    plain["noise"]["seeds"] = [0]
    section = {"ladder": pin(LADDER), "splits": pin(SPLITS), "criteria": pin(CRITERIA), "output_dir": str(out),
               "judged": ["L4_drift_ld"], "ladders": ALLOWED, "grids": {GRID: sha[GRID]}}
    base = _merge(plain, {"lift": {"lidar": {}, "lidar_files": lidar_files, "lidar_file": "depth"}})
    zero = "0" * 64
    variants = {
        "apple_test_depth": {**base, "protect": section},
        "apple_test_depth_open": base,
        "apple_test_depth_outdir": {**base, "protect": section, "output_dir": str(tmp / "eval_depth_b")},
        "apple_test_depth_ladder_sha": {**base, "protect": {**section, "ladder": {"path": LADDER, "sha256": zero}}},
        "apple_test_depth_splits_sha": {**base, "protect": {**section, "splits": {"path": SPLITS, "sha256": zero}}},
        "apple_test_depth_criteria_sha": {**base, "protect": {**section,
                                                              "criteria": {"path": CRITERIA, "sha256": zero}}},
        "apple_test_depth_grid_sha": {**base, "protect": {**section, "grids": {GRID: zero}}},
        "apple_test_depth_judged": {**base, "protect": {**section, "criteria": pin(CRITERIA_OTHER)}},
        "apple_test_depth_widened": {**base, "protect": {**section, "ladder": pin(LADDER_WIDE),
                                                         "grids": {GRID_L3: sha[GRID_L3]}}},
        "apple_test_depth_overlap": {**base, "protect": {**section, "ladders": [ALLOWED[0], ["L3", "L4_drift_ld"]]}},
        "pose_test": plain,
    }
    held = variants["apple_test_depth"]
    once = {**held, "output_dir": str(tmp / "once"), "protect": {**section, "output_dir": str(tmp / "once")}}
    variants.update({
        "apple_test_depth_once": once,
        "apple_test_depth_once_v2": {**once, "run_name": f"{RUN_NAME}_v2"},
        "apple_test_depth_one": {**held, "runs": [str(tests[0])]},
        "apple_test_depth_two": {**held, "runs": [str(root) for root in tests[:2]]},
        "apple_test_depth_variants": {**base, "protect": {
            **section, "ladder": pin(LADDER_VARIANTS), "ladders": [*ALLOWED, *([name] for name in _variant_rungs())],
            "grids": {rel: sha[rel] for rel in (GRID_ZERO, GRID_BIAS, GRID_LIFT, GRID_BARE)}}},
        "apple_test_depth_here": {**held, "output_dir": HERE,
                                  "protect": {**section, "output_dir": HERE, "ladders": [*ALLOWED, ["L3"]]}},
        **{f"apple_test_depth_judged_{v}": {**base, "protect": {**section, "ladder": pin(_judged_ladder(v))}}
           for v in JUDGED_VARIANTS},
        **{f"apple_test_depth_lidar_{case}": conf for case, conf in _lidar_cases(tmp, held, tests, run_ids).items()},
    })
    for name, conf in variants.items():
        put(frozen(name), yaml.safe_dump(conf))
    put(VAL_CONFIG, yaml.safe_dump({**plain, "run_name": "apple_val_depth", "split": "val", "tuning": True,
                                    "runs": [str(val)], "output_dir": str(tmp / "eval_val")}))
    _commit_all(repo)
    log = out / "test_log.csv"
    protect.append_log_row(log, dict(zip(protect.LOG_HEADER, [
        "2026-11-14T00:00:00+00:00", "0" * 12, "ladder", "configs/eval/frozen/other.yaml", "f" * 16, "ladder:x",
        "0" * 64, "a" * 40, "c" * 64, "1/1"])))
    return SimpleNamespace(repo=repo, tmp=tmp, out=out, log=log, experiments=out / RUN_NAME, tests=tests, val=val,
                           run_ids=run_ids, sha=sha,
                           cfg=load_yaml_model(repo / frozen("apple_test_depth"), EvalAppConfig),
                           open_cfg=load_yaml_model(repo / frozen("apple_test_depth_open"), EvalAppConfig))


def _invoke(h, monkeypatch, *args: str):
    """One eval call from inside the stand-in repository (which is `record.counting_repo` for it), with
    `--splits SPLITS` and `--criteria CRITERIA` unless the call gives them."""
    monkeypatch.setattr(record, "counting_repo", lambda: h.repo)
    monkeypatch.chdir(h.repo)
    args = list(args)
    if "--splits" not in args:
        args += ["--splits", SPLITS]
    if "--criteria" not in args:
        args += ["--criteria", CRITERIA]
    return runner.invoke(eval_app, args)


def _check(h, cfg=None, *, files=None):
    """`check_test_request` on the stand-in repository's frozen config `cfg` (default: `h.cfg`) with a clean commit."""
    if files is None:
        files = {"--splits": h.repo / SPLITS, "--criteria": h.repo / CRITERIA}
    return protect.check_test_request(h.cfg if cfg is None else cfg, config_path=h.repo / frozen("apple_test_depth"),
                                      sets=(), runs=(), files=files, grid=None, ladder=None, repo=h.repo, code=CLEAN)


def _listing(h) -> tuple:
    """Whether the experiment folder exists, and the row count of the test log."""
    with h.log.open(newline="", encoding="utf-8") as f:
        return h.experiments.exists(), len(list(csv.DictReader(f)))


def _refused(r, phrase: str) -> None:
    """An eval call refused with exit code 1 and `phrase`, before any bundle was opened."""
    assert r.exit_code == 1 and phrase in r.output and "manifest" not in r.output, (phrase, r.output)


def _late_refused(r, phrase: str, reached: list) -> None:
    """An eval call refused with exit code 1 and `phrase` after the bundles were opened, before scoring
    (`reached` is the `reached` fixture)."""
    assert r.exit_code == 1 and phrase in r.output and reached == [], (phrase, r.output)


@pytest.fixture
def reached(monkeypatch):
    """Scoring replaced: the `allow_test` of each call that reached it, in order; the call then ends with exit
    code 0 and writes nothing."""
    seen = []

    def stub(*args, allow_test=False, **kw):
        seen.append(allow_test)
        raise typer.Exit(0)

    for module, name in ((runner_module, "prepare"), (runner_module, "run_config"), (sweep_module, "run_sweep")):
        monkeypatch.setattr(module, name, stub)
    return seen


@pytest.fixture
def opened(monkeypatch):
    """The folder names of the bundles opened, in order."""
    real, names = SynthRun.open, []

    def spy(path):
        names.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(SynthRun, "open", spy)
    return names


def test_closed_seeds_are_listed_and_refused(depth_held, tmp_path):
    h = depth_held
    assert CLOSED_TEST_SEEDS == LEGACY_TEST_SEEDS == frozenset(range(1000, 1006))
    depth = load_yaml_model(ROOT / SPLITS, SplitsConfig)
    assert depth.test == list(TEST_SEEDS) and not set(depth.test) & CLOSED_TEST_SEEDS
    for cfg in (h.cfg, h.open_cfg):
        for rel, seeds in ((SPLITS_FIRST, [1000, 1001, 1002]), (SPLITS_POSE, [1003, 1004, 1005])):
            with pytest.raises(ProtectionError, match=re.escape(
                    f"split=test: --splits lists held-out seeds that are closed for good: {seeds}")):
                _check(h, cfg, files={"--splits": h.repo / rel, "--criteria": h.repo / CRITERIA})
    with pytest.raises(ProtectionError, match=re.escape("split=test needs --splits") + "$"):
        _check(h, files={"--criteria": h.repo / CRITERIA})
    _check(h)
    bad = tmp_path / "splits.yaml"     # a closed seed listed under val
    bad.write_text(yaml.safe_dump({"dev": [5], "val": [200, 1003], "test": [1006], "exp": [300], "train": [100]}),
                   encoding="utf-8")
    with pytest.raises(ValidationError, match=re.escape("seeds [1003] are closed for good and may only be listed "
                                                        "under test")):
        load_yaml_model(bad, SplitsConfig)


def test_held_out_seeds_are_listed_under_test_only(depth_held, monkeypatch, tmp_path, reached, opened):
    h = depth_held
    plain = {"dev": [5], "val": [200], "test": [], "exp": [300], "train": [100]}
    for split in ("dev", "val", "exp", "train"):
        for seeds in (list(TEST_SEEDS), [1009]):     # the new held-out seeds, and a seed above them
            with pytest.raises(ValidationError, match=re.escape(f"seeds {seeds} are held out")):
                SplitsConfig.model_validate({**plain, split: [*plain[split], *seeds]})
    bad = tmp_path / "splits_val.yaml"     # it would score the held-out bundles as val bundles
    bad.write_text(yaml.safe_dump({**plain, "val": [200, *TEST_SEEDS]}), encoding="utf-8")
    runs = [arg for root in h.tests for arg in ("--run", str(root))]
    r = _invoke(h, monkeypatch, "ladder", "--config", VAL_CONFIG, "--ladder", LADDER, "--rungs", "L3", "--splits",
                str(bad), *runs)
    assert r.exit_code == 2 and "seeds [1006, 1007, 1008] are held out" in r.output, r.output
    assert opened == [] and reached == []


def test_new_held_out_seeds_need_a_protect_section(depth_held, monkeypatch, opened):
    h = depth_held
    before = _listing(h)
    phrase = ("split=test: protect section required for the held-out seeds [1006, 1007, 1008]; write it into the "
              "frozen config")
    for args in (["run", "--config", frozen("apple_test_depth_open")],
                 ["ladder", "--config", frozen("apple_test_depth_open"), "--ladder", POSE_LADDER, "--rungs", "L3"]):
        _refused(_invoke(h, monkeypatch, *args), phrase)
    r = _invoke(h, monkeypatch, "ladder", "--config", frozen("apple_test_depth_open"), "--ladder", POSE_LADDER,
                "--rungs", "L3", "--splits", SPLITS_POSE)    # the closed seeds are refused first
    _refused(r, "split=test: --splits lists held-out seeds that are closed for good: [1003, 1004, 1005]")
    assert opened == [] and _listing(h) == before
    with pytest.raises(ProtectionError, match=re.escape(phrase)):
        _check(h, h.open_cfg)
    unpinned = h.cfg.model_dump()
    unpinned["lift"]["lidar_files"]["nav"]["digests"] = {}
    with pytest.raises(ProtectionError, match=re.escape("split=test: lift.lidar_files.nav.digests is empty")):
        _check(h, EvalAppConfig.model_validate(unpinned))


def test_frozen_request_binds_its_files_and_output(depth_held, monkeypatch, opened):
    h = depth_held
    assert "protect" not in h.open_cfg.model_dump() and "protect" in h.cfg.model_dump()
    assert config_hash(h.cfg) != config_hash(h.open_cfg)
    earlier = load_yaml_model(EVAL / "frozen" / "apple_test_pose.yaml", EvalAppConfig)   # no protect section
    assert "protect" not in earlier.model_dump() and config_hash(earlier) == "a28bf1f9b5f9e100"
    ok = ["ladder", "--rungs", "L3,L4_drift,L4_drift_ld"]
    main = ["--config", frozen("apple_test_depth")]
    zero = "0" * 16
    cases = [
        ([*ok, *main, "--ladder", LADDER_B], f"split=test: --ladder {LADDER_B} is not protect.ladder {LADDER}"),
        ([*ok, "--config", frozen("apple_test_depth_ladder_sha"), "--ladder", LADDER],
         f"split=test: --ladder {LADDER} has sha256 {h.sha[LADDER][:16]}, not the pinned {zero} of protect.ladder"),
        ([*ok, *main, "--ladder", LADDER, "--splits", SPLITS_B],
         f"split=test: --splits {SPLITS_B} is not protect.splits {SPLITS}"),
        ([*ok, "--config", frozen("apple_test_depth_splits_sha"), "--ladder", LADDER],
         f"split=test: --splits {SPLITS} has sha256 {h.sha[SPLITS][:16]}, not the pinned {zero} of "
         "protect.splits"),
        ([*ok, *main, "--ladder", LADDER, "--criteria", CRITERIA_B],
         f"split=test: --criteria {CRITERIA_B} is not protect.criteria {CRITERIA}"),
        ([*ok, "--config", frozen("apple_test_depth_criteria_sha"), "--ladder", LADDER],
         f"split=test: --criteria {CRITERIA} has sha256 {h.sha[CRITERIA][:16]}, not the pinned {zero} of "
         "protect.criteria"),
    ]
    before = _listing(h)
    for args, phrase in cases:
        _refused(_invoke(h, monkeypatch, *args), phrase)
    other = load_yaml_model(h.repo / frozen("apple_test_depth_outdir"), EvalAppConfig)    # not a schema error
    assert other.output_dir != other.protect.output_dir == h.cfg.output_dir
    r = _invoke(h, monkeypatch, *ok, "--config", frozen("apple_test_depth_outdir"), "--ladder", LADDER)
    assert r.exit_code == 1, r.output
    _refused(r, f"split=test: output_dir {other.output_dir} differs from protect.output_dir {h.cfg.output_dir}; a "
                "held-out run writes only into the frozen output folder")
    assert re.search(re.escape("differs from protect.output_dir"), r.output)
    assert opened == [] and _listing(h) == before and not (h.tmp / "eval_depth_b").exists()


def test_frozen_request_binds_rungs_and_grids(depth_held, monkeypatch, reached, opened):
    h = depth_held
    main, ladder = ["--config", frozen("apple_test_depth")], ["--ladder", LADDER]
    ok = ["ladder", *main, *ladder, "--rungs", "L3,L4_drift,L4_drift_ld"]
    widened = ["--config", frozen("apple_test_depth_widened"), "--ladder", LADDER_WIDE]
    lists = "is not one of the protect.ladders lists"
    cases = [
        (["ladder", *main, *ladder, "--rungs", "L3,L4_drift_ld"], f"split=test: --rungs L3,L4_drift_ld {lists}"),
        (["ladder", *main, *ladder],     # without --rungs: every rung and baseline of the ladder
         f"split=test: --rungs L3,L4_drift,L4_drift_ld,L4_drift_ldn,L4_drift_ld_orc,L4_drift_ld_val {lists}"),
        ([*ok, "--with-2x2"], "split=test: a frozen config with a protect section takes no --with-2x2"),
        ([*ok, "--subsets"], "split=test: a frozen config with a protect section takes no --subsets"),
        (["run", *main], "split=test: a frozen config with a protect section serves eval ladder and eval sweep only, "
                         "not eval run"),
        (["sweep", *main, *ladder, "--grid", GRID_B], f"split=test: --grid {GRID_B} is not one of protect.grids"),
        (["sweep", "--config", frozen("apple_test_depth_grid_sha"), *ladder, "--grid", GRID],
         f"split=test: --grid {GRID} has sha256 {h.sha[GRID][:16]}, not the pinned {'0' * 16} of protect.grids"),
        (["ladder", "--config", frozen("apple_test_depth_judged"), *ladder, "--rungs", "L3,L4_drift,L4_drift_ld",
          "--criteria", CRITERIA_OTHER],
         "split=test: the judged rungs of --criteria ['L4_drift'] differ from protect.judged ['L4_drift_ld']"),
        (["ladder", *main, "--ladder", POSE_LADDER, "--rungs", "L3,L4_drift,L4_drift_ld"],
         f"split=test: --ladder {POSE_LADDER} is not protect.ladder {LADDER}"),
        (["ladder", *widened, "--rungs", "L3,L4_drift,L4_drift_ld"],
         "split=test: L3 changes the protect section; held-out rungs keep the frozen config's"),
        (["sweep", *widened, "--grid", GRID_L3], "split=test: baseline @ L3 changes the protect section"),
    ]
    before = _listing(h)
    for args, phrase in cases:
        _refused(_invoke(h, monkeypatch, *args), phrase)
    assert reached == [] and opened == [] and _listing(h) == before
    for args in (ok, ["ladder", *main, *ladder, "--rungs", "L3,L4_drift_ldn,L4_drift_ld_orc"],
                 ["sweep", *main, *ladder, "--grid", GRID]):
        r = _invoke(h, monkeypatch, *args)
        assert r.exit_code == 0 and POSE_LADDER not in r.output, (args, r.output)   # no committed pose ladder rule
    assert reached == [True, True, True] and _listing(h) == before


def test_judged_rung_sits_in_one_allowed_list(depth_held, monkeypatch, reached, opened):
    h = depth_held
    path = h.repo / frozen("apple_test_depth_overlap")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert ProtectConfig.model_validate(raw["protect"]).ladders == [ALLOWED[0], ["L3", "L4_drift_ld"]]
    cfg = load_yaml_model(path, EvalAppConfig)            # the schema does not look at the lists' overlap
    phrase = ("split=test: protect.judged rung L4_drift_ld is in 2 allowed rung lists; a judged rung may be in one "
              "only")
    with pytest.raises(ProtectionError, match=re.escape(phrase)):
        protect.check_test_frozen_request(
            cfg, kind="ladder",
            files={"--ladder": Path(LADDER), "--splits": Path(SPLITS), "--criteria": Path(CRITERIA)},
            ladder_sha256=h.sha[LADDER], grid_sha256=None, criteria_sha256=h.sha[CRITERIA],
            splits_sha256=h.sha[SPLITS], rungs=["L3", "L4_drift_ld"], with_2x2=False, subsets=False,
            criteria=_criteria("L4_drift_ld"))
    before = _listing(h)
    r = _invoke(h, monkeypatch, "ladder", "--config", frozen("apple_test_depth_overlap"), "--ladder", LADDER,
                "--rungs", "L3,L4_drift_ld")
    assert r.exit_code == 1, r.output                     # a protection refusal, not an input error
    _refused(r, phrase)
    assert opened == [] and reached == [] and _listing(h) == before
    assert all("L3" in rungs for rungs in h.cfg.protect.ladders) and len(h.cfg.protect.ladders) == 2
    r = _invoke(h, monkeypatch, "ladder", "--config", frozen("apple_test_depth"), "--ladder", LADDER,
                "--rungs", "L3,L4_drift_ldn,L4_drift_ld_orc")    # a report rung may sit in two lists
    assert r.exit_code == 0 and reached == [True], r.output


def test_val_only_axes_follow_the_first_judged_rung(depth_held):
    h = depth_held
    # A ladder whose two drift rungs keep different values of a val-only axis.
    fake = {"rungs": {"L4_drift": {"pose": {"rig": True}, "noise": {"drift_time_offset_s": 0.01}},
                      "L4_drift_ld": {"pose": {"rig": True}, "noise": {"drift_time_offset_s": 0.02}}}}

    def cell(cfg, base):
        return {"x": (base, rung_config(cfg, fake, base))}

    protect.check_test_pose_rungs(h.cfg, cell(h.cfg, "L4_drift_ld"), fake)
    with pytest.raises(ProtectionError, match=re.escape("split=test: x sets noise.drift_time_offset_s=0.01; that axis "
                                                        "runs on val only (the judged rung keeps 0.02)")):
        protect.check_test_pose_rungs(h.cfg, cell(h.cfg, "L4_drift"), fake)
    protect.check_test_pose_rungs(h.open_cfg, cell(h.open_cfg, "L4_drift"), fake)    # without the section: L4_drift
    with pytest.raises(ProtectionError, match=re.escape("(the judged rung keeps 0.01)")):
        protect.check_test_pose_rungs(h.open_cfg, cell(h.open_cfg, "L4_drift_ld"), fake)
    without_judged = {"rungs": {"L4_drift": fake["rungs"]["L4_drift"]}}
    with pytest.raises(ProtectionError, match=re.escape("split=test: the pose ladder needs rung L4_drift_ld")):
        protect.check_test_pose_rungs(h.cfg, cell(h.cfg, "L4_drift"), without_judged)


def test_refusals_come_before_the_bundles(depth_held, monkeypatch, reached, opened):
    h = depth_held
    cases = [
        (["ladder", "--config", frozen("apple_test_depth_open"), "--ladder", POSE_LADDER, "--rungs", "L3",
          "--splits", SPLITS_POSE], "closed for good"),
        (["ladder", "--config", frozen("apple_test_depth_open"), "--ladder", POSE_LADDER, "--rungs", "L3"],
         "protect section required"),
        (["ladder", "--config", frozen("apple_test_depth"), "--ladder", LADDER, "--rungs", "L3,L4_drift"],
         "is not one of the protect.ladders lists"),
        (["sweep", "--config", frozen("apple_test_depth"), "--ladder", LADDER, "--grid", GRID_B],
         "is not one of protect.grids"),
    ]
    before = _listing(h)
    for args, phrase in cases:
        _refused(_invoke(h, monkeypatch, *args), phrase)
    assert opened == [] and reached == [] and _listing(h) == before
    r = _invoke(h, monkeypatch, "ladder", "--config", frozen("apple_test_depth"), "--ladder", LADDER,
                "--rungs", "L3,L4_drift,L4_drift_ld")
    assert r.exit_code == 0 and reached == [True], r.output
    assert opened == [root.name for root in h.tests] and _listing(h) == before


def test_depth_protection_list(depth_held, monkeypatch, reached, opened):
    h = depth_held
    main, ladder = ["--config", frozen("apple_test_depth")], ["--ladder", LADDER]
    allowed = ["ladder", *main, *ladder, "--rungs", "L3,L4_drift,L4_drift_ld"]
    earlier = ["--config", frozen("pose_test")]     # an earlier frozen pose config: no protect section or LiDAR keys
    cases = [  # (request, the rule that refuses it)
        (["ladder", *main, *ladder, "--rungs", "L3,L4_drift_ld_val"], "is not one of the protect.ladders lists"),
        ([*allowed, "--set", "seed=1"], "split=test takes no --set or --run"),
        (["ladder", *main, "--ladder", POSE_LADDER, "--rungs", "L3,L4_drift,L4_drift_ld"], "is not protect.ladder"),
        ([*allowed, "--splits", SPLITS_POSE], "closed for good"),
        (["ladder", *earlier, *ladder, "--rungs", "L3"], POSE_LADDER_RULE),
        (["sweep", *earlier, *ladder, "--grid", GRID, "--splits", SPLITS_POSE], POSE_LADDER_RULE),
        (["sweep", *earlier, *ladder, "--grid", GRID], POSE_LADDER_RULE),
        (["ladder", *earlier, "--ladder", POSE_LADDER, "--rungs", "L3"], "protect section required"),
        (["ladder", *main, *ladder, "--rungs", "L4_drift_ld,L4_drift,L3"], "is not one of the protect.ladders lists"),
    ]
    before = _listing(h)
    for args, phrase in cases:
        _refused(_invoke(h, monkeypatch, *args), phrase)
    assert opened == [] and reached == [] and _listing(h) == before
    r = _invoke(h, monkeypatch, *allowed)
    assert r.exit_code == 0 and reached == [True], r.output


def test_request_extra_is_unchanged_by_the_move(depth_held, monkeypatch):
    h = depth_held
    extras = []

    def wrapped(*args, **kw):
        extras.append(real(*args, **kw))
        return extras[-1]

    real = record.request_extra
    monkeypatch.setattr(record, "request_extra", wrapped)
    r = _invoke(h, monkeypatch, "ladder", "--config", VAL_CONFIG, "--ladder", LADDER, "--rungs", "L3")
    assert r.exit_code == 0, r.output
    folder = Path(r.output.splitlines()[0])
    res = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    rec = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    extra = real("ladder", ladder_sha256=h.sha[LADDER], rungs=["L3"], with_2x2=False, subsets=False, grid_name=None,
                 grid_sha256=None, criteria_sha256=h.sha[CRITERIA])
    fingerprints = [d["fingerprint"] for d in rec["data"]] + [d["depth_fingerprint"] for d in rec["data"]
                                                              if "depth_fingerprint" in d]
    assert len(fingerprints) == 2                # the bundle and its depth images (box depth from the camera)
    assert record.experiment_id(config_hash=rec["config_hash"], fingerprints=fingerprints, code=rec["code"],
                                seed=rec["seeds"]["seed"], extra=extra) == res["experiment_id"] == folder.name
    assert extras == [extra]
    assert rec["code"]["commit"] is not None and rec["code"]["dirty"] is False


def test_same_request_twice_is_refused(depth_held, monkeypatch, reached, opened):
    h = depth_held
    once = load_yaml_model(h.repo / frozen("apple_test_depth_once"), EvalAppConfig)
    v2 = load_yaml_model(h.repo / frozen("apple_test_depth_once_v2"), EvalAppConfig)
    assert v2.run_name == f"{RUN_NAME}_v2" and config_hash(v2) != config_hash(once)
    assert not set(ALLOWED[1]) & set(once.protect.judged)      # the other allowed list holds no judged rung
    log = Path(once.output_dir) / "test_log.csv"
    date, exp_id, verdict = "2026-11-14T01:02:03+00:00", "e" * 12, "pass 2/3"
    extra = record.request_extra("ladder", ladder_sha256=h.sha[LADDER], rungs=ALLOWED[0], with_2x2=False,
                                 subsets=False, grid_name=None, grid_sha256=None, criteria_sha256=h.sha[CRITERIA])

    def logged(kind: str) -> None:
        """The test log holds one row: the request of `ALLOWED[0]` on `once`, as `kind`."""
        log.unlink(missing_ok=True)
        protect.append_log_row(log, {"date": date, "experiment_id": exp_id, "kind": kind,
                                     "config": frozen("apple_test_depth_once"), "config_hash": config_hash(once),
                                     "extra": extra, "fingerprints": "f" * 64, "commit": "a" * 40,
                                     "criteria_sha256": h.sha[CRITERIA], "verdict": verdict})

    def rows() -> int:
        with log.open(newline="", encoding="utf-8") as f:
            return len(list(csv.DictReader(f)))

    def ask(name: str, rungs: list[str]):
        return _invoke(h, monkeypatch, "ladder", "--config", frozen(name), "--ladder", LADDER, "--rungs",
                       ",".join(rungs))

    logged("ladder")
    r = ask("apple_test_depth_once", ALLOWED[0])
    _refused(r, f"split=test: this request already ran on {date} as {exp_id}; a second run needs a new frozen config")
    assert "verdict" not in r.output and verdict not in r.output
    assert opened == [] and reached == [] and rows() == 1
    for kind, name, rungs in (("ladder", "apple_test_depth_once", ALLOWED[1]),      # another allowed rung list
                              ("sweep", "apple_test_depth_once", ALLOWED[0]),       # a row of another kind
                              ("ladder", "apple_test_depth_once_v2", ALLOWED[0])):  # another config hash
        logged(kind)
        r = ask(name, rungs)
        assert r.exit_code == 0, (kind, name, r.output)
    assert reached == [True, True, True] and rows() == 1


def test_held_out_output_folder_is_the_repository_one(depth_held, monkeypatch, tmp_path):
    h = depth_held
    here = load_yaml_model(h.repo / frozen("apple_test_depth_here"), EvalAppConfig)
    assert here.output_dir == here.protect.output_dir == HERE
    out = h.repo / HERE
    request = ["ladder", "--rungs", "L3"]
    paths = {"--config": frozen("apple_test_depth_here"), "--ladder": LADDER, "--splits": SPLITS,
             "--criteria": CRITERIA}
    monkeypatch.setattr(record, "counting_repo", lambda: h.repo)
    monkeypatch.setattr(fusion, "FUSE_FIXED_PATH", h.repo / FUSE_FIXED)     # by its absolute path too
    monkeypatch.chdir(tmp_path)     # a first run from another directory, with absolute paths
    r = runner.invoke(eval_app, [*request, *(arg for option, rel in paths.items()
                                             for arg in (option, str(h.repo / rel)))])
    assert r.exit_code == 0, r.output
    folder = Path(r.output.splitlines()[0])
    assert folder.parent == out / RUN_NAME and (folder / "results.json").is_file()
    assert not (tmp_path / "runs").exists()
    with (out / "test_log.csv").open(newline="", encoding="utf-8") as f:
        assert [row["experiment_id"] for row in csv.DictReader(f)] == [folder.name]
    r = _invoke(h, monkeypatch, *request, "--config", frozen("apple_test_depth_here"), "--ladder", LADDER)
    _refused(r, f"as {folder.name}; a second run needs a new frozen config")     # the same request from the repository
    with (out / "test_log.csv").open(newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 1
    monkeypatch.chdir(tmp_path)
    assert protect.held_out_output_root(HERE, h.repo) == out
    assert protect.held_out_output_root(str(h.out), h.repo) == h.out
    assert protect.held_out_output_root("~/eval", h.repo) == Path("~/eval").expanduser()


def test_part_of_the_held_out_seeds_is_refused(depth_held, monkeypatch, reached, opened):
    h = depth_held
    request = ["ladder", "--ladder", LADDER, "--rungs", ",".join(ALLOWED[0])]
    for name, n in (("apple_test_depth_one", 1), ("apple_test_depth_two", 2)):
        opened.clear()
        r = _invoke(h, monkeypatch, *request, "--config", frozen(name))
        _late_refused(r, f"split=test: the bundles hold the seeds {list(TEST_SEEDS[:n])}, not the test seeds "
                         f"{list(TEST_SEEDS)} of --splits", reached)
        assert opened == [root.name for root in h.tests[:n]]
    r = _invoke(h, monkeypatch, *request, "--config", frozen("apple_test_depth"))
    assert r.exit_code == 0 and reached == [True], r.output


def test_lift_is_fixed_but_the_depth_choice(depth_held, monkeypatch, reached, opened):
    h = depth_held
    request = ["ladder", "--config", frozen("apple_test_depth_variants"), "--ladder", LADDER_VARIANTS, "--rungs"]
    for rung, key in (("lift_far_min", "far_min_frac"), ("lift_box_frac", "box_frac"),
                      ("lift_offset", "center_offset_m"), ("lift_window", "lidar"), ("lift_files", "lidar_files")):
        _refused(_invoke(h, monkeypatch, *request, rung),
                 f"split=test: {rung}: lift.{key} differs from the frozen config")
    _refused(_invoke(h, monkeypatch, "sweep", *request[1:-1], "--grid", GRID_LIFT),
             "split=test: baseline @ lift_window: lift.lidar differs from the frozen config")
    assert opened == [] and reached == []
    passing = [["lift_det_depth"], *ALLOWED]
    for rungs in passing:
        r = _invoke(h, monkeypatch, *request, ",".join(rungs))
        assert r.exit_code == 0, (rungs, r.output)
    assert reached == [True] * len(passing)
    ladder = yaml.safe_load((h.repo / LADDER_VARIANTS).read_text(encoding="utf-8"))
    configs = [rung_config(h.cfg, ladder, name) for rungs in passing for name in rungs]
    assert {c.lift.depth for c in configs} == {"box_far", "lidar", "lidar_oracle", "det_depth"}
    assert {c.lift.lidar_file for c in configs} == {"depth", "nav"}


def test_held_out_lidar_rungs_need_noise_and_judged_choices(depth_held, monkeypatch, reached, opened):
    h = depth_held
    variants = ["--config", frozen("apple_test_depth_variants"), "--ladder", LADDER_VARIANTS]
    noise = ("split=test: {} takes lift.depth={} with noise.{} 0.0; held-out LiDAR rungs need LiDAR range and angle "
             "noise above 0")
    cases = [
        (["ladder", *variants, "--rungs", "ld_range0"], noise.format("ld_range0", "lidar", "lidar_range_sigma_m")),
        (["ladder", *variants, "--rungs", "ldn_ang0"], noise.format("ldn_ang0", "lidar", "lidar_ang_sigma_deg")),
        (["ladder", *variants, "--rungs", "orc_ang0"],
         noise.format("orc_ang0", "lidar_oracle", "lidar_ang_sigma_deg")),
        (["sweep", *variants, "--grid", GRID_ZERO],     # its point cells keep the base's range sigma 0
         noise.format("noise.depth_rel_sigma=0.03 @ ld_range0", "lidar", "lidar_range_sigma_m")),
    ]
    for variant, need in (("oracle", "lift.depth=lidar, got lidar_oracle"),
                          ("nav", "a depth-unit LiDAR file (lift.lidar_file), got nav"),
                          ("box", "lift.depth=lidar, got box_far"),
                          ("extr", "noise.lidar_extr_rot_deg above 0, got 0.0")):
        cases.append((["ladder", "--config", frozen(f"apple_test_depth_judged_{variant}"), "--ladder",
                       _judged_ladder(variant), "--rungs", ",".join(ALLOWED[0])],
                      f"split=test: judged rung L4_drift_ld needs {need}"))
    for args, phrase in cases:
        _refused(_invoke(h, monkeypatch, *args), phrase)     # exit code 1: a held-out refusal, not a config error
    assert opened == [] and reached == []
    r = _invoke(h, monkeypatch, "ladder", *variants, "--rungs", ",".join(ALLOWED[1]))   # with the pairing report rung
    assert r.exit_code == 0 and reached == [True], r.output


def test_held_out_grid_on_a_noise_free_lidar_base(depth_held, monkeypatch, reached, opened):
    h = depth_held
    variants = ["--config", frozen("apple_test_depth_variants"), "--ladder", LADDER_VARIANTS]
    cfg = load_yaml_model(h.repo / frozen("apple_test_depth_variants"), EvalAppConfig)
    cells = sweep_module.sweep_cells(cfg, sweep_module.load_grid(h.repo / GRID_BARE),
                                     runner_module.load_ladder(h.repo / LADDER_VARIANTS))
    assert [(f"{c.point.name} @ {c.base}", *(c.cfg.noise.factor(k) for k in protect.LIDAR_SIGMA_KEYS))
            for c in cells] == [("baseline @ B_drift_ld", 0.0, 0.0),     # the base as the ladder writes it
                                ("noise.lidar_range_sigma_m=0.01 @ B_drift_ld", 0.01, 0.15),
                                ("noise.lidar_range_sigma_m=0.05 @ B_drift_ld", 0.05, 0.15),
                                ("noise.lidar_ang_sigma_deg=0.05 @ B_drift_ld", 0.02, 0.05),
                                ("noise.lidar_ang_sigma_deg=0.3 @ B_drift_ld", 0.02, 0.3)]
    assert all(c.overridden == () for c in cells)     # each point value reaches its cell
    _refused(_invoke(h, monkeypatch, "ladder", *variants, "--rungs", "B_drift_ld"),     # as a rung of its own
             "split=test: B_drift_ld takes lift.depth=lidar with noise.lidar_range_sigma_m 0.0")
    assert opened == [] and reached == []
    r = _invoke(h, monkeypatch, "sweep", *variants, "--grid", GRID_BARE)
    assert r.exit_code == 0 and reached == [True], r.output


def test_lidar_provenance_accepts_the_pinned_files(depth_held):
    h = depth_held
    bundles = [SynthRun.open(root) for root in h.tests]
    rungs = {name: rung_config(h.cfg, _small_ladder(), name) for name in dict.fromkeys(ALLOWED[0] + ALLOWED[1])}
    out = protect.check_test_lidar_files(bundles, h.cfg, rungs)
    pins = h.cfg.lift.lidar_files
    assert list(pins) == ["depth", "nav"] and [b.meta.seed for b in bundles] == list(TEST_SEEDS)
    assert out == [{"name": name, "run_id": rid, "digest": pins[name].digests[rid], "sim_commit": SIM_COMMIT}
                   for name in ("depth", "nav") for rid in h.run_ids]
    assert all(list(row) == ["name", "run_id", "digest", "sim_commit"] for row in out)


def test_lidar_provenance_refusals(depth_held, monkeypatch, reached):
    h = depth_held
    second = h.tests[1].name
    said = {   # what the refusal of each case says
        "pin": "no pinned lidar.json digest",
        "sidecar": f"{second}/lidar.json: no LiDAR sidecar",
        "sidecar_sha": f"{second}/lidar.json: lidar.json sha256 does not match the pinned digest",
        "format": "format orchard-lidar-sidecar/0, expected orchard-lidar-sidecar/1",
        "variant": "variant tilt, expected judged",
        "device": "device nav, expected depth",
        "crop": "crop pear, expected apple",
        "bundle": f"bundle test_s1009_shake, expected {second}",
        "seed": "seed 1009, expected 1007",
        "manifest": "manifest.json sha256 differs from the bundle",
        "trajectory": "trajectory.json sha256 differs from the bundle",
        "render_commit": f"render_commit {OTHER_COMMIT[:12]}, expected the bundle's orchard_synth commit "
                         f"{SIDECAR_COMMIT[:12]}",
        "npz": f"{second}/lidar_A.npz: sha256 does not match lidar.json files",
        "labels": f"{second}/lidar_B_labels.npz: sha256 does not match lidar.json files",
        "config_sha": f"config sha256 {'ef' * 8} differs from the expected {'ab' * 8}",
        "folder": f"folder {'ef' * 8} is not named after config sha256 {'ab' * 8}",
        "dirty": "orchard_synth dirty is true",
        "commit": f"simulator commit {'3' * 12} is not one of the expected commits",
    }
    assert set(said) == set(PROVENANCE_CASES) - {"one_commit"}
    request = ["ladder", "--ladder", LADDER, "--rungs", ",".join(ALLOWED[0])]
    for case, phrase in said.items():
        run_id = h.run_ids[0 if case == "folder" else 1]     # the folder name is wrong for every bundle
        r = _invoke(h, monkeypatch, *request, "--config", frozen(f"apple_test_depth_lidar_{case}"))
        _late_refused(r, f"split=test: lift.lidar_files.depth for run {run_id}: ", reached)
        assert phrase in r.output and r.output.count("split=test") == 1, (case, r.output)   # led once, not wrapped
    _refused(_invoke(h, monkeypatch, *request, "--config", frozen("apple_test_depth_lidar_unpinned")),
             "split=test: lift.lidar_files.nav.digests is empty")
    bundles = [SynthRun.open(root) for root in h.tests]
    for case, cause, message in (("sidecar", FileNotFoundError, "no LiDAR sidecar"),
                                 ("sidecar_sha", ValueError, "lidar.json sha256 does not match the pinned digest")):
        cfg = load_yaml_model(h.repo / frozen(f"apple_test_depth_lidar_{case}"), EvalAppConfig)
        path = Path(cfg.lift.lidar_files["depth"].dir) / second / "lidar.json"
        with pytest.raises(ProtectionError) as e:
            protect.check_test_lidar_files(bundles, cfg, {})
        assert type(e.value.__cause__) is cause, case
        assert str(e.value).startswith(f"split=test: lift.lidar_files.depth for run {h.run_ids[1]}: {path}: {message}")


def test_one_simulation_commit_across_the_held_out_bundles(depth_held, monkeypatch, reached):
    h = depth_held
    name = "apple_test_depth_lidar_one_commit"
    assert load_yaml_model(h.repo / frozen(name), EvalAppConfig).lift.lidar_files["depth"].sim_commits == [
        SIM_COMMIT, OTHER_COMMIT]                                  # each commit is an expected one
    r = _invoke(h, monkeypatch, "ladder", "--config", frozen(name), "--ladder", LADDER, "--rungs", ",".join(ALLOWED[0]))
    _late_refused(r, f"split=test needs one simulator commit across the LiDAR files, got {[SIM_COMMIT, OTHER_COMMIT]}",
                  reached)


def test_judged_lidar_noise_matches_the_contract(depth_held, monkeypatch, reached):
    h = depth_held
    variants = ["--config", frozen("apple_test_depth_variants"), "--ladder", LADDER_VARIANTS]
    off = "split=test: {}: noise.{} {} differs from the LiDAR contract value {}"
    for rung, key, value, contract in (("ld_range03", "lidar_range_sigma_m", 0.03, 0.02),
                                       ("ld_no_dropout", "lidar_dropout", 0.0, 0.05)):    # unset: 0.0
        _late_refused(_invoke(h, monkeypatch, "ladder", *variants, "--rungs", rung),
                      off.format(rung, key, value, contract), reached)
    passing = [["ladder", *variants, "--rungs", "ld_no_bias"],           # unset: 0.0, the contract value
               ["ladder", *variants, "--rungs", ",".join(ALLOWED[0])],   # every contract value
               ["sweep", *variants, "--grid", GRID_BIAS]]                 # grid cells are not compared
    for args in passing:
        r = _invoke(h, monkeypatch, *args)
        assert r.exit_code == 0, (args, r.output)
    assert reached == [True] * len(passing)
    cfg = load_yaml_model(h.repo / frozen("apple_test_depth_variants"), EvalAppConfig)
    cells = sweep_module.sweep_cells(cfg, sweep_module.load_grid(h.repo / GRID_BIAS),
                                     runner_module.load_ladder(h.repo / LADDER_VARIANTS))
    assert [c.cfg.noise.factor("lidar_range_bias_m") for c in cells] == [0.0, 0.01]   # a cell off the contract
