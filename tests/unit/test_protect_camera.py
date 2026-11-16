"""Held-out requests whose pinned criteria judge a camera configuration: there a judged rung on a camera box depth
passes the LiDAR rule of the judged rungs, whether the frozen config holds the LiDAR file keys or not, while under
criteria that judge a LiDAR configuration or name no judged configuration it is refused as before. A judged rung on
any other depth (the truth depth, or a LiDAR box depth without the extrinsic error, on the truth-label pairing or on
the navigation unit's file) keeps the LiDAR rule whatever the criteria, the noise rule of the LiDAR rungs holds as
before, and a check called without the criteria's reading takes the LiDAR rule.

The requests run on a stand-in counting repository with three fake held-out bundles (seeds 1006 to 1008, shaken,
with the LiDAR files of a depth unit and of a navigation unit beside them). Scoring is replaced by a stub that ends
the command with exit code 0, so a request that reaches it passed every check before scoring."""

import copy
import re
from types import SimpleNamespace

import pytest
import yaml

import counting.evaluation.protect as protect
from counting.config.eval_schema import EvalAppConfig
from counting.config.loader import load_yaml_model
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.runner import ProtectionError, _merge, rung_config
from tests.unit.lidarfake import sidecar_sha256, write_lidar_files
from tests.unit.synthfake import ROOT, SIDECAR_COMMIT, write_run
from tests.unit.test_protect_depth import (
    BUNDLE, DRIFT, FUSE_FIXED, POSE_LADDER, SIM_COMMIT, SPLITS, TEST_SEEDS, UNITS, _commit_all, _invoke,
    _judged_rungs, _refused, _sha256, _small_ladder, frozen, opened, reached,
)

pytestmark = pytest.mark.pose

EVAL = ROOT / "configs" / "eval"

# Files of the stand-in repository, by their repository paths.
CAMERA = "configs/eval/sweeps_depth/ladder_camera.yaml"             # the small ladder
CAMERA_DET = "configs/eval/sweeps_depth/ladder_camera_det.yaml"     # its L4_drift on the truth depth
CAMERA_EXTR = "configs/eval/sweeps_depth/ladder_camera_extr.yaml"   # its L4_drift_ld with no LiDAR extrinsic rotation
CAMERA_ORC = "configs/eval/sweeps_depth/ladder_camera_orc.yaml"     # its L4_drift_ld on the truth-label pairing
CAMERA_NAV = "configs/eval/sweeps_depth/ladder_camera_nav.yaml"     # its L4_drift_ld on the navigation unit's file
CRITERIA_CAMERA = "configs/eval/criteria_camera.yaml"         # judges L4_drift of camera configuration c2
CRITERIA_LIDAR = "configs/eval/criteria_lidar.yaml"           # judges L4_drift of LiDAR configuration c2l
CRITERIA_UNNAMED = "configs/eval/criteria_unnamed.yaml"       # judges L4_drift, no judged configuration
CRITERIA_CAMERA_LD = "configs/eval/criteria_camera_ld.yaml"   # judges L4_drift_ld of camera configuration c2

SPEC = {"apple": {"row_abs_e_pipe": {"le": 0.10}}, "pear": {"row_abs_e_pipe": {"le": 0.10}}}
CRITERIA = {CRITERIA_CAMERA: {"judged_config": "c2", "y6": {"L4_drift": SPEC}},
            CRITERIA_LIDAR: {"judged_config": "c2l", "y6": {"L4_drift": SPEC}},
            CRITERIA_UNNAMED: {"y6": {"L4_drift": SPEC}},
            CRITERIA_CAMERA_LD: {"judged_config": "c2", "y6": {"L4_drift_ld": SPEC}}}
# The frozen configs under configs/eval/frozen/: (ladder, criteria, judged rung, pose.config_label) of each.
HELD = {"camera_c2": (CAMERA, CRITERIA_CAMERA, "L4_drift", "c2"),
        "camera_c2_plain": (CAMERA, CRITERIA_CAMERA, "L4_drift", "c2"),
        "camera_c2l": (CAMERA, CRITERIA_LIDAR, "L4_drift", "c2l"),
        "camera_unnamed": (CAMERA, CRITERIA_UNNAMED, "L4_drift", "c2"),
        "camera_det_c2": (CAMERA_DET, CRITERIA_CAMERA, "L4_drift", "c2"),
        "ld_c2": (CAMERA, CRITERIA_CAMERA_LD, "L4_drift_ld", "c2"),
        "ld_extr_c2": (CAMERA_EXTR, CRITERIA_CAMERA_LD, "L4_drift_ld", "c2"),
        "ld_orc_c2": (CAMERA_ORC, CRITERIA_CAMERA_LD, "L4_drift_ld", "c2"),
        "ld_nav_c2": (CAMERA_NAV, CRITERIA_CAMERA_LD, "L4_drift_ld", "c2")}
BOX_FAR = "split=test: judged rung L4_drift needs lift.depth=lidar, got box_far"


@pytest.fixture(autouse=True)
def reopen_closed_seeds(monkeypatch):
    """Reopen the closed held-out seeds for these tests: the rules they check hold for every held-out seed of the
    fake bundles. The closed list itself is checked in test_protect_depth.py."""
    monkeypatch.setattr(protect, "CLOSED_TEST_SEEDS", frozenset())


@pytest.fixture(scope="module")
def camera_held(tmp_path_factory):
    """The stand-in repository `repo` and the held-out bundles `tests` (seeds 1006 to 1008, shaken, with the LiDAR
    files of both units beside them).

    `repo` commits copies of the depth splits, the committed pose ladder and the fixed fusion settings; the ladders
    `CAMERA` (`_small_ladder`), `CAMERA_DET` (its rung L4_drift on det_depth) and `CAMERA_EXTR`, `CAMERA_ORC` and
    `CAMERA_NAV` (its rung L4_drift_ld replaced by the `extr`, `oracle` and `nav` rungs of `_judged_rungs`); the four
    `CRITERIA` tables; and the `HELD` frozen configs. Each is the c2 dev pose config on the held-out bundles at
    10 deg with one noise seed, its pose section replaced by the reference camera at 10 deg and the configuration
    name of `HELD`, both LiDAR file folders with their pinned `lidar.json` digests (depth unit chosen, box depth
    from the camera) and a protect section that pins its `HELD` ladder, the depth splits and its `HELD` criteria,
    the output folder, its judged rung and the one rung list (L3, the judged rung). `camera_c2_plain` keeps the dev
    config's lift section instead, without the LiDAR keys.
    """
    tmp = tmp_path_factory.mktemp("camera_held")
    repo, out = tmp / "repo", tmp / "eval_camera"
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

    def put(rel: str, text: str) -> None:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")

    for rel in (SPLITS, POSE_LADDER, FUSE_FIXED):
        put(rel, (ROOT / rel).read_text(encoding="utf-8"))
    ladder = _small_ladder()
    ladders = {CAMERA: ladder}
    for rel, rung, patch in ((CAMERA_DET, "L4_drift", _merge(DRIFT, {"lift": {"depth": "det_depth"}})),
                             (CAMERA_EXTR, "L4_drift_ld", _judged_rungs()["extr"]),
                             (CAMERA_ORC, "L4_drift_ld", _judged_rungs()["oracle"]),
                             (CAMERA_NAV, "L4_drift_ld", _judged_rungs()["nav"])):
        ladders[rel] = copy.deepcopy(ladder)
        ladders[rel]["rungs"][rung] = patch
    for rel, text in ladders.items():
        put(rel, yaml.safe_dump(text, sort_keys=False))     # rungs in the order written
    for rel, table in CRITERIA.items():
        put(rel, yaml.safe_dump(table))

    def pin(rel: str) -> dict:
        return {"path": rel, "sha256": _sha256(repo / rel)}

    plain = yaml.safe_load((EVAL / "apple_dev_pose_c2.yaml").read_text(encoding="utf-8"))
    plain.update(run_name="apple_test_camera", split="test", tuning=False, runs=[str(root) for root in tests],
                 output_dir=str(out))
    plain["passes"]["pitches"] = [10.0]
    plain["noise"]["seeds"] = [0]
    lidar = _merge(plain, {"lift": {"lidar": {}, "lidar_files": lidar_files, "lidar_file": "depth"}})
    for name, (ladder_rel, criteria_rel, judged, label) in HELD.items():
        section = {"ladder": pin(ladder_rel), "splits": pin(SPLITS), "criteria": pin(criteria_rel),
                   "output_dir": str(out), "judged": [judged], "ladders": [["L3", judged]]}
        base = plain if name == "camera_c2_plain" else lidar
        put(frozen(name), yaml.safe_dump({**base, "pose": {"rig_ref": 10, "config_label": label}, "protect": section}))
    _commit_all(repo)
    return SimpleNamespace(repo=repo, tests=tests)


def _ladder_request(name: str, *rungs: str) -> list[str]:
    """The eval ladder arguments of frozen config `name` with its pinned ladder and criteria and `rungs`."""
    ladder, criteria, _, _ = HELD[name]
    return ["ladder", "--config", frozen(name), "--ladder", ladder, "--criteria", criteria, "--rungs", ",".join(rungs)]


def test_judges_lidar_reads_the_judged_config():
    for criteria in (None, {}, {"y6": {}}, ["c2"], {"judged_config": "c2l"}, {"judged_config": ""},
                     {"judged_config": False}, {"judged_config": ["c2"]}):
        assert protect.judges_lidar(criteria) is True, criteria
    assert protect.judges_lidar({"judged_config": "c2"}) is False
    assert protect.LIDAR_JUDGED_CONFIGS == frozenset({"c2l"})
    assert protect.CAMERA_BOX_DEPTHS == ("box_median", "box_far")


def test_camera_judged_rung_passes_under_camera_criteria(camera_held, monkeypatch, reached, opened):
    h = camera_held
    for name in ("camera_c2", "camera_c2_plain"):
        r = _invoke(h, monkeypatch, *_ladder_request(name, "L3", "L4_drift"))
        assert r.exit_code == 0, (name, r.output)
    assert reached == [True, True]
    assert len(opened) == 6 and set(opened) == {root.name for root in h.tests}
    lifts = {name: load_yaml_model(h.repo / frozen(name), EvalAppConfig).lift
             for name in ("camera_c2", "camera_c2_plain")}
    assert {name: (lift.depth, lift.lidar_file, lift.lidar_files is None) for name, lift in lifts.items()} == {
        "camera_c2": ("box_far", "depth", False), "camera_c2_plain": ("box_far", None, True)}


def test_camera_judged_rung_is_refused_under_lidar_or_unnamed_criteria(camera_held, monkeypatch, reached, opened):
    h = camera_held
    for name in ("camera_c2l", "camera_unnamed"):
        _refused(_invoke(h, monkeypatch, *_ladder_request(name, "L3", "L4_drift")), BOX_FAR)
    assert opened == [] and reached == []


def test_judged_rungs_off_the_camera_box_depth_keep_the_lidar_rule_under_camera_criteria(camera_held, monkeypatch,
                                                                                         reached, opened):
    h = camera_held
    _refused(_invoke(h, monkeypatch, *_ladder_request("camera_det_c2", "L3", "L4_drift")),
             "split=test: judged rung L4_drift needs lift.depth=lidar, got det_depth")
    for name, need in (("ld_extr_c2", "noise.lidar_extr_rot_deg above 0, got 0.0"),
                       ("ld_orc_c2", "lift.depth=lidar, got lidar_oracle"),
                       ("ld_nav_c2", "a depth-unit LiDAR file (lift.lidar_file), got nav")):
        _refused(_invoke(h, monkeypatch, *_ladder_request(name, "L3", "L4_drift_ld")),
                 f"split=test: judged rung L4_drift_ld needs {need}")
    assert opened == [] and reached == []
    r = _invoke(h, monkeypatch, *_ladder_request("ld_c2", "L3", "L4_drift_ld"))
    assert r.exit_code == 0 and reached == [True], r.output


def test_lidar_rule_is_the_default(camera_held):
    h = camera_held
    cfg = load_yaml_model(h.repo / frozen("camera_c2"), EvalAppConfig)
    ladder = yaml.safe_load((h.repo / CAMERA).read_text(encoding="utf-8"))
    configs = {name: rung_config(cfg, ladder, name) for name in ("L3", "L4_drift")}
    for kwargs in ({}, {"lidar_judged": True}):
        with pytest.raises(ProtectionError, match=re.escape(BOX_FAR)):
            protect.check_test_lidar_rungs(cfg, configs, **kwargs)
    protect.check_test_lidar_rungs(cfg, configs, lidar_judged=False)
    for name, rung, need in (("camera_det_c2", "L4_drift", "lift.depth=lidar, got det_depth"),
                             ("ld_extr_c2", "L4_drift_ld", "noise.lidar_extr_rot_deg above 0, got 0.0")):
        other = load_yaml_model(h.repo / frozen(name), EvalAppConfig)
        rungs = yaml.safe_load((h.repo / HELD[name][0]).read_text(encoding="utf-8"))
        with pytest.raises(ProtectionError, match=re.escape(f"split=test: judged rung {rung} needs {need}")):
            protect.check_test_lidar_rungs(other, {n: rung_config(other, rungs, n) for n in ("L3", rung)},
                                           lidar_judged=False)
    quiet = _merge(_small_ladder(), {"rungs": {"L4_drift_ld": {"noise": {"lidar_range_sigma_m": 0.0}}}})
    with pytest.raises(ProtectionError, match=re.escape(
            "split=test: L4_drift_ld takes lift.depth=lidar with noise.lidar_range_sigma_m 0.0; held-out LiDAR rungs "
            "need LiDAR range and angle noise above 0")):
        protect.check_test_lidar_rungs(cfg, {**configs, "L4_drift_ld": rung_config(cfg, quiet, "L4_drift_ld")},
                                       lidar_judged=False)
