"""Test split protection: the checks a split=test request must pass before it may read a test bundle.

Four checks, each raising `ProtectionError` (exit code 1) at the first broken rule:

- `check_test_request` before any bundle is opened: the config file sits under
  `configs/eval/frozen/` in the counting repository, there is no `--set` or `--run`,
  the work tree is clean and committed, the config and every file option are tracked
  in the repository, a config with `detector.cache` pins one cache digest per bundle
  (`detector.digests`), and a grid changes only noise settings on ladder rungs or baselines.
- `check_test_rungs` after the ladder request or the sweep cells are checked, before any
  bundle is opened: no rung or sweep cell on cached detections (`stages.det=cached`) patches
  the detector or tracker settings, so the cached rungs score with the frozen config's;
  the oracle rungs keep their patches (`last_box`'s `tracker.repair_m`).
- `check_test_bundles` after the bundles are opened and before `prepare`: every bundle
  has a test seed and an `orchard_synth.commit` sidecar, and all bundles share one commit.
- `check_test_caches` right after `check_test_bundles` when the config has `detector.cache`:
  every bundle's cache file is there, its content digest is the pinned one, and the
  `test_log.csv` next to the cache folder (the `counting detect` output folder) has a detect
  row with that digest and the bundle's fingerprint, so every test cache comes from the
  protected `counting detect`.

A split=test request with a pose section passes three more checks. `check_test_request` also needs a
ladder file to be the committed pose ladder (`POSE_LADDER`, for a config without a protect section) and
pinned digests for every pose cache and IMU file the config names (`pose.caches.<name>.digests`,
`pose.imu_files.<name>.digests`); the eval commands add the fixed fusion settings file to the files it
needs tracked. With the rungs or sweep cells (or, for `eval run`, the config itself),
`check_test_pose_rungs` refuses the rungs, grid bases and noise axes that run on val only (the ladder's
`val_only`, `VAL_ONLY_NOISE_KEYS` away from the judged rung's values) and a rung or cell that changes
the pinned files, the reference camera, the configuration name, the alignment knobs (`ALIGN_KNOBS`), the
tracker prediction or the fusion settings (only the odometry factor sigmas, one value across them).
After the bundles are opened, `check_test_trajectories` needs bundles rendered with body shake whose
trajectory files match trajectory.json, and `check_test_pose_caches` needs every pinned pose cache and
IMU file with its content digest and a test log row of the protected `counting pose vo` or
`counting pose imu`. A split=test request without a pose section skips the pose file and bundle checks, so
`check_test_pose_rungs` refuses any of its rungs or sweep cells that adds a pose section.

Every split=test request ends `check_test_request` with three more checks: every name of `lift.lidar_files`
pins its `lidar.json` digests; the `--splits` file lists no seed of `CLOSED_TEST_SEEDS` under test, so held-out
seeds already judged are never scored again; and a frozen config without a protect section may list only seeds
of `LEGACY_TEST_SEEDS`, so every other held-out seed needs one. A frozen config with a protect section serves
only the requests the section names. Right after `check_test_request`, `check_test_frozen_request` refuses a
judged rung in more than one allowed rung list, `eval run`, a `--ladder`, `--splits` or `--criteria` file that is
not the pinned one (by path or sha256), an output folder other than `protect.output_dir`, a `--rungs` list that is
not one of `protect.ladders` (in order), `--with-2x2` and `--subsets`, a grid that is not a pinned one, and
criteria that judge other rungs than `protect.judged`. Once the rung or sweep cell configs are built,
`check_test_rung_protect` refuses one whose patch changes the protect section. Such a config's pinned ladder
takes the place of `POSE_LADDER`, and its first judged rung that of `JUDGED_RUNG` for the val-only axes.

A request of a frozen config with a protect section passes five more checks. Right after `check_test_log`,
`check_test_once` refuses a request the output folder's test log already lists (same config hash, kind and
experiment id extra), so each held-out request runs once and a second run needs a new frozen config. The output
folder of a split=test request is `held_out_output_root`: a relative `output_dir` is taken from the counting
repository, not from the current directory. With the rung or sweep cell configs, after the pose rung checks,
`check_test_lift` refuses one that changes a lift setting other than the depth method and the LiDAR file
(`LIDAR_LIFT_KEYS`), and `check_test_lidar_rungs` refuses one on a LiDAR box depth without LiDAR range and angle
noise (except the cells of a grid's `baseline` point), and a judged rung that does not take the LiDAR depth of the
depth unit with its extrinsic error. Right after `check_test_bundles`, `check_test_seed_set` needs the bundles to
hold every test seed of the splits, so a request on part of them cannot run first. Last before scoring, a config
with `lift.lidar_files` runs `check_test_lidar_files`: each pinned LiDAR file folder of each bundle is the pinned,
judged one made from that bundle by a clean simulator commit the config expects, all of them share one simulator
commit, and every requested ladder rung on the LiDAR depth sets the noise values of its files' contract. Its
refusals, including the read and check errors of the LiDAR files, are all `ProtectionError`.

Every split=test run that writes its results adds one row (`LOG_HEADER` columns) to the
`test_log.csv` of its output folder (`held_out_output_root`) with `append_log_row`;
`earlier_log_runs` counts the earlier rows with the same (config hash, experiment id extra),
the key a rerun warning is given for (a config with a protect section never reruns: `check_test_once`).
`check_test_log` refuses a log whose header is not `LOG_HEADER` (`ValueError`); the CLI runs it
before any bundle is opened, so a log with other columns stops the run before scoring.

`counting detect` on a test bundle runs `check_test_detect` before it reads any image: the
config file sits under `configs/detect/frozen/` in the counting repository, there is no `--set`,
the work tree is clean and committed, the config and every file option are tracked, and the
config's `weights_sha256` equals the sha256 of `--weights`. It then runs `check_test_bundles`
and `check_test_log`, and adds one `detect_log_row` per test bundle right after that bundle's
cache is written (its rerun key is the config hash and `weights:<sha16>`).

`eval truth` is not protected: it prints ground-truth tables only and never calls `prepare`.

`counting pose vo` and `counting pose imu` on a test bundle run `check_test_pose_file`, then
`check_test_bundles` and `check_test_log`, and add one `pose_log_row` per file right after it is
written (kind `pose_vo` with `opencv:<version>`, `pose_imu` with `imu:<profile>`).
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Collection, Mapping, Sequence

from counting.config.eval_schema import (
    CLOSED_TEST_SEEDS, LEGACY_TEST_SEEDS, LIDAR_DEPTHS, LIDAR_EXTR_KEYS, EvalAppConfig, LidarFiles, SplitsConfig,
)
from counting.config.loader import load_yaml_model
from counting.data.formats.orchard_synth import SynthRun
from counting.detection.cached import cache_digest, cache_path
from counting.evaluation.lidarsource import bundle_inputs, contract_noise, sidecar_contract
from counting.evaluation.record import git_state
from counting.evaluation.runner import ProtectionError, rung_config
from counting.evaluation.sweep import Grid, _leaf_paths, grid_points
from counting.geometry.lidarfile import (
    _FIELD_ERRORS, LIDAR_SIDES, LidarFile, _field_error, _read_sidecar, lidar_folder,
)
from counting.odometry.cache import pose_cache_digest, pose_cache_path
from counting.odometry.imufile import imu_file_digest, imu_file_path

FROZEN_DIR = ("configs", "eval", "frozen")
DETECT_FROZEN_DIR = ("configs", "detect", "frozen")
POSE_FROZEN_DIR = ("configs", "pose", "frozen")
POSE_LADDER = ("configs", "eval", "sweeps_pose", "ladder_pose.yaml")   # the one ladder a held-out pose run reads
LOG_HEADER = ("date", "experiment_id", "kind", "config", "config_hash", "extra", "fingerprints", "commit",
              "criteria_sha256", "verdict")
# The noise axes that run on val only: frame time offset, camera-IMU time offset, temperature rise and vibration.
VAL_ONLY_NOISE_KEYS = ("drift_time_offset_s", "imu_time_offset_ms", "imu_temp_k", "imu_vib_acc_ms2", "imu_vib_gyro_dps")
JUDGED_RUNG = "L4_drift"   # the judged pose rung; a held-out rung keeps its values of the val-only axes
ALIGN_KNOBS = ("segment_m", "pair_radius_m", "roll_pitch_sigma_deg", "min_pairs", "oracle_sigma_m")
_SIGMA_KEYS = ("vo_sigma_trans_mm", "vo_sigma_rot_deg")   # the pose.vio keys a held-out rung may set
# (log kind, config section, command, file noun, path and content digest functions) of the pinned pose files
_POSE_FILE_KINDS = (("pose_vo", "caches", "vo", "pose cache file", pose_cache_path, pose_cache_digest),
                    ("pose_imu", "imu_files", "imu", "IMU file", imu_file_path, imu_file_digest))
LIDAR_LIFT_KEYS = ("depth", "lidar_file")   # the lift settings a held-out rung may change: depth method, LiDAR file
# The LiDAR noise a held-out rung on the LiDAR depth needs above 0: the range and the beam angle sigmas.
LIDAR_SIGMA_KEYS = ("lidar_range_sigma_m", "lidar_ang_sigma_deg")
LIDAR_VARIANT = "judged"   # the sidecar variant of the LiDAR files a held-out run reads


@dataclass(frozen=True)
class ProtectedRequest:
    """A split=test request that passed `check_test_request`, a test-bundle detect request that
    passed `check_test_detect`, or a test-bundle pose request that passed `check_test_pose_file`.

    `config_rel` is the config file's path relative to the repository (POSIX), and
    `code` the git state (`commit`, `dirty`, `diff_sha256`) the request was checked against.
    """

    config_rel: str
    code: dict


def repo_relative(path: str | Path, repo: Path) -> str | None:
    """`path` relative to `repo` (POSIX, after `~` and symlinks are resolved), or None when it is outside.

    A relative `path` is taken from the current directory.
    """
    try:
        return Path(path).expanduser().resolve().relative_to(Path(repo).resolve()).as_posix()
    except ValueError:
        return None


def held_out_output_root(output_dir: str, repo: Path) -> Path:
    """The output folder of a split=test request: `output_dir` with `~` expanded, taken from the counting repository
    `repo` when it is relative.

    The test log a held-out request is checked against (`check_test_once`) and the experiment folder it writes then
    do not depend on the directory the request runs from.
    """
    path = Path(output_dir).expanduser()
    return path if path.is_absolute() else Path(repo) / path


def tracked_files(repo: Path) -> frozenset[str]:
    """The paths git tracks in `repo` (`git ls-files`, relative to `repo`); empty when git fails."""
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"], capture_output=True)
    if out.returncode != 0:
        return frozenset()
    return frozenset(p for p in out.stdout.decode().split("\0") if p)


def _test_grid_key(path: tuple[str, ...]) -> bool:
    """True for a leaf a split=test grid may change: `noise.<factor>` (not `noise.seeds`) or `detector.noise.*`."""
    if path[0] == "noise":
        return len(path) == 2 and path[1] != "seeds"
    return path[:2] == ("detector", "noise") and len(path) >= 3


def check_test_grid(grid: Grid, ladder: Mapping | None) -> None:
    """Refuse a split=test grid without bases, with a base the ladder does not name, or changing a non-noise leaf.

    Points go in `grid_points` order and each point's leaves in `_leaf_paths` order;
    the first leaf `_test_grid_key` refuses is named. Without a ladder no base name is known.
    """
    names: set[str] = set()
    if ladder is not None:
        for section in ("rungs", "baselines"):
            names.update(ladder.get(section) or {})
    if not grid.bases or any(b not in names for b in grid.bases):
        raise ProtectionError("split=test grid needs bases, and only ladder rung or baseline names, "
                              f"got {list(grid.bases)}")
    for point in grid_points(grid):
        for path in _leaf_paths(point.patch):
            if not _test_grid_key(path):
                raise ProtectionError("split=test grid may change only noise.* (not noise.seeds) and "
                                      f"detector.noise.*; point '{point.name}' changes {'.'.join(path)}")


def check_test_request(
    cfg: EvalAppConfig, *, config_path: str | Path, sets: Sequence[str], runs: Sequence[str],
    files: Mapping[str, str | Path], grid: Grid | None, ladder: Mapping | None, repo: Path,
    code: Mapping | None = None,
) -> ProtectedRequest:
    """Check a split=test request before any bundle is opened.

    Checks, in order: `config_path` sits under `configs/eval/frozen/` in `repo`; no
    `--set` (`sets`) or `--run` (`runs`); `code` (default: `git_state(repo)`) has a
    commit and is not dirty; `config_path` and each `files` value (keyed by its
    option, such as `--splits`) is a file git tracks in `repo`; a config with a pose
    section and a `--ladder` file but no protect section reads the committed pose ladder
    (`POSE_LADDER`; with one, `check_test_frozen_request` checks the pinned ladder); a config with
    `detector.cache` has `detector.digests`; with a `grid`, `check_test_grid(grid, ladder)`;
    a config with a pose section has non-empty `digests` for every name of `pose.caches`
    and `pose.imu_files`; every name of `lift.lidar_files` has non-empty `digests`; the
    `--splits` file (needed) lists no `CLOSED_TEST_SEEDS` seed under test; and a config without
    a protect section lists only `LEGACY_TEST_SEEDS` seeds under test. A config whose split is
    not test raises `ValueError`.
    """
    if cfg.split != "test":
        raise ValueError("check_test_request needs a split=test config")
    config_rel = repo_relative(config_path, repo)
    parts = PurePosixPath(config_rel).parts if config_rel is not None else ()
    if len(parts) < 4 or parts[:3] != FROZEN_DIR:
        raise ProtectionError("split=test needs a config file under configs/eval/frozen/ in the counting "
                              f"repository, got {config_path}")
    if sets or runs:
        raise ProtectionError("split=test takes no --set or --run; write every setting and bundle into the "
                              "frozen config file")
    code = dict(code) if code is not None else git_state(repo)
    if code["commit"] is None or code["dirty"]:
        raise ProtectionError("split=test needs a clean, committed counting work tree; commit or remove local "
                              "changes first")
    tracked = tracked_files(repo)
    for option, path in [("--config", config_path), *files.items()]:
        if repo_relative(path, repo) not in tracked:
            raise ProtectionError(f"split=test needs {option} to be a file tracked in the counting repository, "
                                  f"got {path}")
    if cfg.pose is not None and cfg.protect is None and "--ladder" in files:
        path = files["--ladder"]
        if repo_relative(path, repo) != "/".join(POSE_LADDER):
            raise ProtectionError(f"split=test with a pose section needs --ladder {'/'.join(POSE_LADDER)}, the "
                                  f"committed pose ladder; got {path}")
    if cfg.detector.cache is not None and not cfg.detector.digests:
        raise ProtectionError("split=test with detector.cache needs detector.digests in the frozen config, "
                              "one pinned cache digest per bundle")
    if grid is not None:
        check_test_grid(grid, ladder)
    if cfg.pose is not None:
        for section in ("caches", "imu_files"):
            for name, pinned in (getattr(cfg.pose, section) or {}).items():
                if not pinned.digests:
                    raise ProtectionError(f"split=test with pose.{section} needs pose.{section}.{name}.digests in "
                                          "the frozen config, one pinned file digest per bundle")
    for name, pinned in (cfg.lift.lidar_files or {}).items():
        if not pinned.digests:
            raise ProtectionError(f"split=test: lift.lidar_files.{name}.digests is empty; pin one lidar.json digest "
                                  "per bundle in the frozen config")
    if "--splits" not in files:
        raise ProtectionError("split=test needs --splits")
    test = set(load_yaml_model(files["--splits"], SplitsConfig).test)
    if closed := sorted(test & CLOSED_TEST_SEEDS):
        raise ProtectionError(f"split=test: --splits lists held-out seeds that are closed for good: {closed}")
    if cfg.protect is None and (new := sorted(test - LEGACY_TEST_SEEDS)):
        raise ProtectionError(f"split=test: protect section required for the held-out seeds {new}; write it into "
                              "the frozen config")
    return ProtectedRequest(config_rel=config_rel, code=code)


def _criteria_rungs(criteria: Mapping | None) -> list[str]:
    """The rungs `criteria` judges: the rung names under its `y1` and `y6` tables, sorted."""
    criteria = criteria or {}
    return sorted({name for kind in ("y1", "y6") for name in (criteria.get(kind) or {})})


def _pinned_file(option: str, key: str, path: Path | None, sha256: str | None, pinned) -> None:
    """Refuse the request file `option` (repository path `path`, bytes `sha256`) unless it is the file `pinned`
    (a `PinnedPath` of the protect section's `key`)."""
    rel = None if path is None else PurePosixPath(path).as_posix()
    if rel != pinned.path:
        raise ProtectionError(f"split=test: {option} {rel or 'none'} is not protect.{key} {pinned.path}")
    if sha256 != pinned.sha256:
        raise ProtectionError(f"split=test: {option} {rel} has sha256 {(sha256 or 'none')[:16]}, not the pinned "
                              f"{pinned.sha256[:16]} of protect.{key}")


def check_test_frozen_request(
    cfg: EvalAppConfig, *, kind: str, files: Mapping[str, Path], ladder_sha256: str | None, grid_sha256: str | None,
    criteria_sha256: str, splits_sha256: str, rungs: Sequence[str] | None, with_2x2: bool, subsets: bool,
    criteria: Mapping,
) -> None:
    """Refuse a split=test request that its frozen config's protect section does not name, before any bundle is opened.

    `kind` is the eval command (`run`, `ladder` or `sweep`), `files` maps each request option (`--ladder`,
    `--splits`, `--criteria`, `--grid`) to its file's path relative to the counting repository, the `*_sha256`
    arguments are those files' sha256, `rungs` the requested rungs (every rung and baseline when `--rungs` is
    not given), `with_2x2` and `subsets` the two flags and `criteria` the criteria table read. Checks, in order:
    each rung of `protect.judged` sits in at most one list of `protect.ladders`; the command is a ladder or a
    sweep; `--ladder` is `protect.ladder` (path, then sha256); `--splits` and `--criteria` are `protect.splits`
    and `protect.criteria`; `cfg.output_dir` is `protect.output_dir`, character for character; for a ladder,
    `rungs` is one of `protect.ladders` (same names in the same order), without `--with-2x2` or `--subsets`; for
    a sweep, `--grid` is a path of `protect.grids` with its sha256; and the rungs the criteria judge (its `y1`
    and `y6` tables) are those of `protect.judged`. A config without a protect section raises `ValueError`.
    """
    section = cfg.protect
    if section is None:
        raise ValueError("check_test_frozen_request needs a config with a protect section")
    for name in section.judged:
        if (k := sum(name in allowed for allowed in section.ladders)) > 1:
            raise ProtectionError(f"split=test: protect.judged rung {name} is in {k} allowed rung lists; a judged "
                                  "rung may be in one only")
    if kind not in ("ladder", "sweep"):
        raise ProtectionError(f"split=test: a frozen config with a protect section serves eval ladder and eval sweep "
                              f"only, not eval {kind}")
    _pinned_file("--ladder", "ladder", files.get("--ladder"), ladder_sha256, section.ladder)
    _pinned_file("--splits", "splits", files.get("--splits"), splits_sha256, section.splits)
    _pinned_file("--criteria", "criteria", files.get("--criteria"), criteria_sha256, section.criteria)
    if cfg.output_dir != section.output_dir:
        raise ProtectionError(f"split=test: output_dir {cfg.output_dir} differs from protect.output_dir "
                              f"{section.output_dir}; a held-out run writes only into the frozen output folder")
    if kind == "ladder":
        if list(rungs or ()) not in section.ladders:
            raise ProtectionError(f"split=test: --rungs {','.join(rungs or ())} is not one of the protect.ladders "
                                  "lists")
        for flag, on in (("--with-2x2", with_2x2), ("--subsets", subsets)):
            if on:
                raise ProtectionError(f"split=test: a frozen config with a protect section takes no {flag}")
    else:
        path = files.get("--grid")
        rel = None if path is None else PurePosixPath(path).as_posix()
        if rel not in section.grids:
            raise ProtectionError(f"split=test: --grid {rel or 'none'} is not one of protect.grids")
        if grid_sha256 != section.grids[rel]:
            raise ProtectionError(f"split=test: --grid {rel} has sha256 {(grid_sha256 or 'none')[:16]}, not the "
                                  f"pinned {section.grids[rel][:16]} of protect.grids")
    if (judged := _criteria_rungs(criteria)) != sorted(set(section.judged)):
        raise ProtectionError(f"split=test: the judged rungs of --criteria {judged} differ from protect.judged "
                              f"{list(section.judged)}")


def check_test_rung_protect(cfg: EvalAppConfig, configs: Mapping[str, EvalAppConfig]) -> None:
    """Refuse a split=test rung or sweep cell whose config changes the frozen config's protect section.

    `configs` maps each requested ladder rung (or sweep cell `"<point> @ <base>"`) name to its config; a rung
    patch is merged deep into the frozen config, so it could widen the allowed requests. The first one whose
    protect section dump differs from `cfg`'s is named.
    """
    frozen = None if cfg.protect is None else cfg.protect.model_dump()
    for name, c in configs.items():
        if (None if c.protect is None else c.protect.model_dump()) != frozen:
            raise ProtectionError(f"split=test: {name} changes the protect section; held-out rungs keep the frozen "
                                  "config's")


def check_test_rungs(cfg: EvalAppConfig, configs: Mapping[str, EvalAppConfig]) -> None:
    """Refuse a split=test rung on cached detections that changes the frozen config's detector or tracker settings.

    `configs` maps each requested ladder rung (or sweep cell) name to its config. Every one
    with `stages.det=cached` must have the `detector` and `tracker` dumps of `cfg`; the first
    one that does not is named. The other rungs may patch both (`last_box`'s `tracker.repair_m`).
    """
    frozen = (cfg.detector.model_dump(), cfg.tracker.model_dump())
    for name, c in configs.items():
        if c.stages.det == "cached" and (c.detector.model_dump(), c.tracker.model_dump()) != frozen:
            raise ProtectionError(f"split=test: rung {name} changes the detector or tracker settings of a cached "
                                  "rung; a test run takes them from the frozen config")


def _pinned_pose_files(c: EvalAppConfig) -> dict | None:
    return None if c.pose is None else c.pose.model_dump(include={"caches", "imu_files"})


def _pose_frame(c: EvalAppConfig) -> tuple:
    return (None, None) if c.pose is None else (c.pose.rig_ref, c.pose.config_label)


def _align_knob(c: EvalAppConfig, key: str) -> object:
    return None if c.align is None else getattr(c.align, key)


def _factor_sigmas(c: EvalAppConfig) -> tuple | None:
    """The odometry factor sigmas of `c`'s `pose.vio` in `_SIGMA_KEYS` order (None for an unset one), or None
    when it sets neither."""
    vio = None if c.pose is None else c.pose.vio
    sigmas = tuple(None if vio is None else getattr(vio, k) for k in _SIGMA_KEYS)
    return None if all(v is None for v in sigmas) else sigmas


def check_test_pose_rungs(
    cfg: EvalAppConfig, configs: Mapping[str, tuple[str | None, EvalAppConfig]], ladder: Mapping | None
) -> None:
    """Refuse a split=test rung or sweep cell that runs on val only or changes what the frozen config pins.

    `configs` maps each name (a ladder rung, or a sweep cell `"<point> @ <base>"`) to the ladder rung
    its config is built on (None for none) and the config. Without a pose section in `cfg`, the first
    config (in `configs` order) that has one is refused, since the pinned pose files, the shaken bundles
    and the rules below all come from the frozen config's pose section; nothing else is checked. With
    one, names go in `configs` order, and for each one the first broken rule is named: with a
    `ladder`, the name or its rung is in the ladder's `val_only`; its `pose.caches` or `pose.imu_files`
    differ from `cfg`'s; its `pose.rig_ref` or `pose.config_label` differ; an `ALIGN_KNOBS` knob differs
    (a config without an align section has none); `tracker.predict` or `tracker.predict_window` differ;
    `pose.vio` sets a key other than the odometry factor sigmas; and, with a `ladder`, it sets a
    `VAL_ONLY_NOISE_KEYS` factor away from the value of the judged rung on `cfg` (`JUDGED_RUNG`, or the
    first rung of `protect.judged` for a config with a protect section; a ladder without that rung is
    refused when this rule first needs it). Then all configs that set factor sigmas must set the
    same two values; the first that differs from the first one is named with it.
    """
    if cfg.pose is None:
        for name, (_, c) in configs.items():
            if c.pose is not None:
                raise ProtectionError(f"split=test: {name} adds a pose section; a held-out pose run needs a frozen "
                                      "config with a pose section")
        return
    val_only = set(ladder.get("val_only") or []) if ladder is not None else set()
    files, frame = _pinned_pose_files(cfg), _pose_frame(cfg)
    predict = (cfg.tracker.predict, cfg.tracker.predict_window)
    judged_rung = cfg.protect.judged[0] if cfg.protect else JUDGED_RUNG
    judged = None   # the noise section of rung_config(cfg, ladder, judged_rung), built on first use

    def judged_noise():
        nonlocal judged
        if judged is None:
            if not any(judged_rung in (ladder.get(s) or {}) for s in ("rungs", "baselines")):
                raise ProtectionError(f"split=test: the pose ladder needs rung {judged_rung}")
            judged = rung_config(cfg, ladder, judged_rung).noise
        return judged

    for name, (base, c) in configs.items():
        if name in val_only or base in val_only:
            raise ProtectionError(f"split=test: {name} runs on val only (val_only of the ladder)")
        if _pinned_pose_files(c) != files:
            raise ProtectionError(f"split=test: {name} changes pose.caches or pose.imu_files; a held-out run reads "
                                  "the frozen config's pinned files")
        if _pose_frame(c) != frame:
            raise ProtectionError(f"split=test: {name} changes pose.rig_ref or pose.config_label")
        for k in ALIGN_KNOBS:
            if _align_knob(c, k) != _align_knob(cfg, k):
                raise ProtectionError(f"split=test: {name} changes the alignment knob align.{k}; held-out rungs keep "
                                      "the frozen config's knobs")
        if (c.tracker.predict, c.tracker.predict_window) != predict:
            raise ProtectionError(f"split=test: {name} changes the tracker prediction")
        vio = None if c.pose is None else c.pose.vio
        if vio is not None:
            for k in type(vio).model_fields:
                if k not in _SIGMA_KEYS and getattr(vio, k) is not None:
                    raise ProtectionError(f"split=test: {name} sets pose.vio.{k}; that setting runs on val only")
        if ladder is not None:
            for k in VAL_ONLY_NOISE_KEYS:
                v = getattr(c.noise, k)
                if v is not None and c.noise.factor(k) != (j := judged_noise().factor(k)):
                    raise ProtectionError(f"split=test: {name} sets noise.{k}={v}; that axis runs on val only "
                                          f"(the judged rung keeps {j})")
    first: tuple[str, tuple] | None = None
    for name, (_, c) in configs.items():
        sigmas = _factor_sigmas(c)
        if sigmas is None:
            continue
        if first is None:
            first = (name, sigmas)
        elif sigmas != first[1]:
            raise ProtectionError(f"split=test: rungs {first[0]} and {name} differ in pose.vio factor sigmas; "
                                  "held-out rungs share the committed ladder's one value")


def check_test_lift(cfg: EvalAppConfig, configs: Mapping[str, EvalAppConfig]) -> None:
    """Refuse a split=test rung or sweep cell that changes a lift setting other than `LIDAR_LIFT_KEYS`.

    `configs` maps each requested ladder rung (or sweep cell `"<point> @ <base>"`) name to its config. A held-out
    rung may choose its depth method and its LiDAR file; every other `lift` setting (the camera box depth knobs,
    the label offsets, the LiDAR depth knobs `lift.lidar` and the pinned LiDAR files `lift.lidar_files`) must dump
    as in `cfg`. The first config that differs is named with its first differing setting, in field order.
    """
    frozen = cfg.lift.model_dump()
    for name, c in configs.items():
        got = c.lift.model_dump()
        for key in type(cfg.lift).model_fields:
            if key not in LIDAR_LIFT_KEYS and got.get(key) != frozen.get(key):
                raise ProtectionError(f"split=test: {name}: lift.{key} differs from the frozen config; held-out rungs "
                                      f"change only {' and '.join(f'lift.{k}' for k in LIDAR_LIFT_KEYS)}")


def check_test_lidar_rungs(
    cfg: EvalAppConfig, configs: Mapping[str, EvalAppConfig], *, baseline_cells: Collection[str] = ()
) -> None:
    """Refuse a split=test LiDAR rung or sweep cell without LiDAR noise, and a judged rung off the LiDAR depth.

    `configs` maps each requested ladder rung (or sweep cell `"<point> @ <base>"`) name to its config. In
    `configs` order, a config on a LiDAR box depth (`LIDAR_DEPTHS`) needs each `LIDAR_SIGMA_KEYS` noise above 0
    (an unset key is 0), and a rung of the frozen config's `protect.judged` (none without a protect section) needs
    `lift.depth=lidar` (not the truth-label pairing), a depth-unit LiDAR file and each `LIDAR_EXTR_KEYS` extrinsic
    error above 0. The first broken rule is named. The rung configs keep the frozen protect section
    (`check_test_rung_protect` runs first), so the judged rungs are read from `cfg`.

    `baseline_cells` names the sweep cells of a grid's `baseline` point, which the noise rule skips: such a cell is
    the grid's base rung as the ladder writes it, which may be a structure rung without noise when the grid's
    `common` block adds the noise to every other point. A sweep has no verdict, and every other cell of that grid
    is still held to the rule, as is the base rung itself when a ladder request names it.
    """
    judged = set(cfg.protect.judged) if cfg.protect is not None else set()
    for name, c in configs.items():
        if c.lift.depth in LIDAR_DEPTHS and name not in baseline_cells:
            for key in LIDAR_SIGMA_KEYS:
                if not c.noise.factor(key) > 0:
                    raise ProtectionError(f"split=test: {name} takes lift.depth={c.lift.depth} with noise.{key} "
                                          f"{c.noise.factor(key)}; held-out LiDAR rungs need LiDAR range and angle "
                                          "noise above 0")
        if name not in judged:
            continue
        if c.lift.depth != "lidar":
            raise ProtectionError(f"split=test: judged rung {name} needs lift.depth=lidar, got {c.lift.depth}")
        if (unit := c.lift.lidar_unit()) != "depth":
            raise ProtectionError(f"split=test: judged rung {name} needs a depth-unit LiDAR file (lift.lidar_file), "
                                  f"got {unit}")
        for key in LIDAR_EXTR_KEYS:
            if not c.noise.factor(key) > 0:
                raise ProtectionError(f"split=test: judged rung {name} needs noise.{key} above 0, got "
                                      f"{c.noise.factor(key)}")


# The first four rules repeat those of `check_test_request` (for configs/detect/frozen/); change them together.
def check_test_detect(
    *, config_path: str | Path, sets: Sequence[str], files: Mapping[str, str | Path], weights_sha256: str,
    pinned_sha256: str | None, repo: Path, code: Mapping | None = None,
) -> ProtectedRequest:
    """Check a `counting detect` request on a test bundle before any image is read.

    Checks, in order: `config_path` sits under `configs/detect/frozen/` in `repo`; no
    `--set` (`sets`); `code` (default: `git_state(repo)`) has a commit and is not dirty;
    `config_path` and each `files` value (keyed by its option, such as `--splits`) is a
    file git tracks in `repo`; and the config's `pinned_sha256` equals `weights_sha256`,
    the sha256 of `--weights`.
    """
    config_rel = repo_relative(config_path, repo)
    parts = PurePosixPath(config_rel).parts if config_rel is not None else ()
    if len(parts) < 4 or parts[:3] != DETECT_FROZEN_DIR:
        raise ProtectionError("counting detect on a test bundle needs a config file under configs/detect/frozen/ "
                              f"in the counting repository, got {config_path}")
    if sets:
        raise ProtectionError("counting detect on a test bundle takes no --set; write every setting into the "
                              "frozen config file")
    code = dict(code) if code is not None else git_state(repo)
    if code["commit"] is None or code["dirty"]:
        raise ProtectionError("counting detect on a test bundle needs a clean, committed counting work tree; "
                              "commit or remove local changes first")
    tracked = tracked_files(repo)
    for option, path in [("--config", config_path), *files.items()]:
        if repo_relative(path, repo) not in tracked:
            raise ProtectionError(f"counting detect on a test bundle needs {option} to be a file tracked in the "
                                  f"counting repository, got {path}")
    if pinned_sha256 != weights_sha256:
        pinned = pinned_sha256[:16] if pinned_sha256 else "none"
        raise ProtectionError("counting detect on a test bundle needs weights_sha256 in the frozen config to equal "
                              f"the sha256 of --weights; got {pinned} and {weights_sha256[:16]}")
    return ProtectedRequest(config_rel=config_rel, code=code)


# The four rules repeat the first four of `check_test_detect` (for configs/pose/frozen/); change them together.
def check_test_pose_file(
    *, command: str, config_path: str | Path, sets: Sequence[str], files: Mapping[str, str | Path], repo: Path,
    code: Mapping | None = None,
) -> ProtectedRequest:
    """Check a `counting pose vo` (`command` "vo") or `counting pose imu` ("imu") request on a test bundle
    before any pose cache or IMU file is made.

    Checks, in order: `config_path` sits under `configs/pose/frozen/` in `repo`; no `--set`
    (`sets`); `code` (default: `git_state(repo)`) has a commit and is not dirty; and
    `config_path` and each `files` value (keyed by its option, such as `--splits`) is a file
    git tracks in `repo`. Another `command` raises `ValueError`.
    """
    if command not in ("vo", "imu"):
        raise ValueError(f"check_test_pose_file needs command vo or imu, got {command}")
    what = "a pose cache" if command == "vo" else "an IMU file"
    config_rel = repo_relative(config_path, repo)
    parts = PurePosixPath(config_rel).parts if config_rel is not None else ()
    if len(parts) < 4 or parts[:3] != POSE_FROZEN_DIR:
        raise ProtectionError(f"counting pose {command} makes {what} of a test bundle only from a config file "
                              f"under configs/pose/frozen/ in the counting repository, got {config_path}")
    if sets:
        raise ProtectionError(f"counting pose {command} on a test bundle takes no --set; write every setting into "
                              "the frozen config file")
    code = dict(code) if code is not None else git_state(repo)
    if code["commit"] is None or code["dirty"]:
        raise ProtectionError(f"counting pose {command} on a test bundle needs a clean, committed counting work "
                              "tree; commit or remove local changes first")
    tracked = tracked_files(repo)
    for option, path in [("--config", config_path), *files.items()]:
        if repo_relative(path, repo) not in tracked:
            raise ProtectionError(f"counting pose {command} on a test bundle needs {option} to be a file tracked "
                                  f"in the counting repository, got {path}")
    return ProtectedRequest(config_rel=config_rel, code=code)


def check_test_bundles(bundles: Sequence[SynthRun], splits: SplitsConfig) -> None:
    """Check opened split=test bundles before `prepare`.

    Checks, in order: every bundle's seed is in the test split, every bundle has an
    `orchard_synth.commit` sidecar, and all bundles share one sidecar commit.
    """
    for run in bundles:
        split = splits.split_of(run.meta.seed) or "none"
        if split != "test":
            raise ProtectionError(f"{run.root}: split=test needs a test seed; seed {run.meta.seed} belongs to "
                                  f"split {split}")
    for run in bundles:
        if run.meta.synth_commit is None:
            raise ProtectionError(f"{run.root}: split=test needs the orchard_synth.commit sidecar")
    commits = sorted({run.meta.synth_commit for run in bundles})
    if len(commits) > 1:
        raise ProtectionError(f"split=test needs one orchard_synth commit across the bundles, got {commits}")


def check_test_seed_set(bundles: Sequence[SynthRun], splits: SplitsConfig) -> None:
    """Refuse opened split=test bundles whose seed set is not the test split's: a held-out request opens every test
    bundle at once, so a request on part of them cannot run first and take the request's one run."""
    got, want = sorted({run.meta.seed for run in bundles}), sorted(set(splits.test))
    if got != want:
        raise ProtectionError(f"split=test: the bundles hold the seeds {got}, not the test seeds {want} of --splits; "
                              "a held-out run opens every test bundle at once")


def check_test_caches(bundles: Sequence[SynthRun], cfg: EvalAppConfig) -> list[dict]:
    """Check the detection caches of opened split=test bundles against `cfg`'s pins and the detect log.

    For each bundle in turn: `cfg.detector.digests` pins a digest for its run id, the
    cache file `cache_path(detector.cache, run_id)` exists, its `cache_digest` is the
    pinned one, and `test_log.csv` in the cache folder's parent (the `counting detect`
    output folder) has a `detect` row whose experiment id lists the pinned digest's first
    16 characters and whose fingerprints list the bundle's. Returns `{"run_id", "digest"}`
    per bundle, in `bundles` order. A cache file `cache_digest` cannot read raises its
    `ValueError`; a config without `detector.cache` raises `ValueError`.
    """
    if cfg.detector.cache is None:
        raise ValueError("check_test_caches needs a config with detector.cache")
    folder = Path(cfg.detector.cache).expanduser()
    log = folder.parent / "test_log.csv"
    pins = cfg.detector.digests or {}
    rows = []
    if log.is_file():
        with log.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    out = []
    for run in bundles:
        run_id, fp = run.meta.run_id, run.meta.fingerprint
        if run_id not in pins:
            raise ProtectionError(f"{run.root}: detector.digests pins no cache for run {run_id}")
        pinned, path = pins[run_id], cache_path(folder, run_id)
        if not path.is_file():
            raise ProtectionError(f"{path}: split=test needs the pinned detection cache file")
        digest = cache_digest(path)
        if digest != pinned:
            raise ProtectionError(f"{path}: cache digest {digest[:16]} differs from the pinned {pinned[:16]}")
        if not any(row.get("kind") == "detect" and pinned[:16] in (row.get("experiment_id") or "").split(";")
                   and fp in (row.get("fingerprints") or "").split(";") for row in rows):
            raise ProtectionError(f"{log}: no counting detect row for cache {pinned[:16]} of bundle {fp[:12]}; "
                                  "a test cache must come from the protected counting detect")
        out.append({"run_id": run_id, "digest": pinned})
    return out


def check_test_pose_caches(bundles: Sequence[SynthRun], cfg: EvalAppConfig) -> list[dict]:
    """Check the pinned pose caches and IMU files of opened split=test bundles against `cfg`'s pins and the
    test logs of the protected pose commands.

    Pose caches (`pose.caches`, log kind `pose_vo`) go first, then IMU files (`pose.imu_files`,
    `pose_imu`); names in the config's order, bundles in `bundles` order. For each: the name's
    `digests` pins a digest for the bundle's run id, the file (`pose_cache_path` or `imu_file_path`
    in the name's `dir`) exists, its content digest is the pinned one, and `test_log.csv` in the
    folder's parent (the pose command's output folder) has a row of that kind whose experiment id
    lists the pinned digest's first 16 characters, whose fingerprints list the bundle's and whose
    config sits under `configs/pose/frozen/`. Returns `{"kind", "name", "run_id", "digest"}` in that
    order. A file the digest function cannot read raises its `ValueError`; a config without a pose
    section raises `ValueError`.
    """
    if cfg.pose is None:
        raise ValueError("check_test_pose_caches needs a config with a pose section")
    frozen = "/".join(POSE_FROZEN_DIR) + "/"
    out = []
    for kind, section, cmd, what, file_path, file_digest in _POSE_FILE_KINDS:
        for name, files in (getattr(cfg.pose, section) or {}).items():
            folder = Path(files.dir).expanduser()
            log = folder.parent / "test_log.csv"
            rows = []
            if log.is_file():
                with log.open(newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
            for run in bundles:
                run_id, fp = run.meta.run_id, run.meta.fingerprint
                if run_id not in files.digests:
                    raise ProtectionError(f"{run.root}: pose.{section}.{name}.digests pins no file for run {run_id}")
                pinned, path = files.digests[run_id], file_path(folder, run_id)
                if not path.is_file():
                    raise ProtectionError(f"{path}: split=test needs the pinned {what}")
                digest = file_digest(path)
                if digest != pinned:
                    raise ProtectionError(f"{path}: content digest {digest[:16]} differs from the pinned {pinned[:16]}")
                if not any(row.get("kind") == kind and pinned[:16] in (row.get("experiment_id") or "").split(";")
                           and fp in (row.get("fingerprints") or "").split(";")
                           and (row.get("config") or "").startswith(frozen) for row in rows):
                    raise ProtectionError(f"{log}: no counting pose {cmd} row from a frozen config for file "
                                          f"{pinned[:16]} of bundle {fp[:12]}; a test file must come from the "
                                          f"protected counting pose {cmd}")
                out.append({"kind": kind, "name": name, "run_id": run_id, "digest": pinned})
    return out


def check_test_trajectories(bundles: Sequence[SynthRun]) -> list[dict]:
    """Check that opened split=test bundles for a config with a pose section were rendered with body shake
    and that their trajectory files still match trajectory.json.

    For each bundle in turn: `meta.shake` is set, trajectory.json is there, and each file its
    `digests` lists (in name order) is there with that sha256. Returns `{"run_id", "trajectory_sha256"}`
    (the sha256 of the trajectory.json bytes) per bundle, in `bundles` order.
    """
    out = []
    for run in bundles:
        if not run.meta.shake:
            raise ProtectionError(f"{run.root}: split=test with a pose section needs a bundle rendered with body "
                                  "shake")
        tpath = run.root / "trajectory.json"
        if not tpath.is_file():
            raise ProtectionError(f"{tpath}: split=test with a pose section needs the bundle's trajectory.json")
        raw = tpath.read_bytes()
        for name, want in sorted(json.loads(raw)["digests"].items()):
            path = run.root / name
            if not path.is_file():
                raise ProtectionError(f"{path}: listed in trajectory.json but missing")
            if hashlib.sha256(path.read_bytes()).hexdigest() != want:
                raise ProtectionError(f"{path}: sha256 differs from trajectory.json")
        out.append({"run_id": run.meta.run_id, "trajectory_sha256": hashlib.sha256(raw).hexdigest()})
    return out


def _lidar_provenance(run: SynthRun, files: LidarFiles, pinned: str, where: str) -> str:
    """Check the LiDAR files of `files` beside the bundle `run` and return their simulator commit.

    `LidarFile` checks the sidecar against `pinned`, the unit, the bundle (crop, name, seed, the sha256 of its
    manifest.json and trajectory.json), the simulation config sha256 and folder name and the expected commits; then
    the sidecar must be the `LIDAR_VARIANT` variant, record the bundle's render commit, list the point and label file
    of each side with their sha256 (the label files are hashed, never opened) and come from a clean simulator tree.
    Those last rules raise `ProtectionError` led by `where`; a file that cannot be read raises `ValueError` or
    `FileNotFoundError`.
    """
    folder = lidar_folder(files.dir, run.root.name)
    lidar = LidarFile(folder, unit=files.unit, bundle=run.root.name, crop=run.meta.crop, seed=run.meta.seed,
                      inputs=bundle_inputs(run), digest=pinned, config_sha256=files.sim_config_sha256,
                      sim_commits=files.sim_commits)
    path, data, sidecar = _read_sidecar(folder)
    if hashlib.sha256(data).hexdigest() != pinned:
        raise ProtectionError(f"{where}{path}: lidar.json changed after it was checked")
    try:
        variant, render_commit, listed = sidecar["variant"], sidecar["inputs"]["render_commit"], sidecar["files"]
    except _FIELD_ERRORS as e:
        raise _field_error(path, e) from None
    if variant != LIDAR_VARIANT:
        raise ProtectionError(f"{where}{path}: variant {variant}, expected {LIDAR_VARIANT}")
    if render_commit != run.meta.synth_commit:
        raise ProtectionError(f"{where}{path}: render_commit {str(render_commit)[:12]}, expected the bundle's "
                              f"orchard_synth commit {str(run.meta.synth_commit)[:12]}")
    for name in (f"lidar_{side}{tail}.npz" for side in LIDAR_SIDES for tail in ("", "_labels")):
        file = folder / name
        if name not in listed:
            raise ProtectionError(f"{where}{path}: lidar.json lists no {name}")
        if not file.is_file():
            raise ProtectionError(f"{where}{file}: no LiDAR file")
        if hashlib.sha256(file.read_bytes()).hexdigest() != listed[name]:
            raise ProtectionError(f"{where}{file}: sha256 does not match lidar.json files")
    if lidar.dirty:
        raise ProtectionError(f"{where}{path}: orchard_synth dirty is true; a held-out run reads LiDAR files of a "
                              "clean simulator commit")
    return lidar.sim_commit


def check_test_lidar_files(
    bundles: Sequence[SynthRun], cfg: EvalAppConfig, rungs: Mapping[str, EvalAppConfig]
) -> list[dict]:
    """Check the pinned LiDAR files of opened split=test bundles, and the LiDAR noise of the rungs on them.

    Names go in the order of `cfg.lift.lidar_files`, bundles in `bundles` order. For each: the name's `digests`
    pins a `lidar.json` digest for the bundle's run id, and `_lidar_provenance` checks the files in the name's
    folder (through `LidarFile`). Then all of them must come from one simulator commit. Last, each rung of `rungs`
    (name -> config; the requested ladder rungs) on `lift.depth=lidar` must set every noise value of its LiDAR
    file's contract (`sidecar_contract`, under counting keys by `contract_noise`) to the contract value, with an
    unset key read as 0 (`NoiseConfig.factor`), against each bundle's sidecar. Returns `{"name", "run_id",
    "digest", "sim_commit"}` per name and bundle, in that order. Every broken rule raises `ProtectionError`; a
    `ValueError` or `FileNotFoundError` from reading or checking a file becomes one naming the name and the run
    (its message and path kept, the error as its cause).
    """
    named = cfg.lift.lidar_files or {}
    out = []

    def where(name: str, run: SynthRun) -> str:
        return f"split=test: lift.lidar_files.{name} for run {run.meta.run_id}: "

    for name, files in named.items():
        for run in bundles:
            run_id = run.meta.run_id
            if run_id not in files.digests:
                raise ProtectionError(f"{where(name, run)}no pinned lidar.json digest")
            pinned = files.digests[run_id]
            try:
                commit = _lidar_provenance(run, files, pinned, where(name, run))
            except ProtectionError:
                raise
            except (ValueError, FileNotFoundError) as e:
                raise ProtectionError(f"{where(name, run)}{e}") from e
            out.append({"name": name, "run_id": run_id, "digest": pinned, "sim_commit": commit})
    commits = sorted({row["sim_commit"] for row in out})
    if len(commits) > 1:
        raise ProtectionError(f"split=test needs one simulator commit across the LiDAR files, got {commits}")
    contracts: dict[tuple[str, str], dict[str, float]] = {}
    for rung, c in rungs.items():
        if c.lift.depth != "lidar":
            continue
        name = c.lift.lidar_file
        if name not in named:
            raise ProtectionError(f"split=test: {rung}: lift.lidar_file {name} is not one of the frozen config's "
                                  "lift.lidar_files")
        files = named[name]
        for run in bundles:
            key = (name, run.meta.run_id)
            if key not in contracts:
                try:
                    contracts[key] = contract_noise(sidecar_contract(lidar_folder(files.dir, run.root.name)),
                                                    files.unit)
                except (ValueError, FileNotFoundError) as e:
                    raise ProtectionError(f"{where(name, run)}{e}") from e
            for k, want in contracts[key].items():
                if (value := c.noise.factor(k)) != want:
                    raise ProtectionError(f"split=test: {rung}: noise.{k} {value} differs from the LiDAR contract "
                                          f"value {want}")
    return out


def check_test_log(path: Path) -> None:
    """Refuse a test log `path` whose header row is not `LOG_HEADER`, in that order (`ValueError` naming
    the file). A missing or empty file passes: `append_log_row` writes the header first.
    """
    path = Path(path)
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)
    if header is not None and tuple(header) != LOG_HEADER:
        raise ValueError(f"{path}: a test log needs the columns {','.join(LOG_HEADER)} in this order, "
                         f"got {','.join(header) or 'none'}")


def check_test_once(path: Path, *, config_hash: str, kind: str, extra: str) -> None:
    """Refuse a split=test request the test log `path` already lists: a row with this `config_hash`, `kind` and
    `extra`. The refusal names the first such row's date and experiment id, never its verdict. A missing file
    passes; the log's header must pass `check_test_log`.
    """
    path = Path(path)
    if not path.is_file():
        return
    check_test_log(path)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["config_hash"], row["kind"], row["extra"]) == (config_hash, kind, extra):
                raise ProtectionError(f"split=test: this request already ran on {row['date']} as "
                                      f"{row['experiment_id']}; a second run needs a new frozen config")


def earlier_log_runs(path: Path, *, config_hash: str, extra: str) -> int:
    """The number of rows of the test log `path` with this `config_hash` and `extra` (0 without the file).

    The log's header must pass `check_test_log`.
    """
    path = Path(path)
    if not path.is_file():
        return 0
    check_test_log(path)
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for row in csv.DictReader(f) if row["config_hash"] == config_hash and row["extra"] == extra)


def append_log_row(path: Path, row: Mapping[str, str]) -> None:
    """Append `row` to the test log `path` in `LOG_HEADER` column order, writing the header first
    when the file is missing or empty (its folder is created). `row` must have exactly the
    `LOG_HEADER` keys, in any order, or `ValueError`.
    """
    if set(row) != set(LOG_HEADER):
        raise ValueError(f"a test log row needs the keys {', '.join(LOG_HEADER)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_HEADER)
        w.writerow([row[k] for k in LOG_HEADER])


def detect_log_row(
    *, date: str, config_rel: str, config_hash: str, weights_sha256: str, fingerprints: Sequence[str],
    digests: Sequence[str], commit: str,
) -> dict:
    """A `test_log.csv` row (`LOG_HEADER` keys) for `counting detect` caches written on test bundles.

    The experiment id lists each cache digest's first 16 characters, `extra` is
    `weights:<sha16>` (the rerun key with `config_hash`), and the fingerprints are sorted;
    a detect row has no criteria or verdict.
    """
    return {"date": date, "experiment_id": ";".join(d[:16] for d in digests), "kind": "detect",
            "config": config_rel, "config_hash": config_hash, "extra": f"weights:{weights_sha256[:16]}",
            "fingerprints": ";".join(sorted(fingerprints)), "commit": commit, "criteria_sha256": "", "verdict": ""}


def pose_log_row(
    *, kind: str, date: str, config_rel: str, config_hash: str, extra: str, fingerprint: str, digest: str,
    commit: str,
) -> dict:
    """A `test_log.csv` row (`LOG_HEADER` keys) for one pose cache (`kind` "pose_vo") or IMU file ("pose_imu")
    written on a test bundle.

    The experiment id is the file's content digest's first 16 characters, the fingerprints are
    the bundle's, and `extra` (`opencv:<version>` or `imu:<profile>`) is the rerun key with
    `config_hash`; a pose row has no criteria or verdict. Another `kind` raises `ValueError`.
    """
    if kind not in ("pose_vo", "pose_imu"):
        raise ValueError("a pose log row's kind must be pose_vo or pose_imu")
    return {"date": date, "experiment_id": digest[:16], "kind": kind, "config": config_rel,
            "config_hash": config_hash, "extra": extra, "fingerprints": fingerprint, "commit": commit,
            "criteria_sha256": "", "verdict": ""}
