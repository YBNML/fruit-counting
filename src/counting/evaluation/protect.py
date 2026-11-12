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

A split=test request with a pose section passes three more checks. `check_test_request` also needs
a ladder file to be the committed pose ladder (`POSE_LADDER`) and pinned digests for every pose cache
and IMU file the config names (`pose.caches.<name>.digests`, `pose.imu_files.<name>.digests`); the
eval commands add the fixed fusion settings file to the files it needs tracked. With the rungs or
sweep cells (or, for `eval run`, the config itself), `check_test_pose_rungs` refuses the
rungs, grid bases and noise axes that run on val only (the ladder's `val_only`, `VAL_ONLY_NOISE_KEYS`
away from the judged rung's values) and a rung or cell that changes the pinned files, the reference
camera, the configuration name, the alignment knobs (`ALIGN_KNOBS`), the tracker prediction or the
fusion settings (only the odometry factor sigmas, one value across them). After the bundles are
opened, `check_test_trajectories` needs bundles rendered with body shake whose trajectory files match
trajectory.json, and `check_test_pose_caches` needs every pinned pose cache and IMU file with its
content digest and a test log row of the protected `counting pose vo` or `counting pose imu`.
A split=test request without a pose section skips the pose file and bundle checks, so
`check_test_pose_rungs` refuses any of its rungs or sweep cells that adds a pose section.

Every split=test run that writes its results adds one row (`LOG_HEADER` columns) to the
output folder's `test_log.csv` with `append_log_row`; `earlier_log_runs` counts the earlier
rows with the same (config hash, experiment id extra), the key a rerun warning is given for.
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
from typing import Mapping, Sequence

from counting.config.eval_schema import EvalAppConfig, SplitsConfig
from counting.data.formats.orchard_synth import SynthRun
from counting.detection.cached import cache_digest, cache_path
from counting.evaluation.record import git_state
from counting.evaluation.runner import ProtectionError, rung_config
from counting.evaluation.sweep import Grid, _leaf_paths, grid_points
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
    section and a `--ladder` file reads the committed pose ladder (`POSE_LADDER`); a config with
    `detector.cache` has `detector.digests`; with a `grid`, `check_test_grid(grid, ladder)`;
    and a config with a pose section has non-empty `digests` for every name of `pose.caches`
    and `pose.imu_files`. A config whose split is not test raises `ValueError`.
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
    if cfg.pose is not None and "--ladder" in files:
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
            for name, files in (getattr(cfg.pose, section) or {}).items():
                if not files.digests:
                    raise ProtectionError(f"split=test with pose.{section} needs pose.{section}.{name}.digests in "
                                          "the frozen config, one pinned file digest per bundle")
    return ProtectedRequest(config_rel=config_rel, code=code)


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
    `VAL_ONLY_NOISE_KEYS` factor away from the value of `JUDGED_RUNG` on `cfg` (a ladder without that
    rung is refused when this rule first needs it). Then all configs that set factor sigmas must set the
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
    judged = None   # the noise section of rung_config(cfg, ladder, JUDGED_RUNG), built on first use

    def judged_noise():
        nonlocal judged
        if judged is None:
            if not any(JUDGED_RUNG in (ladder.get(s) or {}) for s in ("rungs", "baselines")):
                raise ProtectionError(f"split=test: the pose ladder needs rung {JUDGED_RUNG}")
            judged = rung_config(cfg, ladder, JUDGED_RUNG).noise
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
