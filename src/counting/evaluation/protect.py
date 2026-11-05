"""Test split protection: the checks a split=test request must pass before it may read a test bundle.

Two checks, each raising `ProtectionError` (exit code 1) at the first broken rule:

- `check_test_request` before any bundle is opened: the config file sits under
  `configs/eval/frozen/` in the counting repository, there is no `--set` or `--run`,
  the work tree is clean and committed, the config and every file option are tracked
  in the repository, and a grid changes only noise settings on ladder rungs or baselines.
- `check_test_bundles` after the bundles are opened and before `prepare`: every bundle
  has a test seed and an `orchard_synth.commit` sidecar, and all bundles share one commit.

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
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from counting.config.eval_schema import EvalAppConfig, SplitsConfig
from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.record import git_state
from counting.evaluation.runner import ProtectionError
from counting.evaluation.sweep import Grid, _leaf_paths, grid_points

FROZEN_DIR = ("configs", "eval", "frozen")
DETECT_FROZEN_DIR = ("configs", "detect", "frozen")
LOG_HEADER = ("date", "experiment_id", "kind", "config", "config_hash", "extra", "fingerprints", "commit",
              "criteria_sha256", "verdict")


@dataclass(frozen=True)
class ProtectedRequest:
    """A split=test request that passed `check_test_request`, or a test-bundle detect request that
    passed `check_test_detect`.

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
    option, such as `--splits`) is a file git tracks in `repo`; and, with a `grid`,
    `check_test_grid(grid, ladder)`. A config whose split is not test raises `ValueError`.
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
    if grid is not None:
        check_test_grid(grid, ladder)
    return ProtectedRequest(config_rel=config_rel, code=code)


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
