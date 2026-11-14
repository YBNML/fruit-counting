"""Experiment identity: git state, the environment fingerprint and the experiment id/record.
`build_record` assembles a `record.json` payload from pieces its caller
(the CLI) collects; this module does not touch the filesystem beyond reading `repo`'s git state.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import counting
from counting.config.eval_schema import EvalAppConfig
from counting.config.hashing import config_hash

_ENV_PACKAGES = ("numpy", "scipy", "pydantic", "torch", "torchvision")


def _git(repo: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True)


def counting_repo() -> Path:
    """The counting checkout this package is imported from (two levels above `src/counting`)."""
    return Path(counting.__file__).resolve().parents[2]


def git_state(repo: Path | None = None) -> dict:
    """Commit, dirty flag and (if dirty) a diff hash of `repo` (default: `counting_repo()`).

    `{"commit": None, "dirty": False, "diff_sha256": None}` when `repo` is not a git repo.
    Dirty also covers untracked files; `diff_sha256` hashes `git diff HEAD` plus each
    untracked file's (path, content sha256), so it changes when their content changes.
    """
    repo = Path(counting_repo() if repo is None else repo)
    rev = _git(repo, ["rev-parse", "HEAD"])
    if rev.returncode != 0:
        return {"commit": None, "dirty": False, "diff_sha256": None}
    commit = rev.stdout.decode().strip()
    status = _git(repo, ["status", "--porcelain"]).stdout
    dirty = bool(status.strip())
    diff_sha256 = None
    if dirty:
        data = _git(repo, ["diff", "HEAD"]).stdout
        untracked = _git(repo, ["ls-files", "--others", "--exclude-standard"]).stdout.decode()
        for rel in untracked.splitlines():
            if not rel:
                continue
            content_sha256 = hashlib.sha256((repo / rel).read_bytes()).hexdigest()
            data += rel.encode("utf-8") + content_sha256.encode("ascii")
        diff_sha256 = hashlib.sha256(data).hexdigest()
    return {"commit": commit, "dirty": dirty, "diff_sha256": diff_sha256}


def env_info() -> dict:
    """Python plus `_ENV_PACKAGES` versions (`None` if a package is not installed).

    Reads versions through `importlib.metadata`, so torch is never imported.
    """
    info: dict = {"python": platform.python_version()}
    for name in _ENV_PACKAGES:
        try:
            info[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info[name] = None
    return info


def experiment_id(*, config_hash: str, fingerprints: Sequence[str], code: Mapping, seed: int, extra: str = "") -> str:
    """12-char id: sha256 of config hash, sorted data fingerprints, code state and seed."""
    parts = [config_hash, *sorted(fingerprints), code["commit"] or "", str(code["dirty"]),
             code["diff_sha256"] or "", str(seed), extra]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def request_extra(
    kind: str, *, ladder_sha256: str | None = None, rungs: Sequence[str] = (), with_2x2: bool = False,
    subsets: bool = False, grid_name: str | None = None, grid_sha256: str | None = None,
    criteria_sha256: str | None = None,
) -> str:
    """The experiment id's `extra` string for one eval request `kind` (`run`, `ladder` or `sweep`).

    `"run"` carries no extra. `"ladder"` and `"sweep"` extras end in `:criteria:<sha16>` (`-` with
    no criteria file), so a different criteria table changes their experiment id. Any other `kind`
    raises `ValueError`.
    """
    if kind == "run":
        return ""
    if kind == "ladder":
        extra = f"ladder:{ladder_sha256[:16]}:{','.join(rungs)}:{with_2x2}"
        if subsets:
            extra += ":subsets"
    elif kind == "sweep":
        extra = f"sweep:{grid_name}:{grid_sha256[:16]}:{ladder_sha256[:16] if ladder_sha256 else '-'}"
    else:
        raise ValueError(f"unknown kind {kind!r}")
    criteria_part = criteria_sha256[:16] if criteria_sha256 else "-"
    return f"{extra}:criteria:{criteria_part}"


def build_record(
    *, cfg: EvalAppConfig, runs: Sequence[dict], code: dict, timings: dict, started: str, finished: str,
    extra: dict | None = None, detector: dict | None = None, pose_caches: Sequence[dict] | None = None,
    imu_files: Sequence[dict] | None = None, fusion: dict | None = None, lidar_files: Sequence[dict] | None = None,
) -> dict:
    """Assemble a `record.json` payload. Carries no machine name.

    `detector` (the CLI's account of the detection caches it scored) replaces the
    default `detector` entry, which names `cfg.stages.det` and its noise. `pose_caches`
    (the CLI's list of the pose caches the results read) adds a `pose_caches` entry;
    without it the record has none. Likewise `imu_files` (the IMU files the results
    read) adds an `imu_files` entry, `fusion` (the gtsam version, the CPU count and
    the sha256 of each fixed fusion settings file the results ran with) a `fusion` entry
    and `lidar_files` (the LiDAR files the results read) a `lidar_files` entry.
    """
    if detector is None:
        detector = {"name": cfg.stages.det,
                    "noise": None if cfg.detector.noise is None else cfg.detector.noise.model_dump(),
                    "weights_sha256": None, "cache_sha256": None, "pretrained": None}
    record = {
        "config_hash": config_hash(cfg),
        "data": list(runs),
        "code": code,
        "seeds": {"seed": cfg.seed, "noise_seeds": cfg.noise.seeds, "detector_seed": None},
        "detector": detector,
        "env": env_info(),
        "timings": timings,
        "started": started,
        "finished": finished,
        "extra": extra or {},
    }
    if pose_caches is not None:
        record["pose_caches"] = list(pose_caches)
    if imu_files is not None:
        record["imu_files"] = list(imu_files)
    if fusion is not None:
        record["fusion"] = fusion
    if lidar_files is not None:
        record["lidar_files"] = list(lidar_files)
    return record
