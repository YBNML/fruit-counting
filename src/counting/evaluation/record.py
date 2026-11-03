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


def git_state(repo: Path | None = None) -> dict:
    """Commit, dirty flag and (if dirty) a diff hash of `repo` (default: the counting checkout).

    `{"commit": None, "dirty": False, "diff_sha256": None}` when `repo` is not a git repo.
    Dirty also covers untracked files; `diff_sha256` hashes `git diff HEAD` plus each
    untracked file's (path, content sha256), so it changes when their content changes.
    """
    if repo is None:
        repo = Path(counting.__file__).resolve().parents[2]
    repo = Path(repo)
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


def build_record(
    *, cfg: EvalAppConfig, runs: Sequence[dict], code: dict, timings: dict, started: str, finished: str,
    extra: dict | None = None,
) -> dict:
    """Assemble a `record.json` payload. Carries no machine name."""
    return {
        "config_hash": config_hash(cfg),
        "data": list(runs),
        "code": code,
        "seeds": {"seed": cfg.seed, "noise_seeds": cfg.noise.seeds, "detector_seed": None},
        "detector": {"name": "oracle", "weights_sha256": None, "cache_sha256": None, "pretrained": None},
        "env": env_info(),
        "timings": timings,
        "started": started,
        "finished": finished,
        "extra": extra or {},
    }
