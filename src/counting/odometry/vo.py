"""Frame-to-frame RGB-D visual odometry: one bundle's relative camera poses, pass by pass, into a pose cache.

Input. `VoFrame` is all the odometry reads of a frame: its idx, its pass, the intrinsics `K`, the image
size, the RGB image and the 16-bit depth image (mm, plane depth, 0 for no return). It holds no pose:
no frame pose, planned pose, frame time or position number reaches this module.

Features. A frame's RGB image is read with PIL, turned grey (`cv2.cvtColor`), equalised with CLAHE and
given ORB keypoints and descriptors. A keypoint's depth is the nearest depth within `depth_nb` pixels
of its rounded pixel (a (2r+1) x (2r+1) erosion of the depth image); a keypoint without a return or
deeper than `depth_max_m` has no 3D point. Only the previous frame's features are kept.

Pairs. The pair (a -> b) of consecutive frames of a pass gives the motion `T_ba` (a's camera
coordinates to b's) in four steps:

1. Global matching: two nearest ORB descriptors with the ratio test `match_ratio`, then an EPnP
   RANSAC fit (`ransac_px`) from a's 3D points to b's keypoints, refined by Levenberg-Marquardt.
2. When that fit has fewer than `min_inliers` inliers, the motion prior stands in and the pair is
   marked `prior`: the motion of the pass's previous pair, or no motion for its first pair.
3. Guided matching: a's 3D points projected into b with that pose; each takes the nearest descriptor
   within `guided_radius_px` (`prior_radius_px` after the prior), at most `guided_max_hamming` bits
   and below `guided_ratio` times the second nearest; a keypoint of b keeps its nearest match only.
4. A final EPnP RANSAC fit (`final_px`) with the refinement. With fewer than `min_inliers` inliers
   the pair keeps the pose of step 1 or 2 and is marked `failed`. `inliers` is this fit's count.

A pass starts at the identity at its first frame and chains its pairs: `rel[k] = rel[k-1] @ inv(T_ba)`.

Repeatability. `run_vo` runs OpenCV on `threads` threads and, before each pass, seeds OpenCV's global
random generator from (`seed`, pass code) (`cv2.setRNGSeed`, as the design asks). The RANSAC fits do not
draw from it: OpenCV's solvePnPRansac (EPnP, not USAC) seeds its own generator with a fixed value on every
call (checked with OpenCV 5.0.0), so a fit depends on its inputs only, and `seed` today changes the config
hash (the cache folder), not the arrays. With `depth_k` above 0, a's keypoint depths z get Gaussian noise
of sigma `depth_k` z^2 for the pair (a -> b), drawn from a generator keyed by (`noise_seed`,
`DEPTH_NOISE_SALT`, a's idx). The same settings and images give the same arrays, so a cache holds one draw
of that noise.

cv2 and `scipy.spatial.cKDTree` are imported inside `run_vo`, so importing this module does not load cv2.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from counting.config.pose_schema import VoConfig
from counting.data.formats.orchard_synth import PassKey
from counting.geometry.noise import pass_key_code
from counting.odometry.cache import PassOdometry, write_pose_cache

# Part of the depth noise key (the ASCII of "VODN"), so odometry draws never share a key with the
# evaluation noise.
DEPTH_NOISE_SALT = 0x564F444E
_DEPTH_MODES = ("I;16", "I;16B", "I")   # 16-bit grayscale PNG modes
_MIN_FRONT_M = 0.1                      # a 3D point is projected into b only this far in front of it
_NO_SECOND = 999                        # Hamming distance standing in for a missing second candidate


@dataclass(frozen=True, eq=False)
class VoFrame:
    """One frame as the odometry sees it: idx, pass, intrinsics (OpenCV, float64 3x3), image size and
    the paths of its RGB and depth16 images. No pose."""

    idx: int
    pass_key: PassKey
    K: np.ndarray
    width: int
    height: int
    rgb_path: Path
    depth_path: Path


@dataclass(frozen=True, eq=False)
class _Features:
    """A frame's keypoints `xy` float32 (n, 2), ORB descriptors uint8 (n, 32), their bits uint8 (n, 256)
    and keypoint depths `z` float32 (n,) in m, NaN without a 3D point."""

    xy: np.ndarray
    desc: np.ndarray
    bits: np.ndarray
    z: np.ndarray


@dataclass(frozen=True, eq=False)
class _Tools:
    """The OpenCV module and objects and the k-d tree class of one `run_vo` call."""

    cv2: Any
    orb: Any
    clahe: Any
    matcher: Any
    tree: Any


def run_vo(frames: Sequence[VoFrame], cfg: VoConfig) -> dict[tuple[int, int, int], PassOdometry]:
    """The odometry of `frames`, one `PassOdometry` per pass code (`pass_key_code`).

    Passes come in the order of their first frame in `frames`, and a pass runs over its frames in idx
    order; every frame given is used. A frame idx given twice raises `ValueError`, before any image is
    read. An image whose size is not the frame's, or a depth image that is not 16-bit grayscale,
    raises `ValueError` naming the file.
    """
    passes = _passes(frames)

    import cv2
    from scipy.spatial import cKDTree

    cv2.setNumThreads(cfg.threads)
    tools = _Tools(
        cv2=cv2,
        orb=cv2.ORB_create(nfeatures=cfg.n_features, scaleFactor=cfg.orb_scale, nlevels=cfg.orb_levels,
                           fastThreshold=cfg.fast_threshold),
        clahe=cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tile, cfg.clahe_tile)),
        matcher=cv2.BFMatcher(cv2.NORM_HAMMING),
        tree=cKDTree,
    )
    return {code: _run_pass(code, pass_frames, cfg, tools) for code, pass_frames in passes.items()}


def vo_to_cache(frames: Sequence[VoFrame], cfg: VoConfig, path: Path, *, meta: Mapping) -> dict:
    """Run the odometry of `frames` and write it as the pose cache `path` with `meta`; return the summary.

    The summary is also written next to the cache as `path.with_suffix(".json")` (atomically, sorted
    keys): `file` (the cache's file name), `digest`, `meta`, `n_frames`, `passes` (per pass: `pass`
    code, `frames`, `pairs`, `failed` and `prior` pair counts, `inliers_min`, None without a pair)
    and `seconds` (the odometry and the cache write). `meta` goes into the cache as given
    (`write_pose_cache`), so it holds no time and the digest depends on the arrays and `meta` only.
    When the cache or its summary exists, `FileExistsError` is raised before the odometry runs.
    """
    path = Path(path)
    sidecar = path.with_suffix(".json")
    for p in (path, sidecar):
        if p.exists():
            raise FileExistsError(f"{p}: pose cache exists; remove it first to make it again")
    start = time.perf_counter()
    passes = run_vo(frames, cfg)
    digest = write_pose_cache(path, passes, meta)
    summary = {
        "file": path.name,
        "digest": digest,
        "meta": dict(meta),
        "n_frames": sum(len(p.idx) for p in passes.values()),
        "passes": [_pass_summary(code, p) for code, p in passes.items()],
        "seconds": time.perf_counter() - start,
    }
    _write_json(sidecar, summary)
    return summary


def _pass_summary(code: tuple[int, int, int], p: PassOdometry) -> dict:
    n = len(p.idx)
    return {"pass": [int(v) for v in code], "frames": n, "pairs": n - 1, "failed": int(p.failed.sum()),
            "prior": int(p.prior.sum()), "inliers_min": int(p.inliers[1:].min()) if n > 1 else None}


def _write_json(path: Path, obj: dict) -> None:
    """Write `obj` to `path` as JSON (sorted keys, indent 2) through a fsynced sibling `.tmp` file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, sort_keys=True, indent=2) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _passes(frames: Sequence[VoFrame]) -> dict[tuple[int, int, int], list[VoFrame]]:
    """`frames` grouped by pass code in the order of each pass's first frame, frames in idx order."""
    seen: set[int] = set()
    out: dict[tuple[int, int, int], list[VoFrame]] = {}
    for f in frames:
        if f.idx in seen:
            raise ValueError(f"frame idx {f.idx} is given twice")
        seen.add(f.idx)
        out.setdefault(pass_key_code(f.pass_key), []).append(f)
    return {code: sorted(fs, key=lambda f: f.idx) for code, fs in out.items()}


def _cv_seed(seed: int, code: tuple[int, int, int]) -> int:
    """OpenCV's random seed of a pass: (seed, pass code) folded below 2**31 (`cv2.setRNGSeed` takes a C int)."""
    return int(np.random.SeedSequence([seed, *code]).generate_state(1)[0] >> 1)


def _run_pass(code: tuple[int, int, int], frames: list[VoFrame], cfg: VoConfig, tools: _Tools) -> PassOdometry:
    """One pass's chained poses and pair values, its first row the identity with 0, 0, 0."""
    n = len(frames)
    rel = np.empty((n, 4, 4))
    rel[0] = np.eye(4)
    inliers = np.zeros(n, np.int32)
    failed = np.zeros(n, np.uint8)
    prior = np.zeros(n, np.uint8)
    tools.cv2.setRNGSeed(_cv_seed(cfg.seed, code))
    motion = np.eye(4)                                  # the prior of the first pair: no motion
    a = _features(frames[0], cfg, tools)
    for k in range(1, n):
        b = _features(frames[k], cfg, tools)
        za = a.z
        if cfg.depth_k > 0:
            rng = np.random.default_rng([cfg.noise_seed, DEPTH_NOISE_SALT, frames[k - 1].idx])
            za = za + rng.standard_normal(za.shape).astype(np.float32) * (cfg.depth_k * za * za)
        motion, prior[k], failed[k], inliers[k] = _pair(a, b, za, frames[k - 1].K, frames[k].K, motion, cfg, tools)
        rel[k] = rel[k - 1] @ _inverse(motion)
        a = b
    return PassOdometry(idx=np.array([f.idx for f in frames], np.int32), rel_c2w=rel, inliers=inliers,
                        failed=failed, prior=prior)


def _pair(a: _Features, b: _Features, za: np.ndarray, K_a: np.ndarray, K_b: np.ndarray, motion: np.ndarray,
          cfg: VoConfig, tools: _Tools) -> tuple[np.ndarray, int, int, int]:
    """(T_ba, prior, failed, inliers) of the pair (a -> b); `za` are a's keypoint depths for this pair and
    `motion` is the prior."""
    ok = np.isfinite(za)
    matches = _match_global(a.desc, b.desc, cfg.match_ratio, tools.matcher)
    matches = matches[ok[matches[:, 0]]] if len(matches) else matches
    first_guess = None
    if len(matches) >= cfg.min_inliers:
        first_guess, _ = _solve(_backproject(a.xy[matches[:, 0]], za[matches[:, 0]], K_a),
                                b.xy[matches[:, 1]].astype(np.float64), K_b, cfg.ransac_px, cfg, tools.cv2)
    used_prior = first_guess is None
    if used_prior:
        first_guess, radius = motion, cfg.prior_radius_px
    else:
        radius = cfg.guided_radius_px
    ia = np.flatnonzero(ok)
    guided = _match_guided(_backproject(a.xy[ia], za[ia], K_a), ia, a.bits[ia], b, tools.tree(b.xy), first_guess,
                           K_b, radius, cfg)
    final_pose, n_final = _solve(_backproject(a.xy[guided[:, 0]], za[guided[:, 0]], K_a),
                                 b.xy[guided[:, 1]].astype(np.float64), K_b, cfg.final_px, cfg, tools.cv2)
    if final_pose is None:
        return first_guess, int(used_prior), 1, n_final
    return final_pose, int(used_prior), 0, n_final


def _features(frame: VoFrame, cfg: VoConfig, tools: _Tools) -> _Features:
    cv2 = tools.cv2
    grey = tools.clahe.apply(cv2.cvtColor(_read_rgb(frame), cv2.COLOR_RGB2GRAY))
    kps, desc = tools.orb.detectAndCompute(grey, None)
    if desc is None:
        desc = np.zeros((0, 32), np.uint8)
    xy = np.array([kp.pt for kp in kps], np.float32).reshape(-1, 2)
    depth = _read_depth_m(frame)
    if cfg.depth_nb > 0:                                # the nearest depth around each pixel
        side = 2 * cfg.depth_nb + 1
        depth = cv2.erode(depth, np.ones((side, side), np.uint8))
    iu = np.clip(np.rint(xy[:, 0]).astype(int), 0, frame.width - 1)
    iv = np.clip(np.rint(xy[:, 1]).astype(int), 0, frame.height - 1)
    z = depth[iv, iu].astype(np.float32)
    z[~np.isfinite(z) | (z > cfg.depth_max_m)] = np.nan
    return _Features(xy=xy, desc=desc, bits=np.unpackbits(desc, axis=1), z=z)


def _check_size(path: Path, w: int, h: int, frame: VoFrame) -> None:
    if (w, h) != (frame.width, frame.height):
        raise ValueError(f"{path}: image is {w}x{h}, the frame says {frame.width}x{frame.height}")


def _read_rgb(frame: VoFrame) -> np.ndarray:
    """The frame's RGB image as uint8 (h, w, 3)."""
    path = frame.rgb_path
    try:
        with Image.open(path) as im:
            rgb = np.array(im.convert("RGB"))
    except (OSError, SyntaxError) as e:
        raise ValueError(f"{path}: unreadable image ({e})") from None
    _check_size(path, rgb.shape[1], rgb.shape[0], frame)
    return rgb


def _read_depth_m(frame: VoFrame) -> np.ndarray:
    """The frame's depth16 image as float32 m, no return (0) as inf."""
    path = frame.depth_path
    try:
        with Image.open(path) as im:
            mode = im.mode
            raw = np.array(im, dtype=np.uint16) if mode in _DEPTH_MODES else None
    except (OSError, SyntaxError) as e:
        raise ValueError(f"{path}: unreadable image ({e})") from None
    if raw is None:
        raise ValueError(f"{path}: expected a 16-bit grayscale PNG, got mode {mode}")
    _check_size(path, raw.shape[1], raw.shape[0], frame)
    depth = raw.astype(np.float32) / 1000.0
    depth[depth <= 0] = np.inf
    return depth


def _match_global(desc_a: np.ndarray, desc_b: np.ndarray, ratio: float, matcher: Any) -> np.ndarray:
    """Index pairs (n, 2) of a's and b's descriptors that pass the two-nearest ratio test."""
    if len(desc_a) < 2 or len(desc_b) < 2:
        return np.zeros((0, 2), int)
    out = [(m[0].queryIdx, m[0].trainIdx) for m in matcher.knnMatch(desc_a, desc_b, k=2)
           if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    return np.array(out, int).reshape(-1, 2)


def _backproject(xy: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame 3D points of pixels `xy` at plane depths `z`."""
    return np.stack([(xy[:, 0] - K[0, 2]) / K[0, 0] * z, (xy[:, 1] - K[1, 2]) / K[1, 1] * z, z], 1)


def _solve(X: np.ndarray, x: np.ndarray, K: np.ndarray, px: float, cfg: VoConfig,
           cv2: Any) -> tuple[np.ndarray | None, int]:
    """(T, inlier count) of an EPnP RANSAC fit of points `X` to pixels `x` refined by Levenberg-Marquardt
    on its inliers; T is None when there are fewer than `min_inliers` points or inliers."""
    if len(X) < cfg.min_inliers:
        return None, 0
    ok, rvec, tvec, inl = cv2.solvePnPRansac(X, x, K, None, iterationsCount=cfg.ransac_iters, reprojectionError=px,
                                             confidence=cfg.ransac_confidence, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < cfg.min_inliers:
        return None, 0 if inl is None else len(inl)
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(X[inl], x[inl], K, None, rvec, tvec)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    return pose, len(inl)


def _match_guided(Xa: np.ndarray, ia: np.ndarray, bits_a: np.ndarray, b: _Features, tree_b: Any, T_ba: np.ndarray,
                  K: np.ndarray, radius: float, cfg: VoConfig) -> np.ndarray:
    """Index pairs (n, 2) of a's keypoints `ia` (3D points `Xa`, descriptor bits `bits_a`) and b's keypoints:
    a's points moved by `T_ba` and projected into b, each matched to the nearest descriptor among b's
    keypoints within `radius` px, kept when at most `guided_max_hamming` bits and below `guided_ratio`
    times the second nearest; a keypoint of b keeps only its nearest match."""
    Xb = Xa @ T_ba[:3, :3].T + T_ba[:3, 3]
    front = Xb[:, 2] > _MIN_FRONT_M
    uv = np.full((len(Xb), 2), -1e9)
    uv[front] = (Xb[front, :2] / Xb[front, 2:3]) @ np.diag([K[0, 0], K[1, 1]]) + K[:2, 2]
    cand = tree_b.query_ball_point(uv, r=radius)
    qa = np.repeat(np.arange(len(cand)), [len(c) for c in cand])
    if len(qa) == 0:
        return np.zeros((0, 2), int)
    qb = np.concatenate([np.asarray(c, int) for c in cand if len(c)])
    ham = (bits_a[qa] != b.bits[qb]).sum(1)
    order = np.lexsort((ham, qa))
    qa, qb, ham = qa[order], qb[order], ham[order]
    pos = np.flatnonzero(np.r_[True, qa[1:] != qa[:-1]])        # each a point's nearest candidate
    nxt = pos + 1
    has_second = (nxt < len(qa)) & (np.r_[qa[1:], -1][pos] == qa[pos])
    second = np.where(has_second, ham[np.minimum(nxt, len(ham) - 1)], _NO_SECOND)
    keep = (ham[pos] <= cfg.guided_max_hamming) & (ham[pos] < cfg.guided_ratio * second)
    pa, pb = qa[pos][keep], qb[pos][keep]
    o = np.lexsort((ham[pos][keep], pb))                         # b's nearest match first
    pa, pb = pa[o], pb[o]
    u = np.r_[True, pb[1:] != pb[:-1]]
    return np.stack([ia[pa[u]], pb[u]], 1)


def _inverse(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid motion, its last row exactly 0, 0, 0, 1."""
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out
