"""Pose caches and textured bundles for the pose cache and odometry tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

from counting.config.pose_schema import VoConfig
from counting.data.formats.orchard_synth import SynthRun
from counting.geometry.camera import Camera
from counting.geometry.noise import pass_key_code
from counting.geometry.rig import pose_matrix
from counting.odometry.cache import PassOdometry, pose_cache_path, write_pose_cache
from tests.unit.synthfake import write_run

# A straight apple bundle at 10 degrees, 320 x 240, one tree, 0.3 m steps: 14 frames, idx 0-6 side A
# (x -1.0 to 0.8) and 7-13 side B. Every pixel of a 10 degree camera sees the target row's plane.
TEXTURE_RUN = {"crop": "apple", "pitches": (10.0,), "size": (320, 240), "trees_per_row": 1, "step": 0.3}
TEST_VO = VoConfig(n_features=1000, depth_k=0.0)
_TEXEL_M = 0.002                                   # texture texel on the plane (m)
_PLANE_X = (-3.0, 3.5)                             # textured part of the plane: x (m) and z (m), row 0 at the top z
_PLANE_Z = (-0.5, 3.5)


def write_truth_pose_cache(run: SynthRun, cache_dir: Path, *, drop_first: bool = False,
                           meta: Mapping | None = None) -> tuple[Path, str]:
    """Write the true poses of `run` as a pose cache in `cache_dir`; return (its path, its digest).

    Per pass (`run.passes()`, frames in idx order), each frame's true camera-to-world pose relative
    to the pass's first cached frame: that first row is `np.eye(4)` as is, the others are
    `inv(first) @ T`, and every pair value is 0. With `drop_first`, each pass's first frame is left
    out, so its second frame is the first cached one. Meta: `run_id` and `fingerprint` of `run.meta`,
    updated with `meta`. Path: `pose_cache_path(cache_dir, run_id)`.
    """
    passes = {}
    for key, frames in run.passes().items():
        frames = frames[1:] if drop_first else frames
        first = pose_matrix(frames[0].camera)
        rel = [np.eye(4)] + [np.linalg.inv(first) @ pose_matrix(f.camera) for f in frames[1:]]
        n = len(frames)
        passes[pass_key_code(key)] = PassOdometry(
            idx=np.array([f.idx for f in frames], np.int32), rel_c2w=np.stack(rel),
            inliers=np.zeros(n, np.int32), failed=np.zeros(n, np.uint8), prior=np.zeros(n, np.uint8))
    path = pose_cache_path(cache_dir, run.meta.run_id)
    digest = write_pose_cache(path, passes, {"run_id": run.meta.run_id, "fingerprint": run.meta.fingerprint,
                                             **(meta or {})})
    return path, digest


@lru_cache(maxsize=1)
def _texture() -> np.ndarray:
    """Blurred uniform noise (2000 x 3250 texels, rows from the top z down), stretched to 0-255."""
    rows = round((_PLANE_Z[1] - _PLANE_Z[0]) / _TEXEL_M)
    cols = round((_PLANE_X[1] - _PLANE_X[0]) / _TEXEL_M)
    t = gaussian_filter(np.random.default_rng(7).random((rows, cols)) * 255.0, 2.0)
    return (t - t.min()) / (t.max() - t.min()) * 255.0


def paint_plane(root: Path) -> None:
    """Overwrite every frame's RGB and depth16 image of the bundle at `root` with the target row's centre plane.

    Each integer pixel (u, v) of a frame's camera casts the ray d = R_c2w ((u - cx) / fx, (v - cy) / fy, 1)
    from the camera centre C; it meets the plane y = row_y at t = (row_y - C_y) / d_y, which is also the
    plane depth. A ray with a finite t in (0.05, 65) gets the bilinear texture value at its hit point and
    the depth round(1000 t) mm; any other pixel gets grey 128 and depth 0 (no return).
    """
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    row_y = manifest["spec"]["row_ys"][manifest["args"]["target_row"]]
    for path in sorted(root.glob("*_frame.json")):
        frame = json.loads(path.read_text(encoding="utf-8"))
        pose = frame["pose"]
        cam = Camera.from_blender(pose["matrix_world"], pose["width"], pose["height"], pose["hfov_deg"])
        (fx, _, cx), (_, fy, cy) = cam.K[0], cam.K[1]
        u, v = np.meshgrid(np.arange(cam.width, dtype=float), np.arange(cam.height, dtype=float))
        d = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1) @ cam.R_c2w.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (row_y - cam.C[1]) / d[..., 1]
        hit = np.isfinite(t) & (t > 0.05) & (t < 65.0)
        t = np.where(hit, t, 0.0)
        X = cam.C + t[..., None] * d
        rows = (_PLANE_Z[1] - X[..., 2]) / _TEXEL_M
        cols = (X[..., 0] - _PLANE_X[0]) / _TEXEL_M
        grey = map_coordinates(_texture(), [rows.ravel(), cols.ravel()], order=1, mode="nearest")
        grey = np.where(hit, grey.reshape(t.shape), 128.0)
        rgb = np.repeat(np.clip(np.rint(grey), 0, 255).astype(np.uint8)[..., None], 3, axis=2)
        depth = np.where(hit, np.rint(t * 1000.0), 0).astype(np.uint16)
        Image.fromarray(rgb).save(root / f"{frame['idx']:06d}_rgb.png", compress_level=1)
        Image.fromarray(depth).save(root / f"{frame['idx']:06d}_depth16.png", compress_level=1)


def textured_run(root: Path, *, shake: bool = False) -> Path:
    """`write_run(root, **TEXTURE_RUN, shake=shake)` with its images painted by `paint_plane`; returns the bundle."""
    root = write_run(root, **TEXTURE_RUN, shake=shake)
    paint_plane(root)
    return root
