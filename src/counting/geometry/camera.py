"""Blender-to-OpenCV camera model: pose conversion, projection, and back-projection."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, degrees, radians, tan

import numpy as np


@dataclass(frozen=True, eq=False)
class Camera:
    """Pinhole camera in OpenCV convention (+X right, +Y down, +Z forward).

    `R_c2w` and `C` describe the camera pose in world coordinates: a point
    `X_c` in camera coordinates maps to world coordinates as
    `R_c2w @ X_c + C`.
    """

    K: np.ndarray
    R_c2w: np.ndarray
    C: np.ndarray
    width: int
    height: int

    @classmethod
    def from_blender(cls, matrix_world, width: int, height: int, hfov_deg: float) -> "Camera":
        """Build a Camera from a Blender camera-to-world matrix and horizontal FOV.

        `matrix_world` follows Blender convention: the camera looks down its
        local -Z axis with +Y as image-up. This converts the pose to OpenCV
        convention (+X right, +Y down, +Z forward) and builds a distortion-free
        pinhole intrinsic matrix with pixel centers at integer coordinates.
        """
        m = np.asarray(matrix_world, dtype=float)
        fx = (width / 2) / tan(radians(hfov_deg) / 2)
        cx = width / 2 - 0.5
        cy = height / 2 - 0.5
        K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])
        R_c2w = m[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        C = m[:3, 3].copy()
        return cls(K=K, R_c2w=R_c2w, C=C, width=width, height=height)

    def project(self, Xw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to OpenCV pixel coordinates and plane depth.

        Returns `(uv, z)`: `uv` has `u` increasing right and `v` increasing
        down; `z` is the camera-plane depth (distance along the optical axis,
        not ray distance).
        """
        Xc = (Xw - self.C) @ self.R_c2w
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        z = Xc[:, 2]
        u = fx * Xc[:, 0] / z + cx
        v = fy * Xc[:, 1] / z + cy
        return np.stack([u, v], axis=1), z

    def backproject(self, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Lift pixel coordinates at known plane depth `z` to world points."""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x_c = (uv[:, 0] - cx) / fx * z
        y_c = (uv[:, 1] - cy) / fy * z
        Xc = np.stack([x_c, y_c, z], axis=1)
        return Xc @ self.R_c2w.T + self.C

    def ray_dir(self, uv: np.ndarray) -> np.ndarray:
        """Unit-length world-space ray directions through pixel coordinates."""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x_c = (uv[:, 0] - cx) / fx
        y_c = (uv[:, 1] - cy) / fy
        dir_c = np.stack([x_c, y_c, np.ones_like(x_c)], axis=1)
        dir_w = dir_c @ self.R_c2w.T
        return dir_w / np.linalg.norm(dir_w, axis=1, keepdims=True)

    def in_image(self, uv: np.ndarray) -> np.ndarray:
        """Whether each pixel coordinate falls inside the image bounds."""
        u, v = uv[:, 0], uv[:, 1]
        return (u >= -0.5) & (u < self.width - 0.5) & (v >= -0.5) & (v < self.height - 0.5)

    def perturbed(self, rng: np.random.Generator, trans_sigma_m: float, rot_sigma_deg: float,
                  bias_m: np.ndarray | None = None) -> "Camera":
        """A copy of this camera with its pose perturbed by Gaussian translation and rotation noise.

        The new center is `C + trans_sigma_m * t (+ bias_m)` and the new orientation rotates this
        camera's axes by a Rodrigues rotation vector `radians(rot_sigma_deg) * w`, both `t` and `w`
        drawn from `rng`. `K`, `width` and `height` are unchanged.
        """
        t, w = rng.standard_normal(3), rng.standard_normal(3)
        C = self.C + trans_sigma_m * t
        if bias_m is not None:
            C = C + bias_m
        omega = radians(rot_sigma_deg) * w
        theta = np.linalg.norm(omega)
        if theta == 0.0:
            R_c2w = self.R_c2w.copy()
        else:
            axis = omega / theta
            Kx = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
            R_rel = np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * (Kx @ Kx)
            R_c2w = self.R_c2w @ R_rel
        return Camera(K=self.K, R_c2w=R_c2w, C=C, width=self.width, height=self.height)


def box_centers(boxes: np.ndarray) -> np.ndarray:
    """Convert pixel-edge boxes `[x0, y0, x1, y1)` to OpenCV pixel-center coordinates."""
    boxes = np.asarray(boxes, dtype=float)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 - 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 - 0.5
    return np.stack([cx, cy], axis=1)


def pitch_from_blender(matrix_world) -> float:
    """Recover the camera pitch, in degrees, from a Blender camera-to-world matrix."""
    m = np.asarray(matrix_world, dtype=float)
    return degrees(asin(-m[2][2]))
