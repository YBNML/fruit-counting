"""Rig model: cameras at several pitches fixed to one body. The rig is defined here and only here.

Poses are float64 4x4 camera-to-world matrices `[[R_c2w, C], [0, 1]]`, the same pose a `Camera`
holds (`pose_matrix`, `with_pose`).

Body frame B: x along the direction of travel, y to the left, z up. Its origin is the camera
center: the cameras of the rig share one center, so the lever arm between camera and body is zero.

`X_{B->q}` (`pitch_extrinsic(q)`) is the known extrinsic of the camera tilted up by q degrees, the
camera-to-body transform: it maps OpenCV camera coordinates (+X right, +Y down, +Z forward) to body
coordinates. The camera looks to the right of the direction of travel with its image x axis along
-x of the body, and its translation is zero. A camera pose is the body pose times the extrinsic,
`T(q) = B @ X_{B->q}`, and `X_{r->q} = X_{B->r}^-1 @ X_{B->q}` (`relative_extrinsic`) turns the
camera r into the camera q by q - r about the camera x axis.

The believed extrinsics carry calibration errors, as right-multiplied rigid transforms:

- `Xhat_{B->r} = X_{B->r} @ dX_imu` for the reference camera r (`dX_imu`: camera-to-IMU error),
- `Xhat_{B->q} = Xhat_{B->r} @ X_{r->q} @ dX_q` for the camera q (`dX_q`: rig calibration error;
  the reference camera has none).

Taking the body from the reference camera pose, `B = T(r) @ Xhat_{B->r}^-1`
(`body_from_reference`), gives `T(q) = B @ Xhat_{B->q} = T(r) @ X_{r->q} @ dX_q` (`rig_pose`):
`dX_imu` cancels.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from counting.geometry.camera import Camera


def _inverse(T: np.ndarray) -> np.ndarray:
    """Inverse of the rigid 4x4 transform `T`."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ t)
    return out


def pose_matrix(cam: Camera) -> np.ndarray:
    """The pose of `cam` as a 4x4 camera-to-world matrix `[[R_c2w, C], [0, 1]]`."""
    T = np.eye(4)
    T[:3, :3] = cam.R_c2w
    T[:3, 3] = cam.C
    return T


def with_pose(cam: Camera, T: np.ndarray) -> Camera:
    """`cam` moved to the camera-to-world pose `T`: `K`, `width` and `height` are kept, `R_c2w`
    and `C` are copies of `T`'s rotation and translation."""
    T = np.asarray(T, dtype=np.float64)
    return Camera(K=cam.K, R_c2w=T[:3, :3].copy(), C=T[:3, 3].copy(), width=cam.width, height=cam.height)


def pitch_extrinsic(pitch_deg: float) -> np.ndarray:
    """`X_{B->q}`: the camera-to-body transform of the camera tilted up by `pitch_deg` (q).

    Rotation `[[-1, 0, 0], [0, -sin q, -cos q], [0, -cos q, sin q]]`, translation zero.
    """
    q = np.radians(float(pitch_deg))
    s, c = np.sin(q), np.cos(q)
    X = np.eye(4)
    X[:3, :3] = [[-1.0, 0.0, 0.0], [0.0, -s, -c], [0.0, -c, s]]
    return X


def relative_extrinsic(ref_deg: float, pitch_deg: float) -> np.ndarray:
    """`X_{r->q} = X_{B->r}^-1 @ X_{B->q}`: the camera q in the frame of the camera r, a turn by
    q - r about the camera x axis with zero translation."""
    return _inverse(pitch_extrinsic(ref_deg)) @ pitch_extrinsic(pitch_deg)


def believed_extrinsic(pitch_deg: float, ref_deg: float, imu_error: np.ndarray | None = None,
                       rig_error: np.ndarray | None = None) -> np.ndarray:
    """`Xhat_{B->q} = X_{B->r} @ dX_imu @ X_{r->q} @ dX_q`, the extrinsic of the camera q believed
    with the reference camera r.

    `imu_error` is `dX_imu` and `rig_error` is `dX_q`, both rigid 4x4 transforms; None is the
    identity. The reference camera (q = r) takes `rig_error=None`.
    """
    X = pitch_extrinsic(ref_deg)
    if imu_error is not None:
        X = X @ np.asarray(imu_error, dtype=np.float64)
    X = X @ relative_extrinsic(ref_deg, pitch_deg)
    if rig_error is not None:
        X = X @ np.asarray(rig_error, dtype=np.float64)
    return X


def body_from_reference(ref_pose: np.ndarray, ref_believed: np.ndarray) -> np.ndarray:
    """The body pose `B = T(r) @ Xhat_{B->r}^-1` from the reference camera pose `ref_pose` and its
    believed extrinsic `ref_believed`."""
    return np.asarray(ref_pose, dtype=np.float64) @ _inverse(np.asarray(ref_believed, dtype=np.float64))


def rig_pose(body: np.ndarray, believed: np.ndarray) -> np.ndarray:
    """The camera pose `T(q) = B @ Xhat_{B->q}` from the body pose and the believed extrinsic."""
    return np.asarray(body, dtype=np.float64) @ np.asarray(believed, dtype=np.float64)


def rigid(rotvec_rad: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """The rigid transform `[[Exp(rotvec_rad), trans], [0, 1]]`; a zero rotation vector gives exactly
    the identity rotation."""
    rotvec = np.asarray(rotvec_rad, dtype=np.float64)
    T = np.eye(4)
    if rotvec.any():
        T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = np.asarray(trans, dtype=np.float64)
    return T


def interpolate_pose(times: np.ndarray, poses: np.ndarray, t: float) -> np.ndarray:
    """The pose at time `t` from `poses` (N x 4 x 4) at ascending `times` (N).

    Between two neighbouring frames the position is linear in time and the rotation is the slerp
    of the two rotations. Before the first time it is the first pose, after the last time the last.
    """
    times = np.asarray(times, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    if t <= times[0]:
        return poses[0].copy()
    if t >= times[-1]:
        return poses[-1].copy()
    j = int(np.searchsorted(times, t, side="right"))
    a, b = poses[j - 1], poses[j]
    w = (t - times[j - 1]) / (times[j] - times[j - 1])
    turn = Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec()
    out = np.eye(4)
    out[:3, :3] = a[:3, :3] @ rigid(w * turn, np.zeros(3))[:3, :3]
    out[:3, 3] = (1.0 - w) * a[:3, 3] + w * b[:3, 3]
    return out
