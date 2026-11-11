"""Pose quality: the cameras a pose source gave against the true cameras, and the per-pass truth fit.

This module is on the evaluation side: it reads the true camera poses. Poses are 4x4
camera-to-world matrices (`pose_matrix`). For one pass with true poses T_i and used poses U_i
(i = 0 ... n-1, in frame order):

- RPE: for every neighbouring pair, E = (T_i^-1 T_{i+1})^-1 (U_i^-1 U_{i+1}); its translation
  length and rotation angle, pooled over the pairs of every pass (p50, p90, max).
- In-pass drift (`in_pass`): E = (T_0^-1 T_{n-1})^-1 (U_0^-1 U_{n-1}), the end error relative to
  the first frame. `pos_frac` is its translation length over the pass length |C_T(n-1) - C_T(0)|.
- Start error (`start`): E = T_0^-1 U_0. World end error (`world_end`): E = T_{n-1}^-1 U_{n-1}.

Reference camera frame. With `ref_pitch` r, RPE and `in_pass` of a pass at pitch q (more than
1e-6 from r) are measured on poses moved to the camera r: the true poses by the true extrinsic,
T' = T X_{r->q}^-1, and the used poses by the believed one, U' = U dX_q^-1 X_{r->q}^-1, where dX_q
is the rig calibration error of `rig_errors` at q (identity when there is none). A rig
calibration error then leaves the in-pass drift; `in_pass_cam` keeps the in-pass drift in the
pass's own camera frame. Without `ref_pitch`, and for the pass at r, nothing is moved. `start`,
`world_end` and the misalignment between passes stay in each pass's own camera frame.

Components. E is in camera coordinates; M, the rotation of `pitch_extrinsic` (of r for a moved
pass, of q otherwise), turns it to body axes once: e = M E_t gives `along_m` (travel), `side_m`
(left) and `height_m` (up), w = M rotvec(E_R) gives `yaw_deg` (w_z) and `tilt_deg`
(hypot(w_x, w_y), never negative). `pos_m` is |E_t| and `rot_deg` the rotation angle.

Between passes. Every point of a grid on the target row (`interpass_points`) is checked in each
pass at the frame whose true camera sees it nearest the principal point (z > 0 and inside the
image; the earlier frame on a tie). The point in that true camera's coordinates is lifted back to
the world with the used camera, then moved by the pass's alignment transform at the x of that frame's
used camera when there is one; delta = lifted point - point. Two passes that both see a point differ by
|delta_p - delta_q|: `same_side` pairs share side and aisle, `both_sides` pairs are on different sides.

After the alignment. `world_end_aligned` is the world end error after the pass's alignment transform A at
the x of the last frame's used camera: E = T_{n-1}^-1 A U_{n-1}, in the pass's own camera frame; every
value is None for a pass without a transform.

Every value of the result is a Python float, int, str or None.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from counting.data.formats.orchard_synth import OrchardLayout, PassKey
from counting.geometry.camera import Camera
from counting.geometry.rig import pitch_extrinsic, pose_matrix, relative_extrinsic
from counting.multiview.align import PassAlignment

GRID_HALF_WIDTH_M = 0.4    # grid points at the row centre plane and this far to each side of it
GRID_Z_M = (0.5, 3.5)      # lowest and highest grid height
GRID_STEP_M = 0.5          # grid spacing along the row and in height

_PITCH_TOL = 1e-6
_E_KEYS = ("pos_m", "rot_deg", "along_m", "side_m", "height_m", "yaw_deg", "tilt_deg")
_DRIFT_KEYS = (*_E_KEYS, "pos_frac")


def _inverse(T: np.ndarray) -> np.ndarray:
    """Inverse of the rigid 4x4 transform `T`."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ t)
    return out


def fit_truth_rigid(truth: Sequence[Camera], used: Sequence[Camera]) -> np.ndarray:
    """The rigid 4x4 transform A with A @ U_i close to T_i over a pass's frames (true cameras
    `truth`, used cameras `used`, in the same order).

    The rotation is the nearest rotation to the sum of R_true @ R_used^T (SVD, with the sign of
    the determinant fixed), so a straight pass still fixes its roll; the translation is the mean
    of C_true - R @ C_used.
    """
    if len(truth) != len(used) or not truth:
        raise ValueError(f"fit_truth_rigid needs the same number (> 0) of true and used cameras, "
                         f"got {len(truth)} and {len(used)}")
    S = sum(t.R_c2w @ u.R_c2w.T for t, u in zip(truth, used))
    W, _, Vt = np.linalg.svd(S)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(W @ Vt))])
    R = W @ D @ Vt
    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = np.mean([t.C - R @ u.C for t, u in zip(truth, used)], axis=0)
    return A


def apply_rigid(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """The points `X` (n, 3) moved by the rigid 4x4 transform `A`; a row with NaN stays NaN."""
    X = np.asarray(X, dtype=np.float64)
    return X @ A[:3, :3].T + A[:3, 3]


def _steps(lo: float, hi: float, step: float) -> np.ndarray:
    """`lo`, `lo + step`, ... up to `hi` (included when on the grid)."""
    return lo + step * np.arange(int(np.floor((hi - lo) / step + 1e-9)) + 1)


def interpass_points(layout: OrchardLayout) -> np.ndarray:
    """The grid points (m, 3) of the misalignment between passes, x outermost, then y, then z.

    x runs from the first to the last tree of the target row, y is the row centre plane and
    `GRID_HALF_WIDTH_M` to each side of it, z runs over `GRID_Z_M`, all by `GRID_STEP_M`.
    A layout without trees gives (0, 3).
    """
    if not layout.tree_x:
        return np.zeros((0, 3))
    xs = _steps(min(layout.tree_x), max(layout.tree_x), GRID_STEP_M)
    ys = layout.row_y + np.array([-GRID_HALF_WIDTH_M, 0.0, GRID_HALF_WIDTH_M])
    zs = _steps(GRID_Z_M[0], GRID_Z_M[1], GRID_STEP_M)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def _relative_error(T_a: np.ndarray, T_b: np.ndarray, U_a: np.ndarray, U_b: np.ndarray) -> np.ndarray:
    """E = (T_a^-1 T_b)^-1 (U_a^-1 U_b): the error of the used motion from a to b against the true one."""
    return _inverse(_inverse(T_a) @ T_b) @ (_inverse(U_a) @ U_b)


def _angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))


def _components(E: np.ndarray, M: np.ndarray) -> dict:
    """The `_E_KEYS` of the error `E` (camera coordinates), with body axes from the rotation `M`."""
    t = E[:3, 3]
    turn = Rotation.from_matrix(E[:3, :3])
    e = M @ t
    w = np.degrees(M @ turn.as_rotvec())
    return {"pos_m": float(np.linalg.norm(t)), "rot_deg": float(np.degrees(turn.magnitude())),
            "along_m": float(e[0]), "side_m": float(e[1]), "height_m": float(e[2]),
            "yaw_deg": float(w[2]), "tilt_deg": float(np.hypot(w[0], w[1]))}


def _in_pass(T: Sequence[np.ndarray], U: Sequence[np.ndarray], M: np.ndarray, length: float) -> dict:
    """The in-pass drift of a pass (`_DRIFT_KEYS`); every value None for a pass of one frame."""
    if len(T) < 2:
        return dict.fromkeys(_DRIFT_KEYS)
    out = _components(_relative_error(T[0], T[-1], U[0], U[-1]), M)
    out["pos_frac"] = out["pos_m"] / length if length > 0 else None
    return out


def _near(errors: Mapping[float, np.ndarray], pitch: float) -> np.ndarray | None:
    """The value of `errors` whose pitch is within 1e-6 of `pitch`, or None."""
    return next((v for q, v in errors.items() if abs(q - pitch) <= _PITCH_TOL), None)


def _point_deltas(points: np.ndarray, truth: Sequence[Camera], used: Sequence[Camera],
                  alignment: PassAlignment | None) -> tuple[np.ndarray, np.ndarray]:
    """(seen, delta) of one pass: which `points` a true camera sees, and for those the lifted point
    minus the point (NaN rows elsewhere). A lifted point moves by `alignment` at the x of the used
    camera of its frame."""
    m = len(points)
    best, pick = np.full(m, np.inf), np.full(m, -1)
    for k, cam in enumerate(truth):
        with np.errstate(divide="ignore", invalid="ignore"):
            uv, z = cam.project(points)
            dist = np.hypot(uv[:, 0] - cam.K[0, 2], uv[:, 1] - cam.K[1, 2])
        dist = np.where((z > 0) & cam.in_image(uv), dist, np.inf)
        nearer = dist < best   # strict: a tie keeps the earlier frame
        best[nearer], pick[nearer] = dist[nearer], k
    seen = pick >= 0
    delta = np.full((m, 3), np.nan)
    for k in np.unique(pick[seen]):
        sel = pick == k
        T, U = truth[k], used[k]
        lifted = ((points[sel] - T.C) @ T.R_c2w) @ U.R_c2w.T + U.C
        if alignment is not None:
            lifted = alignment.move(lifted, np.full(len(lifted), float(U.C[0])))
        delta[sel] = lifted - points[sel]
    return seen, delta


def _stats(values: Sequence[float], names: tuple[str, str]) -> dict:
    """p50 and p90 of `values` under `names`, None when empty."""
    if not values:
        return dict.fromkeys(names)
    p50, p90 = np.percentile(values, [50, 90])
    return {names[0]: float(p50), names[1]: float(p90)}


def _rpe_stats(values: Sequence[float]) -> dict:
    return {**_stats(values, ("p50", "p90")), "max": float(max(values)) if values else None}


def _summary(rows: Sequence[dict], name: str, keys: tuple[str, ...]) -> dict:
    """Per key, the median over the passes of the absolute value (passes with None left out)."""
    out = {}
    for key in keys:
        values = [abs(r[name][key]) for r in rows if r[name][key] is not None]
        out[key] = float(np.median(values)) if values else None
    return out


def pose_quality(
    passes: Sequence[tuple[PassKey, Sequence[Camera], Sequence[Camera]]],
    layout: OrchardLayout,
    transforms: Mapping[PassKey, PassAlignment] | None = None,
    *,
    ref_pitch: float | None = None,
    rig_errors: Mapping[float, np.ndarray] | None = None,
) -> dict:
    """Pose quality of `passes`, each (pass key, true cameras, used cameras) in frame order.

    `transforms` are the alignment transforms by pass key, applied to the lifted points between
    passes at the x of the used camera of the frame that lifted them, and to the last used camera of
    the pass for `world_end_aligned`. `ref_pitch` is the pitch of the reference camera that RPE and
    `in_pass` are measured in, and `rig_errors` the rig calibration error by pitch (see the module
    docstring); with both None every pass stays in its own camera frame.

    Returns `{"rpe": {"n", "trans_m": {"p50", "p90", "max"}, "rot_deg": {...}}, "passes": [...],
    "summary": {...}, "interpass": {"same_side": {"n", "p50_m", "p90_m"}, "both_sides": {...}}}`.
    A pass row has `pitch`, `side`, `aisle`, `n_frames`, `length_m`, `in_pass` and `in_pass_cam`
    (the error keys and `pos_frac`; every value None for a pass of one frame, `pos_frac` None for
    a pass of zero length), `start`, `world_end` and `world_end_aligned` (every value None for a pass
    without a transform).
    `summary` holds, per entry and key, the median of the absolute values over the passes. An
    empty set gives None (and n 0).
    """
    errors = rig_errors or {}
    points = interpass_points(layout)
    rpe_trans: list[float] = []
    rpe_rot: list[float] = []
    rows: list[dict] = []
    seen: list[tuple[PassKey, np.ndarray, np.ndarray]] = []
    for key, truth, used in passes:
        if len(truth) != len(used) or not truth:
            raise ValueError(f"pass {key}: pose_quality needs the same number (> 0) of true and used cameras, "
                             f"got {len(truth)} and {len(used)}")
        T = [pose_matrix(c) for c in truth]
        U = [pose_matrix(c) for c in used]
        q = float(key.pitch_deg)
        M_own = pitch_extrinsic(q)[:3, :3]
        if ref_pitch is not None and abs(q - ref_pitch) > _PITCH_TOL:
            to_ref = _inverse(relative_extrinsic(ref_pitch, q))
            dX = _near(errors, q)
            believed = to_ref if dX is None else _inverse(dX) @ to_ref
            T_ref, U_ref = [t @ to_ref for t in T], [u @ believed for u in U]
            M_ref = pitch_extrinsic(ref_pitch)[:3, :3]
        else:
            T_ref, U_ref, M_ref = T, U, M_own
        for i in range(len(T) - 1):
            E = _relative_error(T_ref[i], T_ref[i + 1], U_ref[i], U_ref[i + 1])
            rpe_trans.append(float(np.linalg.norm(E[:3, 3])))
            rpe_rot.append(_angle_deg(E[:3, :3]))
        length = float(np.linalg.norm(T[-1][:3, 3] - T[0][:3, 3]))
        alignment = None if transforms is None else transforms.get(key)
        end_aligned = dict.fromkeys(_E_KEYS)
        if alignment is not None:
            A = alignment.at([float(used[-1].C[0])])[0]
            end_aligned = _components(_inverse(T[-1]) @ (A @ U[-1]), M_own)
        rows.append({"pitch": q, "side": key.side, "aisle": int(key.aisle), "n_frames": len(T), "length_m": length,
                     "in_pass": _in_pass(T_ref, U_ref, M_ref, length), "in_pass_cam": _in_pass(T, U, M_own, length),
                     "start": _components(_inverse(T[0]) @ U[0], M_own),
                     "world_end": _components(_inverse(T[-1]) @ U[-1], M_own),
                     "world_end_aligned": end_aligned})
        seen.append((key, *_point_deltas(points, truth, used, alignment)))

    same_side: list[float] = []
    both_sides: list[float] = []
    for a in range(len(seen)):
        for b in range(a + 1, len(seen)):
            (key_a, seen_a, delta_a), (key_b, seen_b, delta_b) = seen[a], seen[b]
            if key_a.side != key_b.side:
                into = both_sides
            elif key_a.aisle == key_b.aisle:
                into = same_side
            else:
                continue
            both = seen_a & seen_b
            into.extend(np.linalg.norm(delta_a[both] - delta_b[both], axis=1).tolist())

    return {
        "rpe": {"n": len(rpe_trans), "trans_m": _rpe_stats(rpe_trans), "rot_deg": _rpe_stats(rpe_rot)},
        "passes": rows,
        "summary": {"in_pass": _summary(rows, "in_pass", _DRIFT_KEYS),
                    "in_pass_cam": _summary(rows, "in_pass_cam", _DRIFT_KEYS),
                    "start": _summary(rows, "start", _E_KEYS), "world_end": _summary(rows, "world_end", _E_KEYS),
                    "world_end_aligned": _summary(rows, "world_end_aligned", _E_KEYS)},
        "interpass": {name: {"n": len(values), **_stats(values, ("p50_m", "p90_m"))}
                      for name, values in (("same_side", same_side), ("both_sides", both_sides))},
    }
