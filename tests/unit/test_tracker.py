from pathlib import Path

import numpy as np

from counting.config.eval_schema import LiftConfig, TrackerConfig
from counting.data.formats.orchard_synth import FrameInput, PassKey
from counting.detection.base import Detections
from counting.geometry.camera import Camera
from counting.multiview.lift import edge_touch, lift_points
from counting.tracking.landmark import edge_cost, predict_boxes, track_pass
from tests.unit.synthfake import robot_matrix

POINTS = np.array([[0.0, 3.95, 1.5], [0.4, 3.55, 1.8], [0.9, 4.0, 1.65], [1.3, 3.6, 2.0], [1.8, 3.9, 1.7]])
KEY = PassKey(30.0, "A", 2)
CFG = TrackerConfig(gate_m=0.05)
CFG5 = TrackerConfig(gate_m=0.05, max_gap=5)
SOFT = TrackerConfig(gate_m=0.05, depth_gate="soft")
EXACT = LiftConfig(depth="det_depth", center_offset_m={})
W, H = 640, 360


def _slide(points, step=0.15, x0=-1.0, x1=3.5, hide=None, score=None, depth=True, r=0.035, zbias=None):
    """Camera slides +x on side A of row y=3.75. One box per visible point, clipped to the image like
    orchard_synth does (a cut fragment is still a record if >= 20 px). owner: (idx, det) -> point.
    zbias: (point idx, frame k) -> metres added to that point's depth in that frame, for both the
    detection depth and the lifted point."""
    frames, dets, pts, owner = [], {}, {}, {}
    for k in range(int(np.floor((x1 - x0) / step + 1e-9)) + 1):
        x = x0 + step * k
        cam = Camera.from_blender(robot_matrix(x, 3.75, "A", 30.0, 1.75, 0.9), W, H, 69.0)
        uv, z = cam.project(points)
        boxes, vis = [], []
        for i in range(len(points)):
            if z[i] <= 0.05 or k in (hide or {}).get(i, ()):
                continue
            rp, c = cam.K[0, 0] * r / z[i], uv[i] + 0.5
            b = np.clip([c[0] - rp, c[1] - rp, c[0] + rp, c[1] + rp], 0, [W, H, W, H])
            if (b[2] - b[0]) * (b[3] - b[1]) >= 20:
                boxes.append(b)
                vis.append(i)
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        sc = np.array([(score or {}).get((i, k), 1.0) for i in vis], np.float32)
        zz = (z[vis] + np.array([(zbias or {}).get((i, k), 0.0) for i in vis])).astype(np.float32)
        d = Detections(boxes, sc, np.zeros(len(vis), np.int8), zz if depth else None)
        frames.append(FrameInput("t", k, KEY, k, cam, x, Path("r"), Path("d")))
        dets[k] = d
        pts[k] = lift_points(boxes, d.labels, zz if depth else np.full(len(vis), np.nan), cam, EXACT)
        owner.update({(k, j): i for j, i in enumerate(vis)})
    return frames, dets, pts, owner


def _groups(tr, owner):
    out = {}
    for key, t in tr.det_track.items():
        out.setdefault(owner[key], set()).add(t)
    return out


def _box_of(dets, owner, i, k):
    """Point i's box in frame k."""
    j = next(j for (kk, j), ii in owner.items() if kk == k and ii == i)
    return dets[k].boxes[j]


def test_landmark_one_track_per_point():
    frames, dets, pts, owner = _slide(POINTS)
    tr = track_pass(frames, dets, pts, CFG)
    g = _groups(tr, owner)
    assert len(tr.X) == 5 and all(len(v) == 1 for v in g.values())
    assert len({next(iter(v)) for v in g.values()}) == 5 and int(tr.n_obs.sum()) == len(owner)
    for i, tids in g.items():
        np.testing.assert_allclose(tr.X[next(iter(tids))], POINTS[i], atol=1e-3)


def test_edge_fragments_stay_on_one_track():
    frames, dets, pts, owner = _slide(POINTS, r=0.06)
    cut, off = set(), set()
    for (k, j), i in owner.items():
        b, cam = dets[k].boxes[j], frames[k].camera
        if b[0] <= 0 or b[2] >= W:
            cut.add(i)
            if not cam.in_image(cam.project(POINTS[i:i + 1])[0])[0]:
                off.add(i)
    assert cut == set(range(5)) and off          # every point is seen cut at the border; some with centre off-screen
    g = _groups(track_pass(frames, dets, pts, CFG), owner)
    assert all(len(v) == 1 for v in g.values()) and len({next(iter(v)) for v in g.values()}) == 5


def test_last_box_splits_when_shift_exceeds_box():
    frames, dets, pts, _ = _slide(POINTS)            # ~2 box sizes of shift per frame
    assert len(track_pass(frames, dets, pts, CFG, method="last_box").X) >= 15


def test_last_box_joins_at_small_step():
    frames, dets, pts, owner = _slide(POINTS, step=0.01, x0=0.0, x1=0.5)
    g = _groups(track_pass(frames, dets, pts, CFG, method="last_box"), owner)
    assert all(len(v) == 1 for v in g.values())


def test_gap_shorter_than_max_gap_keeps_one_track():
    frames, dets, pts, owner = _slide(POINTS, hide={0: range(8, 12)})     # 4 missing frames
    assert len(_groups(track_pass(frames, dets, pts, CFG5), owner)[0]) == 1


def test_gap_of_max_gap_starts_new_track():
    frames, dets, pts, owner = _slide(POINTS, hide={0: range(8, 13)})     # 5 missing frames
    assert len(_groups(track_pass(frames, dets, pts, CFG5), owner)[0]) == 2


def test_long_gap_without_max_gap_keeps_one_track():
    frames, dets, pts, owner = _slide(POINTS, hide={0: range(6, 13)})     # 7 missing frames, still in view
    assert len(_groups(track_pass(frames, dets, pts, CFG), owner)[0]) == 1


def test_low_score_extends_but_never_starts():
    sc = {(0, k): 0.3 for k in range(5, 40)}
    sc.update({(1, k): 0.3 for k in range(40)})
    sc.update({(2, k): 0.05 for k in range(40)})
    frames, dets, pts, owner = _slide(POINTS, score=sc)
    tr = track_pass(frames, dets, pts, CFG)
    g = _groups(tr, owner)
    assert len(g[0]) == 1 and 1 not in g and 2 not in g and len(tr.X) == 3
    assert all(k in tr.det_track for k, i in owner.items() if i == 0)   # low-score boxes extend point 0's track (stage 2)


def test_no_depth_tracks_join_by_iou():
    frames, dets, pts, owner = _slide(POINTS, step=0.01, x0=0.0, x1=0.5, depth=False)
    tr = track_pass(frames, dets, pts, CFG)
    assert all(len(v) == 1 for v in _groups(tr, owner).values()) and np.isnan(tr.X).all()


def test_predict_boxes_marks_behind_outside_and_partial():
    cam = Camera.from_blender(robot_matrix(0.0, 3.75, "A", 30.0, 1.75, 0.9), W, H, 69.0)
    s = cam.K[0, 0] * 0.07 / 2.0
    edge = cam.backproject(np.array([[639.5 + 0.4 * s, 180.0], [639.5 + 1.1 * s, 180.0], [639.5 + 0.8 * s, 180.0]]),
                           np.full(3, 2.0))   # 0.8*s: alive only because the box is grown by gate_px_frac
    X = np.vstack([[[0.0, 3.9, 1.6], [0.0, 7.0, 1.6], [50.0, 3.9, 1.6]], edge])
    _, _, sp, alive = predict_boxes(cam, X, np.full(6, 0.07), grow=0.5)
    assert alive.tolist() == [True, False, False, True, False, True]   # centre off-screen but box still overlaps: alive
    np.testing.assert_allclose(sp[3:], s)


def test_edge_cost_uses_inner_edge():
    boxes = np.array([[0, 40, 12, 60], [20, 40, 40, 60]], float)        # left-cut fragment, interior box
    touch = np.array([[True, False, False, False], [False] * 4])
    c = edge_cost(boxes, touch, uv=np.array([1.5, 49.5]), s=20.0)     # predicted box [-8, 12) x [40, 60)
    np.testing.assert_allclose(c, [0.0, 28.0 / 20.0])


def test_members_frames_and_determinism():
    frames, dets, pts, _ = _slide(POINTS)
    a, b = track_pass(frames, dets, pts, CFG), track_pass(frames, dets, pts, CFG)
    assert a.det_track == b.det_track
    for t, mem in enumerate(a.members()):
        assert a.frames[t] == frozenset(f for f, _ in mem) and a.n_obs[t] == len(mem)


def test_hard_depth_gate_splits_on_wrong_depth():
    for dz, n in ((-0.3, 2), (0.3, 2), (-0.04, 1)):              # gate_m = 0.05
        frames, dets, pts, owner = _slide(POINTS, zbias={(0, 7): dz})
        assert not edge_touch(_box_of(dets, owner, 0, 7)[None], W, H).any()
        assert len(_groups(track_pass(frames, dets, pts, CFG), owner)[0]) == n, dz


def test_border_box_skips_depth_gate():
    frames, dets, pts, owner = _slide(POINTS, zbias={(0, 14): -0.3})
    assert edge_touch(_box_of(dets, owner, 0, 14)[None], W, H).any()      # point 0 leaves the image here
    assert len(_groups(track_pass(frames, dets, pts, CFG), owner)[0]) == 1


def test_edge_only_track_skips_depth_gate():
    frames, dets, pts, owner = _slide(POINTS, zbias={(2, 5): 0.3})
    assert min(k for (k, _), i in owner.items() if i == 2) == 5            # point 2 enters at frame 5,
    assert edge_touch(_box_of(dets, owner, 2, 5)[None], W, H).any()        # cut by the border
    assert len(_groups(track_pass(frames, dets, pts, CFG), owner)[2]) == 1


def test_edge_cost_cut_top_and_bottom():
    boxes = np.array([[40, 0, 60, 12], [40, 348, 60, 360], [40, 0, 60, 360], [40, 20, 60, 40]], float)
    touch = edge_touch(boxes, W, H)
    assert touch[:, 2:].tolist() == [[True, False], [False, True], [True, True], [False, False]]
    c = [edge_cost(boxes[i:i + 1], touch[i:i + 1], np.array(uv), 20.0)[0]
         for i, uv in enumerate([(49.5, 5.5), (49.5, 353.5), (49.5, 100.0), (49.5, 1.5)])]
    np.testing.assert_allclose(c, [0.2, 0.2, 0.0, 1.4])    # cut boxes compare inner edges, not centres


def test_soft_gate_joins_wrong_depth_detection():
    frames, dets, pts, owner = _slide(POINTS, zbias={(0, 7): -0.3})
    assert len(_groups(track_pass(frames, dets, pts, CFG), owner)[0]) == 2
    tr = track_pass(frames, dets, pts, SOFT)
    assert len(_groups(tr, owner)[0]) == 1 and len(tr.X) == 5


def test_soft_gate_matches_hard_when_depths_agree():
    for r, hide in ((0.035, None), (0.06, None), (0.035, {0: range(6, 13)})):
        frames, dets, pts, _ = _slide(POINTS, r=r, hide=hide)
        assert track_pass(frames, dets, pts, SOFT).det_track == track_pass(frames, dets, pts, CFG).det_track


def test_soft_gate_keeps_screen_gate():
    frames, dets, pts, owner = _slide(POINTS, zbias={(0, 7): -0.3})
    j = next(j for (k, j), i in owner.items() if k == 7 and i == 0)
    D = dets[7]
    boxes = D.boxes.copy()
    du = 0.7 * (boxes[j, 2] - boxes[j, 0])            # 0.7 box widths: outside the screen gate (0.5)
    boxes[j] += np.array([du, 0, du, 0], np.float32)
    dets[7] = Detections(boxes, D.scores, D.labels, D.depth_m)
    pts[7] = lift_points(boxes, D.labels, D.depth_m, frames[7].camera, EXACT)
    assert len(_groups(track_pass(frames, dets, pts, SOFT), owner)[0]) == 2


def _with_decoy(frames, dets, pts, owner, i, k, dz, du):
    """Shift point i's box in frame k by `du` px along u and append a decoy at the unshifted box
    whose depth is off by `dz`. Returns (true det index, decoy det index)."""
    j = next(j for (kk, j), ii in owner.items() if kk == k and ii == i)
    D = dets[k]
    boxes = np.vstack([D.boxes, D.boxes[j:j + 1]]).astype(np.float32)
    boxes[j] += np.array([du, 0, du, 0], np.float32)
    z = np.append(D.depth_m, D.depth_m[j] + dz).astype(np.float32)
    n = len(boxes)
    dets[k] = Detections(boxes, np.ones(n, np.float32), np.zeros(n, np.int8), z)
    pts[k] = lift_points(boxes, dets[k].labels, z, frames[k].camera, EXACT)
    owner[(k, n - 1)] = -1
    return j, n - 1


def test_soft_gate_prefers_depth_consistent_detection():
    for soft_cost, decoy_wins in ((0.5, False), (0.0, True)):
        frames, dets, pts, owner = _slide(POINTS)
        prev = next(j for (k, j), i in owner.items() if k == 6 and i == 0)
        jt, jd = _with_decoy(frames, dets, pts, owner, 0, 7, dz=-0.3, du=3.0)   # true box ~0.15 s off
        tr = track_pass(frames, dets, pts, TrackerConfig(gate_m=0.05, depth_gate="soft", soft_cost=soft_cost))
        t0 = tr.det_track[(6, prev)]
        assert (tr.det_track[(7, jd)] == t0, tr.det_track[(7, jt)] == t0) == (decoy_wins, not decoy_wins)
