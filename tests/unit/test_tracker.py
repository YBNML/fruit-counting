from pathlib import Path

import numpy as np

from counting.config.eval_schema import LiftConfig, TrackerConfig
from counting.data.formats.orchard_synth import FrameInput, PassKey
from counting.detection.base import Detections
from counting.geometry.camera import Camera
from counting.multiview.lift import lift_points
from counting.tracking.landmark import edge_cost, predict_boxes, track_pass
from tests.unit.synthfake import robot_matrix

POINTS = np.array([[0.0, 3.95, 1.5], [0.4, 3.55, 1.8], [0.9, 4.0, 1.65], [1.3, 3.6, 2.0], [1.8, 3.9, 1.7]])
KEY = PassKey(30.0, "A", 2)
CFG = TrackerConfig(gate_m=0.05)
CFG5 = TrackerConfig(gate_m=0.05, max_gap=5)
EXACT = LiftConfig(depth="det_depth", center_offset_m={})
W, H = 640, 360


def _slide(points, step=0.15, x0=-1.0, x1=3.5, hide=None, score=None, depth=True, r=0.035):
    """Camera slides +x on side A of row y=3.75. One box per visible point, clipped to the image like
    orchard_synth does (a cut fragment is still a record if >= 20 px). owner: (idx, det) -> point."""
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
        zz = z[vis].astype(np.float32)
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
