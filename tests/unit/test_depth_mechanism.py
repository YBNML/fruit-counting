"""Report-only counts of the tracker and dedup mechanisms: the tracker's trace of the pairs it joined at the soft depth
cost and of why each track was born, the unjoined fragments of one fruit in a pass, the cross-side pairs the band keeps
apart, and the `mechanism` entry of a result with LiDAR files (its seed mean, its sweep cell and its report line)."""

import json
from pathlib import Path

import numpy as np
import pytest

from counting.config.eval_schema import DedupConfig, LiftConfig, TrackerConfig
from counting.data.formats.orchard_synth import FrameInput, OrchardLayout, PassKey
from counting.detection.base import Detections
from counting.evaluation.mechanism import band_blocked_pairs, mechanism_entry, split_fragments
from counting.evaluation.report import render_report
from counting.evaluation.runner import evaluate
from counting.evaluation.sweep import Grid, _kept, sweep_cells
from counting.geometry.camera import Camera
from counting.multiview.dedup import dedup
from counting.multiview.lift import edge_touch, lift_points
from counting.tracking.landmark import PassTracks, TrackTrace, track_pass
from tests.unit.synthfake import robot_matrix
from tests.unit.test_lidar_runner import DRIFT, _exact, _results, lidar_case

pytestmark = pytest.mark.pose

W, H = 640, 360
KEY = PassKey(30.0, "A", 2)
EXACT = LiftConfig(depth="det_depth", center_offset_m={})
HARD = TrackerConfig(gate_m=0.05)
SOFT = TrackerConfig(gate_m=0.05, depth_gate="soft")
FRUIT = np.array([0.3, 3.95, 1.6])
FAR = np.array([0.9, 3.95, 1.6])     # about 150 px from FRUIT on the screen
CAUSES = {"no_track_in_reach", "depth_gate", "other"}
ROW_Y = 3.75
LAYOUT = OrchardLayout(crop="apple", row_ys=(ROW_Y,), target_row=0, row_spacing=3.5, tree_ids=(1,), tree_x=(0.0,))
BAND = DedupConfig(radius_m=0.09, band_m=0.05)


def _scene(*frames, step: float = 0.05) -> tuple[list[FrameInput], dict[int, Detections], dict[int, np.ndarray]]:
    """Frames k = 0, 1, ... of a camera sliding +x by `step` m a frame on side A of the row at y = 3.75. Frame k has
    one box per `(point, dz, du)` of `frames[k]`: the point's 7 cm box shifted `du` px along u, with the point's depth
    plus `dz` m as its depth (NaN: none), lifted at that depth."""
    out, dets, pts = [], {}, {}
    for k, items in enumerate(frames):
        cam = Camera.from_blender(robot_matrix(step * k, ROW_Y, "A", 30.0, 1.75, 0.9), W, H, 69.0)
        boxes, z = [], []
        for point, dz, du in items:
            uv, depth = cam.project(np.asarray(point, dtype=float)[None])
            r, c = cam.K[0, 0] * 0.035 / depth[0], uv[0] + 0.5 + np.array([du, 0.0])
            boxes.append([c[0] - r, c[1] - r, c[0] + r, c[1] + r])
            z.append(depth[0] + dz)
        n = len(items)
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        assert not edge_touch(boxes, W, H).any()        # inner boxes: the depth gate applies to every one
        d = Detections(boxes, np.ones(n, np.float32), np.zeros(n, np.int8), np.asarray(z, np.float32))
        out.append(FrameInput("t", k, KEY, k, cam, step * k, Path("r"), Path("d")))
        dets[k] = d
        pts[k] = lift_points(boxes, d.labels, d.depth_m, cam, EXACT)
    return out, dets, pts


def _traced(scene, cfg: TrackerConfig) -> tuple[PassTracks, TrackTrace]:
    """`track_pass` with a trace, checked to give the tracks it gives without one."""
    frames, dets, pts = scene
    trace = TrackTrace()
    traced, plain = track_pass(frames, dets, pts, cfg, trace=trace), track_pass(frames, dets, pts, cfg)
    assert traced.members() == plain.members() and traced.X.tobytes() == plain.X.tobytes()
    assert traced.det_track == plain.det_track
    return traced, trace


def test_trace_counts_soft_depth_costs_and_births():
    assert TrackTrace().births == dict.fromkeys(("no_track_in_reach", "depth_gate", "other"), 0)
    off = [[(FRUIT, 0.0, 0.0)], [(FRUIT, 0.3, 0.0)]]      # the second box on the track's screen spot, 0.3 m deeper
    tracks, trace = _traced(_scene(*off), SOFT)
    assert len(tracks.X) == 1 and trace.soft_depth_pairs == 1
    assert trace.births == {"no_track_in_reach": 1, "depth_gate": 0, "other": 0}    # the first box starts the track

    far = [*off, [(FAR, 0.0, 0.0)]]                        # a third frame: a box far from every predicted track
    tracks, trace = _traced(_scene(*far), SOFT)
    assert len(tracks.X) == 2 and trace.soft_depth_pairs == 1
    assert trace.births == {"no_track_in_reach": 2, "depth_gate": 0, "other": 0}    # the far box adds one
    tracks, trace = _traced(_scene(*far), HARD)            # the hard gate refuses the deeper box: a track of its own
    assert len(tracks.X) == 3 and trace.soft_depth_pairs == 0
    assert trace.births == {"no_track_in_reach": 2, "depth_gate": 1, "other": 0}

    twin = [[(FRUIT, 0.0, 0.0)], [(FRUIT, 0.0, 0.0), (FRUIT, 0.0, 3.0)]]    # two boxes inside both gates
    for cfg in (HARD, SOFT):
        tracks, trace = _traced(_scene(*twin), cfg)
        assert len(tracks.X) == 2 and trace.soft_depth_pairs == 0
        assert trace.births == {"no_track_in_reach": 1, "depth_gate": 0, "other": 1}   # the shifted one lost the pair
    blind = [[(FRUIT, np.nan, 0.0)], [(FRUIT, np.nan, 0.0), (FRUIT, np.nan, 3.0)]]   # no depth: chained by IoU
    tracks, trace = _traced(_scene(*blind, step=0.0), HARD)
    assert np.isnan(tracks.X).all() and len(tracks.X) == 2         # the IoU gate counts as the track's reach
    assert trace.births == {"no_track_in_reach": 1, "depth_gate": 0, "other": 1} and trace.soft_depth_pairs == 0

    entry = mechanism_entry([_traced(_scene(*far), SOFT)[1], _traced(_scene(*far), HARD)[1]], np.array([]), 2)
    assert entry == {"soft_depth_pairs": 1, "births": {"no_track_in_reach": 4, "depth_gate": 1, "other": 0},
                     "fragments": {"n": 0, "p50_m": None, "p90_m": None}, "band_blocked_pairs": 2}
    fragments = mechanism_entry([TrackTrace()], np.array([0.10, 0.02, 0.04]), None)["fragments"]
    assert fragments["n"] == 3 and fragments["p50_m"] == 0.04
    assert abs(fragments["p90_m"] - 0.088) < 1e-12               # 0.04 + 0.8 * (0.10 - 0.04)


def _pass(key: PassKey, first: int, X, obs) -> tuple[PassTracks, dict[int, np.ndarray]]:
    """A pass `key` whose track t has the point X[t] and one observation per source id of `obs[t]`, each detection 0
    of a frame of its own (from `first` on); returns its tracks and the source ids of those frames."""
    det_track, src, frames = {}, {}, []
    for t, ids in enumerate(obs):
        seen = [first + 10 * t + o for o in range(len(ids))]
        for f, i in zip(seen, ids):
            det_track[(f, 0)] = t
            src[f] = np.array([i])
        frames.append(frozenset(seen))
    n = len(obs)
    tracks = PassTracks(pass_key=key, det_track=det_track, X=np.asarray(X, dtype=float).reshape(-1, 3),
                        labels=np.zeros(n, np.int8), n_obs=np.array([len(ids) for ids in obs]), spread=np.zeros(n),
                        frames=frames)
    return tracks, src


def test_unjoined_fragments_and_band_blocks():
    x0, nan = np.array([1.0, 3.9, 1.5]), [np.nan] * 3
    a, src_a = _pass(PassKey(30.0, "A", 2), 0,
                     [x0, x0 + [0.04, 0, 0], x0 + [0, 0.02, 0], x0 + [0.5, 0, 0], x0 + [0.51, 0, 0], nan],
                     [[7, 7, -1], [-1, 7, 7], [8], [-1], [-1], [7]])   # fruit 7 twice, fruit 8, no fruit twice, no 3D
    b, src_b = _pass(PassKey(10.0, "A", 2), 100, [x0 + [0.01, 0, 0]], [[7]])   # fruit 7 again, in another pass
    got = split_fragments([a, b], {**src_a, **src_b})
    np.testing.assert_allclose(got, [0.04], rtol=0, atol=1e-12)
    assert split_fragments([b], src_b).shape == (0,)

    side_a, side_b = PassKey(30.0, "A", 2), PassKey(30.0, "B", 1)

    def two_sides(ya: float, yb: float):
        A, sa = _pass(side_a, 0, [[2.0, ya, 1.5]], [[7]])
        B, sb = _pass(side_b, 100, [[2.0, yb, 1.5]], [[7]])
        return [A, B], {**sa, **sb}

    for ya, yb, blocked in ((ROW_Y + 0.07, ROW_Y + 0.01, 1), (ROW_Y + 0.04, ROW_Y - 0.02, 0)):   # 0.06 m apart
        passes, src = two_sides(ya, yb)
        assert band_blocked_pairs(passes, src, LAYOUT, BAND) == blocked, (ya, yb)
        keep = [np.ones(1, dtype=bool)] * 2
        lm = dedup(passes, keep, LAYOUT, BAND, mode="hybrid")      # the same pair under the dedup's band rule
        assert len(lm.X) == 1 + blocked

    passes, src = two_sides(ROW_Y + 0.07, ROW_Y + 0.01)
    assert band_blocked_pairs(passes, src, LAYOUT, BAND, keep=[np.zeros(1, dtype=bool), np.ones(1, dtype=bool)]) == 0
    A, sa = _pass(side_a, 0, [[2.0, ROW_Y + 0.07, 1.5], [3.0, ROW_Y + 0.10, 1.5], [4.0, ROW_Y + 0.07, 1.5]],
                  [[7], [9], [10]])
    B, sb = _pass(side_b, 100, [[2.0, ROW_Y + 0.01, 1.5], [3.0, ROW_Y, 1.5], [4.0, ROW_Y + 0.01, 1.5]],
                  [[7], [9], [11]])      # fruit 9 is 0.10 m apart, beyond the radius; 10 and 11 are two fruit
    assert band_blocked_pairs([A, B], {**sa, **sb}, LAYOUT, BAND) == 1
    A2, sa2 = _pass(PassKey(10.0, "A", 2), 200, [[2.0, ROW_Y + 0.01, 1.5]], [[7]])   # the same side: not counted
    assert band_blocked_pairs([A, A2], {**sa, **sa2}, LAYOUT, BAND) == 0

    passes, src = two_sides(ROW_Y + 0.07, ROW_Y + 0.01)     # the blocked pair, and a bridge on side A in the band
    bridge, src_bridge = _pass(PassKey(10.0, "A", 2), 200, [[2.0, ROW_Y + 0.03, 1.5]], [[7]])
    passes, src = [*passes, bridge], {**src, **src_bridge}
    lm = dedup(passes, [np.ones(1, dtype=bool)] * 3, LAYOUT, BAND, mode="hybrid")
    assert len(lm.members) == 1 and len(lm.members[0]) == 3    # the bridge joins all three: one landmark
    assert band_blocked_pairs(passes, src, LAYOUT, BAND) == 0


def test_mechanism_entry_only_with_lidar_files():
    case = lidar_case()
    with_files, plain = case.config(DRIFT), case.config(DRIFT, base=case.plain)
    for seed in (0, 1):
        a, b = case.evaluate(with_files, seed), case.evaluate(plain, seed)
        assert _exact(a, ("y1", "y6")) == _exact(b, ("y1", "y6")), seed
        assert "mechanism" in a and "mechanism" not in b
    first = case.evaluate(with_files, 0)["mechanism"]
    assert set(first) == {"soft_depth_pairs", "births", "fragments", "band_blocked_pairs"}
    assert set(first["births"]) == CAUSES and set(first["fragments"]) == {"n", "p50_m", "p90_m"}
    assert first["band_blocked_pairs"] is None                  # merge dedup: no band
    assert json.dumps(evaluate(case.ctx, with_files, noise_seed=0)["mechanism"]) == json.dumps(first)

    soft = case.config(DRIFT, {"stages": {"dedup": "hybrid"}, "dedup": {"radius_m": 0.09, "band_m": 0.05},
                               "tracker": {"depth_gate": "soft", "repair_m": 0.0}})
    res = case.evaluate(soft, 0)
    entry = res["mechanism"]
    n_tracks = sum(p["n_tracks"] for p in res["y1"]["passes"])    # without repair, one track per birth
    assert sum(entry["births"].values()) == n_tracks and entry["births"]["no_track_in_reach"] > 0
    assert entry["soft_depth_pairs"] > 0 and isinstance(entry["band_blocked_pairs"], int)
    assert entry["fragments"]["n"] > 0 and 0 < entry["fragments"]["p50_m"] <= entry["fragments"]["p90_m"]
    assert json.dumps(evaluate(case.ctx, soft, noise_seed=0)["mechanism"]) == json.dumps(entry)
    assert "mechanism" not in case.evaluate(case.config({"stages": {"track": "oracle"}}))

    combined = case.combined(with_files)
    seeds = [case.evaluate(with_files, s)["mechanism"] for s in (0, 1)]
    assert [p["mechanism"] for p in combined["per_seed"]] == seeds
    mean = combined["mechanism"]
    assert mean["soft_depth_pairs"] == (seeds[0]["soft_depth_pairs"] + seeds[1]["soft_depth_pairs"]) / 2
    assert all(mean["births"][k] == (seeds[0]["births"][k] + seeds[1]["births"][k]) / 2 for k in CAUSES)
    grid = Grid(path=Path("combine.yaml"), bases=(), factors={"lift.lidar.combine": ["replace"]}, product={},
                combos={}, common={})
    kept = _kept(sweep_cells(with_files, grid, None)[0], combined)
    assert kept["mechanism"] == mean and [p["mechanism"] for p in kept["per_seed"]] == seeds

    report = render_report(_results(case, "ladder", {"files": combined, "plain": case.combined(plain)}),
                           criteria=None).splitlines()
    lines = [line for line in report if line.startswith("> 기구")]
    assert len(lines) == 1 and lines[0].startswith("> 기구 files: ")
