import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from counting.config.eval_schema import EvalAppConfig
from counting.config.loader import load_yaml_model
from counting.data.formats.orchard_synth import GtRecord, SynthRun
from counting.detection.base import Detections
from counting.detection.cached import CachedDetector, cache_path, write_cache
from counting.evaluation.cli import eval_app
from counting.evaluation.detmetrics import (
    JUDGED_PITCHES, M_KEYS, average_precision, cross_class_pairs, detection_verdicts, evaluate_frame, fp_kinds,
    greedy_match, score_runs, summarize, summarize_groups,
)
from counting.evaluation.report import results_bytes
from counting.evaluation.runner import describe_run
from counting.evaluation.truth import is_positive
from tests.unit.synthfake import write_run

ROOT = Path(__file__).resolve().parents[2]
SPLITS = str(ROOT / "configs" / "eval" / "splits.yaml")
CRIT_FILE = str(ROOT / "configs" / "eval" / "criteria.yaml")
runner = CliRunner()

COUNT1 = (0.25, 20)
OPS = {"operating": 0.5, "low": 0.1}


def _rec(i, box, *, visib=1.0, vis_px=100, tree=4, depth=2.0):
    return GtRecord(id=i, tree=tree, vis_px=vis_px, full_px=round(vis_px / visib), visib=visib, bbox=tuple(box),
                    trunc=False, depth_med=depth)


def _dets(rows):
    a = np.array(rows, np.float64).reshape(-1, 6)
    return Detections(a[:, :4].astype(np.float32), a[:, 4].astype(np.float32), a[:, 5].astype(np.int8))


RECS = [_rec(1, (0, 5, 10, 15)), _rec(2, (20, 5, 30, 15)), _rec(3, (40, 5, 50, 15), tree=9),
        _rec(4, (60, 5, 70, 15), visib=0.1)]
DETS = _dets([(0, 5, 10, 15, 0.9, 0),       # TP on 1
              (60, 5, 70, 15, 0.8, 0),      # on the ignored record 4: left out
              (1, 5, 11, 15, 0.7, 1),       # IoU 0.82 with 1, already taken: FP dup
              (20, 5, 30, 15, 0.6, 0),      # TP on 2
              (80, 5, 90, 15, 0.5, 0),      # FP background
              (42, 5, 46, 15, 0.4, 0)])     # inside 3 with IoU 0.4: FP part; 3 is missed


FAR = {0: 2.05, 20: 3.2, 80: 9.0}
MED = {0: 2.0, 20: np.nan, 80: 9.0}


def _depth_fn(boxes):
    """Box depths looked up by the box's x0: (box_far, box_median)."""
    return (np.array([FAR[int(x)] for x in boxes[:, 0]]), np.array([MED[int(x)] for x in boxes[:, 0]]))


def _frame(dets=DETS, recs=RECS, *, idx=0, pitch_deg=30.0, **kw):
    return evaluate_frame(dets, recs, idx=idx, pitch_deg=pitch_deg, count1=COUNT1, target_trees=frozenset({3, 4, 5}),
                          width=640, height=360, **kw)


def test_greedy_match_takes_positives_first_and_ignored_records_may_repeat():
    pos = np.array([r.bbox for r in RECS[:3]], np.float64)
    ign = np.array([RECS[3].bbox], np.float64)
    kind, tp_of, iou = greedy_match(DETS.boxes, DETS.scores, pos, ign)
    assert kind.tolist() == [1, -1, 0, 1, 0, 0] and tp_of.tolist() == [0, -1, -1, 1, -1, -1]
    assert iou[0] == 1.0 and math.isnan(iou[2])
    one = np.array([[0, 5, 10, 15]], np.float32)
    k, t, _ = greedy_match(one, np.array([0.9], np.float32), np.array([[0, 5, 10, 23.2]]), np.array([[0, 5, 10, 14]]))
    assert k.tolist() == [1] and t.tolist() == [0]          # IoU 0.55 to a free positive beats 0.9 to an ignored one
    k, _, _ = greedy_match(np.array([[60, 5, 70, 15], [61, 5, 70, 15]], np.float32), np.array([0.8, 0.7], np.float32),
                           np.zeros((0, 4)), ign)
    assert k.tolist() == [-1, -1]
    k, t, _ = greedy_match(np.array([[1, 5, 11, 15], [0, 5, 10, 15]], np.float32), np.array([0.5, 0.5], np.float32),
                           np.array([[0, 5, 10, 15]]), np.zeros((0, 4)))
    assert k.tolist() == [1, 0] and t.tolist() == [0, -1]   # equal scores: stored order


def test_ap50_and_operating_points_by_hand():
    g = summarize([_frame()], OPS)
    assert (g["n_frames"], g["n_pos"], g["n_ignore"]) == (1, 3, 1) and g["ap50"] == pytest.approx(5 / 9)
    op = g["at"]["operating"]
    assert (op["score"], op["tp"], op["fp"], op["ignored"], op["fn"]) == (0.5, 2, 2, 1, 1)
    assert (op["fp_dup"], op["fp_merge"], op["fp_part"], op["fp_ign"], op["fp_bg"]) == (1, 0, 0, 0, 1)
    assert op["precision"] == pytest.approx(0.5) and op["recall"] == pytest.approx(2 / 3)
    assert op["f1"] == pytest.approx(4 / 7) and op["fp_per_frame"] == 2.0
    assert op["split_rate"] == pytest.approx(1 / 3) and op["frag_only_rate"] == 0.0
    assert op["split_rate_low_visib"] is None                 # no positive below visib 0.6
    low = g["at"]["low"]
    assert (low["tp"], low["fp"], low["fp_part"]) == (2, 3, 1) and low["precision"] == pytest.approx(0.4)
    assert low["frag_only_rate"] == pytest.approx(1 / 3)
    best = g["at"]["f1_max"]
    assert best["score"] == pytest.approx(0.6) and best["f1"] == pytest.approx(2 / 3)
    assert average_precision(np.array([0.9, 0.5]), np.array([0, 0], np.int8), 0) is None
    # A false positive ranked first: the first true positive takes the later, higher precision 2/3 (the envelope).
    assert average_precision(np.array([0.9, 0.8, 0.7]), np.array([0, 1, 1], np.int8), 2) == pytest.approx(2 / 3)


def test_ignored_record_with_the_same_box_as_a_positive_leaves_the_detection_to_the_positive():
    recs = [_rec(1, (100, 100, 120, 120)), _rec(2, (100, 100, 120, 120), visib=0.1)]
    g = summarize([_frame(_dets([(100, 100, 120, 120, 0.9, 0)]), recs)], OPS)
    op = g["at"]["operating"]
    assert (g["n_pos"], g["n_ignore"], op["tp"], op["ignored"], op["fp"]) == (1, 1, 1, 0, 0)


def test_metrics_on_empty_and_ignore_only_frames():
    nothing = _frame(_dets([]), [])
    only_ignored = _frame(_dets([(60, 5, 70, 15, 0.9, 0)]), [_rec(4, (60, 5, 70, 15), visib=0.1)], idx=1)
    missed = _frame(_dets([]), [_rec(1, (0, 5, 10, 15))], idx=2)
    g0 = summarize([nothing, only_ignored], OPS)
    op0 = g0["at"]["operating"]
    assert g0["ap50"] is None and op0["precision"] is None and op0["recall"] is None and op0["ignored"] == 1
    assert g0["at"]["f1_max"] == {k: None for k in M_KEYS}
    g1 = summarize([nothing, only_ignored, missed], OPS)
    op1 = g1["at"]["operating"]
    assert g1["ap50"] == 0.0 and op1["recall"] == 0.0 and op1["fn"] == 1
    assert op1["precision"] is None and op1["fp_per_frame"] == 0.0
    assert summarize([], OPS)["n_frames"] == 0
    fps_only = _frame(_dets([(80, 5, 90, 15, 0.9, 0)]), [_rec(1, (0, 5, 10, 15))], idx=3, depth_fn=_depth_fn)
    g2 = summarize([fps_only], OPS)
    op2 = g2["at"]["operating"]
    assert (op2["precision"], op2["recall"], op2["f1"]) == (0.0, 0.0, None)
    assert g2["at"]["f1_max"] == {k: None for k in M_KEYS}       # no threshold has an F1
    assert g2["iou"] == {"p10": None, "p50": None, "p90": None} and g2["other_row_share"] is None
    assert g2["depth"]["box_far"] == {"n": 0, "p50_m": None, "p90_m": None, "p95_m": None, "near_10cm": None,
                                      "far_10cm": None, "nan_rate": None}
    assert g2["depth"]["truth_box_far"]["n"] == 1
    json.loads(results_bytes({"g0": g0, "g1": g1, "g2": g2}))


def test_groups_by_pitch_size_row_label_and_edge():
    recs_b = [_rec(5, (0, 100, 40, 140)), _rec(6, (300, 100, 320, 120), tree=9)]
    dets_b = _dets([(0, 100, 40, 140, 0.9, 1), (300, 100, 320, 120, 0.8, 0), (600, 0, 640, 20, 0.7, 0)])
    groups = summarize_groups([_frame(), _frame(dets_b, recs_b, idx=1, pitch_deg=50.0, bagged_ids=frozenset({5}))],
                              OPS)
    assert sorted(groups) == ["all", "pitch_30", "pitch_50"]
    assert groups["pitch_30"]["ap50"] == pytest.approx(5 / 9) and groups["pitch_50"]["ap50"] == 1.0
    g = groups["all"]
    assert g["ap50"] == pytest.approx(11 / 15)
    assert g["recall_by_size"] == {"<16": {"n": 3, "tp": 2, "recall": pytest.approx(2 / 3)},
                                   "16-32": {"n": 1, "tp": 1, "recall": 1.0}, "32-64": {"n": 1, "tp": 1, "recall": 1.0},
                                   ">=64": {"n": 0, "tp": 0, "recall": None}}
    assert g["precision_by_size"]["<16"] == {"n": 4, "tp": 2, "precision": 0.5}
    assert g["precision_by_size"]["16-32"] == {"n": 2, "tp": 1, "precision": 0.5}
    assert g["recall_by_row"] == {"target": {"n": 3, "tp": 3, "recall": 1.0}, "other": {"n": 2, "tp": 1, "recall": 0.5}}
    assert g["other_row_share"] == 0.25
    assert g["label_confusion"] == {"fruit_fruit": 3, "fruit_bag": 0, "bag_fruit": 0, "bag_bag": 1}
    assert g["edge"] == {"both": 2, "truth_only": 0, "det_only": 0, "neither": 2}
    assert g["recall_by_edge"] == {"edge": {"n": 2, "tp": 2, "recall": 1.0},           # 1 and 5 touch x = 0
                                   "inner": {"n": 3, "tp": 2, "recall": pytest.approx(2 / 3)}}
    assert g["iou"] == {"p10": 1.0, "p50": 1.0, "p90": 1.0} and g["depth"] is None


def test_depth_errors_of_true_positives_and_truth_boxes():
    recs = [_rec(1, (0, 5, 10, 15), depth=2.0), _rec(2, (20, 5, 30, 15), depth=3.0)]
    dets = _dets([(0, 5, 10, 15, 0.9, 0), (20, 5, 30, 15, 0.8, 0), (80, 5, 90, 15, 0.7, 0)])
    d = summarize([_frame(dets, recs, depth_fn=_depth_fn)], OPS)["depth"]
    for det_key, truth_key in (("box_far", "truth_box_far"), ("box_median", "truth_box_median")):
        assert d[det_key] == d[truth_key]                       # here the true positives are the truth boxes
    assert (d["box_far"]["n"], d["box_far"]["far_10cm"], d["box_far"]["nan_rate"]) == (2, 0.5, 0.0)
    assert (d["box_median"]["n"], d["box_median"]["nan_rate"], d["box_median"]["p50_m"]) == (1, 0.5, 0.0)


def test_split_merge_and_ignored_overlap_diagnostics():
    recs = [_rec(1, (0, 0, 20, 20), visib=0.5), _rec(2, (100, 0, 110, 10)), _rec(3, (200, 0, 210, 10), visib=0.3),
            _rec(4, (210, 0, 220, 10), visib=0.4), _rec(5, (300, 0, 320, 20), visib=0.1, vis_px=50),
            _rec(6, (400, 0, 420, 20), visib=0.1, vis_px=300)]
    dets = _dets([(0, 0, 9, 20, 0.9, 0),          # left half of 1 (IoU 0.45): part
                  (11, 0, 20, 20, 0.8, 0),        # right half of 1: part; 1 is split and missed
                  (100, 0, 110, 10, 0.7, 0),      # TP on 2
                  (199, 0, 221, 10, 0.6, 0),      # covers 3 and 4 (IoU 0.45 each): merge
                  (310, 0, 330, 20, 0.55, 0),     # half on the ignored 5 (IoU 0.33): ign
                  (500, 100, 510, 110, 0.52, 0),  # background
                  (400, 0, 420, 20, 0.95, 0)])    # on the ignored 6: left out
    f = _frame(dets, recs)
    assert f.fp_kind.tolist() == ["part", "part", "", "merge", "ign", "bg", ""]
    assert f.near_of.tolist() == [0, 0, -1, 2, -1, -1, -1]
    g = summarize([f], OPS)
    op = g["at"]["operating"]
    assert (op["tp"], op["fp"], op["ignored"], op["fn"]) == (1, 5, 1, 3)
    assert (op["fp_dup"], op["fp_merge"], op["fp_part"], op["fp_ign"], op["fp_bg"]) == (0, 1, 2, 1, 1)
    assert (op["split_rate"], op["frag_only_rate"]) == (0.25, 0.25)
    assert op["split_rate_low_visib"] == pytest.approx(1 / 3)
    visib = {"<0.25": {"n": 0, "tp": 0, "recall": None}, "0.25-0.35": {"n": 1, "tp": 0, "recall": 0.0},
             "0.35-0.5": {"n": 1, "tp": 0, "recall": 0.0}, ">=0.5": {"n": 2, "tp": 1, "recall": 0.5}}
    assert g["recall_by_visib"] == {"operating": visib, "low": visib}     # every detection here is above 0.5
    assert g["ignored_hit"] == {"all": {"n": 2, "hit": 1, "rate": 0.5}, "vis_px>=100": {"n": 1, "hit": 1, "rate": 1.0}}
    kinds, near = fp_kinds(np.array([[0, 0, 9, 20]]), np.array([0], np.int8), np.array([-1]),
                           np.zeros((0, 4)), np.zeros((0, 4)))
    assert kinds.tolist() == ["bg"] and near.tolist() == [-1]
    faint = summarize([_frame(_dets([(0, 0, 20, 20, 0.3, 0)]), [_rec(1, (0, 0, 20, 20), visib=0.3)])], OPS)
    assert faint["recall_by_visib"]["operating"]["0.25-0.35"] == {"n": 1, "tp": 0, "recall": 0.0}
    assert faint["recall_by_visib"]["low"]["0.25-0.35"] == {"n": 1, "tp": 1, "recall": 1.0}   # kept between the two


def test_cross_class_pairs_count_overlapping_fruit_and_bag_boxes():
    raw = _dets([(0, 0, 10, 10, 0.4, 0), (0, 0, 10, 10.5, 0.3, 1),    # both below 0.5, together above
                 (50, 0, 60, 10, 0.9, 0), (50, 0, 60, 10, 0.2, 1),     # a pair, one already above
                 (100, 0, 110, 10, 0.45, 0), (100, 0, 110, 10, 0.45, 0)])   # same class: not a pair
    assert cross_class_pairs(raw, nms_iou=0.5, score=0.5) == (2, 1)
    assert cross_class_pairs(_dets([]), nms_iou=0.5, score=0.5) == (0, 0)


CRIT = {"detection": {"L5": {"pear": {"recall": {"ge": 0.90}, "precision": {"ge": 0.90}},
                             "apple": {"recall": {"ge": 0.85}, "precision": {"ge": 0.90}}}}}


def _groups(values):
    return {name: {"at": {"operating": {"recall": r, "precision": p}}} for name, (r, p) in values.items()}


def test_detection_verdicts_judge_the_val_pitches_only():
    groups = _groups({"all": (0.9, 0.9), "pitch_10": (0.1, 0.1), "pitch_30": (0.90, 0.80), "pitch_50": (0.84, 0.95)})
    lines = detection_verdicts(groups, CRIT, "apple", "val")
    assert [(x["group"], x["metric"], x["op"], x["limit"], x["value"], x["ok"]) for x in lines] == [
        ("pitch_30", "recall", "ge", 0.85, 0.90, True), ("pitch_30", "precision", "ge", 0.90, 0.80, False),
        ("pitch_50", "recall", "ge", 0.85, 0.84, False), ("pitch_50", "precision", "ge", 0.90, 0.95, True)]
    pear = detection_verdicts(_groups({"pitch_45": (0.95, 0.95)}), CRIT, "pear", "val")
    assert [(x["group"], x["value"], x["ok"]) for x in pear] == [
        ("pitch_45", 0.95, True), ("pitch_45", 0.95, True), ("pitch_70", None, None), ("pitch_70", None, None)]
    assert detection_verdicts(groups, CRIT, "apple", "dev") is None
    assert detection_verdicts(groups, None, "apple", "val") is None
    assert detection_verdicts(groups, {"y1": {}}, "apple", "val") is None
    assert JUDGED_PITCHES == {"pear": (45.0, 70.0), "apple": (30.0, 50.0)}


@pytest.fixture(scope="module")
def val_run(tmp_path_factory):
    """Twelve apple frames of the 30 deg passes with val seed 200."""
    return write_run(tmp_path_factory.mktemp("val") / "val_s200", crop="apple", pitches=(30.0,), seed=200,
                     max_frames=12)


def _free_box(recs, side=8, w=640, h=360):
    """The first side x side grid cell that no visible record's box touches."""
    boxes = [r.bbox for r in recs if r.bbox is not None]
    for y in range(0, h - side, side):
        for x in range(0, w - side, side):
            if all(x + side <= b[0] or b[2] <= x or y + side <= b[1] or b[3] <= y for b in boxes):
                return (x, y, x + side, y + side)
    raise AssertionError("no free cell")


@pytest.fixture(scope="module")
def val_cache(val_run, tmp_path_factory):
    """A cache of `val_run`: every count1 box at score 0.9 plus one background box per frame at 0.6."""
    run = SynthRun.open(val_run)
    truth = run.truth()
    dets = {}
    for f in run.frames:
        recs = truth.records[f.idx]
        rows = [(*r.bbox, 0.9, int(truth.fruits[r.id].bagged)) for r in recs if is_positive(r, "count1", truth.count1)]
        dets[f.idx] = _dets([*rows, (*_free_box(recs), 0.6, 0)])
    d = tmp_path_factory.mktemp("cache")
    write_cache(cache_path(d, run.meta.run_id), dets, {"run_id": run.meta.run_id, "fingerprint": run.meta.fingerprint})
    return d


def _val_cfg(tmp_path, run):
    raw = yaml.safe_load((ROOT / "configs" / "eval" / "apple_val.yaml").read_text(encoding="utf-8"))
    raw.update(runs=[str(run)])
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(p)


def test_eval_detmetrics_scores_a_cache_and_is_reproducible(val_run, val_cache, tmp_path):
    cfg_path = _val_cfg(tmp_path, val_run)
    args = ["detmetrics", "--config", cfg_path, "--cache", str(val_cache), "--splits", SPLITS, "--criteria", CRIT_FILE,
            "--nms-iou", "1.0", "--out", str(tmp_path / "out")]
    r1 = runner.invoke(eval_app, args)
    assert r1.exit_code == 0, r1.output
    (folder,) = (tmp_path / "out" / "apple_val").iterdir()
    res = json.loads((folder / "detmetrics.json").read_text(encoding="utf-8"))
    g = res["pooled"]["groups"]["pitch_30"]
    op = g["at"]["operating"]
    assert (res["split"], res["thresholds"]) == ("val", {"operating": 0.5, "low": 0.1})
    assert res["postprocess"] == {"nms_iou": 1.0, "edge_snap_px": 2.0} and g["n_frames"] == 12
    assert g["ap50"] == 1.0 and op["recall"] == 1.0 and op["fp_per_frame"] == 1.0 and op["fp_bg"] == 12
    assert op["precision"] == pytest.approx(g["n_pos"] / (g["n_pos"] + 12), abs=1e-6)
    assert [(x["group"], x["metric"]) for x in res["verdicts"]] == [
        ("pitch_30", "recall"), ("pitch_30", "precision"), ("pitch_50", "recall"), ("pitch_50", "precision")]
    assert res["verdicts"][2]["value"] is None
    assert set(g["depth"]) == {"box_far", "box_median", "truth_box_far", "truth_box_median"}
    run = SynthRun.open(val_run)
    entry = res["runs"][0]
    digest = CachedDetector(cache_path(val_cache, run.meta.run_id)).digest
    assert entry["meta"]["seed"] == 200 and entry["meta"]["depth_fingerprint"] == run.depth_fingerprint()
    assert set(entry["meta"]) - {"depth_fingerprint"} == set(describe_run(SimpleNamespace(run=run, passes={})))
    assert entry["cache"] == {"file": f"{run.meta.run_id}.npz", "digest": digest, "weights_sha256": None}
    assert load_yaml_model(folder / "config.yaml", EvalAppConfig) == load_yaml_model(cfg_path, EvalAppConfig)
    assert set(entry["cross_class"]) == {"pairs", "pairs_per_frame", "split_score"}
    rec = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    assert rec["detector"]["name"] == "cached" and rec["extra"]["caches"][0]["digest"] == digest
    assert (folder / "report.md").read_text(encoding="utf-8").startswith(f"# detmetrics apple_val {folder.name}")
    r2 = runner.invoke(eval_app, args)
    assert r2.exit_code == 0 and "reproduced (same detmetrics.json)" in r2.output, r2.output
    sub = score_runs([run], [CachedDetector(cache_path(val_cache, run.meta.run_id), nms_iou=1.0)],
                     load_yaml_model(cfg_path, EvalAppConfig), frames=[run.frames[0].idx, run.frames[1].idx])
    assert sub["pooled"]["groups"]["all"]["n_frames"] == 2


def test_eval_detmetrics_passes_its_postprocess_options_to_each_cache(val_run, tmp_path, monkeypatch):
    """`--nms-iou` and `--edge-snap-px` reach every `CachedDetector`: a shifted copy of a positive (IoU 2/3) is
    removed by NMS at 0.5 and kept, as a duplicate, at 1.0."""
    import counting.detection.cached as cached

    run = SynthRun.open(val_run)
    truth = run.truth()
    dets, n_copies = {}, 0
    for f in run.frames:
        rows = [(*r.bbox, 0.9, 0) for r in truth.records[f.idx] if is_positive(r, "count1", truth.count1)]
        if rows:
            x0, y0, x1, y1 = rows[0][:4]
            shift = (x1 - x0) / 5
            rows.append((x0 + shift, y0, x1 + shift, y1, 0.8, 0))
            n_copies += 1
        dets[f.idx] = _dets(rows)
    write_cache(cache_path(tmp_path / "cache", run.meta.run_id), dets,
                {"run_id": run.meta.run_id, "fingerprint": run.meta.fingerprint})
    seen = []

    class Spy(cached.CachedDetector):
        def __init__(self, path, **kwargs):
            seen.append(kwargs)
            super().__init__(path, **kwargs)

    monkeypatch.setattr(cached, "CachedDetector", Spy)
    cfg_path, dup = _val_cfg(tmp_path, val_run), {}
    for nms, snap in ((0.5, 2.0), (1.0, 3.0)):
        r = runner.invoke(eval_app, ["detmetrics", "--config", cfg_path, "--cache", str(tmp_path / "cache"), "--splits",
                                     SPLITS, "--nms-iou", f"{nms:g}", "--edge-snap-px", f"{snap:g}",
                                     "--out", str(tmp_path / "out")])
        assert r.exit_code == 0, r.output
        res = json.loads((Path(r.output.splitlines()[0]) / "detmetrics.json").read_text(encoding="utf-8"))
        assert res["postprocess"] == {"nms_iou": nms, "edge_snap_px": snap}
        dup[nms] = res["pooled"]["groups"]["all"]["at"]["operating"]["fp_dup"]
    assert seen == [{"fingerprint": run.meta.fingerprint, "nms_iou": 0.5, "edge_snap_px": 2.0},
                    {"fingerprint": run.meta.fingerprint, "nms_iou": 1.0, "edge_snap_px": 3.0}]
    assert n_copies > 0 and dup == {0.5: 0, 1.0: n_copies}


def test_eval_detmetrics_refuses_test_configs_and_bundles_of_other_splits_or_crops(val_run, val_cache, heldout_run,
                                                                                   apple_run, tmp_path):
    base = ["detmetrics", "--config", _val_cfg(tmp_path, val_run), "--cache", str(val_cache), "--splits", SPLITS,
            "--out", str(tmp_path / "out")]
    for extra in ([], ["--run", str(tmp_path / "missing")]):    # refused before any bundle is opened
        r = runner.invoke(eval_app, [*base, "--set", "split=test", "--set", "tuning=false", *extra])
        assert r.exit_code == 1 and "detection metrics are not computed on split=test; detection quality is judged " \
            "on val" in r.output, r.output
    for run, seed, split in ((heldout_run, 1000, "test"), (apple_run, 5, "dev")):
        r = runner.invoke(eval_app, [*base, "--run", str(run)])
        assert r.exit_code == 1 and f"seed {seed} belongs to split {split}, config split is val" in r.output
    pear = write_run(tmp_path / "pear_s200", crop="pear", pitches=(45.0,), seed=200, max_frames=2)
    r = runner.invoke(eval_app, [*base, "--run", str(pear)])
    assert r.exit_code == 2 and "crop mismatch: run is pear, config is apple" in r.output
    r = runner.invoke(eval_app, [*base, "--set", "tracker=1", "--set", "tracker.gate_m=2"])
    assert r.exit_code == 2 and "--set:" in r.output
    assert not (tmp_path / "out").exists()


def test_eval_detmetrics_needs_the_cache_of_each_bundle(val_run, apple_run, tmp_path):
    args = ["detmetrics", "--config", _val_cfg(tmp_path, val_run), "--splits", SPLITS, "--out", str(tmp_path / "out")]
    (tmp_path / "empty").mkdir()
    r = runner.invoke(eval_app, [*args, "--cache", str(tmp_path / "empty")])
    assert r.exit_code == 2 and "no detection cache for this bundle" in r.output
    run_id, other = SynthRun.open(val_run).meta.run_id, SynthRun.open(apple_run).meta
    write_cache(cache_path(tmp_path / "other", run_id), {0: _dets([])}, {"run_id": run_id, "fingerprint": other.fingerprint})
    r = runner.invoke(eval_app, [*args, "--cache", str(tmp_path / "other")])
    assert r.exit_code == 2 and "cache was made for bundle fingerprint" in r.output
