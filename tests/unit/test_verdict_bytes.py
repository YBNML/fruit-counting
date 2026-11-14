"""Byte digests of what a ladder's judgement gives on fixed hand-made results.

The run entries are plain dicts of hand-written numbers in the shape `evaluate` and `combine_seeds` give.
Nothing here opens a bundle, runs `evaluate` or fuses, so the digests are the same on every machine. Each
digest is the sha256 of `results_bytes` of one output (of the UTF-8 bytes for a string): the pooled
summaries, the error budget, and under two camera configuration names the verdicts, their summary and the
report, per crop; then the request extras of a ladder, a grid and a single run.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from counting.evaluation.budget import ladder_budget
from counting.evaluation.record import request_extra
from counting.evaluation.report import render_report, results_bytes
from counting.evaluation.verdict import ladder_pooled, ladder_verdicts, verdict_summary

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]
CRITERIA = ROOT / "configs" / "eval" / "criteria.yaml"
CRITERIA_POSE = ROOT / "configs" / "eval" / "criteria_pose.yaml"
PITCHES = {"apple": (10.0, 30.0, 50.0), "pear": (20.0, 45.0, 70.0)}
NOISE_SEEDS = (0, 1)
POSE_RUNGS = ["L3", "L4_drift", "LVI", "L5", "L5VI", "LVI_dd", "L4_drift_noimu"]
PLAIN_RUNGS = ["L0", "L1", "L2a", "L2b", "L2c", "L3", "L4_base", "L5", "L3_dd", "L5_L4base"]

# The rungs on a ladder's chain; any other rung is a report rung off the chain.
CHAIN = ("L0", "L1", "L2a", "L2b", "L2c", "L3", "L4_base", "L5", "L4_drift", "LVI", "L5VI")
# Rungs without noise; every other rung is noisy, one view per noise seed.
NOISE_FREE = ("L0", "L1", "L2a", "L2b", "L2c", "L3", "L5", "L3_dd")
# Rungs that lift with the detection depth; the others take the far box depth.
DET_DEPTH = ("L0", "L1", "L2a", "L2b", "L3_dd")
# (det, track, center, dedup, assign) of the rungs before L3; a dedup of None is the ladder's own mode.
EARLY_STAGES = {"L0": ("oracle", "oracle", "truth", "oracle", "truth"),
                "L1": ("oracle", "landmark", "truth", "oracle", "truth"),
                "L2a": ("oracle", "landmark", "truth", None, "truth"),
                "L2b": ("oracle", "landmark", "measured", None, "truth"),
                "L2c": ("oracle", "landmark", "measured", None, "truth")}
SCENES = ((300, "0a1b2c3d"), (301, "4e5f6a7b"))
FINGERPRINTS = ("6c1f0e9a2b3d4c5e", "a07d93be41c25f68")

# Hand-written ledger terms; a pass, or the row of a rung, takes the row that its indices pick.
TERMS = ("fp_bg", "fp_off", "boundary", "dup_track", "dup_cross", "miss_det", "miss_drop", "miss_merge")
TERM_ROWS = (
    (0, 1, 0, 2, 0, 1, 0, 0),
    (1, 0, 0, 0, 1, 0, 2, 1),
    (2, 0, 1, 1, 0, 0, 0, 1),
    (0, 0, 0, 0, 0, 3, 1, 0),
    (1, 2, 1, 0, 2, 0, 0, 0),
    (0, 1, 2, 1, 1, 2, 1, 0),
    (3, 0, 0, 1, 0, 1, 1, 2),
)
# Truth count G1 of each pass per scene (pitches in order, side A then B) and G6 of each tree per scene.
PASS_TRUTH = ((38, 41, 35, 44, 40, 37), (36, 43, 39, 42, 34, 45))
TREE_TRUTH = ((31, 28, 34), (29, 33, 30))
# The count change of the first two trees; the last tree takes the rest of the row's change.
TREE_SHIFT = ((1, -1), (0, 2), (-1, 0), (2, 0))
POS_ERR = ((0.011, 0.024), (0.013, 0.031), (0.009, 0.027), (0.016, 0.038))
# Scale of the pose quality and alignment numbers of a view.
SCALES = (1.0, 1.5, 0.75, 2.0, 1.25)

INDEPENDENT = {"pose_trans_sigma_m": 0.02, "pose_rot_sigma_deg": 0.5, "pass_bias_m": 0.02, "depth_rel_sigma": 0.02,
               "depth_dropout": 0.05, "detector": None}
VO = {"drift_step_trans_mm": 2.1, "drift_step_rot_deg": 0.039, "drift_scale_sigma": 0.0045,
      "drift_rate_deg_per_m": 0.045}
START = {"pass_side_trans_m": 0.05, "pass_side_rot_deg": 0.5}
RIG = {"rig_extr_trans_mm": 14.0, "rig_extr_rot_deg": 0.4}
IMU = {"imu_gyro_n": 0.00116, "imu_acc_n": 0.00497, "imu_gyro_w": 1.82e-05, "imu_acc_w": 0.000475,
       "imu_gyro_b0_dps": 1.0, "imu_acc_b0_mg": 4.0, "imu_temp_k": 1.0}
BELIEF = {"imu_time_offset_ms": 0.0, "imu_extr_trans_mm": 10.0, "imu_extr_rot_deg": 0.3}
ALIGN_SETTINGS = {"segment_m": 3.0, "pair_radius_m": 0.5, "roll_pitch_sigma_deg": 1.0, "min_pairs": 20,
                  "stress_row_bias_m": None, "stress_pair_keep": None, "oracle_sigma_m": None}
CACHE = {"file": "dets.npz", "digest": "5e" * 32, "weights_sha256": "9a" * 32, "pretrained": {"weights": "COCO_V1"}}
CELLS = [{"track": "oracle", "dedup": "oracle", "e_pipe": 0.0, "dup_track": 0, "dup_cross": 0},
         {"track": "oracle", "dedup": "merge", "e_pipe": 0.012, "dup_track": 0, "dup_cross": 1},
         {"track": "landmark", "dedup": "oracle", "e_pipe": -0.021, "dup_track": None, "dup_cross": None},
         {"track": "landmark", "dedup": "merge", "e_pipe": 0.034, "dup_track": 2, "dup_cross": 1}]
LADDER_SHA = "3f9c0b7e5a1d24c86e0f9a3b7c5d1e2f4a6b8c0d2e4f6a8b0c2d4e6f8a0b2c4d"
GRID_SHA = "b1e2c3d4a5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"
CRITERIA_SHA = "7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c5b4a39281706f5e4d3c2b1a09f8e"

# The digests of the outputs on the hand-made results; a change here means a change in what a ladder writes.
POSE_BYTES: dict[str, str] = {
    "apple.pooled": "9f7cec469ea167da1b5d8b9120ea4959a969c11f0a1cd65066801a760b8b493f",
    "apple.budget": "7d6fd0b6abb55314b5094bb94824f6bf7c892bbddce105f012ff2099634a8cf1",
    "apple.verdicts.c2": "4191f5ca05dfe891aa5dacdb86c088d7c945ec624cd1c108592d81fc994c7e23",
    "apple.summary.c2": "106dcc254588c34613d206cd08315889bb83080b26c52a68fdf420783f0801ef",
    "apple.report.c2": "95e940d5567403d43e8e80bbe6c5d57bf38d0e79b18ba78aa7e1ac33f25dcb59",
    "apple.verdicts.c1": "6e0cd712fe29dc919d4f969cd3b6eaea036b141b4088f1e2c05840325718753c",
    "apple.summary.c1": "3973e022e93220f9212c18d0d0c543ae7c309e46640da93a4a0314de999f5112",
    "apple.report.c1": "7d77cdb94336930d17718c9b4a3f946836162c2cf5e543a6353557d6d4c774a6",
    "pear.pooled": "9f7cec469ea167da1b5d8b9120ea4959a969c11f0a1cd65066801a760b8b493f",
    "pear.budget": "e7ab16cd7d3e4cdb1c087e35c4bf11465fe698ce0fb0abf1771a28ea03df3526",
    "pear.verdicts.c2": "9ad1cd6e892f4da5301b96fcc41b235723eae8e09604b2edbcbdfeb71ef1b83f",
    "pear.summary.c2": "15e14d6830730860c2138c7f42f250040a8fbdc1242a1d893095ee4d87ec3c4e",
    "pear.report.c2": "7669985d74011f85e25e98cdb85ea515a922bf91a93d60d9859e8b37c6af9036",
    "pear.verdicts.c1": "aa8400f05eb0354dd958a6202e756658b549122d22d83dd957ebe78c13fcb97a",
    "pear.summary.c1": "3973e022e93220f9212c18d0d0c543ae7c309e46640da93a4a0314de999f5112",
    "pear.report.c1": "055c6bba5a9a66ab0b8f9b5035ac01173212a40ec00346d127231a507868dc20",
    "extra.ladder": "164731a9b87315f6463ea07499a79b6181436fd72081f8202b0acfad222e9f2e",
    "extra.grid": "2de46915b90db5f49655ed2a7ba972a2804ff7d13e31c6528cd7c8de3253c5e7",
    "extra.run": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
PLAIN_BYTES: dict[str, str] = {
    "apple.pooled": "9565d730b48974ea202acd9503341a20a15bf93f17e228977c26b0991a7a9910",
    "apple.budget": "7d6b3a92e3bef0be18adf62dd09981a5eacb51852aa3faddb99aa30f7448e864",
    "apple.verdicts.c2": "0d9dd0257ae1113d3a7f0a70edc43237c7fda2f62be9f8fa4063d6221115d051",
    "apple.summary.c2": "1cf5c0a4fbec86aa26811c24ef8d547e93e2d45390358e4a7e16c8d0f5737d9f",
    "apple.report.c2": "abe80bcea5717fa7adb3bec5b2a5f607eb9c462dffccdfd3429b827cb9734c0f",
    "apple.verdicts.c1": "0d9dd0257ae1113d3a7f0a70edc43237c7fda2f62be9f8fa4063d6221115d051",
    "apple.summary.c1": "1cf5c0a4fbec86aa26811c24ef8d547e93e2d45390358e4a7e16c8d0f5737d9f",
    "apple.report.c1": "abe80bcea5717fa7adb3bec5b2a5f607eb9c462dffccdfd3429b827cb9734c0f",
    "pear.pooled": "9565d730b48974ea202acd9503341a20a15bf93f17e228977c26b0991a7a9910",
    "pear.budget": "99a6f3822815f3e9a2c0d3dc0fd971f10ba1e6cda24c304f2811e1504539aa0c",
    "pear.verdicts.c2": "1be11667e06589ae6998ac89d56c6be7c8be22333d346d5ecece7e1858360031",
    "pear.summary.c2": "b4236e7fae4b1a79f00d33e0b565f3c9f5c0f4f285acf1587090355b6832866e",
    "pear.report.c2": "b06810d6242889ecd6426a91264e9beb62b829d368caafb89dae0d8999d5a145",
    "pear.verdicts.c1": "1be11667e06589ae6998ac89d56c6be7c8be22333d346d5ecece7e1858360031",
    "pear.summary.c1": "b4236e7fae4b1a79f00d33e0b565f3c9f5c0f4f285acf1587090355b6832866e",
    "pear.report.c1": "b06810d6242889ecd6426a91264e9beb62b829d368caafb89dae0d8999d5a145",
    "extra.ladder": "4a5672d7d7020f08eab8fa3621ebb3d0faffe2ad49746445c045bb155a78f194",
    "extra.grid": "9812a2cfea80345e6ab659f2456eae2837074cff570b3b4e73d944511613ddef",
    "extra.run": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}


def _seed_mean(values: list):
    """The seeds' combined value, by the rule of the seed means: equal values stay, None if any is None, dicts
    by key, lists by element and numbers their mean."""
    first = values[0]
    if all(v == first for v in values[1:]):
        return first
    if any(v is None for v in values):
        return None
    if isinstance(first, dict):
        return {k: _seed_mean([v[k] for v in values]) for k in first}
    if isinstance(first, list):
        return [_seed_mean(list(xs)) for xs in zip(*values)]
    return sum(values) / len(values)


def _seed_range(values: list):
    """[min, max] of numbers, None if any is None, dicts by key."""
    if any(v is None for v in values):
        return None
    if isinstance(values[0], dict):
        return {k: _seed_range([v[k] for v in values]) for k in values[0]}
    return [min(values), max(values)]


def _ledger(truth: int, row: tuple[int, ...]) -> dict:
    """A ledger of `truth` true items with the terms `row`; its prediction count keeps the ledger identity."""
    return {"n_pred": truth + sum(row[:5]) - sum(row[5:]), "n_truth": truth, **dict(zip(TERMS, row)),
            "impure": row[0] % 2 + row[3] % 2}


def _rates(ledger: dict) -> dict:
    n = ledger["n_truth"]
    dup = ledger["dup_track"] + ledger["dup_cross"]
    return {"dup_rate": dup / n, "miss_rate": (ledger["miss_det"] + ledger["miss_drop"] + ledger["miss_merge"]) / n,
            "excess_rate": (dup + ledger["fp_off"] + ledger["fp_bg"]) / n, "fp_off_rate": ledger["fp_off"] / n,
            "merge_rate": ledger["impure"] / n}


def _y1(r: int, scene: int, seed: int, pitches, box_depth: bool) -> dict:
    passes = []
    for k, pitch in enumerate(pitches):
        for side_i, side in enumerate("AB"):
            i = 2 * k + side_i
            g1 = PASS_TRUTH[scene][i]
            ledger = _ledger(g1, TERM_ROWS[(r + i + 3 * scene + seed) % len(TERM_ROWS)])
            u1 = ledger["n_pred"]
            passes.append({"pitch": pitch, "side": side, "aisle": side_i, "n_frames": 60 + 4 * k,
                           "n_tracks": u1 + ledger["dup_track"] + 2, "n_no3d": (r + i) % 3, "U1": u1, "G1": g1,
                           "e1": (u1 - g1) / g1, "ledger": ledger})
    e1s = [abs(p["e1"]) for p in passes]
    total = {f: sum(p["ledger"][f] for p in passes) for f in ("n_truth", *TERMS, "impure")}
    rates = _rates(total)
    depth_err = None
    if box_depth:
        depth_err = {"n": 212 + 9 * scene, "p50_m": 0.014, "p90_m": 0.041 + 0.002 * scene, "p95_m": 0.057,
                     "near_10cm": 0.012, "far_10cm": 0.031}
    summary = {"e1_median": float(np.median(e1s)), "e1_max": max(e1s), "dup_rate": rates["dup_rate"],
               "miss_rate": rates["miss_rate"], "merge_rate": rates["merge_rate"],
               "fp_off_rate": rates["fp_off_rate"], "naive_U": sum(p["U1"] for p in passes), "depth_err": depth_err}
    return {"passes": passes, "summary": summary}


def _y6(r: int, scene: int, seed: int, dedup: str, naive_u: int) -> dict:
    g6s = TREE_TRUTH[scene]
    g6 = sum(g6s)
    ledger = _ledger(g6, TERM_ROWS[(2 * r + scene + seed + 1) % len(TERM_ROWS)])
    u = ledger["n_pred"]
    shift = TREE_SHIFT[(r + scene + seed) % len(TREE_SHIFT)]
    trees = []
    for t, (g, d) in enumerate(zip(g6s, (*shift, u - g6 - sum(shift)))):
        mis_in, mis_out = (t + r) % 2, (t + scene) % 2
        trees.append({"tree": t, "U": g + d, "U_truth_attr": g + d - mis_in + mis_out, "G6": g, "G6_count1": g,
                      "G6_any": g + 1, "G6_strict": g - 1, "G6_2": g + 2, "e_pipe": d / g, "e_cover": -2 / (g + 2),
                      "e_total": (d - 2) / (g + 2), "mis_in": mis_in, "mis_out": mis_out})
    row = {"U": u, "U_truth_attr": sum(t["U_truth_attr"] for t in trees), "G6": g6, "G6_count1": g6,
           "G6_any": g6 + 3, "G6_strict": g6 - 3, "G6_2": g6 + 6, "e_pipe": (u - g6) / g6, "e_cover": -6 / (g6 + 6),
           "e_total": (u - g6 - 6) / (g6 + 6), "naive_U": naive_u, "naive_ratio": naive_u / g6}
    rates = _rates(ledger)
    p50, p90 = POS_ERR[(r + scene + seed) % len(POS_ERR)]
    metrics = {"row_abs_e_pipe": abs(row["e_pipe"]),
               "tree_mae_over_mean": sum(abs(t["U"] - t["G6"]) for t in trees) / g6,
               "attribution_error": sum(abs(t["U"] - t["U_truth_attr"]) for t in trees) / g6,
               "pos_err_p50_m": p50, "pos_err_p90_m": p90, "miss_rate": rates["miss_rate"],
               "excess_rate": rates["excess_rate"]}
    ledger_any = {**ledger, "n_truth": g6 + 3, "miss_det": ledger["miss_det"] + 3}
    return {"mode": dedup, "n_landmarks": u + 5, "trees": trees, "row": row, "ledger": ledger,
            "ledger_any": ledger_any, "metrics": metrics}


def _errors(scale: float, *, frac: bool = False) -> dict:
    out = {"pos_m": 0.012 * scale, "rot_deg": 0.21 * scale, "along_m": 0.009 * scale, "side_m": 0.005 * scale,
           "height_m": 0.003 * scale, "yaw_deg": 0.17 * scale, "tilt_deg": 0.06 * scale}
    if frac:
        out["pos_frac"] = 0.0021 * scale
    return out


def _pose_quality(scene: int, scale: float, aligned: bool) -> dict:
    end_aligned = _errors(0.5 * scale) if aligned else dict.fromkeys(_errors(1.0))
    return {"rpe": {"n": 118 + scene, "trans_m": {"p50": 0.0012 * scale, "p90": 0.0031 * scale, "max": 0.0047 * scale},
                    "rot_deg": {"p50": 0.011 * scale, "p90": 0.024 * scale, "max": 0.039 * scale}},
            "passes": [],
            "summary": {"in_pass": _errors(scale, frac=True), "in_pass_cam": _errors(1.1 * scale, frac=True),
                        "start": _errors(2.0 * scale), "world_end": _errors(3.0 * scale),
                        "world_end_aligned": end_aligned},
            "interpass": {"same_side": {"n": 14, "p50_m": 0.006 * scale, "p90_m": 0.013 * scale},
                          "both_sides": {"n": 9 + scene, "p50_m": 0.011 * scale, "p90_m": 0.027 * scale}}}


def _alignment(seed: int, scale: float) -> dict:
    def segments(uses):
        return [{"centre_x_m": 1.5 + 3.0 * j, "n": 24 + 3 * j + seed, "use": use,
                 "residual_m": 0.004 * scale * (j + 1) if use == "own" else None,
                 "move_m": [0.002 * scale, -0.003 * scale * (j + 1), 0.001 * scale],
                 "turn_deg": [0.01 * scale, 0.02 * scale, -0.03 * scale]} for j, use in enumerate(uses)]

    stages = [{"stage": "same_side", "group": "A", "segments": segments(("own", "nearest"))},
              {"stage": "both_sides", "group": "all", "segments": segments(("own", "line"))},
              {"stage": "row_plane", "group": "all", "segments": segments(("own", "identity"))}]
    rows = [(s["stage"], row) for s in stages for row in s["segments"]]
    own = [row for stage, row in rows if stage != "row_plane" and row["use"] == "own"]
    summary = {"segments": len(rows), "own": 3, "nearest": 1, "line": 1, "group": 0, "identity": 1,
               "pairs_p50": float(np.median([row["n"] for row in own])),
               "residual_p50_m": float(np.median([row["residual_m"] for row in own])),
               "row_shift_p50_m": float(np.median([abs(row["move_m"][1]) for stage, row in rows
                                                   if stage == "row_plane"]))}
    return {"mode": ["tracks", "row_plane"], "settings": dict(ALIGN_SETTINGS), "stages": stages, "summary": summary}


def _stages(name: str, dedup: str) -> dict:
    if name in EARLY_STAGES:
        det, track, center, mode, assign = EARLY_STAGES[name]
    else:
        det, track, center, mode, assign = "cached" if name.startswith("L5") else "oracle", "landmark", "measured", \
            None, "grid"
    return {"det": det, "track": track, "center": center, "dedup": mode or dedup, "assign": assign}


def _pose(name: str, pitches) -> dict:
    """The `pose` entry of a pose ladder rung: L3 and L5 on the true poses, `drift` rungs on the drift model and
    the others on a file source; every rung but those two fused with an IMU (not `noimu`) and aligned."""
    truth, drift = name in ("L3", "L5"), "drift" in name
    source_file = not truth and not drift
    fuse = "none" if truth or "noimu" in name else "imu"
    cache = None
    if source_file:
        dd = name.endswith("_dd")
        cache = {"name": "vo_dd" if dd else "vo", "file": "pose.npz", "digest": ("3d" if dd else "2b") * 32,
                 "vo_config_hash": "52ad7d4423dea8fb" if dd else "b16b37e34b609dd2", "opencv": "4.10.0"}
    imu = None
    if fuse == "imu" and source_file:
        imu = {"source": "file", "name": "imu_ext", "file": "imu.npz", "digest": "7c" * 32, "profile": "ext"}
    elif fuse == "imu":
        imu = {"source": "memory", "name": None, "file": None, "digest": None, "profile": None}
    fusion = None
    if fuse == "imu":
        fusion = {"gtsam": "4.2", "fuse_fixed_sha256": "6f" * 32, "estimator": "lab", "standstill_s": None}
    return {"config": "c2", "source": "file" if source_file else "truth", "drift": drift, "anchor": "nominal",
            "start": None if truth else "nominal_from_plan", "rig": True, "rig_ref": pitches[0], "fuse": fuse,
            "align": "none" if truth else ["tracks", "row_plane"], "predict": "pose", "predict_window": None,
            "dedup": "hybrid", "dedup_radius_m": 0.09, "dedup_band_m": 0.05, "gate_px_frac": 0.5, "cache": cache,
            "imu": imu, "fusion": fusion, "flow": None}


def _noise(name: str, pose: dict | None) -> dict:
    if pose is None:
        return dict(INDEPENDENT)
    depth = {} if name.endswith("_dd") else {"depth_rel_sigma": 0.02, "depth_dropout": 0.05}
    base = {"pose_trans_sigma_m": 0.0, "pose_rot_sigma_deg": 0.0, "pass_bias_m": 0.0, "depth_rel_sigma": 0.0,
            "depth_dropout": 0.0, **depth, "detector": None}
    if pose["drift"]:
        return {**base, **VO, **START, **RIG, **({} if pose["fuse"] == "none" else {**IMU, **BELIEF})}
    return {**base, **START, **RIG, **BELIEF}


def _rung(name: str, r: int, scene: int, pitches, seeds, pose_ladder: bool) -> dict:
    """Rung `name` (index `r` in its ladder) on scene `scene`: one view, or under noise the seed mean of one view
    per noise seed, in the shape `combine_seeds` gives."""
    stages = _stages(name, "hybrid" if pose_ladder else "merge")
    lift_depth = "det_depth" if name in DET_DEPTH else "box_far"
    pose = _pose(name, pitches) if pose_ladder else None
    aligned = pose is not None and pose["align"] != "none"
    noisy = name not in NOISE_FREE
    views = []
    for seed in range(len(seeds)) if noisy else [0]:
        y1 = _y1(r, scene, seed, pitches, lift_depth != "det_depth")
        view = {"y1": y1, "y6": _y6(r, scene, seed, stages["dedup"], y1["summary"]["naive_U"])}
        if pose is not None:
            scale = SCALES[(r + 2 * scene + seed) % len(SCALES)]
            view["pose_quality"] = _pose_quality(scene, scale, aligned)
            view["alignment"] = _alignment(seed, scale) if aligned else None
        views.append(view)
    rung = {"stages": stages, "lift_depth": lift_depth}
    if not noisy:
        rung.update(y1=views[0]["y1"], y6=views[0]["y6"])
        if pose is not None:
            rung.update(pose=pose, pose_quality=views[0]["pose_quality"], alignment=views[0]["alignment"])
    else:
        spread = {"y1": {"summary": _seed_range([v["y1"]["summary"] for v in views])},
                  "y6": {k: _seed_range([v["y6"][k] for v in views]) for k in ("row", "metrics")}}
        rung.update(y1=_seed_mean([v["y1"] for v in views]), y6=_seed_mean([v["y6"] for v in views]), noisy=True,
                    seeds=list(seeds), noise=_noise(name, pose), spread=spread,
                    per_seed=[{"noise_seed": s, "y1": v["y1"], "y6": v["y6"]} for s, v in zip(seeds, views)])
        if pose is not None:
            rung.update(pose=pose, pose_quality=_seed_mean([v["pose_quality"] for v in views]))
            spread["pose_quality"] = {k: _seed_range([v["pose_quality"][k] for v in views])
                                      for k in ("rpe", "summary", "interpass")}
            rung["alignment"] = None
            if aligned:
                summaries = [v["alignment"]["summary"] for v in views]
                rung["alignment"] = {"mode": views[0]["alignment"]["mode"],
                                     "settings": views[0]["alignment"]["settings"], "summary": _seed_mean(summaries)}
                spread["alignment"] = {"summary": _seed_range(summaries)}
            for item, v in zip(rung["per_seed"], views):
                item.update(pose_quality=v["pose_quality"], alignment=v["alignment"])
    if stages["det"] == "cached":
        rung["cache"] = dict(CACHE)
    return rung


def _ladder_entries(names, crop: str, pitches, seeds) -> list[dict]:
    """Two scenes' ladder entries of rungs `names` for `crop`, with passes at `pitches` on both sides and noisy
    rungs over noise `seeds`. A ladder with a rung outside `PLAIN_RUNGS` is a pose ladder: each of its rungs has
    a pose entry and pose quality, and each scene a pass coverage; the others have a 2 x 2 table instead."""
    pose_ladder = not set(names) <= set(PLAIN_RUNGS)
    entries = []
    for scene, (seed, tag) in enumerate(SCENES):
        meta = {"run_id": f"{crop}_s{seed}_{tag}", "crop": crop, "preset": "row", "seed": seed, "samples": 64,
                "time": "noon", "n_frames": 360 + 12 * scene, "is_sequence": True, "base_step_m": 0.15,
                "fingerprint": FINGERPRINTS[scene], "synth_commit": "0f1e2d3"}
        if pose_ladder:
            g6 = sum(TREE_TRUTH[scene])
            meta["coverage"] = {"g6": g6, "g6_all": g6 + 8, "ratio": g6 / (g6 + 8)}
        entries.append({"meta": meta, "chain": [n for n in names if n in CHAIN],
                        "rungs": {n: _rung(n, r, scene, pitches, seeds, pose_ladder) for r, n in enumerate(names)},
                        "interaction_2x2": None if pose_ladder else {"cells": [dict(c) for c in CELLS]}})
    return entries


def _digest(value) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else results_bytes(value)
    return hashlib.sha256(data).hexdigest()


def _digests(names, criteria_path: Path, extras: dict[str, str]) -> dict[str, str]:
    criteria = yaml.safe_load(criteria_path.read_text(encoding="utf-8"))
    out: dict = {}
    for crop in ("apple", "pear"):
        entries = _ladder_entries(names, crop, PITCHES[crop], NOISE_SEEDS)
        pooled = ladder_pooled(entries, crop=crop)
        budget = ladder_budget(entries, crop)
        out[f"{crop}.pooled"], out[f"{crop}.budget"] = pooled, budget
        for label in ("c2", "c1"):
            verdicts = ladder_verdicts(entries, pooled, criteria, crop, config_label=label)
            results = {"schema_version": 11, "kind": "ladder", "run_name": f"{crop}_ladder",
                       "experiment_id": "5a17c0de42b9", "split": "test", "crop": crop, "runs": entries,
                       "pooled": pooled, "verdicts": verdicts, "budget": budget}
            out[f"{crop}.verdicts.{label}"] = verdicts
            out[f"{crop}.summary.{label}"] = verdict_summary(verdicts)
            out[f"{crop}.report.{label}"] = render_report(results, criteria=criteria)
    return {name: _digest(value) for name, value in {**out, **extras}.items()}


def _check(got: dict[str, str], pinned: dict[str, str], name: str) -> None:
    assert got == pinned, f"{name} = " + json.dumps(got, indent=4)


def test_pose_ladder_bytes_are_pinned():
    extras = {"extra.ladder": request_extra("ladder", ladder_sha256=LADDER_SHA, rungs=POSE_RUNGS,
                                            criteria_sha256=CRITERIA_SHA),
              "extra.grid": request_extra("sweep", grid_name="drift_vi.yaml", grid_sha256=GRID_SHA,
                                          ladder_sha256=LADDER_SHA, criteria_sha256=CRITERIA_SHA),
              "extra.run": request_extra("run")}
    _check(_digests(POSE_RUNGS, CRITERIA_POSE, extras), POSE_BYTES, "POSE_BYTES")


def test_plain_ladder_bytes_are_pinned():
    extras = {"extra.ladder": request_extra("ladder", ladder_sha256=LADDER_SHA, rungs=PLAIN_RUNGS, with_2x2=True,
                                            subsets=True, criteria_sha256=CRITERIA_SHA),
              "extra.grid": request_extra("sweep", grid_name="pose_noise.yaml", grid_sha256=GRID_SHA),
              "extra.run": request_extra("run")}
    _check(_digests(PLAIN_RUNGS, CRITERIA, extras), PLAIN_BYTES, "PLAIN_BYTES")
