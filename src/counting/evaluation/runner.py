"""Stage runner: check a bundle, run one stage setting through Y1 and Y6, and walk the oracle ladder.

`prepare` checks a bundle against the config and builds what every stage
setting shares: frames by pass, the oracle, noisy oracle or cached detections,
source ids and truth tables.
`evaluate` runs one setting of `cfg.stages` and scores it with the exact error
ledger. `evaluate_seeds` and `run_config` run a setting with noise once per
noise seed and combine the results (`combine_seeds`). `run_ladder` evaluates
the ladder rungs and the (track, dedup) 2 x 2 table of
`configs/eval/sweeps/ladder.yaml`; a rung that changes the detections (L5 of
`configs/eval/sweeps/ladder_l5.yaml`) runs on a context of its own.
Results are JSON-shaped dicts of plain Python values (str, int, float, bool, None).
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import yaml

from counting.config.eval_schema import EvalAppConfig, LiftConfig, SplitsConfig
from counting.data.formats.orchard_synth import (
    FrameInput, OrchardLayout, PassKey, SceneTruth, SynthRun, read_depth_m, select_frames,
)
from counting.detection.base import Detections
from counting.detection.cached import CachedDetector, cache_path
from counting.detection.oracle import OracleDetector
from counting.evaluation.ledger import Ledger, UnitView, build_ledger, combine
from counting.evaluation.match import match_frame, unit_majority
from counting.evaluation.oracle_stages import (
    oracle_dedup, oracle_tracks, truth_centers, truth_row_mask, truth_trees,
)
from counting.evaluation.posequality import apply_rigid, fit_truth_rigid, pose_quality
from counting.evaluation.posesource import (
    check_pose_supported, model_cameras, pose_cache_entry, pose_settings, rig_errors,
)
from counting.evaluation.seeds import combine_seeds
from counting.evaluation.truth import (
    SummaryMismatch, TruthTables, check_summary, is_positive, pass_subsets, truth_sets,
)
from counting.geometry.camera import Camera
from counting.geometry.noise import (
    STEP_BIAS, STEP_DEPTH, STEP_DEPTH_K, STEP_DET, STEP_POSE, depth_k_noise, noise_rng, pass_bias, pass_key_code,
    perturb_depth,
)
from counting.geometry.posefile import PoseFile
from counting.multiview.assign import assign_trees, in_target_row
from counting.multiview.dedup import dedup, repair_pass
from counting.multiview.lift import box_far_depth, box_median_depth, lift_points
from counting.odometry.cache import pose_cache_path
from counting.tracking.landmark import PassTracks, track_pass

VARIANTS = ("count1", "any", "strict")
FRAME_KEYS = ("crop", "split", "passes", "detector.positives")   # the frames and truth a ladder rung may not change
SUBSET_RUNGS = ("L3", "L5")   # the ladder rungs that get the pass-subset table under `subsets`


class CheckFailed(ValueError):
    """A bundle check failed (seed in another split, summary.json missing); exit code 1."""


class ProtectionError(CheckFailed):
    """A split=test request broke a test protection rule; exit code 1."""


@dataclass
class RunContext:
    """One checked bundle and everything stage settings share. `tables` keys are `VARIANTS`.

    `prepared` holds the config settings `prepare` checked or built from
    (`_prepared_settings`); `evaluate` refuses a config that differs in any.
    `cache` identifies the detection cache under `stages.det=cached`
    (`file`, `digest`, and the cache meta's `weights_sha256` and `pretrained`,
    None when the meta has none); it is None otherwise.
    `_pose_files` holds the pose caches `open_pose_file` opened, by (path, pinned digest or None).
    """

    run: SynthRun
    truth: SceneTruth
    layout: OrchardLayout
    passes: dict[PassKey, list[FrameInput]]
    dets: dict[int, Detections]
    src: dict[int, np.ndarray]
    tables: dict[str, TruthTables]
    prepared: dict[str, object]
    cache: dict | None = None
    _frame_of: dict[int, FrameInput] = field(init=False, repr=False)
    _box_depth: dict[tuple, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _pose_files: dict[tuple[str, str | None], PoseFile] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._frame_of = {f.idx: f for frames in self.passes.values() for f in frames}

    def box_depth(self, idx: int, lift: LiftConfig) -> np.ndarray:
        """Box depth of frame `idx`'s detections by `lift.depth` (`box_median` or `box_far`), read-only.

        Cached by (idx, method, the method's own parameters), so the depth16
        image is read once per key and a parameter the method does not use
        shares the cached array.
        """
        if lift.depth == "box_median":
            key: tuple = (idx, "box_median", float(lift.box_frac))
        elif lift.depth == "box_far":
            key = (idx, "box_far", float(lift.far_scale), float(lift.far_gap_m), float(lift.far_min_frac))
        else:
            raise ValueError(f"box_depth needs a box depth method, got lift.depth={lift.depth}")
        if key not in self._box_depth:
            depth = read_depth_m(self._frame_of[idx].depth_path)
            boxes = self.dets[idx].boxes
            if lift.depth == "box_median":
                z = box_median_depth(boxes, depth, lift.box_frac)
            else:
                z = box_far_depth(boxes, depth, lift.far_scale, lift.far_gap_m, lift.far_min_frac)
            z.flags.writeable = False
            self._box_depth[key] = z
        return self._box_depth[key]


def _prepared_settings(cfg: EvalAppConfig, det_seed: int | None = None) -> dict[str, object]:
    """The settings `prepare` checks (crop, split) or fixes in the context (frames by pass, detections, source ids).

    `det_seed` (the detection noise seed) is kept under `stages.det=oracle_noisy` only, else None.
    The cache folder, post-processing, pinned digests and score thresholds (`score_lo` drops
    boxes, `score_hi` orders the matching) are kept under `stages.det=cached` only, else None.
    """
    det, cached = cfg.detector, cfg.stages.det == "cached"
    return {"crop": cfg.crop, "split": cfg.split, "passes": cfg.passes.model_dump(),
            "detector.positives": det.positives, "stages.det": cfg.stages.det,
            "detector.noise": None if det.noise is None else det.noise.model_dump(),
            "det_seed": det_seed if cfg.stages.det == "oracle_noisy" else None,
            "detector.cache": det.cache if cached else None,
            "detector.post": det.post_settings().model_dump() if cached else None,
            "detector.digests": det.digests if cached else None,
            "tracker.score_hi": cfg.tracker.score_hi if cached else None,
            "tracker.score_lo": cfg.tracker.score_lo if cached else None}


def _check_config(ctx: RunContext, cfg: EvalAppConfig, det_seed: int | None = None) -> None:
    """Refuse a config that differs from the config `ctx` was prepared with.

    `evaluate` scores the frames, detections and source ids that `prepare`
    built, so a stage setting (a ladder rung) may change what `evaluate` runs
    (`stages` other than `det`, `lift`, `noise`, `tracker`, `dedup`, `truth`,
    `stop_after`) but none of `_prepared_settings(cfg, det_seed)`, which under
    `stages.det=cached` also holds the tracker's score thresholds.
    """
    for key, value in _prepared_settings(cfg, det_seed).items():
        had = ctx.prepared.get(key)
        if value != had:
            raise ValueError(f"{ctx.run.root}: {key}={value} differs from the prepared {key}={had}; "
                             "run prepare with this config")


def _check_frames(ctx: RunContext, cfg: EvalAppConfig) -> None:
    """Refuse a config whose `FRAME_KEYS` settings differ from the ones `ctx` was prepared with.

    A ladder rung may run on a context of its own (other detections), but on
    the same frames, passes and truth as the ladder's base context.
    """
    settings = _prepared_settings(cfg)
    for key in FRAME_KEYS:
        value, had = settings[key], ctx.prepared.get(key)
        if value != had:
            raise ValueError(f"{ctx.run.root}: {key}={value} differs from the prepared {key}={had}; "
                             "a rung may change the detector, not the frames")


def open_cache(run: SynthRun, cfg: EvalAppConfig) -> CachedDetector:
    """The detection cache of `run` in the folder `cfg.detector.cache`, opened as a `CachedDetector`.

    The file is `cache_path(detector.cache, run_id)`; a missing one raises
    `FileNotFoundError`, and with `detector.digests` a run it does not pin
    raises `ValueError`, both before the file is read. `CachedDetector` then
    checks the file, the bundle fingerprint and, with `detector.digests`, the
    pinned content digest (`ValueError` naming the cache path), and
    post-processes with `detector.post`.
    """
    run_id, digests = run.meta.run_id, cfg.detector.digests
    path = cache_path(Path(cfg.detector.cache).expanduser(), run_id)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no detection cache for this bundle")
    if digests is not None and run_id not in digests:
        raise ValueError(f"{run.root}: detector.digests pins no cache for run {run_id}")
    return CachedDetector(path, fingerprint=run.meta.fingerprint, digest=None if digests is None else digests[run_id],
                          **cfg.detector.post_settings().model_dump())


def open_pose_file(ctx: RunContext, cfg: EvalAppConfig) -> PoseFile | None:
    """The pose cache of `ctx`'s bundle under `pose.source=file`, opened once per context; None otherwise.

    The folder is `pose.caches[pose.cache].dir` and the file `pose_cache_path(dir, run_id)`; a
    missing one raises `FileNotFoundError`, and with that cache's `digests` set a run it does not
    pin raises `ValueError`, both before the file is read. `PoseFile` then checks the bundle
    fingerprint, the run id and, with `digests`, the pinned content digest (`ValueError` naming the
    cache path). The opened file is kept in the context by (path, pinned digest or None).
    """
    if cfg.pose is None or cfg.pose.source != "file":
        return None
    run, name = ctx.run, cfg.pose.cache
    files, run_id = cfg.pose.caches[name], run.meta.run_id
    path = pose_cache_path(Path(files.dir).expanduser(), run_id)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no pose cache for this bundle")
    if files.digests and run_id not in files.digests:
        raise ValueError(f"{run.root}: pose.caches.{name}.digests pins no cache for run {run_id}")
    pin = files.digests.get(run_id)
    key = (str(path), pin)
    if key not in ctx._pose_files:
        ctx._pose_files[key] = PoseFile(path, fingerprint=run.meta.fingerprint, run_id=run_id, digest=pin)
    return ctx._pose_files[key]


def _match_by_score(dets, recs, score_hi, prefer):
    """Source ids of `dets`: the boxes scoring at least `score_hi` are matched first, then the rest
    against the records still free in this frame (both with `match_frame`)."""
    out = np.full(len(dets), -1, dtype=np.int64)
    hi, lo = np.flatnonzero(dets.scores >= score_hi), np.flatnonzero(dets.scores < score_hi)
    out[hi] = match_frame(dets.subset(hi), recs, prefer=prefer)
    used = set(out[hi][out[hi] >= 0].tolist())
    if len(lo):
        out[lo] = match_frame(dets.subset(lo), [r for r in recs if r.id not in used], prefer=prefer)
    return out


def prepare(
    run: SynthRun, cfg: EvalAppConfig, splits: SplitsConfig, *, truth: SceneTruth | None = None,
    det_seed: int | None = None, allow_test: bool = False,
) -> RunContext:
    """Check `run` against `cfg`, then build its detections, source ids and truth tables.

    Checks, in order (each message starts with the bundle or cache path): crop,
    split=test without `allow_test` (`ProtectionError`), the seed's split
    (`CheckFailed`), sequence bundle, the summary.json self-check
    (`SummaryMismatch`, or `CheckFailed` when the file is missing), that
    `det_seed` is given exactly when `stages.det=oracle_noisy`, and, under
    `stages.det=cached`, the cache (`open_cache`: the file exists
    (`FileNotFoundError`), `detector.digests` pins the run, then the file, the
    bundle fingerprint and the pinned digest (`ValueError`)).
    The detections are the oracle's, the noisy oracle's or, under
    `stages.det=cached`, the cache's after `detector.post`, without the boxes
    scoring under `tracker.score_lo`. Source ids come from `match_frame`; for
    cached detections it runs twice per frame (`_match_by_score`): the boxes
    scoring at least `tracker.score_hi` first, then the rest against the
    records they left free.
    `allow_test` lets a split=test config on to the seed check; only the protected eval commands pass it.
    `det_seed` is the detection noise seed under `stages.det=oracle_noisy`, and only there.
    `truth` replaces `run.truth()` (leak tests pass a relabelled copy).
    """
    meta, root = run.meta, run.root
    if meta.crop != cfg.crop:
        raise ValueError(f"{root}: crop mismatch: run is {meta.crop}, config is {cfg.crop}")
    if cfg.split == "test" and not allow_test:
        raise ProtectionError(f"{root}: split=test needs prepare(..., allow_test=True), "
                              "which only the protected eval commands pass")
    actual = splits.split_of(meta.seed) or "none"
    if actual != cfg.split:
        raise CheckFailed(f"{root}: seed {meta.seed} belongs to split {actual}, config split is {cfg.split}")
    if not meta.is_sequence:
        raise ValueError(f"{root}: not a sequence bundle (max_frames or irregular step)")
    truth = run.truth() if truth is None else truth
    try:
        check_summary(truth, run.layout, run.passes())
    except SummaryMismatch as e:
        raise SummaryMismatch(f"{root}: {e}") from e
    except ValueError as e:
        raise CheckFailed(f"{root}: {e}") from e
    noisy_det, cached = cfg.stages.det == "oracle_noisy", cfg.stages.det == "cached"
    if noisy_det and det_seed is None:
        raise ValueError(f"{root}: stages.det=oracle_noisy needs a detection noise seed")
    if not noisy_det and det_seed is not None:
        raise ValueError(f"{root}: det_seed={det_seed} given, but stages.det={cfg.stages.det} adds no detection noise")

    frames = select_frames(run.frames, base_step_m=meta.base_step_m, pitches=cfg.passes.pitches,
                           sides=cfg.passes.sides, step_m=cfg.passes.step_m)
    passes: dict[PassKey, list[FrameInput]] = {}
    for f in frames:
        passes.setdefault(f.pass_key, []).append(f)
    cache = None
    if cached:
        detector = open_cache(run, cfg)
        cache = {"file": detector.path.name, "digest": detector.digest,
                 "weights_sha256": detector.meta.get("weights_sha256"), "pretrained": detector.meta.get("pretrained")}
    elif noisy_det:
        detector = OracleDetector(truth, cfg.detector.positives, cfg.detector.noise,
                                  rng_key=(cfg.seed, det_seed, STEP_DET, meta.seed))
    else:
        detector = OracleDetector(truth, cfg.detector.positives)
    dets: dict[int, Detections] = {}
    for f in frames:
        D = detector.detect(f)
        dets[f.idx] = D.subset(np.flatnonzero(D.scores >= cfg.tracker.score_lo)) if cached else D

    def prefer(r) -> bool:   # first tie-break: prefer the detector's own positives
        return is_positive(r, cfg.detector.positives, truth.count1)

    if cached:
        src = {f.idx: _match_by_score(dets[f.idx], truth.records[f.idx], cfg.tracker.score_hi, prefer)
               for f in frames}
    else:
        src = {f.idx: match_frame(dets[f.idx], truth.records[f.idx], prefer=prefer) for f in frames}
    tables = {v: truth_sets(truth, run.layout, passes, v) for v in VARIANTS}
    return RunContext(run=run, truth=truth, layout=run.layout, passes=passes, dets=dets, src=src, tables=tables,
                      prepared=_prepared_settings(cfg, det_seed), cache=cache)


def describe_run(ctx: RunContext) -> dict:
    """Bundle identity for results (no paths). `n_frames` counts the frames scored."""
    m = ctx.run.meta
    return {"run_id": m.run_id, "crop": m.crop, "preset": m.preset, "seed": m.seed, "samples": m.samples,
            "time": m.time, "n_frames": sum(len(f) for f in ctx.passes.values()), "is_sequence": m.is_sequence,
            "base_step_m": m.base_step_m, "fingerprint": m.fingerprint, "synth_commit": m.synth_commit}


def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def _median(xs: Sequence[float]) -> float | None:
    return float(np.median(xs)) if len(xs) else None


def _percentile(xs: Sequence[float], q: float) -> float | None:
    return float(np.percentile(xs, q)) if len(xs) else None


def depth_error_stats(err: np.ndarray) -> dict | None:
    """Stats of `err` (box depth minus detection depth, m; finite values only), or None when empty.

    `near_10cm`/`far_10cm` are the fraction of `err` strictly beyond -0.10 m / +0.10 m.
    """
    if err.size == 0:
        return None
    abs_err = np.abs(err)
    n = int(err.size)
    return {"n": n, "p50_m": float(np.percentile(abs_err, 50)), "p90_m": float(np.percentile(abs_err, 90)),
            "p95_m": float(np.percentile(abs_err, 95)), "near_10cm": float(np.count_nonzero(err < -0.10) / n),
            "far_10cm": float(np.count_nonzero(err > 0.10) / n)}


def _units(majority: np.ndarray, purity: np.ndarray, obs: Sequence[tuple[int, ...]], sel) -> list[UnitView]:
    return [UnitView(int(majority[u]), float(purity[u]), obs[u]) for u in sel]


@dataclass(frozen=True, eq=False)
class _PassStage:
    """One pass's stage results that Y6 reuses; `detected` is the set of truth ids its detections matched."""

    key: PassKey
    tracks: PassTracks
    keep: np.ndarray
    maj: np.ndarray
    obs: list[tuple[int, ...]]
    detected: set[int]
    U1: int


@dataclass(frozen=True, eq=False)
class _PassTrack:
    """One pass after tracking: the cameras the lift and the tracker used, and the tracks they gave.

    `frames` are the pass's frames in `ctx.passes`, with the truth cameras that scoring uses;
    `cameras` maps each frame idx to the camera the lift and the tracker used. `tracks` are the
    tracks after in-pass repair (or the oracle tracks). `depth_errs` are the pass's box-depth
    errors in frame order, and `depth_err` is their `depth_error_stats`.
    """

    key: PassKey
    frames: list[FrameInput]
    cameras: dict[int, Camera]
    tracks: PassTracks
    depth_errs: list[np.ndarray]
    depth_err: dict | None


def _track_passes(ctx: RunContext, cfg: EvalAppConfig, noise_seed: int | None) -> list[_PassTrack]:
    """Lift and track every pass of `ctx`, in `ctx.passes` order (the first stage of `evaluate`).

    Each frame starts from the camera of the pose source (`model_cameras`: the rig and the drift
    model under `cfg.pose`, or the pose cache of `open_pose_file` under `pose.source=file`), or
    from its true camera when that gives None. With `noise_seed`, a pass's bias is drawn before
    its frames, and each frame's camera then gets `Camera.perturbed`, then its chosen depth
    `perturb_depth` (after the depth error is taken) and, with `noise.depth_k` set,
    `depth_k_noise` on the depth before noise. The lift and the tracker use these cameras;
    `track_pass` runs once per pass on the frames that carry them.
    """
    st, scene, nz = cfg.stages, ctx.run.meta.seed, cfg.noise
    cams = model_cameras(ctx.run.frames, ctx.passes, cfg, noise_seed, scene=scene,
                         base_step_m=ctx.run.meta.base_step_m, plans=ctx.run.plans(),
                         pose_file=open_pose_file(ctx, cfg))
    out: list[_PassTrack] = []
    for key, frames in ctx.passes.items():
        bias = None if noise_seed is None else pass_bias(
            noise_rng(cfg.seed, noise_seed, STEP_BIAS, scene, *pass_key_code(key)), nz.pass_bias_m)
        points, seen = {}, []   # `seen`: the frames with the cameras the tracker and the lift use
        depth_errs: list[np.ndarray] = []
        for f in frames:
            D = ctx.dets[f.idx]
            z = D.depth_m if cfg.lift.depth == "det_depth" else ctx.box_depth(f.idx, cfg.lift)
            if cfg.lift.depth != "det_depth" and D.depth_m is not None:
                e = z - D.depth_m
                depth_errs.append(e[np.isfinite(e)])
            if cams is not None:
                f = replace(f, camera=cams[key][f.idx])
            if noise_seed is not None:
                f = replace(f, camera=f.camera.perturbed(noise_rng(cfg.seed, noise_seed, STEP_POSE, scene, f.idx),
                                                         nz.pose_trans_sigma_m, nz.pose_rot_sigma_deg, bias))
                clean = z
                z = perturb_depth(z, noise_rng(cfg.seed, noise_seed, STEP_DEPTH, scene, f.idx),
                                  nz.depth_rel_sigma, nz.depth_dropout)
                if nz.depth_k is not None:
                    z = z + depth_k_noise((cfg.seed, noise_seed, STEP_DEPTH_K, scene, f.idx), clean, nz.depth_k)
            points[f.idx] = lift_points(D.boxes, D.labels, z, f.camera, cfg.lift)
            seen.append(f)
        depth_err = depth_error_stats(np.concatenate(depth_errs)) if depth_errs else None
        if st.track == "oracle":   # one track per id already: no in-pass repair
            tracks = oracle_tracks(key, seen, ctx.dets, ctx.src, points)
        else:
            tracks = repair_pass(track_pass(seen, ctx.dets, points, cfg.tracker, method=st.track),
                                 cfg.tracker.repair_m)
        out.append(_PassTrack(key=key, frames=frames, cameras={f.idx: f.camera for f in seen}, tracks=tracks,
                              depth_errs=depth_errs, depth_err=depth_err))
    return out


def _align_passes(
    ctx: RunContext, cfg: EvalAppConfig, passes: list[_PassTrack],
) -> tuple[list[_PassTrack], dict[PassKey, np.ndarray] | None]:
    """Align the tracked passes by `cfg.align.mode` (the second stage of `evaluate`).

    Returns the passes in the same order, with their track centres moved, and the transform
    applied to each pass by pass key. Without an `align` section, or with `align.mode=none`,
    the passes come back unchanged with None. `fit_truth_pass` moves each pass's track centres by
    the rigid transform that best takes its used cameras to its true cameras (`fit_truth_rigid`);
    a centre without 3D stays NaN. Any other mode raises `NotImplementedError`.
    """
    if cfg.align is None or cfg.align.mode == "none":
        return passes, None
    if cfg.align.mode == "fit_truth_pass":
        out, transforms = [], {}
        for p in passes:
            A = fit_truth_rigid([f.camera for f in p.frames], [p.cameras[f.idx] for f in p.frames])
            out.append(replace(p, tracks=replace(p.tracks, X=apply_rigid(A, p.tracks.X))))
            transforms[p.key] = A
        return out, transforms
    mode = ",".join(cfg.align.mode) if isinstance(cfg.align.mode, list) else cfg.align.mode
    raise NotImplementedError(f"align.mode={mode} is not supported")


def _score_pass(ctx: RunContext, cfg: EvalAppConfig, p: _PassTrack) -> tuple[dict, Ledger, _PassStage]:
    """Score one tracked pass against the truth of `ctx` (the third stage of `evaluate`).

    Keeps the tracks with at least `cfg.tracker.min_obs_for(stages.det)` observations, a 3D
    centre and a place in the target row. Returns the pass's `y1.passes` row, its
    `cfg.truth.variant` ledger and the `_PassStage` that Y6 reuses.
    """
    st, key, frames = cfg.stages, p.key, p.frames
    G, G_any = ctx.tables[cfg.truth.variant], ctx.tables["any"]
    obs = [tuple(int(ctx.src[f][d]) for f, d in m) for m in p.tracks.members()]
    maj, pur = unit_majority(obs)
    X = truth_centers(maj, p.tracks.X, ctx.truth) if st.center == "truth" else p.tracks.X
    tracks = replace(p.tracks, X=X)
    if st.assign == "truth":
        row, tree = truth_row_mask(maj, X, ctx.layout, ctx.truth), truth_trees(maj, X, ctx.layout, ctx.truth)
    else:
        row, tree = in_target_row(X, ctx.layout), assign_trees(X, ctx.layout)
    enough = tracks.n_obs >= cfg.tracker.min_obs_for(st.det)
    has3d = np.isfinite(X).all(axis=1)
    keep = enough & has3d & row
    kept = np.flatnonzero(keep)

    detected = {int(s) for f in frames for s in ctx.src[f.idx].tolist() if s >= 0}
    units = _units(maj, pur, obs, kept)
    common = dict(units=units, track_majorities=[(0, int(maj[t])) for t in kept],
                  kept_obs_ids={i for t in kept for i in obs[t]}, detected_ids=detected)
    L = build_ledger(truth_ids=G.g1[key], target_ids=G.target_ids, **common)
    L_any = build_ledger(truth_ids=G_any.g1[key], target_ids=G_any.target_ids, **common)

    U1, G1 = len(kept), len(G.g1[key])
    g1_tree = G.per_tree(G.g1[key])
    row_out = {
        "pitch": key.pitch_deg, "side": key.side, "aisle": key.aisle, "n_frames": len(frames),
        "n_tracks": len(tracks.X), "n_no3d": int((enough & ~has3d).sum()),
        "U1": U1, "G1": G1, "e1": _ratio(U1 - G1, G1),
        "trees": {str(t): {"U1": int((tree[kept] == t).sum()), "G1": g1_tree[t]} for t in ctx.layout.tree_ids},
        "ledger": L.as_dict(), "ledger_any": L_any.as_dict(), "depth_err": p.depth_err,
    }
    return row_out, L, _PassStage(key, tracks, keep, maj, obs, detected, U1)


def evaluate(ctx: RunContext, cfg: EvalAppConfig, *, subsets: bool = False, noise_seed: int | None = None) -> dict:
    """Run the stage setting `cfg.stages` on `ctx` and score Y1 (per pass) and Y6 (per tree and row).

    `G1`/`G6` are the `cfg.truth.variant` truth; a ratio with a zero
    denominator, and the median/max/percentile of an empty set, is None.
    A pose setting the runner does not run raises `NotImplementedError` before
    anything else (`check_pose_supported`).
    `cfg` is then checked (`_check_config`, against `noise_seed` as the
    detection noise seed, or the prepared one without it): a config that
    differs from the prepared one raises `ValueError`. Then `noise_seed` must
    be given exactly when `cfg.is_noisy()` (`ValueError`).
    The passes go through three stages, each in `ctx.passes` order: every pass
    is lifted and tracked (`_track_passes`), the passes are aligned
    (`_align_passes`), and each pass is scored (`_score_pass`).
    A track is kept with at least `cfg.tracker.min_obs_for(stages.det)`
    observations. A cached context's result also has `cache` (`ctx.cache`).
    Each pass's and the summary's `depth_err` is `depth_error_stats` of box depth
    minus detection depth (`None` under `lift.depth=det_depth`, without detector
    depth, or with no detections).
    The tracker and the lift use the cameras of the pose source, `model_cameras`:
    the rig and the drift model under `cfg.pose` (under `pose.source=file`, the
    pose cache that `open_pose_file` opens once per context), else the true
    cameras. With `noise_seed`, every such camera gets `Camera.perturbed`, and
    every chosen depth gets `perturb_depth` (and, with `noise.depth_k`,
    `depth_k_noise`) after `depth_err` is taken; scoring always uses the truth.
    With a pose section the result also has `pose` (`pose_settings`; its `cache`
    is `pose_cache_entry` of the pose cache read, None under `pose.source=truth`)
    and `pose_quality` (`pose_quality` of the cameras the lift and the tracker used
    against the true cameras, with the alignment transforms; under `pose.rig`
    the in-pass drift and RPE are measured in the frame of the reference camera
    `pose.rig_ref`, with the rig calibration errors of the passes' pitches).
    With `subsets` (which needs `stop_after=y6`), the result also has `subsets`
    (`_score_subsets`): the passes' tracks are reused, and Y6 is scored again on
    each subset of `pass_subsets`.
    """
    check_pose_supported(cfg)
    _check_config(ctx, cfg, ctx.prepared.get("det_seed") if noise_seed is None else noise_seed)
    root = ctx.run.root
    if cfg.is_noisy() and noise_seed is None:
        raise ValueError(f"{root}: the config has noise, so evaluate needs a noise_seed")
    if not cfg.is_noisy() and noise_seed is not None:
        raise ValueError(f"{root}: noise_seed={noise_seed} given, but the config has no noise")
    if subsets and cfg.stop_after == "y1":
        raise ValueError("--subsets needs stop_after=y6")
    st = cfg.stages
    staged = _track_passes(ctx, cfg, noise_seed)
    aligned, transforms = _align_passes(ctx, cfg, staged)
    passes_out, ledgers = [], []
    stages: list[_PassStage] = []
    depth_errs_all: list[np.ndarray] = []
    for p in aligned:
        row_out, L, stage = _score_pass(ctx, cfg, p)
        passes_out.append(row_out)
        ledgers.append(L)
        stages.append(stage)
        depth_errs_all.extend(p.depth_errs)

    e1s = [abs(p["e1"]) for p in passes_out if p["e1"] is not None]
    rates = combine(ledgers).rates()
    naive_U = sum(p["U1"] for p in passes_out)
    depth_err_summary = depth_error_stats(np.concatenate(depth_errs_all)) if depth_errs_all else None
    y1 = {"passes": passes_out,
          "summary": {"e1_median": _median(e1s), "e1_max": max(e1s) if e1s else None,
                      "dup_rate": rates["dup_rate"], "miss_rate": rates["miss_rate"],
                      "merge_rate": rates["merge_rate"], "fp_off_rate": rates["fp_off_rate"], "naive_U": naive_U,
                      "depth_err": depth_err_summary}}
    y6 = None
    if cfg.stop_after != "y1":
        y6 = _y6(ctx, cfg, stages, ctx.tables)
    out = {"stages": st.model_dump(), "lift_depth": cfg.lift.depth, "y1": y1, "y6": y6}
    if cfg.pose is not None:
        out["pose"] = pose_settings(cfg, noise_seed, planned=ctx.run.meta.shake,
                                    cache=pose_cache_entry(cfg, open_pose_file(ctx, cfg)))
        out["pose_quality"] = pose_quality(
            [(p.key, [f.camera for f in p.frames], [p.cameras[f.idx] for f in p.frames]) for p in staged],
            ctx.layout, transforms, ref_pitch=cfg.pose.rig_ref if cfg.pose.rig else None,
            rig_errors=rig_errors(cfg, noise_seed, scene=ctx.run.meta.seed, pitches=[p.key.pitch_deg for p in staged]))
    if ctx.cache is not None:
        out["cache"] = _cache_result(ctx)
    if subsets:
        out["subsets"] = _score_subsets(ctx, cfg, stages, y6)
    return out


def _score_subsets(ctx: RunContext, cfg: EvalAppConfig, stages: Sequence[_PassStage], y6_all: dict) -> list[dict]:
    """One row per subset S of `pass_subsets`: Y6 scored on S's passes alone, against S's truth and the whole truth.

    S's truth tables are rebuilt from S's frames. `U`, `G6`, `e_pipe`, `ledger`
    and `tree_mae_over_mean` (the `cfg.truth.variant` row ledger and its Y6
    metric, both against S's own truth) are S's Y6 row; `e_all` and
    `tree_mae_over_mean_all` compare S's counts with the `G6` of every pass
    (`y6_all`). A subset with no pass has None everywhere but `S` and `n_passes`.
    """
    g_all = {t["tree"]: t["G6"] for t in y6_all["trees"]}
    G_all = y6_all["row"]["G6"]
    rows = []
    for name, keys in pass_subsets(list(ctx.passes)):
        if not keys:
            rows.append({"S": name, "n_passes": 0, "U": None, "G6": None, "e_pipe": None, "e_all": None,
                         "tree_mae_over_mean_all": None, "tree_mae_over_mean": None, "ledger": None})
            continue
        sel = [s for s in stages if s.key in keys]
        tables = {v: truth_sets(ctx.truth, ctx.layout, {k: ctx.passes[k] for k in keys}, v) for v in VARIANTS}
        y6 = _y6(ctx, cfg, sel, tables)
        U = y6["row"]["U"]
        rows.append({"S": name, "n_passes": len(keys), "U": U, "G6": y6["row"]["G6"], "e_pipe": y6["row"]["e_pipe"],
                     "e_all": _ratio(U - G_all, G_all),
                     "tree_mae_over_mean_all": _ratio(sum(abs(t["U"] - g_all[t["tree"]]) for t in y6["trees"]),
                                                      sum(g_all.values())),
                     "tree_mae_over_mean": y6["metrics"]["tree_mae_over_mean"], "ledger": y6["ledger"]})
    return rows


def _y6(ctx: RunContext, cfg: EvalAppConfig, stages: Sequence[_PassStage], tables: Mapping[str, TruthTables]) -> dict:
    """Dedup the kept tracks of `stages` into landmarks, attribute them to trees, and score the row.

    `tables` (keys `VARIANTS`) is the truth of the frames of `stages`' passes: `ctx.tables` for every pass.
    """
    st, layout, truth = cfg.stages, ctx.layout, ctx.truth
    G, G_any = tables[cfg.truth.variant], tables["any"]
    T = G.target_ids
    tracks_list, keep_list = [s.tracks for s in stages], [s.keep for s in stages]
    maj_list, obs_list = [s.maj for s in stages], [s.obs for s in stages]
    detected_all = set().union(*(s.detected for s in stages))
    naive_U = sum(s.U1 for s in stages)
    if st.dedup == "oracle":
        lm, mode = oracle_dedup(tracks_list, keep_list, maj_list), "oracle"
    else:
        lm, mode = dedup(tracks_list, keep_list, layout, cfg.dedup, mode=st.dedup), st.dedup
    pass_index = {t.pass_key: i for i, t in enumerate(tracks_list)}
    members = [[(pass_index[k], t) for k, t in m] for m in lm.members]
    lm_obs = [tuple(o for i, t in m for o in obs_list[i][t]) for m in members]
    lm_maj, lm_pur = unit_majority(lm_obs)
    truth_tree = truth_trees(lm_maj, lm.X, layout, truth)
    pred_tree = truth_tree if st.assign == "truth" else assign_trees(lm.X, layout)

    row_sel = np.flatnonzero(pred_tree >= 0)
    kept_obs = {o for u in row_sel for o in lm_obs[u]}

    def ledger(truth_ids, target_ids, sel):
        return build_ledger(truth_ids=truth_ids, target_ids=target_ids, units=_units(lm_maj, lm_pur, lm_obs, sel),
                            track_majorities=[(i, int(maj_list[i][t])) for u in sel for i, t in members[u]],
                            kept_obs_ids=kept_obs, detected_ids=detected_all)

    def errors(U, G6, G6_2):
        return {"e_pipe": _ratio(U - G6, G6), "e_cover": _ratio(G6 - G6_2, G6_2), "e_total": _ratio(U - G6_2, G6_2)}

    trees = []
    for t in layout.tree_ids:
        ids_t = frozenset(g for g in T if G.tree_of[g] == t)
        U, U_truth = int((pred_tree == t).sum()), int((truth_tree == t).sum())
        G6, G6_2 = len(G.g6 & ids_t), G.g6_total[t]
        trees.append({
            "tree": t, "U": U, "U_truth_attr": U_truth, "G6": G6,
            **{f"G6_{v}": len(tables[v].g6 & ids_t) for v in VARIANTS}, "G6_2": G6_2,
            **errors(U, G6, G6_2),
            "mis_in": int(((pred_tree == t) & (truth_tree != t)).sum()),
            "mis_out": int(((truth_tree == t) & (pred_tree != t)).sum()),
            "ledger": ledger(G.g6 & ids_t, ids_t, np.flatnonzero(truth_tree == t)).as_dict(),
        })

    U_row, G6_row, G6_2_row = len(row_sel), len(G.g6), sum(G.g6_total.values())
    row_ledger = ledger(G.g6, T, row_sel)
    row = {"U": U_row, "U_truth_attr": int((truth_tree >= 0).sum()), "G6": G6_row,
           **{f"G6_{v}": len(tables[v].g6) for v in VARIANTS}, "G6_2": G6_2_row,
           **errors(U_row, G6_row, G6_2_row), "naive_U": naive_U, "naive_ratio": _ratio(naive_U, G6_row)}

    sum_g6 = sum(t["G6"] for t in trees)
    pos_err = [float(np.linalg.norm(lm.X[u] - truth.fruits[int(m)].center))
               for u, m in enumerate(lm_maj.tolist()) if m in T]
    row_rates = row_ledger.rates()
    metrics = {
        "row_abs_e_pipe": None if row["e_pipe"] is None else abs(row["e_pipe"]),
        "tree_mae_over_mean": _ratio(sum(abs(t["U"] - t["G6"]) for t in trees), sum_g6),
        "attribution_error": _ratio(sum(abs(t["U"] - t["U_truth_attr"]) for t in trees), sum_g6),
        "pos_err_p50_m": _percentile(pos_err, 50), "pos_err_p90_m": _percentile(pos_err, 90),
        "miss_rate": row_rates["miss_rate"], "excess_rate": row_rates["excess_rate"],
    }
    return {"mode": mode, "n_landmarks": len(lm.X), "trees": trees, "row": row,
            "ledger": row_ledger.as_dict(), "ledger_any": ledger(G_any.g6, G_any.target_ids, row_sel).as_dict(),
            "metrics": metrics}


def _cache_result(ctx: RunContext) -> dict:
    """The `cache` entry of a cached context's result."""
    return {k: ctx.cache[k] for k in ("file", "digest", "weights_sha256", "pretrained")}


def evaluate_seeds(ctx: RunContext, cfg: EvalAppConfig, *, subsets: bool = False) -> dict:
    """Evaluate `cfg` on `ctx` once per `cfg.noise.seeds` entry and combine the results; without noise, once.

    Under `stages.det=oracle_noisy` each seed needs its own context
    (`run_config`), so `ctx` serves only the seed it was prepared with.
    The pass-subset table (`subsets`) needs a setting without noise (`ValueError`).
    A cached context's combined result also has `cache`, as `evaluate`'s has.
    """
    if not cfg.is_noisy():
        return evaluate(ctx, cfg, subsets=subsets)
    if subsets:
        raise ValueError("--subsets needs a stage setting without noise")
    out = combine_seeds(cfg, [(s, evaluate(ctx, cfg, noise_seed=s)) for s in cfg.noise.seeds])
    if ctx.cache is not None:
        out["cache"] = _cache_result(ctx)
    return out


def run_config(
    run: SynthRun, cfg: EvalAppConfig, splits: SplitsConfig, *, allow_test: bool = False,
) -> tuple[RunContext, dict, dict[str, float]]:
    """`prepare` and evaluate `run` under `cfg`, once per noise seed when `cfg` has noise.

    Under `stages.det=oracle_noisy` every seed `s` of `cfg.noise.seeds` gets its
    own `prepare(..., det_seed=s)` and `evaluate(..., noise_seed=s)`, one context
    at a time, and the results are combined (`combine_seeds`); otherwise one
    `prepare` serves `evaluate_seeds`. Every `prepare` gets `allow_test`.
    Returns the last context, the result and the seconds spent,
    `{"prepare": ..., "evaluate": ...}`.
    """
    t_prepare = t_evaluate = 0.0
    if cfg.stages.det == "oracle_noisy":
        per_seed = []
        for s in cfg.noise.seeds:
            ctx = None   # one context at a time: drop the previous seed's before the next prepare
            t0 = time.perf_counter()
            ctx = prepare(run, cfg, splits, det_seed=s, allow_test=allow_test)
            t1 = time.perf_counter()
            per_seed.append((s, evaluate(ctx, cfg, noise_seed=s)))
            t_prepare += t1 - t0
            t_evaluate += time.perf_counter() - t1
        res = combine_seeds(cfg, per_seed)
    else:
        t0 = time.perf_counter()
        ctx = prepare(run, cfg, splits, allow_test=allow_test)
        t1 = time.perf_counter()
        res = evaluate_seeds(ctx, cfg)
        t_prepare = t1 - t0
        t_evaluate = time.perf_counter() - t1
    return ctx, res, {"prepare": t_prepare, "evaluate": t_evaluate}


class Ladder(dict):
    """A ladder file's mapping; `path` is the file it was read from, for error messages."""

    path: Path | None = None


def load_ladder(path: str | Path) -> Ladder:
    """Read a ladder file (`rungs`, optional `baselines` and `interaction_2x2`)."""
    p = Path(path).expanduser()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("rungs"), dict):
        raise ValueError(f"{p}: a ladder file needs a 'rungs' mapping")
    ladder = Ladder(raw)
    ladder.path = p
    return ladder


def _merge(base: dict, patch: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        out[k] = _merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _patched(cfg: EvalAppConfig, patch: Mapping) -> EvalAppConfig:
    return EvalAppConfig.model_validate(_merge(cfg.model_dump(), patch))


def rung_config(cfg: EvalAppConfig, ladder: dict, name: str) -> EvalAppConfig:
    """`cfg` with ladder rung (or baseline) `name` deep-merged in, validated again."""
    for section in ("rungs", "baselines"):
        if name in (ladder.get(section) or {}):
            return _patched(cfg, ladder[section][name])
    names = [*(ladder.get("rungs") or {}), *(ladder.get("baselines") or {})]
    raise ValueError(f"unknown rung {name!r}; available: {', '.join(names)}")


def check_ladder_request(
    cfg: EvalAppConfig, ladder: Mapping, rungs: Sequence[str], *, with_2x2: bool = False, subsets: bool = False
) -> None:
    """Validate a ladder request against `cfg` and `ladder` alone, before any bundle is opened.

    Checks `cfg`'s base config (`stages.dedup`, `lift.depth`, `stages.det`, `noise`, and, with a
    `pose` section, the true poses without the rig or fusion, and with an `align` section,
    `align.mode=none`), every name in `rungs` (`rung_config` raises `unknown rung ...` for one it
    does not know), with `with_2x2`, the `interaction_2x2` section and its base rung, and,
    with `subsets`, that `rungs` has a rung of `SUBSET_RUNGS` and `cfg` runs to `stop_after=y6`.
    """
    if cfg.stages.dedup == "oracle":
        raise ValueError("ladder base config has stages.dedup=oracle; rungs need the crop default dedup mode")
    if cfg.lift.depth == "det_depth":
        raise ValueError("ladder base config has lift.depth=det_depth; rungs L2c and L3 use the base box depth "
                         "method")
    if cfg.stages.det != "oracle":
        raise ValueError(f"ladder base config has stages.det={cfg.stages.det}; ladder rungs need oracle detections")
    if not cfg.noise.is_zero():
        raise ValueError("ladder base config has noise set; only the L4 rungs add pose and depth noise")
    pose = cfg.pose
    if pose is not None and (pose.source != "truth" or pose.rig or pose.fuse != "none"):
        raise ValueError("ladder base config has pose.source, pose.rig or pose.fuse set; rungs choose the pose "
                         "source, the rig and the fusion")
    if cfg.align is not None and cfg.align.mode != "none":
        raise ValueError("ladder base config has align.mode set; rungs choose the alignment")
    for name in rungs:
        rung_config(cfg, ladder, name)
    if with_2x2:
        spec = ladder.get("interaction_2x2")
        if not isinstance(spec, Mapping) or not {"base", "track", "dedup"} <= set(spec):
            where = getattr(ladder, "path", None) or "ladder"
            raise ValueError(f"{where}: with_2x2 needs an 'interaction_2x2' mapping with base, track and dedup")
        rung_config(cfg, ladder, spec["base"])
    if subsets:
        if not any(name in rungs for name in SUBSET_RUNGS):
            raise ValueError("--subsets needs rung L3 or L5")
        if cfg.stop_after == "y1":
            raise ValueError("--subsets needs stop_after=y6")


def check_verdict_request(cfg: EvalAppConfig, rungs: Sequence[str], *, subsets: bool, kind: str = "ladder") -> None:
    """With `cfg.verdict_subsets` set, the request must be a ladder (`kind`, the eval command: `eval run`
    and `eval sweep` judge no pass subset), `subsets` must be true and `rungs` must include rung L5
    (`ValueError` otherwise)."""
    if not cfg.verdict_subsets:
        return
    if kind != "ladder":
        raise ValueError(f"verdict_subsets is judged by eval ladder only; eval {kind} does not read it")
    if not (subsets and "L5" in rungs):
        raise ValueError("verdict_subsets needs --subsets and rung L5")


def check_verdict_subsets(bundles: Sequence[SynthRun], cfg: EvalAppConfig) -> None:
    """With `cfg.verdict_subsets` set, every name must be one of `bundles`' pass-subset names
    (`pass_subsets` of the frames `select_frames` selects under `cfg.passes`, the same selection
    `prepare` makes), or `ValueError` naming the bundle. Reads no video or depth."""
    if not cfg.verdict_subsets:
        return
    for run in bundles:
        frames = select_frames(run.frames, base_step_m=run.meta.base_step_m, pitches=cfg.passes.pitches,
                               sides=cfg.passes.sides, step_m=cfg.passes.step_m)
        keys = list(dict.fromkeys(f.pass_key for f in frames))
        names = [name for name, _ in pass_subsets(keys)]
        for name in cfg.verdict_subsets:
            if name not in names:
                raise ValueError(f"{run.root}: verdict_subsets names {name!r}, not a pass subset of this "
                                 f"bundle ({', '.join(names)})")


def run_ladder(
    ctx: RunContext, cfg: EvalAppConfig, ladder: dict, rungs: Sequence[str], *, with_2x2: bool = False,
    subsets: bool = False, context_for: Callable[[EvalAppConfig], RunContext] | None = None,
) -> dict:
    """Evaluate `rungs` (and, with `with_2x2`, the (track, dedup) 2 x 2 table on the ladder's base rung).

    `chain` lists the requested rungs in ladder order (baselines are not on
    it); `rungs` holds their results in the requested order. A rung inherits
    `cfg.stages.det` unless it sets its own (L5: `cached`); a rung without its
    own `stages.dedup` (`L2c`, `L3`, `L3_dd`, the L4 and L5 rungs) inherits
    `cfg`'s; a rung without its own `lift` section (`L2c`, `L3`, the L4 and L5
    rungs) inherits `cfg.lift`, so it uses `cfg`'s box depth method and
    parameters. So `cfg` must use `stages.det=oracle`, `lift.depth` other than
    `det_depth`, `stages.dedup` other than `oracle`, and no noise. A rung with
    noise (the L4 rungs, `L5_L4base`) runs once per `cfg.noise.seeds` entry and
    is combined (`combine_seeds`). Each 2 x 2 cell is evaluated with its own
    config on `ctx`; its values are None under `stop_after=y1`.

    A rung whose `_prepared_settings` equal `ctx.prepared` runs on `ctx`. The
    others are grouped by their `_prepared_settings` (first-seen order); each
    group runs on `context_for(the group's first config)`, one such context at
    a time, so L5 and `L5_L4base` (which differ in noise only) share one. Such a
    rung may change the detector, not `FRAME_KEYS` (`_check_frames`); without
    `context_for` it is refused as `evaluate` refuses it (`_check_config`).
    Every rung and cell config is checked before any of them runs (first
    `check_pose_supported`, right after the configs are built), and the
    rungs of a group against their own context before they run. With
    `subsets`, the rungs of `SUBSET_RUNGS` alone also get the pass-subset table
    (`evaluate(..., subsets=True)`).
    """
    check_ladder_request(cfg, ladder, rungs, with_2x2=with_2x2, subsets=subsets)
    configs = {name: rung_config(cfg, ladder, name) for name in rungs}
    cells: list[tuple[str, str, EvalAppConfig]] = []
    if with_2x2:
        spec = ladder["interaction_2x2"]
        base = rung_config(cfg, ladder, spec["base"])
        for track in spec["track"]:
            for d in spec["dedup"]:
                d = cfg.stages.dedup if d == "crop_default" else d
                cells.append((track, d, _patched(base, {"stages": {"track": track, "dedup": d}})))
    for c in [*configs.values(), *(c for _, _, c in cells)]:
        check_pose_supported(c)
    on_ctx: list[str] = []
    groups: dict[str, list[str]] = {}
    for name, c in configs.items():
        settings = _prepared_settings(c)
        if settings == ctx.prepared:
            on_ctx.append(name)
            continue
        _check_frames(ctx, c)
        if context_for is None:
            _check_config(ctx, c)
        groups.setdefault(json.dumps(settings, sort_keys=True), []).append(name)
    for c in [*(configs[name] for name in on_ctx), *(c for _, _, c in cells)]:
        _check_config(ctx, c)

    def run_rung(rung_ctx: RunContext, name: str) -> dict:
        return evaluate_seeds(rung_ctx, configs[name], subsets=subsets and name in SUBSET_RUNGS)

    results = {name: run_rung(ctx, name) for name in on_ctx}
    interaction = None
    if with_2x2:
        interaction = {"cells": [_cell(track, d, evaluate(ctx, c)["y6"]) for track, d, c in cells]}
    for names in groups.values():
        group_ctx = context_for(configs[names[0]])
        for name in names:
            _check_config(group_ctx, configs[name])
        results.update((name, run_rung(group_ctx, name)) for name in names)
        group_ctx = None   # one extra context at a time: drop it before the next group's
    return {"chain": [n for n in (ladder.get("rungs") or {}) if n in rungs],
            "rungs": {name: results[name] for name in configs},
            "interaction_2x2": interaction}


def _cell(track: str, dedup_mode: str, y6: dict | None) -> dict:
    return {"track": track, "dedup": dedup_mode,
            "e_pipe": None if y6 is None else y6["row"]["e_pipe"],
            "dup_track": None if y6 is None else y6["ledger"]["dup_track"],
            "dup_cross": None if y6 is None else y6["ledger"]["dup_cross"]}
