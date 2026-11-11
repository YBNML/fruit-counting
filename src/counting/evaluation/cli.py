"""`counting eval` sub-app: `eval truth` prints pass/tree ground-truth tables and runs
the summary/geometry/split checks for one or more orchard_synth bundles and, for bundles
rendered with body shake, the rig, trajectory and IMU checks; `eval run` and
`eval ladder` score a stage setting (or walk the oracle ladder) and write the experiment
folder; `eval sweep` scores a grid of stage-setting points (optionally crossed with
ladder bases) and writes the same kind of experiment folder with a `sweep.csv`. A split=test
request to these three must pass the test protection checks (`counting.evaluation.protect`)
before any bundle is scored, and each one that writes its folder is logged in `test_log.csv`.
`eval detmetrics` scores the detection caches of a config's bundles (AP50, precision, recall
and the false-positive kinds) and writes a `detmetrics.json` folder; it refuses split=test.

Module-level imports are `typer` and `rich` only, so importing `counting.cli` (which
imports this module to register the `eval` sub-app) never loads torch. Everything else
is imported inside the command function.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

eval_app = typer.Typer(help="Synthetic Y1/Y6 scoring", no_args_is_help=True)


# Suppresses Typer's single-command auto-promotion; keep even when other commands are added.
@eval_app.callback()
def _eval() -> None:
    """Synthetic Y1/Y6 scoring."""


@eval_app.command()
def truth(
    runs: list[str] = typer.Argument(..., help="orchard_synth run directories"),
    pitches: str = typer.Option(None, "--pitches", help="Comma-separated pitch degrees, e.g. 45,70"),
    sides: str = typer.Option(None, "--sides", help="Comma-separated sides, e.g. A,B"),
    step: float = typer.Option(None, "--step", help="Camera step in metres for the sequence thinning rule"),
    variant: str = typer.Option("count1", "--variant", help="count1 | any | strict"),
    subsets: bool = typer.Option(False, "--subsets", help="Print the pass-subset upper-bound table"),
    check_geometry_flag: bool = typer.Option(
        False, "--check-geometry", help="Run the adapter's camera/depth sanity check"
    ),
    split: str = typer.Option(None, "--split", help="Expected split name for every run's seed"),
    splits_path: str = typer.Option(
        "configs/eval/splits.yaml", "--splits", help="Splits YAML (for split_of and --split)"
    ),
) -> None:
    """Print pass/tree ground-truth tables and run the summary/geometry/split checks.

    For a bundle rendered with body shake it also checks that the cameras at one position
    share their body and time (rig) and that the recorded trajectory gives each frame pose
    (trajectory), within 1e-6 m and 1e-4 deg, and that noiseless IMU samples of each side's
    trajectory, integrated from its first frame to its last, end within 2 cm of it (imu).

    Exit code 0 is success, 1 is a check failure (summary mismatch/missing, geometry,
    split, rig, trajectory, imu), 2 is an input error. A bundle whose sampled frames show no
    ground reports ground=n/a; the centre check still applies.
    """
    from counting.config.eval_schema import SplitsConfig
    from counting.config.loader import load_yaml_model
    from counting.data.formats.orchard_synth import FrameInput, PassKey, SynthRun, select_frames
    from counting.evaluation.truth import (
        SummaryMismatch,
        check_geometry,
        check_imu,
        check_rig,
        check_summary,
        check_trajectory,
        subset_table,
        truth_sets,
    )

    def group_passes(frames: list[FrameInput]) -> dict[PassKey, list[FrameInput]]:
        """Frames grouped by pass, passes in first-appearance order, frames in idx order."""
        out: dict[PassKey, list[FrameInput]] = {}
        for f in frames:
            out.setdefault(f.pass_key, []).append(f)
        return out

    pitch_list = [float(p) for p in pitches.split(",")] if pitches else None
    side_list = [s.strip() for s in sides.split(",")] if sides else None
    filtered = pitch_list is not None or side_list is not None or step is not None

    console = Console(soft_wrap=True)
    failed = False
    try:
        splits_cfg = load_yaml_model(splits_path, SplitsConfig)

        for run_path in runs:
            run = SynthRun.open(run_path)
            scene = run.truth()
            seed = run.meta.seed
            seed_split = splits_cfg.split_of(seed)
            step_text = "-" if run.meta.base_step_m is None else f"{run.meta.base_step_m:g}"
            typer.echo(
                f"{run.meta.run_id}  crop={run.meta.crop}  seed={seed} ({seed_split or 'unknown'})  "
                f"is_sequence: {str(run.meta.is_sequence).lower()}  "
                f"base_step_m={step_text}  frames={len(run.frames)}"
            )

            selected = select_frames(
                run.frames, base_step_m=run.meta.base_step_m, pitches=pitch_list, sides=side_list, step_m=step
            )
            passes_sel = group_passes(selected)
            tt = truth_sets(scene, run.layout, passes_sel, variant=variant)
            tt_any = tt if variant == "any" else truth_sets(scene, run.layout, passes_sel, variant="any")
            tt_strict = (
                tt if variant == "strict" else truth_sets(scene, run.layout, passes_sel, variant="strict")
            )

            pass_table = Table(title="passes")
            for col in ("pitch", "side", "aisle", "frames", "G1"):
                pass_table.add_column(col)
            for key in sorted(passes_sel, key=lambda k: (k.pitch_deg, k.side, k.aisle)):
                pass_table.add_row(
                    f"{key.pitch_deg:g}", key.side, str(key.aisle),
                    str(len(passes_sel[key])), str(len(tt.g1[key])),
                )
            console.print(pass_table)

            tree_table = Table(title="trees")
            for col in ("tree", "G6", "total", "any", "strict"):
                tree_table.add_column(col)
            g6_counts = tt.per_tree(tt.g6)
            any_counts = tt_any.per_tree(tt_any.g6)
            strict_counts = tt_strict.per_tree(tt_strict.g6)
            for t in tt.tree_ids:
                tree_table.add_row(
                    str(t), str(g6_counts[t]), str(tt.g6_total[t]), str(any_counts[t]), str(strict_counts[t])
                )
            console.print(tree_table)

            if filtered:
                typer.echo("summary skipped (filtered)")
            else:
                try:
                    check_summary(scene, run.layout, run.passes())
                except SummaryMismatch as e:
                    typer.echo(f"summary MISMATCH: {e}")
                    failed = True
                except ValueError:
                    typer.echo("summary MISSING")
                    failed = True
                else:
                    typer.echo("summary OK")

            if subsets:
                sub_table = Table(title="subsets")
                for col in ("S", "n_passes", "G6", "ratio_all", "ratio_total"):
                    sub_table.add_column(col)
                for row in subset_table(scene, run.layout, passes_sel):
                    ratio_all = "-" if row["ratio_all"] is None else f"{row['ratio_all']:.3f}"
                    ratio_total = "-" if row["ratio_total"] is None else f"{row['ratio_total']:.3f}"
                    sub_table.add_row(row["S"], str(row["n_passes"]), str(row["G6"]), ratio_all, ratio_total)
                console.print(sub_table)

            if check_geometry_flag:
                res = check_geometry(run, scene)
                ground_text = "n/a (no ground pixels)" if res["ground"] == "n/a" else res["ground"]
                detail = (
                    f"center_in_bbox={res['center_in_bbox']}, n_unoccluded={res['n_unoccluded']}, "
                    f"ground={ground_text}, "
                    f"ground_abs_z_median_m={res['ground_abs_z_median_m']}, n_ground_px={res['n_ground_px']}"
                )
                if res["passed"]:
                    typer.echo(f"geometry OK ({detail})")
                else:
                    typer.echo(f"geometry FAIL ({detail})")
                    failed = True

            if split is not None:
                if seed_split == split:
                    typer.echo("split OK")
                else:
                    typer.echo(f"split MISMATCH: seed {seed} is {seed_split}")
                    failed = True

            # The whole bundle, whatever --pitches, --sides and --step select.
            if run.meta.shake:
                rig = check_rig(run)
                traj = check_trajectory(run)
                imu = check_imu(run)
                rig_detail = (
                    f"positions={rig['positions']}, pairs={rig['pairs']}, max_pos_m={rig['max_pos_m']:.3g}, "
                    f"max_rot_deg={rig['max_rot_deg']:.3g}, max_dt_s={rig['max_dt_s']:.3g}"
                )
                traj_detail = (
                    f"frames={traj['frames']}, max_pos_m={traj['max_pos_m']:.3g}, "
                    f"max_rot_deg={traj['max_rot_deg']:.3g}"
                )
                imu_detail = (
                    f"drives={imu['drives']}, max_end_m={imu['max_end_m']:.3g}, "
                    f"max_end_deg={imu['max_end_deg']:.3g}"
                )
                typer.echo(f"rig {'OK' if rig['passed'] else 'FAIL'} ({rig_detail})")
                typer.echo(f"trajectory {'OK' if traj['passed'] else 'FAIL'} ({traj_detail})")
                typer.echo(f"imu {'OK' if imu['passed'] else 'FAIL'} ({imu_detail})")
                if not (rig["passed"] and traj["passed"] and imu["passed"]):
                    failed = True
    except (ValueError, FileNotFoundError, NotImplementedError) as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=2)

    if failed:
        raise typer.Exit(code=1)


def _load_criteria(path: str) -> tuple[dict | None, str | None]:
    """(YAML criteria at `path`, sha256 of the file's bytes), or `(None, None)` if the file does not exist."""
    import hashlib
    from pathlib import Path

    import yaml

    p = Path(path).expanduser()
    if not p.is_file():
        return None, None
    data = p.read_bytes()
    return yaml.safe_load(data.decode("utf-8")), hashlib.sha256(data).hexdigest()


def _row_text(label: str, res: dict) -> str:
    """`"{label}: row U=... G6=... e_pipe=..."` (`-` under stop_after=y1); a noisy `res` names its seed mean."""
    from counting.evaluation.report import _count

    y6 = res["y6"]
    if y6 is None:
        text = f"{label}: row U=- G6=- e_pipe=-"
    else:
        row = y6["row"]
        e_pipe = "-" if row["e_pipe"] is None else f"{row['e_pipe']:+.1%}"
        text = f"{label}: row U={_count(row['U'])} G6={_count(row['G6'])} e_pipe={e_pipe}"
    if res.get("noisy"):
        text += f" (mean over noise seeds {','.join(map(str, res['seeds']))})"
    return text


def _echo_rung_lines(run_entries: list[dict]) -> None:
    """One line per (bundle, rung), from `_row_text` (`-` under stop_after=y1)."""
    multi = len(run_entries) > 1
    for entry in run_entries:
        prefix = f"{entry['meta']['run_id']}: " if multi else ""
        for name, rung in entry["rungs"].items():
            typer.echo(_row_text(f"{prefix}{name}", rung))


def _echo_sweep_lines(run_entries: list[dict]) -> None:
    """One line per (bundle, cell), from `_row_text` (label `[{run_id}: ]{point} @ {base or -}`)."""
    multi = len(run_entries) > 1
    for entry in run_entries:
        prefix = f"{entry['meta']['run_id']}: " if multi else ""
        for cell in entry["cells"]:
            typer.echo(_row_text(f"{prefix}{cell['point']} @ {cell['base'] or '-'}", cell))


def _cell_results(entry: dict) -> list[dict]:
    """Every scored result of one bundle's entry: the rungs of a run or ladder, the cells of a sweep."""
    return [*(entry.get("rungs") or {}).values(), *(entry.get("cells") or [])]


def _cache_identities(run_entries: list[dict]) -> list[dict]:
    """The detection caches the results were scored on, from the `cache` of every rung and sweep cell.

    One `{run_id, file, digest, weights_sha256}` per (run_id, digest), in that order.
    """
    found: dict[tuple[str, str], dict] = {}
    for entry in run_entries:
        run_id = entry["meta"]["run_id"]
        for res in _cell_results(entry):
            cache = res.get("cache")
            if cache is not None:
                found.setdefault((run_id, cache["digest"]), {
                    "run_id": run_id, "file": cache["file"], "digest": cache["digest"],
                    "weights_sha256": cache["weights_sha256"]})
    return [found[key] for key in sorted(found)]


def _pose_cache_identities(run_entries: list[dict]) -> list[dict]:
    """The pose caches the results read, from the `pose.cache` of every rung and sweep cell.

    One `{run_id, name, file, digest, vo_config_hash, opencv}` per (run_id, digest), in that order.
    A result with no `pose` entry, or a `pose` entry with no `cache`, adds none.
    """
    found: dict[tuple[str, str], dict] = {}
    for entry in run_entries:
        run_id = entry["meta"]["run_id"]
        for res in _cell_results(entry):
            cache = (res.get("pose") or {}).get("cache")
            if cache is not None:
                found.setdefault((run_id, cache["digest"]), {
                    "run_id": run_id, "name": cache["name"], "file": cache["file"], "digest": cache["digest"],
                    "vo_config_hash": cache["vo_config_hash"], "opencv": cache["opencv"]})
    return [found[key] for key in sorted(found)]


def _imu_file_identities(run_entries: list[dict]) -> list[dict]:
    """The IMU files the results read, from the `pose.imu` of every rung and sweep cell whose `source` is
    `file`.

    One `{run_id, name, file, digest, profile}` per (run_id, digest), in that order. A result with no
    `pose` entry, or with IMU streams made in memory or none, adds none.
    """
    found: dict[tuple[str, str], dict] = {}
    for entry in run_entries:
        run_id = entry["meta"]["run_id"]
        for res in _cell_results(entry):
            imu = (res.get("pose") or {}).get("imu")
            if imu is not None and imu["source"] == "file":
                found.setdefault((run_id, imu["digest"]), {
                    "run_id": run_id, "name": imu["name"], "file": imu["file"], "digest": imu["digest"],
                    "profile": imu["profile"]})
    return [found[key] for key in sorted(found)]


def _fusion_identity(run_entries: list[dict]) -> dict | None:
    """The fusion the results ran, from the `pose.fusion` of every rung and sweep cell, or None when none
    was fused: `{gtsam, fuse_fixed_sha256}`, the gtsam version they share (None when they differ) and the
    sorted distinct sha256s of the fixed fusion settings."""
    fusions = [f for entry in run_entries for res in _cell_results(entry)
               if (f := (res.get("pose") or {}).get("fusion")) is not None]
    if not fusions:
        return None
    return {"gtsam": _same([f["gtsam"] for f in fusions]),
            "fuse_fixed_sha256": sorted({f["fuse_fixed_sha256"] for f in fusions})}


def _same(values: list):
    """The value every entry of `values` shares, else None."""
    return values[0] if values and all(v == values[0] for v in values) else None


def _run_eval(
    *, config: str, run_: list[str] | None, set_: list[str] | None, splits_path: str, criteria_path: str,
    kind: str, rungs: list[str] | None = None, ladder_path: str | None = None, grid_path: str | None = None,
    with_2x2: bool = False, subsets: bool = False,
) -> None:
    """Shared body of `eval run` (`kind="run"`), `eval ladder` (`kind="ladder"`) and `eval sweep` (`kind="sweep"`).

    Exit code 0 is success (including a reproduced rerun), 1 is a check failure
    (seed/split mismatch, summary mismatch, a split=test protection check, failed
    reproducibility), 2 is an input error. `CheckFailed` (and so `ProtectionError`) is a
    `ValueError`, so it is caught before the input-error branch.

    In order:
    1. read the config (with `--set`, then `--run`), the splits and the criteria;
    2. read the ladder (and its default rungs), or the grid and its points (and, with grid
       bases, the ladder);
    3. under split=test, `check_test_request` on the config and every file read so far, then
       `check_test_log` on the output folder's `test_log.csv`;
    4. check the pass-subset verdict request (`check_verdict_request`: only a ladder reads
       `cfg.verdict_subsets`), then the ladder request, or the `--set` overlap and the sweep cells;
       under split=test, `check_test_rungs` on the requested rungs or the sweep cells; then the
       pose settings the runner does not run (`check_pose_supported`) of the config, of every
       requested rung (and the 2 x 2 base rung) or of every sweep cell;
    5. open every bundle in `cfg.runs` (a sweep checks each one's frame selection, split=test
       runs `check_test_bundles` and, with `detector.cache`, `check_test_caches`) and sort them
       by (fingerprint, run_id), so the results do not depend on the order the bundles were
       listed in; a ladder then checks `cfg.verdict_subsets` against them (`check_verdict_subsets`);
    6. score the bundles one at a time, one context at a time; only a request that passed the
       split=test checks lets `prepare` take a split=test config (`allow_test`); a ladder whose
       requested rung L5 runs on cached detections also scores those caches as detections
       (`detection`: `score_runs` at rung L5's thresholds and post-processing);
    7. build the experiment id, `pooled` (run and ladder, with `y1_judged` when rung L5 ran),
       `verdicts` (a ladder with criteria, plus the L5 `detection` rows and, with
       `cfg.verdict_subsets`, the `L5[<name>]` rows of `subset_verdicts`) and, for a ladder with
       rung L5, the checked error `budget` (`ladder_budget`), and write the experiment folder; the
       digests of the detection caches (`_cache_identities`), of the pose caches
       (`_pose_cache_identities`) and of the IMU files (`_imu_file_identities`) the results name,
       and the sha256 of each fixed fusion settings file they ran with (`_fusion_identity`), join
       the bundle fingerprints, and the record lists them (`fusion` with the gtsam version and the
       CPU count);
    8. under split=test, add a `test_log.csv` row, and warn when the log already lists a run
       with the same config hash and request (`extra`).
    Under `stages.det=oracle_noisy`, `eval run` prepares the bundle once per noise seed. A ladder
    prepares a second context for the rungs that change the detections (`run_ladder`'s `context_for`).
    """
    import hashlib
    import os
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    from counting.config.eval_schema import EvalAppConfig, SplitsConfig
    from counting.config.hashing import config_hash
    from counting.config.loader import load_yaml_model
    from counting.data.formats.orchard_synth import SynthRun
    from counting.evaluation.budget import ladder_budget
    from counting.evaluation.detmetrics import detection_rows, score_runs
    from counting.evaluation.posesource import check_pose_supported
    from counting.evaluation.protect import (
        append_log_row, check_test_bundles, check_test_caches, check_test_log, check_test_request, check_test_rungs,
        earlier_log_runs,
    )
    from counting.evaluation.record import build_record, counting_repo, experiment_id, git_state, request_extra
    from counting.evaluation.report import SCHEMA_VERSION, ReproducibilityError, write_experiment
    from counting.evaluation.runner import (
        CheckFailed, RunContext, check_ladder_request, check_verdict_request, check_verdict_subsets, describe_run,
        load_ladder, open_cache, prepare, run_config, run_ladder, rung_config,
    )
    from counting.evaluation.sweep import (
        check_frame_selection, check_set_overlap, grid_points, load_grid, run_sweep, sweep_cells,
    )
    from counting.evaluation.truth import SummaryMismatch
    from counting.evaluation.verdict import ladder_pooled, ladder_verdicts, subset_verdicts, verdict_summary

    def file_sha256(path: str) -> str:
        return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()

    earlier = 0
    try:
        try:
            cfg = load_yaml_model(config, EvalAppConfig, overrides=set_)
        except KeyError as e:  # a --set path that cannot be descended
            raise ValueError(f"--set: {e.args[0]}") from e
        if run_:
            cfg = cfg.model_copy(update={"runs": list(run_)})
        splits_cfg = load_yaml_model(splits_path, SplitsConfig)
        criteria, criteria_sha256 = _load_criteria(criteria_path)

        ladder = ladder_sha256 = grid = grid_sha256 = cells = None
        points = []
        if kind == "sweep":
            grid_sha256 = file_sha256(grid_path)
            grid = load_grid(grid_path)
            points = grid_points(grid)
        if kind == "ladder" or (grid is not None and grid.bases):
            ladder_sha256 = file_sha256(ladder_path)
            ladder = load_ladder(ladder_path)
            if kind == "ladder" and rungs is None:
                rungs = [*(ladder.get("rungs") or {}), *(ladder.get("baselines") or {})]

        protected = None
        if cfg.split == "test":
            files = {"--splits": splits_path, "--criteria": criteria_path}
            if ladder is not None:
                files["--ladder"] = ladder_path
            if grid is not None:
                files["--grid"] = grid_path
            protected = check_test_request(cfg, config_path=config, sets=set_ or [], runs=run_ or [], files=files,
                                           grid=grid, ladder=ladder, repo=counting_repo())
            check_test_log(Path(cfg.output_dir).expanduser() / "test_log.csv")

        check_verdict_request(cfg, rungs or (), subsets=subsets, kind=kind)
        if kind == "ladder":
            check_ladder_request(cfg, ladder, rungs, with_2x2=with_2x2, subsets=subsets)
            if protected is not None:
                check_test_rungs(cfg, {name: rung_config(cfg, ladder, name) for name in rungs})
        elif kind == "sweep":
            check_set_overlap([s.split("=", 1)[0] for s in (set_ or [])], points)
            cells = sweep_cells(cfg, grid, ladder)
            if protected is not None:
                check_test_rungs(cfg, {f"{c.point.name} @ {c.base}": c.cfg for c in cells})
        if kind == "run":
            check_pose_supported(cfg)
        elif kind == "ladder":
            names = [*rungs, ladder["interaction_2x2"]["base"]] if with_2x2 else rungs
            for name in names:
                check_pose_supported(rung_config(cfg, ladder, name))
        else:
            for cell in cells:
                check_pose_supported(cell.cfg)

        started = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        bundles = [SynthRun.open(run_path) for run_path in cfg.runs]
        t_prepare, t_score = time.perf_counter() - t0, 0.0
        if kind == "sweep":
            for bundle in bundles:
                check_frame_selection(bundle, cells)
        if protected is not None:
            check_test_bundles(bundles, splits_cfg)
            if cfg.detector.cache is not None:
                check_test_caches(bundles, cfg)
        bundles.sort(key=lambda b: (b.meta.fingerprint, b.meta.run_id))
        if kind == "ladder":
            check_verdict_subsets(bundles, cfg)

        allow_test = protected is not None
        run_entries = []
        for bundle in bundles:
            ctx = None   # one context at a time: drop the previous bundle's before the next prepare
            if kind == "sweep":
                entry, timings = run_sweep(bundle, cells, splits_cfg, allow_test=allow_test)
            elif kind == "run":
                ctx, res, timings = run_config(bundle, cfg, splits_cfg, allow_test=allow_test)
                entry = {"meta": describe_run(ctx), "chain": [], "rungs": {"run": res}, "interaction_2x2": None}
            else:
                t_other = []   # seconds of each `context_for` prepare, counted as prepare, not evaluate

                def context_for(c: EvalAppConfig) -> RunContext:
                    t = time.perf_counter()
                    try:
                        return prepare(bundle, c, splits_cfg, allow_test=allow_test)
                    finally:
                        t_other.append(time.perf_counter() - t)

                t0 = time.perf_counter()
                ctx = prepare(bundle, cfg, splits_cfg, allow_test=allow_test)
                t1 = time.perf_counter()
                out = run_ladder(ctx, cfg, ladder, rungs, with_2x2=with_2x2, subsets=subsets, context_for=context_for)
                timings = {"prepare": t1 - t0 + sum(t_other), "evaluate": time.perf_counter() - t1 - sum(t_other)}
                entry = {"meta": describe_run(ctx), **out}
            t_prepare += timings["prepare"]
            t_score += timings["evaluate"]
            run_entries.append(entry)
        detection = None
        if kind == "ladder" and "L5" in rungs:
            l5 = rung_config(cfg, ladder, "L5")
            if l5.stages.det == "cached":
                t0 = time.perf_counter()
                dets = [open_cache(bundle, l5) for bundle in bundles]
                t1 = time.perf_counter()
                detection = {"rung": "L5", "thresholds": {"operating": l5.tracker.score_hi, "low": l5.tracker.score_lo},
                             "postprocess": l5.detector.post_settings().model_dump(),
                             **score_runs(bundles, dets, l5)}
                t_prepare, t_score = t_prepare + (t1 - t0), t_score + (time.perf_counter() - t1)
        finished = datetime.now(timezone.utc).isoformat()

        if kind == "run":
            uses_box_depth = cfg.lift.depth != "det_depth"
        elif kind == "ladder":
            lift_depths = [rung["lift_depth"] for entry in run_entries for rung in entry["rungs"].values()]
            if with_2x2:
                base_name = ladder["interaction_2x2"]["base"]
                base_patch = (ladder.get("rungs") or {}).get(base_name) \
                    or (ladder.get("baselines") or {}).get(base_name) or {}
                lift_depths.append((base_patch.get("lift") or {}).get("depth", cfg.lift.depth))
            uses_box_depth = any(d != "det_depth" for d in lift_depths)
        else:
            uses_box_depth = any(cell.cfg.lift.depth != "det_depth" for cell in cells)
        extra = request_extra(kind, ladder_sha256=ladder_sha256, rungs=rungs or (), with_2x2=with_2x2,
                              subsets=subsets, grid_name=grid.path.name if grid else None, grid_sha256=grid_sha256,
                              criteria_sha256=criteria_sha256)

        fingerprints = [bundle.meta.fingerprint for bundle in bundles]
        data = [entry["meta"] for entry in run_entries]
        if uses_box_depth:
            depth_fps = [bundle.depth_fingerprint() for bundle in bundles]
            fingerprints += depth_fps
            data = [{**d, "depth_fingerprint": fp} for d, fp in zip(data, depth_fps)]
        identities = _cache_identities(run_entries)
        detector = None
        if identities:
            fingerprints += [c["digest"] for c in identities]
            for i, d in enumerate(data):
                digests = [c["digest"] for c in identities if c["run_id"] == d["run_id"]]
                if digests:
                    data[i] = {**d, "cache_digests": digests}
            pretrained = [res["cache"]["pretrained"] for entry in run_entries for res in _cell_results(entry)
                          if res.get("cache") is not None]
            detector = {"name": "cached", "noise": None,
                        "weights_sha256": _same([c["weights_sha256"] for c in identities]),
                        "cache_sha256": identities[0]["digest"] if len(identities) == 1 else None,
                        "caches": identities, "pretrained": _same(pretrained)}
        pose_ids = _pose_cache_identities(run_entries)
        fingerprints += [c["digest"] for c in pose_ids]
        for i, d in enumerate(data):
            digests = [c["digest"] for c in pose_ids if c["run_id"] == d["run_id"]]
            if digests:
                data[i] = {**d, "pose_cache_digests": digests}
        imu_ids = _imu_file_identities(run_entries)
        fingerprints += [c["digest"] for c in imu_ids]
        for i, d in enumerate(data):
            digests = [c["digest"] for c in imu_ids if c["run_id"] == d["run_id"]]
            if digests:
                data[i] = {**d, "imu_file_digests": digests}
        fusion = None
        identity = _fusion_identity(run_entries)
        if identity is not None:
            fingerprints += identity["fuse_fixed_sha256"]
            fusion = {"gtsam": identity["gtsam"], "cpus": os.cpu_count(),
                      "fuse_fixed_sha256": identity["fuse_fixed_sha256"]}

        code = protected.code if protected is not None else git_state()
        chash = config_hash(cfg)
        exp_id = experiment_id(config_hash=chash, fingerprints=fingerprints, code=code, seed=cfg.seed, extra=extra)
        results = {"schema_version": SCHEMA_VERSION, "kind": kind, "run_name": cfg.run_name,
                   "experiment_id": exp_id, "split": cfg.split, "crop": cfg.crop, "runs": run_entries,
                   "pooled": None, "verdicts": None}
        if kind != "sweep":
            results["pooled"] = ladder_pooled(run_entries, crop=cfg.crop)
        if detection is not None:
            results["detection"] = detection
        # run ids carry no criteria hash (verdicts would break run reruns); sweep cells keep tree tables at the
        # baseline point only, so only a ladder gets verdicts
        if kind == "ladder" and criteria is not None:
            results["verdicts"] = ladder_verdicts(run_entries, results["pooled"], criteria, cfg.crop)
            if detection is not None:
                det_rows = detection_rows(detection, criteria, cfg.crop)
                if det_rows:
                    results["verdicts"]["detection"] = det_rows
            if cfg.verdict_subsets:
                results["verdicts"].update(subset_verdicts(run_entries, criteria, cfg.crop, cfg.verdict_subsets))
        if kind == "ladder":
            # checked before any file is written, so a budget that does not add up leaves no folder
            budget = ladder_budget(run_entries, cfg.crop)
            if budget is not None:
                results["budget"] = budget
        record_extra = None
        if kind == "sweep":
            results["sweep"] = {"grid": grid.path.name, "bases": list(grid.bases),
                                "points": [{"name": p.name, "patch": p.patch, "factor": p.factor, "value": p.value}
                                          for p in points]}
            record_extra = {"grid_sha256": grid_sha256, "ladder_sha256": ladder_sha256,
                            "detectors": sorted({cell.cfg.stages.det for cell in cells}),
                            "criteria_sha256": criteria_sha256}
        elif kind == "ladder":
            record_extra = {"ladder_sha256": ladder_sha256,
                            "detectors": sorted({rung["stages"]["det"] for entry in run_entries
                                                 for rung in entry["rungs"].values()}),
                            "criteria_sha256": criteria_sha256}
        record = build_record(cfg=cfg, runs=data, code=code,
                              timings={"prepare": round(t_prepare, 3), "evaluate": round(t_score, 3)},
                              started=started, finished=finished, extra=record_extra, detector=detector,
                              pose_caches=pose_ids or None, imu_files=imu_ids or None, fusion=fusion)
        out_root = Path(cfg.output_dir).expanduser()
        folder, reproduced = write_experiment(out_root, cfg=cfg, results=results, record=record, criteria=criteria)

        if protected is not None:
            log = out_root / "test_log.csv"
            earlier = earlier_log_runs(log, config_hash=chash, extra=extra)
            append_log_row(log, {"date": started, "experiment_id": exp_id, "kind": kind,
                                 "config": protected.config_rel, "config_hash": chash, "extra": extra,
                                 "fingerprints": ";".join(sorted(fingerprints)), "commit": code["commit"],
                                 "criteria_sha256": criteria_sha256,
                                 "verdict": verdict_summary(results["verdicts"])})
    except SummaryMismatch as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)
    except ReproducibilityError as e:
        typer.echo(f"reproducibility FAILED {e}")
        raise typer.Exit(code=1)
    except CheckFailed as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)
    except (ValueError, FileNotFoundError, NotImplementedError) as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=2)

    typer.echo(str(folder))
    if reproduced:
        typer.echo("reproduced (same results.json)")
    if earlier > 0:
        typer.echo(f"warning: test_log.csv already lists {earlier} test run(s) of config {chash} with this request; "
                   "this run is logged as a new line")
    if kind == "sweep":
        _echo_sweep_lines(run_entries)
    else:
        _echo_rung_lines(run_entries)


@eval_app.command()
def run(
    config: str = typer.Option(..., "--config", help="Eval config YAML"),
    run_: list[str] = typer.Option(None, "--run", help="Override cfg.runs with these bundle paths"),
    set_: list[str] = typer.Option(None, "--set", help="Dot-path override: key.path=value", show_default=False),
    splits_path: str = typer.Option(
        "configs/eval/splits.yaml", "--splits", help="Splits YAML (for prepare's split check)"
    ),
    criteria_path: str = typer.Option(
        "configs/eval/criteria.yaml", "--criteria",
        help="Criteria YAML; ignored if missing (a split=test run needs a tracked one)",
    ),
) -> None:
    """Run one stage setting (`cfg.stages`) over the config's bundles and write the experiment folder."""
    _run_eval(config=config, run_=run_, set_=set_, splits_path=splits_path, criteria_path=criteria_path, kind="run")


@eval_app.command()
def ladder(
    config: str = typer.Option(..., "--config", help="Eval config YAML"),
    rungs: str = typer.Option(None, "--rungs", help="Comma-separated rung/baseline names (default: all)"),
    with_2x2: bool = typer.Option(False, "--with-2x2", help="Also run the (track, dedup) 2x2 table"),
    subsets: bool = typer.Option(False, "--subsets", help="Add the pass-subset prediction table to rungs L3 and L5"),
    ladder_path: str = typer.Option(
        "configs/eval/sweeps/ladder.yaml", "--ladder", help="Ladder YAML (rungs, baselines, interaction_2x2)"
    ),
    run_: list[str] = typer.Option(None, "--run", help="Override cfg.runs with these bundle paths"),
    set_: list[str] = typer.Option(None, "--set", help="Dot-path override: key.path=value", show_default=False),
    splits_path: str = typer.Option(
        "configs/eval/splits.yaml", "--splits", help="Splits YAML (for prepare's split check)"
    ),
    criteria_path: str = typer.Option(
        "configs/eval/criteria.yaml", "--criteria",
        help="Criteria YAML; ignored if missing (a split=test run needs a tracked one)",
    ),
) -> None:
    """Walk the ladder (`run_ladder`) over the config's bundles and write the experiment folder."""
    rung_list = [r.strip() for r in rungs.split(",")] if rungs else None
    _run_eval(config=config, run_=run_, set_=set_, splits_path=splits_path, criteria_path=criteria_path,
             kind="ladder", rungs=rung_list, ladder_path=ladder_path, with_2x2=with_2x2, subsets=subsets)


@eval_app.command()
def sweep(
    config: str = typer.Option(..., "--config", help="Eval config YAML"),
    grid_path: str = typer.Option(..., "--grid", help="Sweep grid YAML (bases, factors/product, combos)"),
    ladder_path: str = typer.Option(
        "configs/eval/sweeps/ladder.yaml", "--ladder", help="Ladder YAML; read only if the grid has bases"
    ),
    run_: list[str] = typer.Option(None, "--run", help="Override cfg.runs with these bundle paths"),
    set_: list[str] = typer.Option(None, "--set", help="Dot-path override: key.path=value", show_default=False),
    splits_path: str = typer.Option(
        "configs/eval/splits.yaml", "--splits", help="Splits YAML (for prepare's split check)"
    ),
    criteria_path: str = typer.Option(
        "configs/eval/criteria.yaml", "--criteria",
        help="Criteria YAML; ignored if missing (a split=test run needs a tracked one)",
    ),
) -> None:
    """Score a grid of stage-setting points (optionally crossed with ladder bases) over the config's
    bundles and write the experiment folder (`sweep.csv` instead of `per_pass.csv`/`per_tree.csv`)."""
    _run_eval(config=config, run_=run_, set_=set_, splits_path=splits_path, criteria_path=criteria_path,
             kind="sweep", grid_path=grid_path, ladder_path=ladder_path)


def _echo_detmetrics_lines(result: dict) -> None:
    """One line per (bundle, group) at the operating threshold, then `verdict <OK>/<rows>` when judged."""
    def num(x) -> str:
        return "-" if x is None else f"{x:.3f}"

    t_hi = result["thresholds"]["operating"]
    for entry in result["runs"]:
        for name, g in entry["groups"].items():
            op = g["at"]["operating"]
            typer.echo(f"{entry['meta']['run_id']} {name}: AP50 {num(g['ap50'])} | at {t_hi:g}: "
                       f"P {num(op['precision'])} R {num(op['recall'])} F1 {num(op['f1'])} "
                       f"FP/frame {num(op['fp_per_frame'])} (dup {op['fp_dup']}, merge {op['fp_merge']}, "
                       f"part {op['fp_part']}, ign {op['fp_ign']}, bg {op['fp_bg']}) split {num(op['split_rate'])}")
    verdicts = result["verdicts"]
    if verdicts:
        typer.echo(f"verdict {sum(1 for v in verdicts if v['ok'])}/{len(verdicts)}")


@eval_app.command("detmetrics")
def detmetrics_cmd(
    config: str = typer.Option(..., "--config", help="Eval config YAML (crop, split, passes, lift, tracker)"),
    cache_dir: str = typer.Option(..., "--cache", help="Folder of detection caches, one <run_id>.npz per bundle"),
    run_: list[str] = typer.Option(None, "--run", help="Override cfg.runs with these bundle paths"),
    set_: list[str] = typer.Option(None, "--set", help="Dot-path override: key.path=value", show_default=False),
    splits_path: str = typer.Option(
        "configs/eval/splits.yaml", "--splits", help="Splits YAML (each bundle's seed must be in the config split)"
    ),
    criteria_path: str = typer.Option(
        "configs/eval/criteria.yaml", "--criteria", help="Criteria YAML for the val verdicts; ignored if missing"
    ),
    nms_iou: float = typer.Option(
        None, "--nms-iou", help="Class-agnostic NMS IoU applied to the cached boxes (default: detector.post)"
    ),
    edge_snap_px: float = typer.Option(
        None, "--edge-snap-px", help="Snap box sides this close to the border onto it (default: detector.post)"
    ),
    out: str = typer.Option("runs/detmetrics", "--out", help="Output root; writes <out>/<run_name>/<experiment_id>/"),
) -> None:
    """Score the detection caches of the config's bundles and write the detmetrics folder.

    Exit code 0 is success (including a reproduced rerun), 1 is a check failure (split=test,
    a bundle seed outside the config split, failed reproducibility), 2 is an input error
    (crop mismatch, a missing or mismatched cache, a bad `--set`). split=test is refused
    before any bundle is opened; the bundles are scored in (fingerprint, run_id) order.
    The caches are post-processed with `detector.post` of the config; `--nms-iou` and
    `--edge-snap-px` replace its two values when given.
    """
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    from counting.config.eval_schema import EvalAppConfig, SplitsConfig
    from counting.config.hashing import config_hash
    from counting.config.loader import load_yaml_model
    from counting.data.formats.orchard_synth import SynthRun
    from counting.detection.cached import CachedDetector, cache_path, file_sha256
    from counting.evaluation.detmetrics import (
        DETMETRICS_SCHEMA, detection_verdicts, score_runs, write_detmetrics,
    )
    from counting.evaluation.record import build_record, experiment_id, git_state
    from counting.evaluation.report import ReproducibilityError
    from counting.evaluation.runner import CheckFailed, ProtectionError

    try:
        try:
            cfg = load_yaml_model(config, EvalAppConfig, overrides=set_)
        except KeyError as e:  # a --set path that cannot be descended
            raise ValueError(f"--set: {e.args[0]}") from e
        if run_:
            cfg = cfg.model_copy(update={"runs": list(run_)})
        if cfg.split == "test":
            raise ProtectionError("counting eval detmetrics does not run on split=test; "
                                  "the protected eval ladder scores the test caches once with rung L5")
        post = cfg.detector.post_settings().model_dump()
        for key, value in (("nms_iou", nms_iou), ("edge_snap_px", edge_snap_px)):
            if value is not None:
                post[key] = value
        postprocess = {"nms_iou": post["nms_iou"], "edge_snap_px": post["edge_snap_px"]}
        if post["min_short_px"] > 0:
            postprocess["min_short_px"] = post["min_short_px"]
        splits_cfg = load_yaml_model(splits_path, SplitsConfig)
        criteria, criteria_sha256 = _load_criteria(criteria_path)

        started = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        caches_root = Path(cache_dir).expanduser()
        bundles = [SynthRun.open(run_path) for run_path in cfg.runs]
        for bundle in bundles:
            meta, root = bundle.meta, bundle.root
            if meta.crop != cfg.crop:
                raise ValueError(f"{root}: crop mismatch: run is {meta.crop}, config is {cfg.crop}")
            actual = splits_cfg.split_of(meta.seed) or "none"
            if actual != cfg.split:
                raise CheckFailed(f"{root}: seed {meta.seed} belongs to split {actual}, config split is {cfg.split}")
            path = cache_path(caches_root, meta.run_id)
            if not path.is_file():
                raise FileNotFoundError(f"{path}: no detection cache for this bundle")
        bundles.sort(key=lambda b: (b.meta.fingerprint, b.meta.run_id))
        caches = [CachedDetector(cache_path(caches_root, b.meta.run_id), fingerprint=b.meta.fingerprint, **post)
                  for b in bundles]
        t1 = time.perf_counter()
        scored = score_runs(bundles, caches, cfg)
        t2 = time.perf_counter()
        finished = datetime.now(timezone.utc).isoformat()
        verdicts = detection_verdicts(scored["pooled"]["groups"], criteria, cfg.crop, cfg.split)

        code = git_state()
        fingerprints = [b.meta.fingerprint for b in bundles]
        fingerprints += [entry["meta"]["depth_fingerprint"] for entry in scored["runs"]]
        fingerprints += [c.digest for c in caches]
        criteria_part = criteria_sha256[:16] if criteria_sha256 else "-"
        post_part = f"nms{post['nms_iou']:g}:snap{post['edge_snap_px']:g}"
        if post["min_short_px"] > 0:
            post_part += f":short{post['min_short_px']:g}"
        exp_id = experiment_id(config_hash=config_hash(cfg), fingerprints=fingerprints, code=code, seed=cfg.seed,
                               extra=f"detmetrics:{post_part}:criteria:{criteria_part}")
        result = {"schema_version": DETMETRICS_SCHEMA, "kind": "detmetrics", "experiment_id": exp_id,
                  "run_name": cfg.run_name, "crop": cfg.crop, "split": cfg.split,
                  "thresholds": {"operating": cfg.tracker.score_hi, "low": cfg.tracker.score_lo},
                  "postprocess": postprocess,
                  "runs": scored["runs"], "pooled": scored["pooled"], "verdicts": verdicts}

        weights = [c.meta.get("weights_sha256") for c in caches]
        cache_rows = [{"run_id": b.meta.run_id, "file": c.path.name, "digest": c.digest,
                       "weights_sha256": c.meta.get("weights_sha256")} for b, c in zip(bundles, caches)]
        record = build_record(cfg=cfg, runs=[entry["meta"] for entry in scored["runs"]], code=code,
                              timings={"prepare": round(t1 - t0, 3), "evaluate": round(t2 - t1, 3)},
                              started=started, finished=finished,
                              extra={"caches": cache_rows, **postprocess, "criteria_sha256": criteria_sha256,
                                     "splits_sha256": file_sha256(Path(splits_path).expanduser())})
        record["detector"] = {"name": "cached", "noise": None,
                              "weights_sha256": weights[0] if all(w == weights[0] for w in weights) else None,
                              "cache_sha256": caches[0].digest if len(caches) == 1 else None, "pretrained": None}
        folder, reproduced = write_detmetrics(Path(out).expanduser(), cfg=cfg, result=result, record=record)
    except ReproducibilityError as e:
        typer.echo(f"reproducibility FAILED {e}")
        raise typer.Exit(code=1)
    except CheckFailed as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)
    except (ValueError, FileNotFoundError) as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=2)

    typer.echo(str(folder))
    if reproduced:
        typer.echo("reproduced (same detmetrics.json)")
    _echo_detmetrics_lines(result)
