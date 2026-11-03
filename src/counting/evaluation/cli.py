"""`counting eval` sub-app: `eval truth` prints pass/tree ground-truth tables and runs
the summary/geometry/split checks for one or more orchard_synth bundles; `eval run` and
`eval ladder` score a stage setting (or walk the oracle ladder) and write the experiment
folder.

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

    Exit code 0 is success, 1 is a check failure (summary mismatch/missing, geometry,
    split), 2 is an input error.
    """
    from counting.config.eval_schema import SplitsConfig
    from counting.config.loader import load_yaml_model
    from counting.data.formats.orchard_synth import FrameInput, PassKey, SynthRun, select_frames
    from counting.evaluation.truth import (
        SummaryMismatch,
        check_geometry,
        check_summary,
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
                detail = (
                    f"center_in_bbox={res['center_in_bbox']}, n_unoccluded={res['n_unoccluded']}, "
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
    except (ValueError, FileNotFoundError, NotImplementedError) as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=2)

    if failed:
        raise typer.Exit(code=1)


def _load_criteria(path: str):
    """YAML criteria at `path`, or `None` if the file does not exist."""
    import yaml
    from pathlib import Path

    p = Path(path).expanduser()
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else None


def _echo_rung_lines(run_entries: list[dict]) -> None:
    """One line per (bundle, rung): `<rung>: row U=... G6=... e_pipe=...` (`-` under stop_after=y1)."""
    multi = len(run_entries) > 1
    for entry in run_entries:
        prefix = f"{entry['meta']['run_id']}: " if multi else ""
        for name, rung in entry["rungs"].items():
            y6 = rung["y6"]
            if y6 is None:
                typer.echo(f"{prefix}{name}: row U=- G6=- e_pipe=-")
                continue
            row = y6["row"]
            e_pipe = "-" if row["e_pipe"] is None else f"{row['e_pipe']:+.1%}"
            typer.echo(f"{prefix}{name}: row U={row['U']} G6={row['G6']} e_pipe={e_pipe}")


def _run_eval(
    *, config: str, run_: list[str] | None, set_: list[str] | None, splits_path: str, criteria_path: str,
    kind: str, rungs: list[str] | None = None, ladder_path: str | None = None, with_2x2: bool = False,
) -> None:
    """Shared body of `eval run` (`kind="run"`) and `eval ladder` (`kind="ladder"`).

    Exit code 0 is success (including a reproduced rerun), 1 is a check failure
    (seed/split mismatch, summary mismatch, failed reproducibility), 2 is an
    input error. `CheckFailed` is a `ValueError`, so it is caught before the input-error branch.
    """
    import hashlib
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    from counting.config.eval_schema import EvalAppConfig, SplitsConfig
    from counting.config.hashing import config_hash
    from counting.config.loader import load_yaml_model
    from counting.data.formats.orchard_synth import SynthRun
    from counting.evaluation.record import build_record, experiment_id, git_state
    from counting.evaluation.report import SCHEMA_VERSION, ReproducibilityError, write_experiment
    from counting.evaluation.runner import CheckFailed, describe_run, evaluate, load_ladder, prepare, run_ladder
    from counting.evaluation.truth import SummaryMismatch

    try:
        cfg = load_yaml_model(config, EvalAppConfig, overrides=set_)
        if run_:
            cfg = cfg.model_copy(update={"runs": list(run_)})
        splits_cfg = load_yaml_model(splits_path, SplitsConfig)
        criteria = _load_criteria(criteria_path)

        ladder, ladder_bytes = None, b""
        if kind == "ladder":
            ladder_bytes = Path(ladder_path).expanduser().read_bytes()
            ladder = load_ladder(ladder_path)
            if rungs is None:
                rungs = [*(ladder.get("rungs") or {}), *(ladder.get("baselines") or {})]

        started = datetime.now(timezone.utc).isoformat()
        t_prepare = t_score = 0.0
        contexts, run_entries = [], []
        for run_path in cfg.runs:
            t0 = time.perf_counter()
            ctx = prepare(SynthRun.open(run_path), cfg, splits_cfg)
            t_prepare += time.perf_counter() - t0
            contexts.append(ctx)

            t0 = time.perf_counter()
            if kind == "run":
                out = {"chain": [], "rungs": {"run": evaluate(ctx, cfg)}, "interaction_2x2": None}
            else:
                out = run_ladder(ctx, cfg, ladder, rungs, with_2x2=with_2x2)
            t_score += time.perf_counter() - t0
            run_entries.append({"meta": describe_run(ctx), **out})
        finished = datetime.now(timezone.utc).isoformat()

        # Sort bundles by identity (fingerprint, run_id), not by cfg.runs/--run order, so
        # results.json's `runs` list is the same regardless of the order bundles were listed
        # in (the experiment id already ignores that order via sorted fingerprints).
        order = sorted(
            range(len(contexts)), key=lambda i: (contexts[i].run.meta.fingerprint, contexts[i].run.meta.run_id)
        )
        contexts = [contexts[i] for i in order]
        run_entries = [run_entries[i] for i in order]

        if kind == "run":
            uses_box_median = cfg.lift.depth == "box_median"
            extra = ""
        else:
            lift_depths = [rung["lift_depth"] for entry in run_entries for rung in entry["rungs"].values()]
            if with_2x2:
                base_name = ladder["interaction_2x2"]["base"]
                base_patch = (ladder.get("rungs") or {}).get(base_name) \
                    or (ladder.get("baselines") or {}).get(base_name) or {}
                lift_depths.append((base_patch.get("lift") or {}).get("depth", cfg.lift.depth))
            uses_box_median = "box_median" in lift_depths
            extra = f"ladder:{hashlib.sha256(ladder_bytes).hexdigest()[:16]}:{','.join(rungs)}:{with_2x2}"

        fingerprints = [ctx.run.meta.fingerprint for ctx in contexts]
        data = [describe_run(ctx) for ctx in contexts]
        if uses_box_median:
            depth_fps = [ctx.run.depth_fingerprint() for ctx in contexts]
            fingerprints += depth_fps
            data = [{**d, "depth_fingerprint": fp} for d, fp in zip(data, depth_fps)]

        code = git_state()
        exp_id = experiment_id(config_hash=config_hash(cfg), fingerprints=fingerprints, code=code,
                               seed=cfg.seed, extra=extra)
        results = {"schema_version": SCHEMA_VERSION, "kind": kind, "run_name": cfg.run_name,
                   "experiment_id": exp_id, "split": cfg.split, "crop": cfg.crop, "runs": run_entries}
        record = build_record(cfg=cfg, runs=data, code=code,
                              timings={"prepare": round(t_prepare, 3), "evaluate": round(t_score, 3)},
                              started=started, finished=finished)
        folder, reproduced = write_experiment(Path(cfg.output_dir).expanduser(), cfg=cfg, results=results,
                                              record=record, criteria=criteria)
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
        "configs/eval/criteria.yaml", "--criteria", help="Criteria YAML; ignored if missing"
    ),
) -> None:
    """Run one stage setting (`cfg.stages`) over the config's bundles and write the experiment folder."""
    _run_eval(config=config, run_=run_, set_=set_, splits_path=splits_path, criteria_path=criteria_path, kind="run")


@eval_app.command()
def ladder(
    config: str = typer.Option(..., "--config", help="Eval config YAML"),
    rungs: str = typer.Option(None, "--rungs", help="Comma-separated rung/baseline names (default: all)"),
    with_2x2: bool = typer.Option(False, "--with-2x2", help="Also run the (track, dedup) 2x2 table"),
    ladder_path: str = typer.Option(
        "configs/eval/sweeps/ladder.yaml", "--ladder", help="Ladder YAML (rungs, baselines, interaction_2x2)"
    ),
    run_: list[str] = typer.Option(None, "--run", help="Override cfg.runs with these bundle paths"),
    set_: list[str] = typer.Option(None, "--set", help="Dot-path override: key.path=value", show_default=False),
    splits_path: str = typer.Option(
        "configs/eval/splits.yaml", "--splits", help="Splits YAML (for prepare's split check)"
    ),
    criteria_path: str = typer.Option(
        "configs/eval/criteria.yaml", "--criteria", help="Criteria YAML; ignored if missing"
    ),
) -> None:
    """Walk the oracle ladder (`run_ladder`) over the config's bundles and write the experiment folder."""
    rung_list = [r.strip() for r in rungs.split(",")] if rungs else None
    _run_eval(config=config, run_=run_, set_=set_, splits_path=splits_path, criteria_path=criteria_path,
             kind="ladder", rungs=rung_list, ladder_path=ladder_path, with_2x2=with_2x2)
