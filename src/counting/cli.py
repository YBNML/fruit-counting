"""Typer-based CLI entrypoint."""

from __future__ import annotations

import platform
import sys

import typer
from rich.console import Console
from rich.table import Table

from counting import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


# Suppresses Typer's single-command auto-promotion; keep even when other commands are added.
@app.callback()
def _main() -> None:
    """Fruit counting pipeline CLI."""


@app.command()
def info() -> None:
    """Show environment and device information."""
    import torch

    from counting.utils.device import resolve_device

    table = Table(title=f"counting {__version__}")
    table.add_column("key")
    table.add_column("value")
    table.add_row("python", sys.version.split()[0])
    table.add_row("platform", platform.platform())
    table.add_row("torch", torch.__version__)
    table.add_row("cuda.available", str(torch.cuda.is_available()))
    mps = getattr(torch.backends, "mps", None)
    table.add_row("mps.available", str(bool(mps and mps.is_available())))
    table.add_row("resolved(auto)", resolve_device("auto"))
    console.print(table)


@app.command("validate-config")
def validate_config(
    path: str = typer.Argument(..., help="Path to YAML config"),
    set_: list[str] = typer.Option(
        None, "--set", help="Dot-path override: key.path=value", show_default=False
    ),
) -> None:
    """Load and validate a YAML config. Prints resolved device."""
    from counting.config.loader import load_config
    from counting.utils.device import resolve_device

    cfg = load_config(path, overrides=set_ or None)
    device = resolve_device(cfg.device)
    console.print(f"[green]OK[/green] {path} (device → {device})")


@app.command()
def diagnose(
    image_dir: str = typer.Argument(..., help="Directory of images"),
    report_dir: str = typer.Option("./reports/diagnostics", help="Where to write outputs"),
) -> None:
    """Compute resolution/blur/exposure diagnostics for a directory."""
    from counting.data.diagnostics import diagnose_directory

    r = diagnose_directory(image_dir, report_dir=report_dir)
    console.print(
        f"[green]OK[/green] {r.image_count} images → {report_dir}\n"
        f"  low_blur_ratio={r.blur['low_blur_ratio']}  "
        f"under={r.exposure['underexposed_ratio']}  over={r.exposure['overexposed_ratio']}"
    )


@app.command()
def infer(
    image: str = typer.Argument(..., help="Path to image"),
    config: str = typer.Option(..., "--config", "-c", help="Pipeline YAML"),
    set_: list[str] = typer.Option(None, "--set", help="Override key.path=value"),
    output: str = typer.Option(None, "--output", "-o", help="Output JSON path (optional)"),
) -> None:
    """Run inference on a single image."""
    from counting.config.loader import load_config
    from counting.io.serialize import write_batch_json
    from counting.pipeline import build_pipeline
    from counting.utils.image import read_image_rgb

    cfg = load_config(config, overrides=set_ or None)
    pipe = build_pipeline(cfg)
    arr = read_image_rgb(image)
    result = pipe.run_numpy(arr, image_path=str(image))

    console.print(
        f"image={image} raw_count={result.raw_count} verified={result.verified_count} "
        f"device={result.device} err={result.error or '-'}"
    )
    if output:
        write_batch_json([result], output)
        console.print(f"[green]saved[/green] {output}")


@app.command()
def batch(
    image_dir: str = typer.Argument(..., help="Directory of images"),
    config: str = typer.Option(..., "--config", "-c", help="Pipeline YAML"),
    set_: list[str] = typer.Option(None, "--set", help="Override key.path=value"),
    output: str = typer.Option("./runs/last_batch", help="Output directory"),
    fmt: str = typer.Option("json", "--format", help="json | csv | both"),
) -> None:
    """Run inference over a directory."""
    from pathlib import Path

    from counting.config.loader import load_config
    from counting.data.formats.imagefolder import ImageFolderDataset
    from counting.io.serialize import write_batch_csv, write_batch_json
    from counting.pipeline import build_pipeline

    cfg = load_config(config, overrides=set_ or None)
    if fmt not in {"json", "csv", "both"}:
        raise typer.BadParameter("--format must be json|csv|both")

    ds = ImageFolderDataset(image_dir)
    pipe = build_pipeline(cfg)
    results = []
    for i, rec in enumerate(ds, 1):
        arr = rec.read_rgb()
        r = pipe.run_numpy(arr, image_path=str(rec.path))
        results.append(r)
        console.print(
            f"[{i}/{len(ds)}] {rec.relpath} raw={r.raw_count} verified={r.verified_count} "
            f"err={r.error or '-'}"
        )

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt in {"json", "both"}:
        write_batch_json(results, out_dir / "results.json")
    if fmt in {"csv", "both"}:
        write_batch_csv(results, out_dir / "results.csv")
    console.print(f"[green]done[/green] {len(results)} images → {out_dir}")


@app.command("cache-embeddings")
def cache_embeddings(
    config: str = typer.Option(..., "--config", "-c", help="Training YAML"),
    set_: list[str] = typer.Option(None, "--set", help="Override key.path=value"),
    limit: int = typer.Option(0, help="If >0, only process this many images (smoke)"),
) -> None:
    """Precompute SAM embeddings and write to the on-disk feature cache."""
    import hashlib
    import yaml

    from tqdm import tqdm

    from counting.config.loader import apply_overrides
    from counting.config.train_schema import TrainAppConfig
    from counting.data.cache import FeatureCacheWriter, compute_cache_meta_hash
    from counting.data.formats.fsc147 import FSC147Dataset
    from counting.models.pseco.embedding import SAMImageEmbedder
    from counting.utils.device import resolve_device

    raw = yaml.safe_load(open(config, "r", encoding="utf-8"))
    if set_:
        raw = apply_overrides(raw, list(set_))
    cfg = TrainAppConfig.model_validate(raw)

    device = resolve_device(cfg.device)
    if device not in {"cuda", "mps", "cpu"}:
        raise typer.BadParameter(f"Unexpected device: {device}")

    if cfg.data.format != "fsc147":
        raise typer.BadParameter(
            f"Only fsc147 is supported for embedding caching; got {cfg.data.format!r}"
        )

    # Cache train + val into the SAME cache dir so the trainer's val loader
    # can read the same file. Deduplicate in case someone sets them equal.
    splits_to_cache: list[str] = []
    for s in (cfg.data.train_split, cfg.data.val_split):
        if s not in splits_to_cache:
            splits_to_cache.append(s)

    # Content hash over fields that, when changed, invalidate the cache.
    sam_hash = ""
    sam_ckpt_path = cfg.model.sam_checkpoint
    if sam_ckpt_path:
        h = hashlib.sha256()
        with open(sam_ckpt_path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        sam_hash = h.hexdigest()[:16]

    meta_for_hash = {
        "sam_ckpt_hash": sam_hash,
        "image_size": cfg.data.image_size,
        "dtype": cfg.cache.dtype,
        "dataset_root": cfg.data.root,
        "splits": splits_to_cache,
        "augment_variants": cfg.cache.augment_variants,
    }
    cache_hash = compute_cache_meta_hash(meta_for_hash)

    embedder = SAMImageEmbedder(cfg.model.sam_checkpoint, device=device)
    console.print(f"[loading] SAM ViT-H on {device}")
    embedder.prepare()

    writer = FeatureCacheWriter(
        cache_dir=cfg.cache.dir,
        meta={**meta_for_hash, "hash": cache_hash},
        shard_size=256,
        dtype=cfg.cache.dtype,
    )
    writer.open()

    total = 0
    for split in splits_to_cache:
        ds = FSC147Dataset(cfg.data.root, split=split)
        records = list(ds)
        if limit > 0:
            records = records[:limit]
        for rec in tqdm(records, desc=f"embed[{split}]"):
            arr = rec.read_rgb()
            emb = embedder.embed(arr).numpy()
            writer.write(rec.relpath, emb)
        total += len(records)

    writer.close()
    embedder.cleanup()

    console.print(
        f"[green]done[/green] cached {total} images "
        f"across splits {splits_to_cache} at {cfg.cache.dir}"
    )
    console.print(f"  hash={cache_hash}")


_train_app = typer.Typer(help="Training entry points", no_args_is_help=True)
app.add_typer(_train_app, name="train")

from counting.evaluation.cli import eval_app  # noqa: E402
app.add_typer(eval_app, name="eval")

pose_app = typer.Typer(help="Pose files of render bundles", no_args_is_help=True)
app.add_typer(pose_app, name="pose")


# Suppresses Typer's single-command auto-promotion; keep even when other commands are added.
@pose_app.callback()
def _pose() -> None:
    """Pose files of render bundles."""


@_train_app.command("detector")
def train_detector_cli(
    config: str = typer.Option(..., "--config", "-c", help="Detector training YAML"),
    set_: list[str] = typer.Option(None, "--set", help="Dot-path override: key.path=value", show_default=False),
    splits_path: str = typer.Option(
        "configs/eval/splits.yaml", "--splits", help="Splits YAML (every training bundle must be a train seed)"
    ),
    max_iters: int = typer.Option(None, "--max-iters", min=1, help="Stop after this many steps"),
    densest_first: bool = typer.Option(
        False, "--densest-first", help="Benchmark: one epoch of the views with the most boxes (needs --max-iters)"
    ),
    resume: str = typer.Option(None, "--resume", help="A training folder to continue from its last epoch"),
) -> None:
    """Train the Faster R-CNN detector on the synthetic train bundles.

    Exit code 0 is success, 1 is a check failure (a bundle outside the train split, a run
    folder that is finished, unfinished or not resumable by this request, a non-finite loss,
    out of GPU memory), 2 is an input error (missing bundle, bad config or `--set`, bad options).
    """
    from pathlib import Path

    from counting.config.det_schema import DetTrainConfig
    from counting.config.eval_schema import SplitsConfig
    from counting.config.loader import load_yaml_model
    from counting.detection.cached import file_sha256
    from counting.evaluation.runner import CheckFailed
    from counting.training.detector import train_detector

    try:
        try:
            cfg = load_yaml_model(config, DetTrainConfig, overrides=set_)
        except KeyError as e:  # a --set path that cannot be descended
            raise ValueError(f"--set: {e.args[0]}") from e
        splits = load_yaml_model(splits_path, SplitsConfig)
        result = train_detector(cfg, splits, max_iters=max_iters, densest_first=densest_first,
                                resume=Path(resume) if resume else None,
                                splits_sha256=file_sha256(Path(splits_path).expanduser()), echo=typer.echo)
    except CheckFailed as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)
    except (ValueError, FileNotFoundError) as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=2)

    def mib(x: float | None) -> str:
        return "-" if x is None else f"{x:.0f}"

    sec = "-" if result.sec_per_iter is None else f"{result.sec_per_iter:.3f}"
    typer.echo(f"weights {result.weights_sha256}")
    typer.echo(f"steps {result.steps}, {sec} s/iter, max memory {mib(result.max_memory_mb)} MiB, "
               f"reserved {mib(result.max_reserved_mb)} MiB")


@app.command()
def detect(
    config: str = typer.Option(..., "--config", help="Detector inference YAML"),
    weights: str = typer.Option(..., "--weights", help="Trained weights; its training config.yaml sits next to it"),
    run_: list[str] = typer.Option(..., "--run", help="A bundle to run the detector on (repeat for more)"),
    set_: list[str] = typer.Option(None, "--set", help="Dot-path override: key.path=value", show_default=False),
    splits_path: str = typer.Option(
        "configs/eval/splits.yaml", "--splits", help="Splits YAML (every bundle's seed must be in a split)"
    ),
) -> None:
    """Run the trained detector over bundles and write one detection cache per bundle.

    Each cache goes to `<output_dir>/<weights sha256[:16]>/<run_id>.npz`, with a `.json` summary
    next to it, and is written once: an existing cache stops the command before the model loads.
    Test bundles need a frozen, committed config that pins the weights' sha256 and no `--set`;
    each test bundle's cache adds a row to `<output_dir>/test_log.csv` as soon as it is written.

    Exit code 0 is success, 1 is a check failure (a seed in no split, weights trained with other
    model settings or not the ones the config pins, a test-bundle protection check, an existing
    cache), 2 is an input error (missing weights, training config, bundle or RGB image, a bad
    config or `--set`, weights that do not fit the model).
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from counting.config.det_schema import DetInferConfig, DetTrainConfig, model_mismatch
    from counting.config.eval_schema import SplitsConfig
    from counting.config.hashing import config_hash
    from counting.config.loader import load_yaml_model
    from counting.data.formats.orchard_synth import SynthRun
    from counting.detection import frcnn
    from counting.detection.cached import cache_path, detect_to_cache, file_sha256, rgb_fingerprint
    from counting.evaluation.protect import (
        append_log_row, check_test_bundles, check_test_detect, check_test_log, detect_log_row, earlier_log_runs,
    )
    from counting.evaluation.record import counting_repo, git_state
    from counting.evaluation.runner import CheckFailed
    from counting.training.detector import train_config_hash

    try:
        try:
            cfg = load_yaml_model(config, DetInferConfig, overrides=set_)
        except KeyError as e:  # a --set path that cannot be descended
            raise ValueError(f"--set: {e.args[0]}") from e
        splits = load_yaml_model(splits_path, SplitsConfig)

        w_path = Path(weights).expanduser()
        if not w_path.is_file():
            raise FileNotFoundError(f"{weights}: no weights file")
        w_sha = file_sha256(w_path)

        train_dir = w_path.absolute().parent
        train_path = train_dir / "config.yaml"
        if not train_path.is_file():
            raise FileNotFoundError(f"{train_path}: no training config next to the weights")
        train_cfg = load_yaml_model(train_path, DetTrainConfig)
        keys = model_mismatch(train_cfg.model, cfg.model)
        if keys:
            raise CheckFailed(f"{train_path}: the weights were trained with other model settings: {', '.join(keys)}")
        pretrained = None
        record_path = train_dir / "record.json"
        if record_path.is_file():
            try:
                pretrained = json.loads(record_path.read_text(encoding="utf-8")).get("pretrained")
            except (ValueError, AttributeError) as e:
                raise ValueError(f"{record_path}: not a readable training record ({e})") from None

        bundles = [SynthRun.open(path) for path in run_]
        for bundle in bundles:
            if splits.split_of(bundle.meta.seed) is None:
                raise CheckFailed(f"{bundle.root}: seed {bundle.meta.seed} is in no split")
        bundles.sort(key=lambda b: (b.meta.fingerprint, b.meta.run_id))

        out_root = Path(cfg.output_dir).expanduser()
        log = out_root / "test_log.csv"
        chash = config_hash(cfg)
        protected, earlier = None, 0
        if any(splits.split_of(b.meta.seed) == "test" for b in bundles):
            protected = check_test_detect(config_path=config, sets=set_ or [], files={"--splits": splits_path},
                                          weights_sha256=w_sha, pinned_sha256=cfg.weights_sha256,
                                          repo=counting_repo())
            check_test_bundles(bundles, splits)
            check_test_log(log)
            earlier = earlier_log_runs(log, config_hash=chash, extra=f"weights:{w_sha[:16]}")
        elif cfg.weights_sha256 is not None and cfg.weights_sha256 != w_sha:
            raise CheckFailed(f"--weights {weights} has sha256 {w_sha[:16]}, but the config pins weights_sha256 "
                              f"{cfg.weights_sha256[:16]}")
        code = protected.code if protected is not None else git_state()

        paths = [cache_path(out_root / w_sha[:16], b.meta.run_id) for b in bundles]
        for path in paths:
            for p in (path, path.with_suffix(".json")):
                if p.exists():
                    raise CheckFailed(f"{p}: cache exists; a cache is written once per weights, so remove it first "
                                      "to make it again")
        rgb_shas = [rgb_fingerprint(b.frames) for b in bundles]

        detector = frcnn.load_detector(w_path, cfg.model, device=cfg.device)
        import torch
        import torchvision

        common = {"weights_sha256": w_sha, "train_id": train_dir.name,
                  "train_config_hash": train_config_hash(train_cfg), "pretrained": pretrained,
                  "score_min": cfg.model.score_min, "max_dets": cfg.model.max_dets,
                  "model": cfg.model.model_dump(), "torch": str(torch.__version__),
                  "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda, "device": cfg.device,
                  "counting_commit": code["commit"], "counting_dirty": code["dirty"],
                  "splits_sha256": file_sha256(Path(splits_path).expanduser())}
        started = datetime.now(timezone.utc).isoformat()
        if earlier > 0:
            typer.echo(f"warning: test_log.csv already lists {earlier} test detect row(s) of config {chash} with "
                       "these weights; this run adds one row per bundle")
        for bundle, rgb_sha, path in zip(bundles, rgb_shas, paths):
            m, cam = bundle.meta, bundle.frames[0].camera
            meta = {"run_id": m.run_id, "fingerprint": m.fingerprint, "rgb_sha256": rgb_sha, **common,
                    "image_size": [cam.width, cam.height]}
            summary = detect_to_cache(bundle.frames, detector, path, meta=meta, max_dets=cfg.model.max_dets)
            if protected is not None:
                append_log_row(log, detect_log_row(date=started, config_rel=protected.config_rel, config_hash=chash,
                                                   weights_sha256=w_sha, fingerprints=[m.fingerprint],
                                                   digests=[summary["digest"]], commit=code["commit"]))
            typer.echo(str(path))
            typer.echo(f"{m.run_id}: frames {summary['n_frames']}, detections {summary['n_dets']}, "
                       f"digest {summary['digest'][:16]}, {summary['seconds']:.1f} s")
            if summary["saturated"]:
                typer.echo(f"warning: {m.run_id}: {len(summary['saturated'])} frame(s) reached "
                           f"max_dets={cfg.model.max_dets}; detections past the cap are not in the cache")
    except (CheckFailed, FileExistsError) as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)
    except (ValueError, FileNotFoundError) as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=2)


@pose_app.command("vo")
def pose_vo(
    config: str = typer.Option(..., "--config", help="Visual odometry YAML"),
    run_: list[str] = typer.Option(..., "--run", help="A bundle to run the odometry on (repeat for more)"),
    set_: list[str] = typer.Option(None, "--set", help="Dot-path override: key.path=value", show_default=False),
    splits_path: str = typer.Option(
        "configs/eval/splits_pose.yaml", "--splits", help="Splits YAML (every bundle's seed must be in a split)",
    ),
) -> None:
    """Run the frame-to-frame RGB-D visual odometry over bundles and write one pose cache per bundle.

    Each cache goes to `<output_dir>/<VO config hash>/<run_id>.npz`, with a `.json` summary next to it,
    and is written once: an existing cache or summary stops the command before the odometry runs.
    With `step_m` set, the odometry runs on the thinned frames only. The cache meta names the bundle
    (run id, fingerprint, sha256 of all its RGB and depth images), the settings and their hash, the
    OpenCV and numpy versions, the code state, the splits file's sha256 and the image size; the time
    taken goes to the summary only. Every bundle must be a sequence bundle with its seed in a split.
    A test bundle needs a frozen config under configs/pose/frozen/, a clean committed tree and no --set;
    each test file adds a row to `<output_dir>/test_log.csv` as soon as it is written.

    Exit code 0 is success, 1 is a check failure (a seed in no split, a test-bundle protection check, an
    existing cache or summary), 2 is an input error (missing bundle or image, a bundle that is not a
    sequence, a bad config or `--set`).
    """
    from datetime import datetime, timezone
    from pathlib import Path

    import cv2
    import numpy as np

    from counting.config.eval_schema import SplitsConfig
    from counting.config.hashing import config_hash
    from counting.config.loader import load_yaml_model
    from counting.config.pose_schema import VoConfig
    from counting.data.formats.orchard_synth import SynthRun, select_frames
    from counting.detection.cached import file_sha256, rgb_fingerprint
    from counting.evaluation.posesource import vo_inputs
    from counting.evaluation.protect import (
        append_log_row, check_test_bundles, check_test_log, check_test_pose_file, earlier_log_runs, pose_log_row,
    )
    from counting.evaluation.record import counting_repo, git_state
    from counting.evaluation.runner import CheckFailed, ProtectionError
    from counting.odometry.cache import pose_cache_path
    from counting.odometry.vo import vo_to_cache

    started = datetime.now(timezone.utc).isoformat()
    try:
        try:
            cfg = load_yaml_model(config, VoConfig, overrides=set_)
        except KeyError as e:  # a --set path that cannot be descended
            raise ValueError(f"--set: {e.args[0]}") from e
        splits = load_yaml_model(splits_path, SplitsConfig)

        bundles = [SynthRun.open(path) for path in run_]
        for bundle in bundles:
            seed = bundle.meta.seed
            if splits.split_of(seed) is None:
                raise CheckFailed(f"{bundle.root}: seed {seed} is in no split")
            if not bundle.meta.is_sequence:
                raise ValueError(f"{bundle.root}: not a sequence bundle (max_frames or irregular step)")
        held_out = [b for b in bundles if splits.split_of(b.meta.seed) == "test"]
        bundles.sort(key=lambda b: (b.meta.fingerprint, b.meta.run_id))

        out_root = Path(cfg.output_dir).expanduser()
        log = out_root / "test_log.csv"
        chash = config_hash(cfg)
        extra = f"opencv:{cv2.__version__}"
        protected, earlier = None, 0
        if held_out:
            try:
                protected = check_test_pose_file(command="vo", config_path=config, sets=set_ or [],
                                                 files={"--splits": splits_path}, repo=counting_repo())
            except ProtectionError as e:
                raise CheckFailed(f"{held_out[0].root}: seed {held_out[0].meta.seed} is a test seed; {e}") from e
            check_test_bundles(bundles, splits)
            check_test_log(log)
            earlier = earlier_log_runs(log, config_hash=chash, extra=extra)

        paths = [pose_cache_path(out_root / chash, b.meta.run_id) for b in bundles]
        for path in paths:
            for p in (path, path.with_suffix(".json")):
                if p.exists():
                    raise CheckFailed(f"{p}: pose cache exists; remove it first to make it again")
        # Every bundle's frames and fingerprints come before the first odometry run, so a bad bundle
        # stops the command before any cache is written.
        inputs = [vo_inputs(select_frames(b.frames, base_step_m=b.meta.base_step_m, step_m=cfg.step_m))
                  for b in bundles]
        rgb_shas = [rgb_fingerprint(b.frames) for b in bundles]
        depth_shas = [b.depth_fingerprint() for b in bundles]

        code = protected.code if protected is not None else git_state()
        common = {"vo": {k: v for k, v in cfg.model_dump(mode="json").items() if k != "output_dir"},
                  "vo_config_hash": chash, "opencv": cv2.__version__, "numpy": np.__version__,
                  "counting_commit": code["commit"], "counting_dirty": code["dirty"],
                  "splits_sha256": file_sha256(Path(splits_path).expanduser())}
        if earlier > 0:
            typer.echo(f"warning: test_log.csv already lists {earlier} test pose row(s) of config {chash} with "
                       f"{extra}; this run adds one row per bundle")
        for bundle, frames, rgb_sha, depth_sha, path in zip(bundles, inputs, rgb_shas, depth_shas, paths):
            m, cam = bundle.meta, bundle.frames[0].camera
            meta = {"run_id": m.run_id, "fingerprint": m.fingerprint, "rgb_sha256": rgb_sha,
                    "depth_sha256": depth_sha, **common, "image_size": [cam.width, cam.height]}
            summary = vo_to_cache(frames, cfg, path, meta=meta)
            if protected is not None:
                append_log_row(log, pose_log_row(kind="pose_vo", date=started, config_rel=protected.config_rel,
                                                 config_hash=chash, extra=extra, fingerprint=m.fingerprint,
                                                 digest=summary["digest"], commit=code["commit"]))
            passes = summary["passes"]
            typer.echo(str(path))
            typer.echo(f"{m.run_id}: frames {summary['n_frames']}, pairs {sum(p['pairs'] for p in passes)}, "
                       f"failed {sum(p['failed'] for p in passes)}, prior {sum(p['prior'] for p in passes)}, "
                       f"digest {summary['digest'][:16]}, {summary['seconds']:.1f} s")
    except (CheckFailed, FileExistsError) as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)
    except (ValueError, FileNotFoundError) as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=2)


@pose_app.command("imu")
def pose_imu(
    config: str = typer.Option(..., "--config", help="Simulated IMU YAML"),
    run_: list[str] = typer.Option(..., "--run", help="A bundle to make the IMU file of (repeat for more)"),
    set_: list[str] = typer.Option(None, "--set", help="Dot-path override: key.path=value", show_default=False),
    splits_path: str = typer.Option(
        "configs/eval/splits_pose.yaml", "--splits", help="Splits YAML (every bundle's seed must be in a split)",
    ),
) -> None:
    """Simulate the IMU of bundles and write one IMU file per bundle.

    Each file goes to `<output_dir>/<IMU config hash>/<run_id>.npz`, with a `.json` summary next to it,
    and is written once: an existing file or summary stops the command before any stream is made. A file
    holds a measured stream for every drive of both layouts (`rig`: one drive per side; `pass`: one drive
    per pass) under every noise seed of the config: the noiseless samples of the bundle's body motion (its
    recorded trajectory, or the assumed straight drive of a bundle without one) plus the profile's noise,
    drawn under the config `seed`, the noise seed and the bundle seed. Every bundle's drives are made
    before the first file is written. The file meta names the bundle (run id, fingerprint), the profile
    and its noise, the sampling, the assumed motion, the trajectory source, the settings hash, the numpy
    and SciPy versions, the code state and the splits file's sha256; the time taken goes to the summary
    only. Every bundle must be a sequence bundle with its seed in a split. A test bundle needs a frozen
    config under configs/pose/frozen/, a clean committed tree and no --set; each test file adds a row to
    `<output_dir>/test_log.csv` as soon as it is written. A profile without white noise or turn-on bias is
    refused when the config is read, since a noiseless IMU is the truth.

    Exit code 0 is success, 1 is a check failure (a seed in no split, a test-bundle protection check, an
    existing file or summary), 2 is an input error (missing bundle, a bundle that is not a sequence, a bad
    config or `--set`, a profile without white noise or turn-on bias).
    """
    import importlib.metadata
    import json
    import os
    import time
    from dataclasses import asdict
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np

    from counting.config.eval_schema import SplitsConfig
    from counting.config.hashing import config_hash
    from counting.config.loader import load_yaml_model
    from counting.config.pose_schema import ImuConfig
    from counting.data.formats.orchard_synth import SynthRun
    from counting.detection.cached import file_sha256
    from counting.evaluation.imusim import AssumedMotion, ImuProfile, ImuSampling, imu_session, session_drives
    from counting.evaluation.protect import (
        append_log_row, check_test_bundles, check_test_log, check_test_pose_file, earlier_log_runs, pose_log_row,
    )
    from counting.evaluation.record import counting_repo, git_state
    from counting.evaluation.runner import CheckFailed, ProtectionError
    from counting.odometry.imufile import LAYOUTS, imu_file_path, write_imu_file

    started = datetime.now(timezone.utc).isoformat()
    try:
        try:
            cfg = load_yaml_model(config, ImuConfig, overrides=set_)
        except KeyError as e:  # a --set path that cannot be descended
            raise ValueError(f"--set: {e.args[0]}") from e
        splits = load_yaml_model(splits_path, SplitsConfig)

        bundles = [SynthRun.open(path) for path in run_]
        for bundle in bundles:
            seed = bundle.meta.seed
            if splits.split_of(seed) is None:
                raise CheckFailed(f"{bundle.root}: seed {seed} is in no split")
            if not bundle.meta.is_sequence:
                raise ValueError(f"{bundle.root}: not a sequence bundle (max_frames or irregular step)")
        held_out = [b for b in bundles if splits.split_of(b.meta.seed) == "test"]
        bundles.sort(key=lambda b: (b.meta.fingerprint, b.meta.run_id))

        out_root = Path(cfg.output_dir).expanduser()
        log = out_root / "test_log.csv"
        chash = config_hash(cfg)
        extra = f"imu:{cfg.profile}"
        protected, earlier = None, 0
        if held_out:
            try:
                protected = check_test_pose_file(command="imu", config_path=config, sets=set_ or [],
                                                 files={"--splits": splits_path}, repo=counting_repo())
            except ProtectionError as e:
                raise CheckFailed(f"{held_out[0].root}: seed {held_out[0].meta.seed} is a test seed; {e}") from e
            check_test_bundles(bundles, splits)
            check_test_log(log)
            earlier = earlier_log_runs(log, config_hash=chash, extra=extra)

        paths = [imu_file_path(out_root / chash, b.meta.run_id) for b in bundles]
        for path in paths:
            for p in (path, path.with_suffix(".json")):
                if p.exists():
                    raise CheckFailed(f"{p}: IMU file exists; remove it first to make it again")
        # Every bundle's drives come before the first write, so a bad bundle stops the command before any
        # file is written.
        motion = AssumedMotion.of(cfg)
        drives = [{layout: session_drives(b, layout, motion) for layout in LAYOUTS} for b in bundles]

        profile, sampling = ImuProfile.from_config(cfg.noise), ImuSampling.of(cfg)
        code = protected.code if protected is not None else git_state()
        common = {"imu_seed": cfg.seed, "noise_seeds": list(cfg.noise_seeds), "profile": cfg.profile,
                  "noise": cfg.noise.model_dump(), "rate_hz": cfg.rate_hz, "acc_range_g": cfg.acc_range_g,
                  "row_end_s": cfg.row_end_s, "motion": asdict(motion), "imu_config_hash": chash,
                  "numpy": np.__version__, "scipy": importlib.metadata.version("scipy"),
                  "counting_commit": code["commit"], "counting_dirty": code["dirty"],
                  "splits_sha256": file_sha256(Path(splits_path).expanduser())}
        if earlier > 0:
            typer.echo(f"warning: test_log.csv already lists {earlier} test pose row(s) of config {chash} with "
                       f"{extra}; this run adds one row per bundle")
        for bundle, by_layout, path in zip(bundles, drives, paths):
            m = bundle.meta
            start = time.perf_counter()
            streams, rows = {}, []   # keys in layout, noise seed, drive order
            for layout in LAYOUTS:
                for noise_seed in cfg.noise_seeds:
                    measured = imu_session(by_layout[layout], profile, sampling, imu_seed=cfg.seed,
                                           noise_seed=noise_seed, scene=m.seed)
                    for d, stream in zip(by_layout[layout], measured):
                        streams[d.stream_key(noise_seed)] = stream
                        rows.append({"layout": layout, "noise_seed": noise_seed, "side": d.side, "aisle": d.aisle,
                                     "pitch": d.pitch_deg, "samples": len(stream.t),
                                     "t_first": float(stream.t[0]), "t_last": float(stream.t[-1])})
            meta = {"run_id": m.run_id, "fingerprint": m.fingerprint, **common,
                    "traj_source": by_layout["rig"][0].source}
            digest = write_imu_file(path, streams, meta)
            seconds = time.perf_counter() - start
            if protected is not None:
                append_log_row(log, pose_log_row(kind="pose_imu", date=started, config_rel=protected.config_rel,
                                                 config_hash=chash, extra=extra, fingerprint=m.fingerprint,
                                                 digest=digest, commit=code["commit"]))
            summary = {"digest": digest, "file": path.name, "meta": meta, "streams": rows, "seconds": seconds}
            sidecar = path.with_suffix(".json")
            tmp = sidecar.with_name(sidecar.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(summary, sort_keys=True, indent=2) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, sidecar)
            typer.echo(str(path))
            typer.echo(f"{m.run_id}: profile {cfg.profile}, drives rig {len(by_layout['rig'])} pass "
                       f"{len(by_layout['pass'])}, noise seeds {','.join(map(str, cfg.noise_seeds))}, "
                       f"samples {sum(r['samples'] for r in rows)}, digest {digest[:16]}, {seconds:.1f} s")
    except (CheckFailed, FileExistsError) as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)
    except (ValueError, FileNotFoundError) as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=2)


@_train_app.command("pseco-head")
def train_pseco_head_cli(
    config: str = typer.Option(..., "--config", "-c"),
    set_: list[str] = typer.Option(None, "--set"),
    resume: str = typer.Option(None, "--resume", help="Path to last.ckpt"),
) -> None:
    """Fine-tune the PseCo ROIHeadMLP using cached SAM features."""
    import yaml

    from counting.config.loader import apply_overrides
    from counting.config.train_schema import TrainAppConfig
    from counting.models.pseco.trainer import train_pseco_head

    raw = yaml.safe_load(open(config, "r", encoding="utf-8"))
    if set_:
        raw = apply_overrides(raw, list(set_))
    cfg = TrainAppConfig.model_validate(raw)

    if resume:
        console.print(f"[yellow]--resume is not yet supported; running from init_mlp[/yellow]")

    train_pseco_head(cfg)
    console.print("[green]training complete[/green]")


@app.command("extract-clip-features")
def extract_clip_features(
    dataset: str = typer.Option(
        "fsc147",
        "--dataset",
        help="Either 'fsc147' (auto-derive from ImageClasses_FSC147.txt) or a path to a newline-delimited text file of class names.",
    ),
    dataset_root: str = typer.Option(
        "./datasets/fsc147",
        "--dataset-root",
        help="FSC-147 root (used when --dataset=fsc147).",
    ),
    out: str = typer.Option(..., "--out", "-o", help="Output .pt file"),
    device: str = typer.Option(
        "cpu", "--device", help="cpu | mps | cuda | auto"
    ),
) -> None:
    """Extract CLIP ViT-B/32 text features for a set of class names."""
    from pathlib import Path

    from counting.models.pseco.clip_features import (
        encode_class_names,
        save_text_features,
    )
    from counting.utils.device import resolve_device

    if dataset == "fsc147":
        classes_file = Path(dataset_root) / "ImageClasses_FSC147.txt"
        if not classes_file.exists():
            raise typer.BadParameter(f"Classes file not found: {classes_file}")
        names: list[str] = []
        seen: set[str] = set()
        with classes_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                cname = parts[1]
                if cname not in seen:
                    seen.add(cname)
                    names.append(cname)
    else:
        p = Path(dataset)
        if not p.exists():
            raise typer.BadParameter(f"Class list file not found: {p}")
        names = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    resolved_device = resolve_device(device)
    console.print(f"[loading] open_clip ViT-B/32 on {resolved_device}")
    features = encode_class_names(names, device=resolved_device)
    save_text_features(features, out)
    console.print(f"[green]done[/green] {len(features)} classes → {out}")


if __name__ == "__main__":
    app()
