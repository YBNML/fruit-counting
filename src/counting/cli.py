"""Typer-based CLI entrypoint."""

from __future__ import annotations

import platform
import sys

import typer
from rich.console import Console
from rich.table import Table

from counting import __version__
from counting.utils.device import resolve_device

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
            f"Only fsc147 is supported in Plan 2; got {cfg.data.format!r}"
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
