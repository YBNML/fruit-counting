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


if __name__ == "__main__":
    app()
