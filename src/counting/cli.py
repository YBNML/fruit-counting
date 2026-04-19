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


if __name__ == "__main__":
    app()
