"""A directory of images, no labels. Inference/diagnostics only."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from counting.data.base import ImageRecord

_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImageFolderDataset:
    def __init__(self, root: str | Path, *, extensions: set[str] | None = None) -> None:
        self.root = Path(root)
        self.extensions = extensions or _EXTS
        self._paths = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in self.extensions
        )
        if not self._paths:
            raise FileNotFoundError(f"No images found under: {self.root}")

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> ImageRecord:
        p = self._paths[idx]
        return ImageRecord(path=p, relpath=str(p.relative_to(self.root)))

    def __iter__(self) -> Iterator[ImageRecord]:
        for i in range(len(self)):
            yield self[i]
