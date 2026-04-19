"""Dataset protocol. Plan 2 extends this with point/box labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from counting.utils.image import read_image_rgb


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    relpath: str

    def read_rgb(self) -> np.ndarray:
        return read_image_rgb(self.path)


@runtime_checkable
class CountingDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> ImageRecord: ...
    def __iter__(self): ...
