"""FSC-147 dataset adapter.

Expected layout (matches upstream PseCo):
    root/
      images_384_VarV2/<image_id>.jpg
      annotation_FSC147_384.json     # per-image points + box examples
      Train_Test_Val_FSC_147.json    # split mapping
      ImageClasses_FSC147.txt        # tab-separated <image>\t<class_name>
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from counting.utils.image import read_image_rgb

_VALID_SPLITS = {"train", "val", "test"}


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc


def _load_image_classes(path: Path) -> dict[str, str]:
    """Parse tab-separated `<image_filename>\t<class_name>` file."""
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{line_no}: expected tab-separated <image>\\t<class>, got {line!r}"
                )
            mapping[parts[0]] = parts[1]
    return mapping


@dataclass(frozen=True)
class FSC147Record:
    path: Path
    relpath: str
    points: list[tuple[float, float]]
    box_examples: list[list[tuple[float, float]]]
    count: int
    class_name: str

    def read_rgb(self) -> np.ndarray:
        return read_image_rgb(self.path)


class FSC147Dataset:
    def __init__(self, root: str | Path, *, split: str = "train") -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"Unknown split {split!r}. Valid: {sorted(_VALID_SPLITS)}")
        self.root = Path(root)
        self.split = split

        img_dir = self.root / "images_384_VarV2"
        ann_path = self.root / "annotation_FSC147_384.json"
        split_path = self.root / "Train_Test_Val_FSC_147.json"
        classes_path = self.root / "ImageClasses_FSC147.txt"
        for p in (img_dir, ann_path, split_path, classes_path):
            if not p.exists():
                raise FileNotFoundError(f"FSC-147 artifact missing: {p}")

        splits = _load_json(split_path)
        annotations = _load_json(ann_path)
        image_classes = _load_image_classes(classes_path)

        self._records: list[FSC147Record] = []
        for name in splits.get(split, []):
            ann = annotations.get(name, {})
            pts = [(float(x), float(y)) for x, y in ann.get("points", [])]
            boxes = [
                [(float(x), float(y)) for x, y in box]
                for box in ann.get("box_examples_coordinates", [])
            ]
            self._records.append(FSC147Record(
                path=img_dir / name,
                relpath=name,
                points=pts,
                box_examples=boxes,
                count=len(pts),
                class_name=image_classes.get(name, ""),
            ))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> FSC147Record:
        return self._records[idx]

    def __iter__(self) -> Iterator[FSC147Record]:
        return iter(self._records)
