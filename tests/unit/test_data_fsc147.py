import json
from pathlib import Path

import numpy as np
from PIL import Image

from counting.data.formats.fsc147 import FSC147Dataset


def _make_annotations(root: Path):
    imgs = root / "images_384_VarV2"
    imgs.mkdir(parents=True)
    for name in ("1.jpg", "2.jpg", "3.jpg"):
        arr = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(imgs / name)

    (root / "annotation_FSC147_384.json").write_text(json.dumps({
        "1.jpg": {
            "points": [[10, 10], [20, 20]],
            "box_examples_coordinates": [[[0, 0], [0, 30], [30, 30], [30, 0]]],
        },
        "2.jpg": {
            "points": [[5, 5]],
            "box_examples_coordinates": [[[0, 0], [0, 10], [10, 10], [10, 0]]],
        },
        "3.jpg": {"points": [], "box_examples_coordinates": []},
    }))
    (root / "Train_Test_Val_FSC_147.json").write_text(json.dumps({
        "train": ["1.jpg", "2.jpg"],
        "val": ["3.jpg"],
        "test": [],
    }))
    # Tab-separated filename<TAB>class_name, one per line
    (root / "ImageClasses_FSC147.txt").write_text(
        "1.jpg\tapples\n2.jpg\tapples\n3.jpg\tsea shells\n"
    )


def test_fsc147_loads_split(tmp_path):
    _make_annotations(tmp_path)
    ds = FSC147Dataset(tmp_path, split="train")

    assert len(ds) == 2
    rec = ds[0]
    assert rec.relpath in {"1.jpg", "2.jpg"}
    img = rec.read_rgb()
    assert img.shape[:2] == (48, 64)
    assert len(rec.points) >= 0
    assert isinstance(rec.count, int)


def test_fsc147_points_match_annotations(tmp_path):
    _make_annotations(tmp_path)
    ds = FSC147Dataset(tmp_path, split="train")
    by_name = {r.relpath: r for r in ds}
    assert by_name["1.jpg"].count == 2
    assert by_name["2.jpg"].count == 1
    assert by_name["1.jpg"].points == [(10.0, 10.0), (20.0, 20.0)]


def test_fsc147_bad_split_raises(tmp_path):
    import pytest

    _make_annotations(tmp_path)
    with pytest.raises(ValueError, match="split"):
        FSC147Dataset(tmp_path, split="bogus")


def test_fsc147_missing_files_raise(tmp_path):
    import pytest

    (tmp_path / "images_384_VarV2").mkdir()
    with pytest.raises(FileNotFoundError):
        FSC147Dataset(tmp_path, split="train")


def test_fsc147_malformed_json_raises_with_path(tmp_path):
    import pytest

    _make_annotations(tmp_path)
    (tmp_path / "annotation_FSC147_384.json").write_text("{ not valid json")

    with pytest.raises(ValueError, match="annotation_FSC147_384.json"):
        FSC147Dataset(tmp_path, split="train")


def test_fsc147_class_name_populated(tmp_path):
    _make_annotations(tmp_path)
    ds = FSC147Dataset(tmp_path, split="train")
    by_name = {r.relpath: r for r in ds}
    assert by_name["1.jpg"].class_name == "apples"
    assert by_name["2.jpg"].class_name == "apples"


def test_fsc147_missing_classes_file_raises(tmp_path):
    import pytest

    _make_annotations(tmp_path)
    (tmp_path / "ImageClasses_FSC147.txt").unlink()
    with pytest.raises(FileNotFoundError, match="ImageClasses_FSC147.txt"):
        FSC147Dataset(tmp_path, split="train")
