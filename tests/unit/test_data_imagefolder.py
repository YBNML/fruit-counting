import numpy as np
from PIL import Image

from counting.data.formats.imagefolder import ImageFolderDataset


def _make_image(path, size=(8, 8), color=(0, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_enumerates_supported_extensions(tmp_path):
    _make_image(tmp_path / "a.jpg")
    _make_image(tmp_path / "sub" / "b.png")
    (tmp_path / "readme.txt").write_text("x")

    ds = ImageFolderDataset(tmp_path)
    assert len(ds) == 2
    rels = sorted(r.relpath for r in ds)
    assert rels == ["a.jpg", "sub/b.png"]


def test_getitem_returns_record_and_loads_image(tmp_path):
    _make_image(tmp_path / "a.png", color=(255, 0, 0))

    ds = ImageFolderDataset(tmp_path)
    rec = ds[0]
    assert rec.relpath == "a.png"
    assert rec.path.exists()
    img = rec.read_rgb()
    assert isinstance(img, np.ndarray)
    assert img.shape == (8, 8, 3)
    assert img[0, 0, 0] == 255


def test_empty_directory_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        ImageFolderDataset(tmp_path)
