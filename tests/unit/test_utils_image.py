import numpy as np
from PIL import Image

from counting.utils.image import ensure_np_rgb, ensure_pil, read_image_rgb


def test_read_image_rgb(tmp_path):
    arr = (np.random.rand(10, 12, 3) * 255).astype(np.uint8)
    p = tmp_path / "x.png"
    Image.fromarray(arr).save(p)

    out = read_image_rgb(p)
    assert out.shape == (10, 12, 3)
    assert out.dtype == np.uint8


def test_ensure_pil_from_numpy_rgb():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    pil = ensure_pil(arr, assume="rgb")
    assert isinstance(pil, Image.Image)
    assert pil.size == (4, 4)


def test_ensure_pil_from_pil():
    im = Image.new("RGB", (2, 3))
    assert ensure_pil(im) is im


def test_ensure_np_rgb_from_pil():
    im = Image.new("RGB", (3, 2), (255, 0, 0))
    arr = ensure_np_rgb(im)
    assert arr.shape == (2, 3, 3)
    assert arr[0, 0, 0] == 255


def test_ensure_np_rgb_from_ndarray_passthrough():
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    assert ensure_np_rgb(arr) is arr
