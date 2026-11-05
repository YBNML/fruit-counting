from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from counting.config.det_schema import DetTrainConfig
from counting.data.formats.orchard_synth import SynthRun, run_fingerprint
from counting.evaluation.runner import CheckFailed
from counting.evaluation.truth import is_positive
from counting.training.detector import (
    TrainFrame, crop_targets, densest_views, epoch_order, flip_boxes, load_train_frames, train_bundle_paths, view_of,
    view_rng, window_box_count,
)
from tests.unit.synthfake import RGB_PALETTE, load_splits, write_run

DIMMED = 1001   # tree 4, fruit 1 of the fake orchard, shown at 10% visibility: an ignored record


def _frame(boxes, labels, ids=None, width=640, height=360):
    boxes = np.array(boxes, np.float32).reshape(-1, 4)
    ids = np.arange(1, len(boxes) + 1) if ids is None else ids
    return TrainFrame(rgb_path=Path("x_rgb.png"), inst_path=Path("x_instances.png"), run_id="apple_s100_x", idx=0,
                      pitch_deg=30.0, width=width, height=height, px_min=20, ids=np.array(ids, np.int64),
                      boxes=boxes, labels=np.array(labels, np.int64))


def test_train_frames_hold_count1_records_of_every_row_with_torchvision_labels(tmp_path):
    root = write_run(tmp_path / "apple_s100", crop="apple", pitches=(30.0,), seed=100, max_frames=6,
                     dim={DIMMED: 0.1}, images=True)
    frames, bundles = load_train_frames([("apple", root)], load_splits(), crop_size=256)
    run = SynthRun.open(root)
    truth = run.truth()
    assert [f.idx for f in frames] == [f.idx for f in run.frames] and len(frames) == 6
    ignored = 0
    for f in frames:
        recs = truth.records[f.idx]
        pos = [r for r in recs if is_positive(r, "count1", truth.count1)]
        ignored += sum(r.vis_px > 0 and not is_positive(r, "count1", truth.count1) for r in recs)
        assert (f.boxes.dtype, f.labels.dtype, f.ids.dtype) == (np.float32, np.int64, np.int64)
        assert f.boxes.tolist() == [[float(c) for c in r.bbox] for r in pos] and f.ids.tolist() == [r.id for r in pos]
        assert f.labels.tolist() == [2 if truth.fruits[r.id].bagged else 1 for r in pos]
        assert f.rgb_path.is_file() and f.inst_path == root / f"{f.idx:06d}_instances.png" and f.inst_path.is_file()
        assert (f.width, f.height, f.px_min, f.pitch_deg) == (640, 360, 20, 30.0)
    assert ignored > 0
    assert bundles == [{"run_id": run.meta.run_id, "crop": "apple", "seed": 100, "fingerprint": run.meta.fingerprint,
                        "synth_commit": None, "n_frames": 6, "n_boxes": sum(len(f.boxes) for f in frames)}]


def test_training_refuses_bundles_outside_the_train_split(tmp_path, apple_run):
    val = write_run(tmp_path / "apple_s200", crop="apple", pitches=(30.0,), seed=200, max_frames=2)
    with pytest.raises(CheckFailed, match="seed 200 is in split val; the detector trains on train seeds only"):
        load_train_frames([("apple", val)], load_splits(), crop_size=256)
    with pytest.raises(CheckFailed, match="seed 5 is in split dev; the detector trains on train seeds only"):
        load_train_frames([("apple", apple_run)], load_splits(), crop_size=256)
    pear = write_run(tmp_path / "pear_s101", crop="pear", pitches=(45.0,), seed=101, max_frames=2)
    with pytest.raises(ValueError, match="crop mismatch: bundle is pear, expected apple"):
        load_train_frames([("apple", pear)], load_splits(), crop_size=256)
    with pytest.raises(ValueError, match="crop 1024 is larger than the 640x360 frames"):
        load_train_frames([("pear", pear)], load_splits(), crop_size=1024)


def test_train_bundle_paths_follow_the_train_split():
    paths = train_bundle_paths(DetTrainConfig(), load_splits())
    home = Path("~/datasets/orchard_synth").expanduser()
    assert len(paths) == 40 and paths[0] == ("apple", home / "apple" / "train_s100")
    assert paths[20] == ("pear", home / "pear" / "train_s100")
    assert [p.name for _, p in paths[:20]] == [f"train_s{s}" for s in range(100, 120)]
    cfg = DetTrainConfig.model_validate({"data": {"crops": ["pear", "apple"], "seeds": [101, 100]}})
    assert train_bundle_paths(cfg, load_splits()) == [
        (c, home / c / f"train_s{s}") for c in ("pear", "apple") for s in (101, 100)]


def test_crop_targets_use_the_visible_pixels_inside_the_window():
    inst = np.zeros((360, 640), np.uint16)
    inst[60:100, 90:100] = 1          # 1: L-shaped mask; inside the window only its bottom strip shows
    inst[90:100, 100:130] = 1
    inst[100:120, 350:370] = 2        # 2: full square, 6 of its 20 columns inside
    inst[300:320, 200:203] = 3        # 3: 18 pixels inside: dropped
    inst[296:316, 250:252] = 4        # 4: exactly 20 pixels inside: kept
    inst[60:80, 360:380] = 5          # 5: box overlaps the window, mask does not
    f = _frame([[90, 60, 130, 100], [350, 100, 370, 120], [200, 300, 220, 320], [250, 296, 260, 316],
                [340, 60, 380, 80], [200, 100, 220, 120], [500, 200, 520, 220]], [1, 1, 1, 2, 1, 2, 1])
    boxes, labels = crop_targets(f, inst, 100, 50, 256)   # 6: inside the window, kept as is; 7: outside
    assert (boxes.dtype, labels.dtype) == (np.float32, np.int64)
    assert boxes.tolist() == [[0, 40, 30, 50], [250, 50, 256, 70], [150, 246, 152, 256], [100, 50, 120, 70]]
    assert labels.tolist() == [1, 1, 2, 2]
    empty_boxes, empty_labels = crop_targets(f, inst, 500, 0, 16)
    assert empty_boxes.shape == (0, 4) and empty_labels.shape == (0,)
    assert window_box_count(f, 100, 50, 256) == 6


def test_flips_views_and_epoch_orders_are_seeded():
    assert flip_boxes(np.array([[0, 10, 30, 50]], np.float32), 256).tolist() == [[226.0, 10.0, 256.0, 50.0]]
    f = _frame([[90, 60, 130, 100]], [1])
    r = np.random.default_rng([7, 2, 3])
    assert view_of(f, 256, 0.5, view_rng(7, 2, 3)) == (int(r.integers(0, 385)), int(r.integers(0, 105)),
                                                        bool(r.random() < 0.5))
    views = [view_of(f, 256, 0.5, view_rng(0, 0, i)) for i in range(200)]
    assert all(0 <= x0 <= 640 - 256 and 0 <= y0 <= 360 - 256 for x0, y0, _ in views)
    assert 60 <= sum(flip for _, _, flip in views) <= 140
    assert not any(view_of(f, 256, 0.0, view_rng(0, 0, i))[2] for i in range(50))
    assert epoch_order(5, seed=3, epoch=1).tolist() == np.random.default_rng([3, 1]).permutation(5).tolist()


def test_densest_views_put_the_fullest_windows_first():
    grid = [[x, y, x + 10, y + 10] for x in range(0, 640, 40) for y in range(0, 360, 40)]
    frames = [_frame([[600, 300, 610, 310]], [1]), _frame(grid, [1] * len(grid))]
    views = densest_views(frames, size=256, hflip_p=0.5, seed=0, epochs=3)
    assert sorted(views) == [(e, i) for e in range(3) for i in range(2)]
    counts = [window_box_count(frames[i], *view_of(frames[i], 256, 0.5, view_rng(0, e, i))[:2], 256) for e, i in views]
    assert counts == sorted(counts, reverse=True) and [i for _, i in views[:3]] == [1, 1, 1]


def test_fake_images_follow_the_semantic_image(tmp_path):
    plain = write_run(tmp_path / "plain", crop="apple", pitches=(30.0,), max_frames=2)
    img = write_run(tmp_path / "img", crop="apple", pitches=(30.0,), max_frames=2, images=True)
    assert not list(plain.glob("*_rgb.png")) and not list(plain.glob("*_instances.png"))
    assert len(list(img.glob("*_rgb.png"))) == len(list(img.glob("*_instances.png"))) == 2
    rgb = np.asarray(Image.open(img / "000000_rgb.png"))
    sem = np.asarray(Image.open(img / "000000_semantic.png"))
    inst = np.asarray(Image.open(img / "000000_instances.png"))
    assert rgb.shape == (360, 640, 3) and rgb.dtype == np.uint8 and (rgb == RGB_PALETTE[sem]).all()
    assert inst.dtype == np.uint16 and ((inst > 0) == np.isin(sem, (3, 4))).all()
    run = SynthRun.open(img)
    seen = {r.id for r in run.truth().records[0] if r.bbox is not None}
    assert set(np.unique(inst[inst > 0]).tolist()) <= seen
    assert run_fingerprint(plain) == run_fingerprint(img)


@pytest.mark.slow
def test_training_views_hold_the_window_image_and_its_targets(tmp_path):
    """Without colour jitter a view is its RGB window / 255 (mirrored when drawn so) and its targets are that
    window's `crop_targets` (mirrored with it); every target box covers painted fruit or bag pixels only."""
    import torch

    from counting.data.formats.orchard_synth import read_png_u16
    from counting.training.detector import ViewDataset

    root = write_run(tmp_path / "apple_s100", crop="apple", pitches=(30.0,), seed=100, max_frames=6, images=True)
    frames, _ = load_train_frames([("apple", root)], load_splits(), crop_size=160)
    counted = RGB_PALETTE[[3, 4]]                   # the fruit and bag colours
    views = [(e, i) for e in range(2) for i in range(len(frames))]
    boxes_seen = {}
    for hflip_p in (0.0, 1.0):
        cfg = DetTrainConfig.model_validate({"model": {"train_size": 160}, "seed": 3, "aug": {
            "hflip_p": hflip_p, "brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0}})
        ds = ViewDataset(frames, views, cfg)
        boxes_seen[hflip_p] = 0
        for k, (e, i) in enumerate(views):
            image, target = ds[k]
            f = frames[i]
            x0, y0, flip = view_of(f, 160, hflip_p, view_rng(3, e, i))
            assert flip == (hflip_p == 1.0)
            rgb = np.asarray(Image.open(f.rgb_path).convert("RGB"))[y0:y0 + 160, x0:x0 + 160]
            boxes, labels = crop_targets(f, read_png_u16(f.inst_path), x0, y0, 160)
            if flip:
                rgb, boxes = rgb[:, ::-1], flip_boxes(boxes, 160)
            assert image.dtype == torch.float32 and tuple(image.shape) == (3, 160, 160)
            assert np.array_equal(image.numpy(), rgb.transpose(2, 0, 1).astype(np.float32) / np.float32(255))
            assert (target["boxes"].dtype, target["labels"].dtype) == (torch.float32, torch.int64)
            assert np.array_equal(target["boxes"].numpy(), boxes) and np.array_equal(target["labels"].numpy(), labels)
            shown = np.rint(image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            for bx0, by0, bx1, by1 in boxes.astype(int):
                patch = shown[by0:by1, bx0:bx1].reshape(-1, 1, 3)
                assert patch.size and (patch == counted[None]).all(axis=2).any(axis=1).all()
            boxes_seen[hflip_p] += len(boxes)
    assert boxes_seen[0.0] > 0 and boxes_seen[1.0] > 0
