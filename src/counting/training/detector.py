"""Faster R-CNN detector training: targets from orchard_synth truth, crop windows, flips and orders.

Training targets read ground truth (records and instance masks), so they live
here and nowhere on the prediction path. A frame's targets are the count1
records of every row, in the order the bundle stores them, labelled for
torchvision (1 fruit, 2 bag; 0 is its background class). A training view is a
square window of a frame, maybe flipped left to right; windows, flips and epoch
orders come from seeded generators, so a view depends only on (seed, epoch,
frame position). Epoch indices run from 0 in those generators; the training log
and checkpoints count epochs from 1.

`train_detector` trains in a run folder named by the training identity (config,
bundles, code, seed and request), and resumes from the folder's last epoch
checkpoint. Nothing at module level imports torch.
"""

from __future__ import annotations

import csv
import json
import pickle
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil, floor, inf
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import yaml
from PIL import Image

from counting.config.det_schema import DetTrainConfig
from counting.config.eval_schema import SplitsConfig
from counting.config.hashing import config_hash
from counting.data.formats.orchard_synth import SynthRun, read_png_u16
from counting.detection import frcnn
from counting.detection.cached import file_sha256
from counting.evaluation.record import env_info, experiment_id, git_state
from counting.evaluation.runner import CheckFailed
from counting.evaluation.truth import is_positive
from counting.training.callbacks import CosineWithWarmup

LOSS_KEYS = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")
LOG_HEADER = ("step", "epoch", "lr", "loss", *LOSS_KEYS, "sec", "max_mem_mb")
# The `start.json` keys a resume reads.
START_KEYS = ("train_id", "config_hash", "fingerprints", "args", "segments")


@dataclass(frozen=True, eq=False)
class TrainFrame:
    """One training frame and its full-frame targets.

    `ids` are the record ids (the values in the instances image), `boxes` the
    records' xyxy pixel boxes and `labels` their torchvision classes (1 fruit,
    2 bag), in record order. `px_min` is the bundle's count1 pixel minimum.
    """

    rgb_path: Path
    inst_path: Path
    run_id: str
    idx: int
    pitch_deg: float
    width: int
    height: int
    px_min: int
    ids: np.ndarray      # (K,) int64
    boxes: np.ndarray    # (K, 4) float32 xyxy
    labels: np.ndarray   # (K,) int64


def train_bundle_paths(cfg: DetTrainConfig, splits: SplitsConfig) -> list[tuple[str, Path]]:
    """(crop, bundle path) pairs: crops in `cfg.data.crops` order, then seeds as given (`cfg.data.seeds` or
    `splits.train`, not sorted)."""
    seeds = splits.train if cfg.data.seeds is None else cfg.data.seeds
    return [(c, Path(cfg.data.root.format(crop=c, seed=s)).expanduser()) for c in cfg.data.crops for s in seeds]


def frame_targets(run: SynthRun) -> list[TrainFrame]:
    """The `TrainFrame` of every frame of `run`, in idx order."""
    truth = run.truth()
    out = []
    for f in run.frames:
        pos = [r for r in truth.records[f.idx] if is_positive(r, "count1", truth.count1)]
        out.append(TrainFrame(
            rgb_path=f.rgb_path, inst_path=run.root / f"{f.idx:06d}_instances.png", run_id=f.run_id, idx=f.idx,
            pitch_deg=f.pass_key.pitch_deg, width=f.camera.width, height=f.camera.height, px_min=truth.count1[1],
            ids=np.array([r.id for r in pos], np.int64),
            boxes=np.array([r.bbox for r in pos], np.float32).reshape(-1, 4),
            labels=np.array([2 if truth.fruits[r.id].bagged else 1 for r in pos], np.int64)))
    return out


def load_train_frames(
    bundles: Sequence[tuple[str, Path]], splits: SplitsConfig, *, crop_size: int
) -> tuple[list[TrainFrame], list[dict]]:
    """The training frames of `bundles` (bundle order, then idx order) and one summary dict per bundle.

    Each bundle is opened and checked in turn: its crop must be the one it is
    listed under (`ValueError`), its seed must be in the train split
    (`CheckFailed`), and its frames must be at least `crop_size` on each side
    (`ValueError`). Every message starts with the bundle path.
    """
    frames: list[TrainFrame] = []
    summaries: list[dict] = []
    for crop, path in bundles:
        run = SynthRun.open(path)
        meta, root = run.meta, run.root
        if meta.crop != crop:
            raise ValueError(f"{root}: crop mismatch: bundle is {meta.crop}, expected {crop}")
        split = splits.split_of(meta.seed)
        if split != "train":
            raise CheckFailed(f"{root}: seed {meta.seed} is in split {split or 'none'}; "
                              "the detector trains on train seeds only")
        for f in run.frames:
            if min(f.camera.width, f.camera.height) < crop_size:
                raise ValueError(f"{root}: crop {crop_size} is larger than the "
                                 f"{f.camera.width}x{f.camera.height} frames")
        targets = frame_targets(run)
        frames.extend(targets)
        summaries.append({"run_id": meta.run_id, "crop": meta.crop, "seed": meta.seed,
                          "fingerprint": meta.fingerprint, "synth_commit": meta.synth_commit,
                          "n_frames": len(targets), "n_boxes": sum(len(t.boxes) for t in targets)})
    return frames, summaries


def _overlaps(boxes: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    """Which of `boxes` share some area with the window `[x0, x0+size) x [y0, y0+size)`."""
    return ((boxes[:, 0] < x0 + size) & (boxes[:, 2] > x0) & (boxes[:, 1] < y0 + size) & (boxes[:, 3] > y0))


def window_box_count(frame: TrainFrame, x0: int, y0: int, size: int) -> int:
    """How many of `frame`'s boxes overlap the window: an upper bound on what `crop_targets` keeps."""
    return int(_overlaps(frame.boxes, x0, y0, size).sum())


def crop_targets(
    frame: TrainFrame, inst: np.ndarray | None, x0: int, y0: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """The (M, 4) float32 boxes, in window coordinates, and (M,) int64 labels of the window, in record order.

    A box that misses the window is dropped and one wholly inside is kept as
    is. A box on the window edge is replaced by the tight box of its own
    instance pixels (`inst == id`) inside the window, or dropped when fewer
    than `frame.px_min` of them are there. `inst` is the frame's (H, W)
    instances image; it may be None when no box is on the edge.
    """
    x1, y1 = x0 + size, y0 + size
    out_boxes, out_labels = [], []
    for k in np.flatnonzero(_overlaps(frame.boxes, x0, y0, size)):
        bx0, by0, bx1, by1 = (float(v) for v in frame.boxes[k])
        if bx0 >= x0 and by0 >= y0 and bx1 <= x1 and by1 <= y1:
            out_boxes.append((bx0 - x0, by0 - y0, bx1 - x0, by1 - y0))
            out_labels.append(frame.labels[k])
            continue
        if inst is None:
            raise ValueError(f"{frame.inst_path}: a box crosses the window edge, so its instance mask is needed")
        cx0, cy0 = max(floor(bx0), x0), max(floor(by0), y0)
        ys, xs = np.nonzero(inst[cy0:min(ceil(by1), y1), cx0:min(ceil(bx1), x1)] == frame.ids[k])
        if len(xs) < frame.px_min:
            continue
        ox, oy = cx0 - x0, cy0 - y0   # the clipped region's corner in window coordinates
        out_boxes.append((ox + xs.min(), oy + ys.min(), ox + xs.max() + 1, oy + ys.max() + 1))
        out_labels.append(frame.labels[k])
    return np.array(out_boxes, np.float32).reshape(-1, 4), np.array(out_labels, np.int64)


def flip_boxes(boxes: np.ndarray, width: int) -> np.ndarray:
    """`boxes` (xyxy) mirrored left to right in an image `width` pixels wide."""
    out = boxes.copy()
    out[:, 0] = width - boxes[:, 2]
    out[:, 2] = width - boxes[:, 0]
    return out


def view_rng(seed: int, epoch: int, index: int) -> np.random.Generator:
    """The generator of the view of training frame `index` (its position in the frame list) in `epoch`."""
    return np.random.default_rng([seed, epoch, index])


def view_of(frame: TrainFrame, size: int, hflip_p: float, rng: np.random.Generator) -> tuple[int, int, bool]:
    """(x0, y0, flip) of a `size` window of `frame`, drawn from `rng` in that order."""
    x0 = int(rng.integers(0, frame.width - size + 1))
    y0 = int(rng.integers(0, frame.height - size + 1))
    flip = bool(rng.random() < hflip_p)
    return x0, y0, flip


def epoch_order(n: int, *, seed: int, epoch: int) -> np.ndarray:
    """The order in which `epoch` visits the `n` training frames."""
    return np.random.default_rng([seed, epoch]).permutation(n)


def densest_views(
    frames: Sequence[TrainFrame], *, size: int, hflip_p: float, seed: int, epochs: int
) -> list[tuple[int, int]]:
    """Every (epoch, frame position) pair, those whose view window holds the most boxes first.

    Pairs start in epoch order, then frame order; the sort by
    `window_box_count` (descending) is stable.
    """
    pairs = [(e, i) for e in range(epochs) for i in range(len(frames))]
    count = {}
    for e, i in pairs:
        x0, y0, _ = view_of(frames[i], size, hflip_p, view_rng(seed, e, i))
        count[(e, i)] = window_box_count(frames[i], x0, y0, size)
    return sorted(pairs, key=lambda p: -count[p])


# --- training ---------------------------------------------------------------------------------


def train_config_hash(cfg: DetTrainConfig) -> str:
    """The config hash of a training: `config_hash` without the runtime settings (device, output folder,
    data loader workers, log interval), so changing those neither moves the run folder nor blocks a resume."""
    return config_hash(cfg.model_copy(update={"num_workers": 0, "log_every": 1}))


def run_folder(
    cfg: DetTrainConfig, splits: SplitsConfig, *, max_iters: int | None, densest_first: bool
) -> tuple[Path, list[TrainFrame], list[dict], dict]:
    """(run folder, training frames, bundle summaries, code state) of a training request.

    Opens and checks the training bundles (`load_train_frames`) and names the
    folder `<output_dir>/<run_name>/<training id>`; the id covers the config,
    the bundle fingerprints, the code, the seed and the request. Writes nothing.
    """
    frames, bundles = load_train_frames(train_bundle_paths(cfg, splits), splits, crop_size=cfg.model.train_size)
    code = git_state()
    train_id = experiment_id(config_hash=train_config_hash(cfg), fingerprints=[b["fingerprint"] for b in bundles],
                             code=code, seed=cfg.seed, extra=f"train:{max_iters or '-'}:{densest_first}")
    return Path(cfg.output_dir).expanduser() / cfg.run_name / train_id, frames, bundles, code


def collate_views(batch) -> tuple[list, list]:
    """(images, targets) of a batch of `ViewDataset` items."""
    return tuple(zip(*batch))


def _crosses_edge(frame: TrainFrame, x0: int, y0: int, size: int) -> bool:
    """Whether a box of `frame` overlaps the window but is not wholly inside it (so its mask is needed)."""
    b = frame.boxes[_overlaps(frame.boxes, x0, y0, size)]
    return bool(((b[:, 0] < x0) | (b[:, 1] < y0) | (b[:, 2] > x0 + size) | (b[:, 3] > y0 + size)).any())


class ViewDataset:
    """The training views `(epoch, frame position)` as (CHW float image, target dict) pairs.

    Each view is the `view_of` window of its frame, flipped when drawn so, then
    colour-jittered. Targets hold `boxes` (M, 4) float32 in window coordinates
    and `labels` (M,) int64; M may be 0. torch is imported only when an item is read.
    """

    def __init__(self, frames: Sequence[TrainFrame], views: Sequence[tuple[int, int]], cfg: DetTrainConfig) -> None:
        self.frames = frames
        self.views = list(views)
        self.cfg = cfg
        self._jitter = None

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, k: int):
        import torch
        from torchvision.transforms import v2

        epoch, i = self.views[k]
        frame, size, aug = self.frames[i], self.cfg.model.train_size, self.cfg.aug
        x0, y0, flip = view_of(frame, size, aug.hflip_p, view_rng(self.cfg.seed, epoch, i))
        with Image.open(frame.rgb_path) as im:
            if im.size != (frame.width, frame.height):
                raise ValueError(f"{frame.rgb_path}: image is {im.size[0]}x{im.size[1]}, "
                                 f"the frame camera is {frame.width}x{frame.height}")
            rgb = np.asarray(im.convert("RGB").crop((x0, y0, x0 + size, y0 + size)))
        inst = None
        if _crosses_edge(frame, x0, y0, size):
            inst = read_png_u16(frame.inst_path)
            if inst.shape != (frame.height, frame.width):
                raise ValueError(f"{frame.inst_path}: instances image is {inst.shape[1]}x{inst.shape[0]}, "
                                 f"the frame camera is {frame.width}x{frame.height}")
        boxes, labels = crop_targets(frame, inst, x0, y0, size)
        if flip:
            rgb = rgb[:, ::-1]
            boxes = flip_boxes(boxes, size)
        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / np.float32(255))
        if self._jitter is None:
            self._jitter = v2.ColorJitter(brightness=aug.brightness, contrast=aug.contrast,
                                          saturation=aug.saturation, hue=aug.hue)
        return self._jitter(image), {"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels)}


def save_epoch_checkpoint(path: Path, model, optimizer, *, epoch: int, step: int) -> None:
    """Write the model and optimizer state after `epoch` epochs and `step` steps to `path` (tmp file, then replace)."""
    import torch

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step}, tmp)
    tmp.replace(path)


@dataclass(frozen=True)
class TrainResult:
    """A finished training: its run folder, weights sha256, total steps and this run's timing and peak CUDA memory."""

    folder: Path
    weights_sha256: str
    steps: int
    sec_per_iter: float | None
    max_memory_mb: float | None
    max_reserved_mb: float | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json(path: Path, obj) -> None:
    _write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def _code_label(code: dict) -> str:
    return (code["commit"] or "none")[:12] + ("+dirty" if code["dirty"] else "")


def _same_code(a: dict, b: dict) -> bool:
    return all(a[k] == b[k] for k in ("commit", "dirty", "diff_sha256"))


def _check_fresh(folder: Path) -> None:
    if (folder / "weights.pt").exists():
        raise CheckFailed(f"{folder}: this training already finished (weights.pt exists)")
    if (folder / "last.pt").exists():
        raise CheckFailed(f"{folder}: an unfinished training is here; pass --resume {folder} to continue it")


def _args_label(args) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items()) if isinstance(args, dict) else str(args)


def _read_start(path: Path) -> dict:
    """The training start record at `path`; `CheckFailed` naming the file when it is not JSON or lacks
    `START_KEYS` or the first segment's code state."""
    try:
        start = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise CheckFailed(f"{path}: not a readable training start record ({e})") from None
    try:
        first = start["segments"][0]["code"]
        ok = all(k in start for k in START_KEYS) and all(k in first for k in ("commit", "dirty", "diff_sha256"))
    except (KeyError, IndexError, TypeError):
        ok = False
    if not ok:
        raise CheckFailed(f"{path}: not a readable training start record (it needs {', '.join(START_KEYS)}, "
                          "and the code state of its first segment)")
    return start


def _check_resume(folder: Path, *, config: str, fingerprints: list[str], args: dict, code: dict,
                  echo: Callable) -> dict:
    """The `start.json` of the training in `folder`, after checking that the request may resume it: the same
    config, bundles and request arguments (`args`) as the training started with."""
    if not (folder / "start.json").is_file():
        raise CheckFailed(f"{folder}: not a detector training folder (no start.json)")
    if (folder / "weights.pt").exists():
        raise CheckFailed(f"{folder}: this training already finished (weights.pt exists)")
    if not (folder / "last.pt").is_file():
        raise CheckFailed(f"{folder}: nothing to resume (no last.pt)")
    start = _read_start(folder / "start.json")
    if start["config_hash"] != config:
        raise CheckFailed(f"{folder}: this training was started with config {start['config_hash'][:16]}, "
                          f"the request has {config[:16]}")
    if start["fingerprints"] != fingerprints:
        raise CheckFailed(f"{folder}: the training bundles changed since this training started")
    if start["args"] != args:
        raise CheckFailed(f"{folder}: this training was started with {_args_label(start['args'])}; "
                          f"the request has {_args_label(args)}")
    first = start["segments"][0]["code"]
    if not _same_code(code, first):
        echo(f"warning: {folder}: resuming with code {_code_label(code)} (started with {_code_label(first)})")
    return start


def _truncate_log(path: Path, step: int) -> None:
    """Drop the `log.csv` rows past `step` (written after the checkpoint being resumed)."""
    if not path.is_file():
        return
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:            # not even a header: start the log again
        path.unlink()
        return
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows([rows[0], *(r for r in rows[1:] if r and int(r[0]) <= step)])
    tmp.replace(path)


def _log_row(path: Path, row: Sequence) -> None:
    """Append one row to `log.csv`, writing the header first when the file is new."""
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        if new:
            w.writerow(LOG_HEADER)
        w.writerow(row)


def train_detector(
    cfg: DetTrainConfig,
    splits: SplitsConfig,
    *,
    max_iters: int | None = None,
    densest_first: bool = False,
    resume: Path | None = None,
    splits_sha256: str | None = None,
    echo: Callable[[str], object] = print,
) -> TrainResult:
    """Train the detector of `cfg` on its train bundles and write `weights.pt` and `record.json` in its run folder.

    A fresh start refuses a folder that already holds a finished or an
    unfinished training; `resume` (a run folder) continues from its `last.pt`
    when the config, bundles and `max_iters`/`densest_first` are those it
    started with. Every folder check runs before the model is built. Training
    stops after all epochs or once `max_iters` steps are done, and before an
    optimizer step whose loss or gradient is not finite; `densest_first`
    (with `max_iters`, not with `resume`) trains one epoch of the views holding
    the most boxes, as a speed benchmark.
    """
    if densest_first and max_iters is None:
        raise ValueError("--densest-first needs --max-iters (it is a benchmark mode)")
    if densest_first and resume is not None:
        raise ValueError("--densest-first cannot be combined with --resume")
    started, t_start = _now(), time.perf_counter()
    folder, frames, bundles, code = run_folder(cfg, splits, max_iters=max_iters, densest_first=densest_first)
    fingerprints = [b["fingerprint"] for b in bundles]
    chash = train_config_hash(cfg)
    args = {"max_iters": max_iters, "densest_first": densest_first}
    if resume is None:
        _check_fresh(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "log.csv").unlink(missing_ok=True)
        _write_text(folder / "config.yaml", yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
        start = {"train_id": folder.name, "config_hash": chash, "fingerprints": fingerprints, "data": bundles,
                 "args": args, "splits_sha256": splits_sha256,
                 "segments": [{"started": started, "code": code, "from_step": 0}]}
        _write_json(folder / "start.json", start)
    else:
        folder = Path(resume).expanduser()
        start = _check_resume(folder, config=chash, fingerprints=fingerprints, args=args, code=code, echo=echo)
    echo(f"folder {folder}")

    model = frcnn.build_model(cfg.model, pretrained=cfg.pretrained, mode="train", seed=cfg.seed)
    import torch
    from torch.utils.data import DataLoader

    device, opt = cfg.device, cfg.optim
    cuda = torch.device(device).type == "cuda"
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=opt.base_lr(), momentum=opt.momentum, weight_decay=opt.weight_decay)
    first_epoch = step = 0
    if resume is not None:
        ckpt_path = folder / "last.pt"
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            first_epoch, step = int(ckpt["epoch"]), int(ckpt["step"])
        except (pickle.UnpicklingError, EOFError, RuntimeError, ValueError, KeyError, IndexError, TypeError) as e:
            raise CheckFailed(f"{ckpt_path}: not a readable training checkpoint ({type(e).__name__})") from None
        _truncate_log(folder / "log.csv", step)
        start["segments"].append({"started": started, "code": code, "from_step": step})
        _write_json(folder / "start.json", start)

    n, batch, size = len(frames), opt.batch_size, cfg.model.train_size
    if densest_first:
        schedule = [densest_views(frames, size=size, hflip_p=cfg.aug.hflip_p, seed=cfg.seed,
                                  epochs=opt.epochs)[:max_iters * batch]]
    else:
        schedule = [[(e, int(i)) for i in epoch_order(n, seed=cfg.seed, epoch=e)] for e in range(opt.epochs)]
    last_step = sum(ceil(len(views) / batch) for views in schedule)
    if max_iters is not None:
        last_step = min(last_step, max_iters)
    lr_at = CosineWithWarmup(opt.base_lr(), opt.warmup_iters, opt.epochs * ceil(n / batch), opt.min_lr).lr_at
    log, times = folder / "log.csv", []

    done = max_iters is not None and step >= max_iters        # a resumed training may have no step left
    for e in range(first_epoch, len(schedule)):
        if done:
            break
        loader = DataLoader(ViewDataset(frames, schedule[e], cfg), batch_size=batch, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate_views,
                            generator=torch.Generator().manual_seed(cfg.seed + e))
        for images, targets in loader:
            lr = lr_at(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            images = [x.to(device) for x in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                losses = model(images, targets)
                loss = sum(losses.values())
                if not bool(torch.isfinite(loss)):
                    shown = ", ".join(f"{k}={float(v):.4g}" for k, v in losses.items())
                    raise CheckFailed(f"non-finite loss at step {step + 1}: {shown}")
                optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, inf if opt.clip_grad_norm is None else opt.clip_grad_norm)
                if not bool(torch.isfinite(norm)):
                    raise CheckFailed(f"non-finite gradient at step {step + 1}")
                optimizer.step()
            except torch.cuda.OutOfMemoryError:
                raise CheckFailed(f"out of GPU memory at step {step + 1} with batch size {batch}") from None
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            step += 1
            if step % cfg.log_every == 0 or step == last_step:
                mem = f"{torch.cuda.max_memory_allocated(device) / 2**20:.1f}" if cuda else ""
                _log_row(log, [step, e + 1, f"{lr:.6g}", f"{float(loss):.6f}",
                               *(f"{float(losses[k]):.6f}" for k in LOSS_KEYS), f"{times[-1]:.4f}", mem])
            done = max_iters is not None and step >= max_iters
            if done:
                break
        if done:
            break
        save_epoch_checkpoint(folder / "last.pt", model, optimizer, epoch=e + 1, step=step)

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    tmp = folder / "weights.pt.tmp"
    torch.save(state, tmp)
    weights_sha256 = file_sha256(tmp)
    _write_text(folder / "weights.sha256", f"{weights_sha256}  weights.pt\n")
    skip = min(10, len(times) - 1)
    sec_per_iter = statistics.median(times[skip:]) if times else None
    max_mb = torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None
    reserved_mb = torch.cuda.max_memory_reserved(device) / 2**20 if cuda else None
    record = {
        "train_id": start["train_id"], "config_hash": chash, "code": code, "segments": start["segments"],
        "env": {**env_info(), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "device_name": torch.cuda.get_device_name(device) if cuda else device},
        "data": bundles, "seeds": {"seed": cfg.seed},
        "pretrained": frcnn.pretrained_info(cfg.model) if cfg.pretrained else None,
        "weights_sha256": weights_sha256, "splits_sha256": splits_sha256,
        "timings": {"steps": step, "sec_per_iter_median": sec_per_iter, "max_memory_mb": max_mb,
                    "max_reserved_mb": reserved_mb, "seconds": round(time.perf_counter() - t_start, 3)},
        "args": {"max_iters": max_iters, "densest_first": densest_first, "resume": resume is not None},
        "started": started, "finished": _now(),
    }
    _write_json(folder / "record.json", record)
    tmp.replace(folder / "weights.pt")   # last: weights.pt marks the training finished
    return TrainResult(folder=folder, weights_sha256=weights_sha256, steps=step, sec_per_iter=sec_per_iter,
                       max_memory_mb=max_mb, max_reserved_mb=reserved_mb)
