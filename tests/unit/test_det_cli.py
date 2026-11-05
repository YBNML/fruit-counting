import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

import counting.detection.frcnn as frcnn
import counting.evaluation.record as record
from counting.cli import app
from counting.config.det_schema import DetInferConfig, DetTrainConfig
from counting.config.loader import load_yaml_model
from counting.data.formats.orchard_synth import SynthRun
from counting.detection.base import Detections
from counting.detection.cached import CachedDetector, cache_path
from counting.evaluation.protect import (
    LOG_HEADER, append_log_row, check_test_detect, detect_log_row, earlier_log_runs,
)
from counting.evaluation.record import git_state
from counting.evaluation.runner import ProtectionError
from counting.training.detector import run_folder, train_config_hash
from tests.unit.synthfake import SIDECAR_COMMIT, load_splits, write_run

ROOT = Path(__file__).resolve().parents[2]
SPLITS = str(ROOT / "configs" / "eval" / "splits.yaml")
TRAIN_CFG = str(ROOT / "configs" / "train" / "frcnn_synth.yaml")
runner = CliRunner()


def _gate(*args, **kwargs):
    raise RuntimeError("model reached")


def _sets(tmp_path, seeds="[100]"):
    return [f"data.root={tmp_path}/{{crop}}_s{{seed}}", "data.crops=[apple]", f"data.seeds={seeds}",
            "model.train_size=128", f"output_dir={tmp_path}/train", "device=cpu", "pretrained=false", "num_workers=0"]


def _args(tmp_path, *extra, seeds="[100]"):
    return ["train", "detector", "--config", TRAIN_CFG, "--splits", SPLITS,
            *[x for s in [*_sets(tmp_path, seeds), *extra] for x in ("--set", s)]]


def test_train_detector_refuses_bundles_outside_the_train_split_before_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(frcnn, "build_model", _gate)
    write_run(tmp_path / "apple_s200", crop="apple", pitches=(30.0,), seed=200, max_frames=2)
    r = runner.invoke(app, _args(tmp_path, seeds="[200]"))
    assert r.exit_code == 1 and "seed 200 is in split val; the detector trains on train seeds only" in r.output
    write_run(tmp_path / "apple_s100", crop="apple", pitches=(30.0,), seed=100, max_frames=2)
    r = runner.invoke(app, _args(tmp_path))
    assert isinstance(r.exception, RuntimeError) and "model reached" in str(r.exception)


def test_train_detector_input_errors_exit_2(tmp_path, monkeypatch):
    monkeypatch.setattr(frcnn, "build_model", _gate)
    r = runner.invoke(app, _args(tmp_path))
    assert r.exit_code == 2 and "no manifest.json" in r.output
    write_run(tmp_path / "apple_s100", crop="apple", pitches=(30.0,), seed=100, max_frames=2)
    r = runner.invoke(app, _args(tmp_path, "model.train_size=1024"))
    assert r.exit_code == 2 and "crop 1024 is larger than the 640x360 frames" in r.output
    r = runner.invoke(app, _args(tmp_path, "optim.epochs=0"))
    assert r.exit_code == 2 and "optim.epochs" in r.output
    r = runner.invoke(app, _args(tmp_path, f"data.root={tmp_path}/{{crop}}_s{{seed}}/{{run}}"))
    assert r.exit_code == 2 and "data.root may use only the {crop} and {seed} fields" in r.output, r.output
    r = runner.invoke(app, [*_args(tmp_path), "--densest-first"])
    assert r.exit_code == 2 and "--densest-first needs --max-iters" in r.output


def test_train_detector_guards_its_run_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(frcnn, "build_model", _gate)
    write_run(tmp_path / "apple_s100", crop="apple", pitches=(30.0,), seed=100, max_frames=2)
    r = runner.invoke(app, _args(tmp_path))                     # a fresh start writes its settings, then builds
    assert isinstance(r.exception, RuntimeError) and "model reached" in str(r.exception)
    cfg = load_yaml_model(TRAIN_CFG, DetTrainConfig, overrides=_sets(tmp_path))
    folder = run_folder(cfg, load_splits(), max_iters=None, densest_first=False)[0]
    assert load_yaml_model(folder / "config.yaml", DetTrainConfig) == cfg
    start = json.loads((folder / "start.json").read_text(encoding="utf-8"))
    assert start["config_hash"] == train_config_hash(cfg)
    assert start["fingerprints"] == [SynthRun.open(tmp_path / "apple_s100").meta.fingerprint]
    assert [s["from_step"] for s in start["segments"]] == [0]
    resume = ["--resume", str(folder)]
    r = runner.invoke(app, [*_args(tmp_path), "--resume", str(tmp_path / "nowhere")])
    assert r.exit_code == 1 and "not a detector training folder (no start.json)" in r.output
    r = runner.invoke(app, [*_args(tmp_path), *resume])
    assert r.exit_code == 1 and "nothing to resume (no last.pt)" in r.output
    (folder / "last.pt").write_bytes(b"x")
    r = runner.invoke(app, _args(tmp_path))
    assert r.exit_code == 1 and "an unfinished training is here; pass --resume" in r.output
    r = runner.invoke(app, [*_args(tmp_path, "optim.epochs=3"), *resume])
    assert r.exit_code == 1 and "this training was started with config" in r.output
    r = runner.invoke(app, [*_args(tmp_path, "num_workers=2"), *resume])   # runtime settings may change
    assert isinstance(r.exception, RuntimeError) and "model reached" in str(r.exception)
    r = runner.invoke(app, [*_args(tmp_path), *resume, "--densest-first", "--max-iters", "5"])
    assert r.exit_code == 2 and "--densest-first cannot be combined with --resume" in r.output
    (folder / "weights.pt").write_bytes(b"x")
    for args in ([*_args(tmp_path), *resume], _args(tmp_path)):
        r = runner.invoke(app, args)
        assert r.exit_code == 1 and "this training already finished (weights.pt exists)" in r.output


def test_train_detector_resumes_only_the_training_it_started(tmp_path, monkeypatch):
    monkeypatch.setattr(frcnn, "build_model", _gate)
    write_run(tmp_path / "apple_s100", crop="apple", pitches=(30.0,), seed=100, max_frames=2)
    r = runner.invoke(app, [*_args(tmp_path), "--max-iters", "5"])
    assert isinstance(r.exception, RuntimeError) and "model reached" in str(r.exception)
    cfg = load_yaml_model(TRAIN_CFG, DetTrainConfig, overrides=_sets(tmp_path))
    folder = run_folder(cfg, load_splits(), max_iters=5, densest_first=False)[0]
    (folder / "last.pt").write_bytes(b"x")
    resume = [*_args(tmp_path), "--resume", str(folder)]
    for extra, asked in (([], "None"), (["--max-iters", "7"], "7")):
        r = runner.invoke(app, [*resume, *extra])
        assert r.exit_code == 1 and (f"{folder}: this training was started with max_iters=5, densest_first=False; "
                                     f"the request has max_iters={asked}, densest_first=False") in r.output, r.output
    r = runner.invoke(app, [*resume, "--max-iters", "5"])
    assert isinstance(r.exception, RuntimeError) and "model reached" in str(r.exception)
    assert "warning:" not in r.output
    start = json.loads((folder / "start.json").read_text(encoding="utf-8"))
    start["segments"][0]["code"] = {"commit": "0" * 40, "dirty": False, "diff_sha256": None}
    (folder / "start.json").write_text(json.dumps(start), encoding="utf-8")
    r = runner.invoke(app, [*resume, "--max-iters", "5"])
    assert isinstance(r.exception, RuntimeError) and "model reached" in str(r.exception)
    assert re.search(rf"^warning: {re.escape(str(folder))}: resuming with code \S+ \(started with 000000000000\)$",
                     r.output, re.M), r.output
    write_run(tmp_path / "apple_s100", crop="apple", pitches=(30.0,), seed=100, max_frames=3)
    r = runner.invoke(app, [*resume, "--max-iters", "5"])
    assert r.exit_code == 1 and f"{folder}: the training bundles changed since this training started" in r.output


def test_train_detector_refuses_an_unreadable_start_record(tmp_path, monkeypatch):
    monkeypatch.setattr(frcnn, "build_model", _gate)
    write_run(tmp_path / "apple_s100", crop="apple", pitches=(30.0,), seed=100, max_frames=2)
    r = runner.invoke(app, _args(tmp_path))
    assert isinstance(r.exception, RuntimeError) and "model reached" in str(r.exception)
    cfg = load_yaml_model(TRAIN_CFG, DetTrainConfig, overrides=_sets(tmp_path))
    folder = run_folder(cfg, load_splits(), max_iters=None, densest_first=False)[0]
    (folder / "last.pt").write_bytes(b"x")
    for text in ("not json", "{}", "[1]", '{"config_hash": "h", "fingerprints": [], "segments": []}'):
        (folder / "start.json").write_text(text, encoding="utf-8")
        r = runner.invoke(app, [*_args(tmp_path), "--resume", str(folder)])
        assert r.exit_code == 1 and f"error: {folder / 'start.json'}: not a readable training start record" \
            in r.output, r.output


DETECT_CFG = str(ROOT / "configs" / "detect" / "frcnn_synth.yaml")
W = "a" * 64


def _commit_all(repo):
    """Make `repo` a git repository with every file in it committed."""
    g = ["git", "-C", str(repo)]
    subprocess.run([*g, "init", "-q"], check=True)
    subprocess.run([*g, "add", "-A"], check=True)
    subprocess.run([*g, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "commit", "-qm", "c"],
                   check=True)


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs" / "detect" / "frozen").mkdir(parents=True)
    for rel in ("configs/detect/frozen/frozen.yaml", "configs/detect/loose.yaml", "splits.yaml"):
        (repo / rel).write_text("x: 1\n", encoding="utf-8")
    _commit_all(repo)
    return repo


def _write_train_config(folder, **model):
    """A training config next to the weights: the committed inference model, changed by `model`."""
    base = load_yaml_model(DETECT_CFG, DetInferConfig).model
    cfg = DetTrainConfig(model=base.model_copy(update=model))
    (folder / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")


def test_test_detect_request_refusals_name_each_rule(tmp_path):
    repo = _git_repo(tmp_path)
    ok = dict(config_path=repo / "configs" / "detect" / "frozen" / "frozen.yaml", sets=[],
              files={"--splits": repo / "splits.yaml"}, weights_sha256=W, pinned_sha256=W, repo=repo)
    req = check_test_detect(**ok)
    assert req.config_rel == "configs/detect/frozen/frozen.yaml" and req.code["dirty"] is False
    cases = [
        ({"config_path": repo / "configs" / "detect" / "loose.yaml"},
         "counting detect on a test bundle needs a config file under configs/detect/frozen/ in the counting repository"),
        ({"sets": ["device=cpu"]}, "counting detect on a test bundle takes no --set"),
        ({"code": {"commit": "c" * 40, "dirty": True, "diff_sha256": "d"}},
         "counting detect on a test bundle needs a clean, committed counting work tree"),
        ({"files": {"--splits": tmp_path / "outside.yaml"}},
         "counting detect on a test bundle needs --splits to be a file tracked in the counting repository"),
        ({"pinned_sha256": None}, "needs weights_sha256 in the frozen config to equal the sha256 of --weights; got none"),
        ({"pinned_sha256": "b" * 64}, "got bbbbbbbbbbbbbbbb and aaaaaaaaaaaaaaaa"),
    ]
    for change, message in cases:
        with pytest.raises(ProtectionError, match=re.escape(message)):
            check_test_detect(**{**ok, **change})


def test_detect_log_row_fits_the_test_log(tmp_path):
    row = detect_log_row(date="2026-11-06T00:00:00+00:00", config_rel="configs/detect/frozen/frozen.yaml",
                         config_hash="h" * 16, weights_sha256=W, fingerprints=["f2", "f1"], digests=["d" * 64, "e" * 64],
                         commit="c" * 40)
    assert set(row) == set(LOG_HEADER) and row["kind"] == "detect" and row["extra"] == f"weights:{W[:16]}"
    assert row["fingerprints"] == "f1;f2" and row["experiment_id"] == f"{'d' * 16};{'e' * 16}"
    log = tmp_path / "test_log.csv"
    append_log_row(log, row)
    assert earlier_log_runs(log, config_hash="h" * 16, extra=f"weights:{W[:16]}") == 1


def test_detect_refuses_before_loading_the_model(tmp_path, monkeypatch, heldout_run):
    monkeypatch.setattr(frcnn, "load_detector", _gate)
    val = write_run(tmp_path / "val_s200", crop="apple", pitches=(30.0,), seed=200, max_frames=2, images=True)
    plain = write_run(tmp_path / "plain_s200", crop="apple", pitches=(30.0,), seed=200, max_frames=2)
    w = tmp_path / "w" / "weights.pt"
    w.parent.mkdir()
    w.write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest()
    out = tmp_path / "detect"

    def call(*extra, weights=w):
        return runner.invoke(app, ["detect", "--config", DETECT_CFG, "--weights", str(weights), "--splits", SPLITS,
                                   "--set", f"output_dir={out}", *extra])

    r = call("--run", str(val))
    assert r.exit_code == 2 and "no training config next to the weights" in r.output
    _write_train_config(w.parent, anchor_sizes=[8, 16, 32, 64, 128])
    r = call("--run", str(val))
    assert r.exit_code == 1 and "the weights were trained with other model settings: anchor_sizes" in r.output
    _write_train_config(w.parent)
    r = call("--run", str(heldout_run))
    assert r.exit_code == 1 and "counting detect on a test bundle needs a config file under configs/detect/frozen/" \
        in r.output
    r = call("--run", str(val), "--set", f"weights_sha256={'b' * 64}")
    assert r.exit_code == 1 and f"has sha256 {sha[:16]}, but the config pins weights_sha256 bbbbbbbbbbbbbbbb" in r.output
    r = call("--run", str(val), weights=tmp_path / "none.pt")
    assert r.exit_code == 2 and "none.pt" in r.output
    r = call("--run", str(plain))
    assert r.exit_code == 2 and "no RGB image" in r.output
    r = call("--run", str(val), "--set", "model=1", "--set", "model.max_dets=2")
    assert r.exit_code == 2 and "--set:" in r.output
    npz = out / sha[:16] / f"{SynthRun.open(val).meta.run_id}.npz"
    npz.parent.mkdir(parents=True)
    npz.write_bytes(b"x")
    rgb = SynthRun.open(val).frames[0].rgb_path
    rgb.rename(rgb.with_name("aside.png"))          # an existing cache stops the run before any RGB image is read
    r = call("--run", str(val))
    assert r.exit_code == 1 and "cache exists; a cache is written once per weights" in r.output, r.output
    rgb.with_name("aside.png").rename(rgb)
    npz.unlink()
    r = call("--run", str(val))
    assert isinstance(r.exception, RuntimeError) and "model reached" in str(r.exception)


class _StubDetector:
    """Stands in for the loaded Faster R-CNN: one box per frame, and an error on the frames of run `fail_run`."""

    name = "stub"

    def __init__(self, fail_run=None):
        self.fail_run = fail_run

    def detect(self, frame):
        if frame.run_id == self.fail_run:
            raise ValueError(f"{frame.rgb_path}: detector failed")
        return Detections(np.array([[1, 1, 9, 9]], np.float32), np.array([0.9], np.float32), np.array([0], np.int8))


def _frozen_setup(tmp_path, monkeypatch):
    """(repo, weights, sha256, bundles) of a test-bundle detect request that passes every protection check.

    `repo` stands for the counting repository: it commits a frozen detect config that pins the weights'
    sha256 (device cpu, output under `tmp_path/detect`) and a splits copy. The weights sit next to their
    training config. The two bundles have test seed 1000, RGB images and the sidecar, and are listed in
    the reverse of the (fingerprint, run_id) order in which detect writes them. The model is a stub.
    """
    w = tmp_path / "train" / "weights.pt"
    w.parent.mkdir()
    w.write_bytes(b"weights")
    _write_train_config(w.parent)
    sha = hashlib.sha256(b"weights").hexdigest()
    repo = tmp_path / "repo"
    (repo / "configs" / "detect" / "frozen").mkdir(parents=True)
    (repo / "configs" / "eval").mkdir(parents=True)
    raw = yaml.safe_load(Path(DETECT_CFG).read_text(encoding="utf-8"))
    raw.update(weights_sha256=sha, device="cpu", output_dir=str(tmp_path / "detect"))
    (repo / "configs" / "detect" / "frozen" / "frozen.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    shutil.copy(SPLITS, repo / "configs" / "eval" / "splits.yaml")
    _commit_all(repo)
    monkeypatch.setattr(record, "counting_repo", lambda: repo)
    monkeypatch.setattr(frcnn, "load_detector", lambda *args, **kwargs: _StubDetector())
    bundles = []
    for name, step in (("test_a", 0.15), ("test_b", 0.2)):
        root = write_run(tmp_path / name, crop="apple", pitches=(30.0,), seed=1000, max_frames=2, images=True,
                         step=step)
        (root / "orchard_synth.commit").write_text(SIDECAR_COMMIT + "\n", encoding="utf-8")
        bundles.append(root)
    bundles.sort(key=lambda b: (SynthRun.open(b).meta.fingerprint, SynthRun.open(b).meta.run_id), reverse=True)
    return repo, w, sha, bundles


def _detect_args(repo, weights, bundles, *extra, splits=None):
    return ["detect", "--config", str(repo / "configs" / "detect" / "frozen" / "frozen.yaml"),
            "--weights", str(weights), "--splits", str(splits or repo / "configs" / "eval" / "splits.yaml"),
            *[x for b in bundles for x in ("--run", str(b))], *extra]


def _log_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _metas(bundles):
    """The bundles' metas in the order detect writes their caches."""
    return sorted((SynthRun.open(b).meta for b in bundles), key=lambda m: (m.fingerprint, m.run_id))


@pytest.mark.slow
def test_detect_on_test_bundles_logs_a_row_per_cache_and_warns_on_a_rerun(tmp_path, monkeypatch):
    repo, w, sha, bundles = _frozen_setup(tmp_path, monkeypatch)
    out, log, metas = tmp_path / "detect" / sha[:16], tmp_path / "detect" / "test_log.csv", _metas(bundles)
    r = runner.invoke(app, _detect_args(repo, w, bundles))
    assert r.exit_code == 0, r.output
    rows = _log_rows(log)
    assert [(x["kind"], x["extra"], x["config"], x["commit"]) for x in rows] == \
        [("detect", f"weights:{sha[:16]}", "configs/detect/frozen/frozen.yaml", git_state(repo)["commit"])] * 2
    assert [x["fingerprints"] for x in rows] == [m.fingerprint for m in metas]
    assert [x["experiment_id"] for x in rows] == [CachedDetector(cache_path(out, m.run_id)).digest[:16] for m in metas]
    assert "warning: test_log.csv" not in r.output
    for p in out.iterdir():
        p.unlink()
    r = runner.invoke(app, _detect_args(repo, w, bundles))
    assert r.exit_code == 0, r.output
    assert r.output.index("warning: test_log.csv already lists 2 test detect row(s) of config ") < \
        r.output.index(str(cache_path(out, metas[0].run_id)))
    assert len(_log_rows(log)) == 4


@pytest.mark.slow
def test_detect_on_test_bundles_keeps_the_rows_of_caches_written_before_a_failure(tmp_path, monkeypatch):
    repo, w, sha, bundles = _frozen_setup(tmp_path, monkeypatch)
    first, second = _metas(bundles)
    monkeypatch.setattr(frcnn, "load_detector", lambda *args, **kwargs: _StubDetector(fail_run=second.run_id))
    r = runner.invoke(app, _detect_args(repo, w, bundles))
    assert r.exit_code == 2 and "detector failed" in r.output, r.output
    assert [x["fingerprints"] for x in _log_rows(tmp_path / "detect" / "test_log.csv")] == [first.fingerprint]
    out = tmp_path / "detect" / sha[:16]
    assert cache_path(out, first.run_id).is_file() and not cache_path(out, second.run_id).exists()


def test_detect_on_test_bundles_refuses_a_loose_request_before_the_model(tmp_path, monkeypatch):
    repo, w, sha, bundles = _frozen_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(frcnn, "load_detector", _gate)
    other = tmp_path / "other" / "weights.pt"
    other.parent.mkdir()
    other.write_bytes(b"other weights")
    _write_train_config(other.parent)
    val = write_run(tmp_path / "v", crop="apple", pitches=(30.0,), seed=200, max_frames=2, images=True)
    loose = tmp_path / "splits.yaml"                # the same splits, but not a file of the repository
    shutil.copy(SPLITS, loose)
    cases = [
        (_detect_args(repo, w, bundles, "--set", "device=cpu"), "counting detect on a test bundle takes no --set"),
        (_detect_args(repo, other, bundles),
         f"needs weights_sha256 in the frozen config to equal the sha256 of --weights; got {sha[:16]} and "),
        (_detect_args(repo, w, [*bundles, val]), "split=test needs a test seed; seed 200 belongs to split val"),
        (_detect_args(repo, w, bundles, splits=loose),
         f"needs --splits to be a file tracked in the counting repository, got {loose}"),
    ]
    for args, message in cases:
        r = runner.invoke(app, args)
        assert r.exit_code == 1 and message in r.output, r.output
    (repo / "notes.txt").write_text("x", encoding="utf-8")       # an untracked file dirties the work tree
    r = runner.invoke(app, _detect_args(repo, w, bundles))
    assert r.exit_code == 1 and "needs a clean, committed counting work tree" in r.output, r.output
    (repo / "notes.txt").unlink()
    (tmp_path / "detect").mkdir()
    (tmp_path / "detect" / "test_log.csv").write_text("a,b\n", encoding="utf-8")
    r = runner.invoke(app, _detect_args(repo, w, bundles))
    assert r.exit_code == 2 and "a test log needs the columns" in r.output, r.output
    assert not (tmp_path / "detect" / sha[:16]).exists()


@pytest.mark.slow
def test_detect_refuses_weights_that_are_not_a_torch_file(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(frcnn, "build_model", lambda *args, **kwargs: torch.nn.Linear(2, 2))
    val = write_run(tmp_path / "val_s200", crop="apple", pitches=(30.0,), seed=200, max_frames=2, images=True)
    w = tmp_path / "w" / "weights.pt"
    w.parent.mkdir()
    w.write_bytes(b"weights")
    _write_train_config(w.parent)
    r = runner.invoke(app, ["detect", "--config", DETECT_CFG, "--weights", str(w), "--splits", SPLITS,
                            "--run", str(val), "--set", f"output_dir={tmp_path / 'detect'}", "--set", "device=cpu"])
    assert r.exit_code == 2 and f"error: {w}: not a readable torch weights file" in r.output, r.output
