"""LiDAR files beside a bundle: the prediction-side reader and its checks, what it leaves unread, the scoring-side
reads of the sidecar and the label files, and the bundle identity the files leave alone."""

import hashlib
import inspect
import json
import re
import shutil
from math import cos, radians, sin
from pathlib import Path

import numpy as np
import pytest

from counting.data.formats.orchard_synth import SynthRun
from counting.evaluation.lidarsource import (
    LIDAR_LABELS_FORMAT, bundle_inputs, contract_noise, read_labels, sidecar_contract, sidecar_mount, sidecar_mount_rpy,
)
from counting.geometry import lidarfile
from counting.geometry.lidarfile import (
    LIDAR_ARRAYS, LIDAR_FILE_FORMAT, LIDAR_SIDECAR, LIDAR_SIDECAR_FORMAT, LIDAR_SIDES, LIDAR_TEXT_KEYS, LIDAR_UNITS,
    LidarFile, lidar_folder,
)
from tests.unit.lidarfake import dump_json, mount_matrix, shared_lidar_bundle, sidecar_sha256, write_lidar_files

pytestmark = pytest.mark.pose

SIM_SHA = "ab" * 32
COMMIT = "1" * 40


def _copy(folder: Path, tmp: Path, sha16: str = SIM_SHA[:16]) -> Path:
    """A copy of the LiDAR file folder `folder` at `tmp/lidar/<sha16>/<folder name>`."""
    dest = tmp / "lidar" / sha16 / folder.name
    shutil.copytree(folder, dest)
    return dest


def _edit_sidecar(folder: Path, edit) -> None:
    """Apply `edit` to the sidecar JSON object of `folder` and write it back as deterministic JSON."""
    sidecar = json.loads((folder / LIDAR_SIDECAR).read_bytes())
    edit(sidecar)
    (folder / LIDAR_SIDECAR).write_bytes(dump_json(sidecar))


def _edit_npz(folder: Path, name: str, edit) -> None:
    """Apply `edit` to the arrays of `folder/name`, write them back and record the new sha256 in the sidecar."""
    with np.load(folder / name) as z:
        arrays = {k: z[k] for k in z.files}
    edit(arrays)
    np.savez_compressed(folder / name, **arrays)
    sha = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    _edit_sidecar(folder, lambda s: s["files"].__setitem__(name, sha))


def _sidecar(folder: Path) -> dict:
    return json.loads((folder / LIDAR_SIDECAR).read_bytes())


def _open(folder: Path, run: SynthRun, **changes) -> LidarFile:
    """`LidarFile` of the depth unit on `folder` for `run`, pinned to the sidecar as it is now, with `changes`."""
    kw = dict(unit="depth", bundle=run.root.name, crop=run.meta.crop, seed=run.meta.seed, inputs=bundle_inputs(run),
              digest=sidecar_sha256(folder) if (folder / LIDAR_SIDECAR).is_file() else None,
              config_sha256=SIM_SHA, sim_commits=[COMMIT])
    return LidarFile(folder, **{**kw, **changes})


def test_lidar_file_reads_the_prediction_arrays():
    assert (LIDAR_FILE_FORMAT, LIDAR_SIDECAR_FORMAT, LIDAR_SIDECAR) == \
        ("orchard-lidar/1", "orchard-lidar-sidecar/1", "lidar.json")
    assert (LIDAR_TEXT_KEYS, LIDAR_SIDES, LIDAR_UNITS) == (("format", "device", "side"), ("A", "B"), ("depth", "nav"))
    assert LIDAR_ARRAYS == {"dir": (np.float32, (-1, 3)), "range_m": (np.float32, (-1,)), "t_ns": (np.int64, (-1,)),
                            "row": (np.uint32, (-1,))}
    shared = shared_lidar_bundle()
    run = SynthRun.open(shared.bundle)
    assert lidar_folder(shared.bundle.parent / "lidar" / SIM_SHA[:16], run.root.name) == shared.depth
    assert lidar_folder("~/lidar", "b") == Path.home() / "lidar" / "b"

    file = _open(shared.depth, run)
    assert {name for name in vars(file) if not name.startswith("_")} == \
        {"folder", "digest", "device", "config_sha256", "sim_commit", "dirty"}
    assert (file.folder, file.digest, file.device, file.config_sha256, file.sim_commit, file.dirty) == \
        (shared.depth, sidecar_sha256(shared.depth), "depth", SIM_SHA, COMMIT, False)
    spans = _sidecar(shared.depth)["time"]["sides"]
    for side in LIDAR_SIDES:
        points = file.points(side)
        n = spans[side]["n_points"]
        assert points.side == side and n > 0
        for name, (dtype, shape) in LIDAR_ARRAYS.items():
            arr = getattr(points, name)
            assert arr.dtype == dtype and arr.shape == (n, *shape[1:]) and not arr.flags.writeable, (side, name)
        assert (points.first_ns, points.last_ns) == (0, spans[side]["last_ns"])
        assert points.first_ns <= points.t_ns.min() and points.t_ns.max() <= points.last_ns
        assert np.all(np.diff(points.t_ns) >= 0)
        assert file.points(side) is points
    assert file.points("A") is not file.points("B")
    with pytest.raises(ValueError, match=re.escape(f"{shared.depth}: no LiDAR side 'C'")):
        file.points("C")
    nav = LidarFile(shared.nav, unit="nav", bundle=run.root.name, crop="apple", seed=5, inputs=bundle_inputs(run),
                    digest=sidecar_sha256(shared.nav), config_sha256="cd" * 32, sim_commits=[COMMIT])
    assert nav.device == "nav" and len(nav.points("B").t_ns) == _sidecar(shared.nav)["time"]["sides"]["B"]["n_points"]


def _set(*keys_and_value):
    """An edit that sets `obj[k0][k1]... = value`."""
    *keys, value = keys_and_value

    def edit(obj):
        for key in keys[:-1]:
            obj = obj[key]
        obj[keys[-1]] = value
    return edit


def _decrease(arrays):
    arrays["t_ns"][0] = arrays["t_ns"][-1]


def _past_the_end(arrays):
    arrays["t_ns"][-1] += 5_000


# (case, edit of the copied folder, LidarFile changes, file named in the message, exception, message)
REFUSALS = [
    ("no sidecar", lambda f: (f / LIDAR_SIDECAR).unlink(), {}, LIDAR_SIDECAR, FileNotFoundError, "no LiDAR sidecar"),
    ("pin", None, {"digest": "0" * 64}, LIDAR_SIDECAR, ValueError, "lidar.json sha256 does not match"),
    ("format", lambda f: _edit_sidecar(f, _set("format", "orchard-lidar-sidecar/0")), {}, LIDAR_SIDECAR, ValueError,
     "format orchard-lidar-sidecar/0, expected orchard-lidar-sidecar/1"),
    ("unit", lambda f: _edit_sidecar(f, _set("device", "nav")), {}, LIDAR_SIDECAR, ValueError,
     "device nav, expected depth"),
    ("crop", None, {"crop": "pear"}, LIDAR_SIDECAR, ValueError, "crop apple, expected pear"),
    ("bundle", None, {"bundle": "dev_s6_shake"}, LIDAR_SIDECAR, ValueError,
     "bundle dev_s5_shake, expected dev_s6_shake"),
    ("seed", None, {"seed": 6}, LIDAR_SIDECAR, ValueError, "seed 5, expected 6"),
    ("another bundle's manifest", lambda f: _edit_sidecar(f, _set("inputs", "manifest.json", "9" * 64)), {},
     LIDAR_SIDECAR, ValueError, "manifest.json sha256 differs from the bundle"),
    ("another bundle's body record", lambda f: _edit_sidecar(f, _set("inputs", "trajectory.json", "9" * 64)), {},
     LIDAR_SIDECAR, ValueError, "trajectory.json sha256 differs from the bundle"),
    ("config sha256", None, {"config_sha256": "cd" * 32}, LIDAR_SIDECAR, ValueError,
     "config sha256 abababababababab differs from the expected cdcdcdcdcdcdcdcd"),
    ("folder name", "0" * 16, {}, LIDAR_SIDECAR, ValueError,
     "folder 0000000000000000 is not named after config sha256 abababababababab"),
    ("commit", None, {"sim_commits": ["2" * 40]}, LIDAR_SIDECAR, ValueError,
     "simulator commit 111111111111 is not one of the expected commits"),
    ("npz sha256", lambda f: (f / "lidar_A.npz").write_bytes((f / "lidar_B.npz").read_bytes()), {}, "lidar_A.npz",
     ValueError, "sha256 does not match lidar.json files"),
    ("key added", lambda f: _edit_npz(f, "lidar_A.npz", lambda a: a.__setitem__("reflectivity", a["range_m"])), {},
     "lidar_A.npz", ValueError, "arrays ['device', 'dir', 'format', 'range_m', 'reflectivity', 'row', 'side', 't_ns'], "
     "expected ['device', 'dir', 'format', 'range_m', 'row', 'side', 't_ns']"),
    ("key missing", lambda f: _edit_npz(f, "lidar_A.npz", lambda a: a.pop("row")), {}, "lidar_A.npz", ValueError,
     "arrays ['device', 'dir', 'format', 'range_m', 'side', 't_ns'], "
     "expected ['device', 'dir', 'format', 'range_m', 'row', 'side', 't_ns']"),
    ("dir float64", lambda f: _edit_npz(f, "lidar_A.npz", lambda a: a.__setitem__("dir", a["dir"].astype(np.float64))),
     {}, "lidar_A.npz", ValueError, "dir has dtype float64, expected float32"),
    ("side text", lambda f: _edit_npz(f, "lidar_A.npz", lambda a: a.__setitem__("side", np.array("B"))), {},
     "lidar_A.npz", ValueError, "side B, expected A"),
    ("point count", lambda f: _edit_sidecar(f, lambda s: s["time"]["sides"]["A"].__setitem__(
        "n_points", s["time"]["sides"]["A"]["n_points"] + 1)), {}, "lidar_A.npz", ValueError,
     "points, lidar.json time.sides.A.n_points is"),
    ("t_ns decreases", lambda f: _edit_npz(f, "lidar_A.npz", _decrease), {}, "lidar_A.npz", ValueError,
     "t_ns decreases"),
    ("t_ns past last_ns", lambda f: _edit_npz(f, "lidar_A.npz", _past_the_end), {}, "lidar_A.npz", ValueError,
     "t_ns runs outside [first_ns, last_ns]"),
]


def test_lidar_file_refusals(tmp_path):
    shared = shared_lidar_bundle()
    run = SynthRun.open(shared.bundle)
    assert len(REFUSALS) == 20
    for i, (case, edit, changes, name, error, message) in enumerate(REFUSALS):
        folder = _copy(shared.depth, tmp_path / str(i), edit if isinstance(edit, str) else SIM_SHA[:16])
        if callable(edit):
            edit(folder)
        refused = pytest.raises(error, match=re.escape(f"{folder / name}: ") + ".*" + re.escape(message))
        if name == LIDAR_SIDECAR:   # refused on opening, before any point file is read
            with refused:
                _open(folder, run, **changes)
        else:
            file = _open(folder, run, **changes)
            with refused:
                file.points("A")


def test_prediction_side_never_reads_labels_or_mount(tmp_path):
    shared = shared_lidar_bundle()
    run = SynthRun.open(shared.bundle)
    plain = _open(shared.depth, run).points("A")
    folder = _copy(shared.depth, tmp_path)
    for side in LIDAR_SIDES:   # rubbish labels, the sidecar and its sha256 unchanged
        (folder / f"lidar_{side}_labels.npz").write_bytes(b"not an npz")
    _edit_sidecar(folder, _set("mount", "T_body_sensor", [[0.0, -1.0, 0.0, 0.3], [1.0, 0.0, 0.0, 0.2],
                                                           [0.0, 0.0, 1.0, 0.9], [0.0, 0.0, 0.0, 1.0]]))
    _edit_sidecar(folder, _set("config", "values", "mount", "apple", {"xyz_m": [0.3, 0.2, 0.9],
                                                                      "rpy_deg": [0.0, 0.0, 90.0]}))
    moved = _open(folder, run).points("A")
    for name in LIDAR_ARRAYS:
        assert getattr(moved, name).tobytes() == getattr(plain, name).tobytes(), name
    assert (moved.first_ns, moved.last_ns) == (plain.first_ns, plain.last_ns)
    source = inspect.getsource(lidarfile)
    for pattern in (r"_labels", r"\bmount\b", r"counting_contract", r'"values"', r"\btrajectory\b"):
        assert not re.search(pattern, source), pattern


def test_evaluation_side_reads_mount_contract_and_labels(tmp_path):
    shared = shared_lidar_bundle()
    want = np.eye(4)
    want[:3, 3] = (0.0, 0.0, 0.1)
    assert np.abs(sidecar_mount(shared.depth) - want).max() <= 1e-12
    assert sidecar_mount_rpy(shared.depth, "apple") == (0.0, 0.0, 0.0)

    bent = _copy(shared.depth, tmp_path / "bent")
    for not_rigid in ([[1.0, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.1], [0.0, 0.0, 0.0, 1.0]],
                      [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.1], [0.0, 0.0, 0.0, 1.0]],
                      [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.1], [0.0, 0.0, 0.1, 1.0]]):
        _edit_sidecar(bent, _set("mount", "T_body_sensor", not_rigid))   # sheared, mirrored, last row moved
        with pytest.raises(ValueError, match=re.escape(f"{bent / LIDAR_SIDECAR}: mount T_body_sensor is not rigid")):
            sidecar_mount(bent)

    bundle = shutil.copytree(shared.bundle, tmp_path / "apple" / shared.bundle.name)
    tilted = write_lidar_files(bundle, mount_xyz=(0.0, 0.0, 0.3), mount_rpy_deg=(-26.5, 0.0, 0.0))
    assert sidecar_mount_rpy(tilted, "apple") == (-26.5, 0.0, 0.0)
    c, s = cos(radians(-26.5)), sin(radians(-26.5))
    want = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.3], [0.0, 0.0, 0.0, 1.0]])
    assert np.abs(sidecar_mount(tilted) - want).max() <= 1e-12
    _edit_sidecar(tilted, _set("config", "values", "mount", "apple", "rpy_deg", [-26.5, 0.0, 5.0]))
    with pytest.raises(ValueError, match=re.escape(f"{tilted / LIDAR_SIDECAR}: mount rpy_deg does not match "
                                                   f"T_body_sensor")):
        sidecar_mount_rpy(tilted, "apple")
    turned = _copy(shared.depth, tmp_path / "turned")   # three angles fix the order Rz(yaw) Ry(pitch) Rx(roll)
    _edit_sidecar(turned, _set("mount", "T_body_sensor", mount_matrix((0.1, 0.2, 0.3), (10.0, 20.0, 30.0)).tolist()))
    _edit_sidecar(turned, _set("config", "values", "mount", "apple", {"xyz_m": [0.1, 0.2, 0.3],
                                                                      "rpy_deg": [10.0, 20.0, 30.0]}))
    assert sidecar_mount_rpy(turned, "apple") == (10.0, 20.0, 30.0)

    assert set(sidecar_contract(shared.depth)) == {"range_sigma_m", "range_sigma_near_m", "near_range_m",
                                                   "ang_sigma_deg", "dropout", "range_bias_m", "extrinsic",
                                                   "time_offset_ms"}

    assert LIDAR_LABELS_FORMAT == "orchard-lidar-labels/1"
    n = _sidecar(shared.depth)["time"]["sides"]["A"]["n_points"]
    labels = read_labels(shared.depth, "A", n_points=n)
    assert labels.side == "A"
    for name, dtype in (("fruit_id", np.uint16), ("part", np.uint8), ("mixed", np.uint8)):
        arr = getattr(labels, name)
        assert arr.dtype == dtype and arr.shape == (n,) and not arr.flags.writeable, name
    fruits = set(SynthRun.open(shared.bundle).truth().fruits)
    assert set(labels.fruit_id[labels.fruit_id > 0].tolist()) <= fruits
    assert np.array_equal(labels.fruit_id > 0, np.isin(labels.part, (3, 4))) and np.any(labels.fruit_id > 0)
    with pytest.raises(ValueError, match=re.escape(f"{shared.depth / 'lidar_A_labels.npz'}: {n} labels, expected 1")):
        read_labels(shared.depth, "A", n_points=1)
    swapped = _copy(shared.depth, tmp_path / "swapped")
    (swapped / "lidar_A_labels.npz").write_bytes((swapped / "lidar_B_labels.npz").read_bytes())
    with pytest.raises(ValueError, match=re.escape(f"{swapped / 'lidar_A_labels.npz'}: sha256 does not match "
                                                   f"lidar.json files")):
        read_labels(swapped, "A", n_points=n)


def test_contract_maps_to_noise_keys():
    shared = shared_lidar_bundle()
    depth = {"lidar_range_sigma_m": 0.02, "lidar_range_sigma_near_m": 0.03, "lidar_near_range_m": 1.0,
             "lidar_ang_sigma_deg": 0.15, "lidar_dropout": 0.05, "lidar_range_bias_m": 0.0}
    assert contract_noise(sidecar_contract(shared.depth), "depth") == \
        {**depth, "lidar_extr_trans_mm": 10.0, "lidar_extr_rot_deg": 0.3}
    assert contract_noise(sidecar_contract(shared.nav), "nav") == \
        {**depth, "imu_extr_trans_mm": 10.0, "imu_extr_rot_deg": 0.3}
    with pytest.raises(ValueError, match="time_offset_ms"):
        contract_noise({**sidecar_contract(shared.depth), "time_offset_ms": 1.0}, "depth")


def test_bundle_identity_ignores_lidar_files_beside_it(tmp_path):
    shared = shared_lidar_bundle()
    beside = SynthRun.open(shared.bundle)   # its LiDAR files are written
    bundle = shutil.copytree(shared.bundle, tmp_path / "apple" / shared.bundle.name)
    alone = SynthRun.open(bundle)
    assert not (bundle.parent / "lidar").exists()
    write_lidar_files(bundle)
    write_lidar_files(bundle, unit="nav", mount_xyz=(0.0, 0.0, 0.5), config_sha256="cd" * 32)
    after = SynthRun.open(bundle)
    assert sorted(p.name for p in (bundle.parent / "lidar").iterdir()) == ["abababababababab", "cdcdcdcdcdcdcdcd"]
    assert (alone.meta.run_id, alone.meta.fingerprint) == (after.meta.run_id, after.meta.fingerprint) == \
        (beside.meta.run_id, beside.meta.fingerprint)
    sha = {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in ("manifest.json",
                                                                                     "trajectory.json")}
    assert bundle_inputs(after) == sha
    assert {k: v for k, v in _sidecar(shared.depth)["inputs"].items() if k in sha} == sha
