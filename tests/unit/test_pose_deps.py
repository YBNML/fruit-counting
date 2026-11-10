"""Version pins of third-party libraries and the pytest marker of the pose tests.

OpenCV is pinned to one minor version in both conda environment files and in pyproject.toml, so every
machine installs the same OpenCV release. The CPU environment takes every conda package from conda-forge,
torch included, so the environment loads one OpenMP runtime.

GTSAM comes from conda-forge through the environment files only: the PyPI wheel lists pytest among its
runtime requirements.
"""

import re
import tomllib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.pose

ROOT = Path(__file__).resolve().parents[2]
ENV_FILES = ("environment.yml", "environment-cuda.yml")


def _env(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def _conda_specs(name: str) -> list[str]:
    """The conda (not pip) dependency specs of the environment file `name`."""
    return [d for d in _env(name)["dependencies"] if isinstance(d, str)]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_conda_environments_pin_one_opencv_minor():
    for name in ENV_FILES:
        assert [d for d in _conda_specs(name) if d.split("=")[0] == "opencv"] == ["opencv=5.0.*"], name


def test_pyproject_pins_the_same_opencv_minor():
    deps = _pyproject()["project"]["dependencies"]
    assert [d for d in deps if d.startswith("opencv")] == ["opencv-python>=5.0,<5.1"]


def test_cpu_environment_takes_conda_packages_from_conda_forge_only():
    env = _env("environment.yml")
    assert env["channels"] == ["conda-forge"]
    specs = _conda_specs("environment.yml")
    assert "opencv=5.0.*" in specs
    assert [d for d in specs if d.split("=")[0] in ("pytorch", "torchvision")] == ["pytorch=2.14.*", "torchvision=0.29.*"]
    pip = next(d["pip"] for d in env["dependencies"] if isinstance(d, dict))
    assert not [d for d in pip if re.match(r"[A-Za-z0-9_.-]+", d).group() in ("torch", "torchvision")]


def test_pose_marker_is_registered():
    ini = _pyproject()["tool"]["pytest"]["ini_options"]
    assert [m.split(":")[0] for m in ini["markers"]] == ["slow", "synth", "pose"]
    assert ini["addopts"] == "-q -m 'not synth'"   # pose tests stay in the default run


def test_gtsam_comes_from_the_conda_environments_only():
    for name in ENV_FILES:
        assert [d for d in _conda_specs(name) if d.split("=")[0] == "gtsam"] == ["gtsam=4.3.*"], name
    project = _pyproject()["project"]
    specs = [*project["dependencies"], *(s for group in project["optional-dependencies"].values() for s in group)]
    assert not [s for s in specs if "gtsam" in s]
