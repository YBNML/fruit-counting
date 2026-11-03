import numpy as np
import pytest

from counting.geometry.camera import Camera, box_centers, pitch_from_blender
from tests.unit.synthfake import robot_matrix

FX = 960 / np.tan(np.radians(34.5))


def _id_cam():
    return Camera.from_blender(np.eye(4), 1920, 1080, 69.0)


def test_intrinsics_robot_preset():
    cam = _id_cam()
    assert cam.K[0, 0] == pytest.approx(1396.808668, abs=1e-6) and cam.K[1, 1] == cam.K[0, 0]
    assert (cam.K[0, 2], cam.K[1, 2]) == (959.5, 539.5)


def test_identity_camera_projects_axis_point_to_center():
    uv, z = _id_cam().project(np.array([[0.0, 0.0, -2.0]]))
    assert uv[0] == pytest.approx([959.5, 539.5], abs=1e-12) and z[0] == pytest.approx(2.0)


def test_blender_up_is_image_up():
    uv, _ = _id_cam().project(np.array([[0.1, 0.2, -2.0]]))
    assert uv[0] == pytest.approx([959.5 + 0.05 * FX, 539.5 - 0.1 * FX], abs=1e-9)


def _pitched():
    return Camera.from_blender(robot_matrix(0.4, 3.75, "A", 30.0, 1.75, 0.9), 1920, 1080, 69.0)


def _pts():
    rng = np.random.default_rng(0)
    return np.c_[rng.uniform(-1, 1, 50), rng.uniform(3.4, 4.1, 50), rng.uniform(1.2, 2.4, 50)]


def test_project_backproject_roundtrip():
    cam, X = _pitched(), _pts()
    uv, z = cam.project(X)
    assert np.all(z > 0)
    np.testing.assert_allclose(cam.backproject(uv, z), X, atol=1e-9)


def test_ray_dir_is_unit_and_points_at_X():
    cam, X = _pitched(), _pts()
    d = cam.ray_dir(cam.project(X)[0])
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)
    v = X - cam.C
    np.testing.assert_allclose(np.cross(d, v), 0.0, atol=1e-9)
    assert np.all(np.einsum("ij,ij->i", d, v) > 0)


def test_pitch_from_blender():
    assert pitch_from_blender(robot_matrix(0.0, 0.0, "A", 30.0, 2.0, 0.5)) == pytest.approx(30.0)
    assert pitch_from_blender(robot_matrix(0.0, 0.0, "B", 70.0, 2.0, 0.5)) == pytest.approx(70.0)


def test_in_image_bounds():
    uv = np.array([[-0.5, -0.5], [1919.49, 1079.49], [1919.5, 0.0], [-0.51, 0.0]])
    assert _id_cam().in_image(uv).tolist() == [True, True, False, False]


def test_box_centers_edge_to_opencv():
    np.testing.assert_allclose(box_centers(np.array([[10, 20, 14, 30]], float)), [[11.5, 24.5]])
