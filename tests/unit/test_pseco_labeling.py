import pytest
import torch

from counting.models.pseco.labeling import (
    assign_targets_from_points,
    points_in_box,
)


def test_point_inside_box():
    assert points_in_box([(5.0, 5.0)], (0.0, 0.0, 10.0, 10.0)) is True


def test_point_outside_box():
    assert points_in_box([(11.0, 5.0)], (0.0, 0.0, 10.0, 10.0)) is False


def test_point_on_left_edge_included():
    assert points_in_box([(0.0, 5.0)], (0.0, 0.0, 10.0, 10.0)) is True


def test_point_on_right_edge_excluded():
    assert points_in_box([(10.0, 5.0)], (0.0, 0.0, 10.0, 10.0)) is False


def test_empty_points_is_false():
    assert points_in_box([], (0.0, 0.0, 10.0, 10.0)) is False


def test_multiple_points_any_inside():
    pts = [(50.0, 50.0), (5.0, 5.0)]
    assert points_in_box(pts, (0.0, 0.0, 10.0, 10.0)) is True


def test_assign_targets_shape_and_dtype():
    boxes = [torch.tensor([[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]])]
    points = [[(5.0, 5.0)]]
    targets = assign_targets_from_points(boxes, points)
    assert targets.shape == (2,)
    assert targets.dtype == torch.long
    assert targets.tolist() == [1, 0]


def test_assign_targets_two_images_flattened_row_major():
    # Image 0: 2 boxes; first contains a point, second does not.
    # Image 1: 2 boxes; neither contains a point.
    boxes = [
        torch.tensor([[[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]]),
        torch.tensor([[[100.0, 100.0, 110.0, 110.0], [200.0, 200.0, 210.0, 210.0]]]),
    ]
    points = [
        [(5.0, 5.0)],
        [(0.0, 0.0)],  # not inside either image-1 box
    ]
    targets = assign_targets_from_points(boxes, points)
    # Row-major: image0_box0, image0_box1, image1_box0, image1_box1
    assert targets.tolist() == [1, 0, 0, 0]


def test_assign_targets_mismatched_lengths_raises():
    boxes = [torch.zeros((1, 2, 4))]
    points: list[list[tuple[float, float]]] = [[], []]  # 2 images worth
    with pytest.raises(ValueError, match="length"):
        assign_targets_from_points(boxes, points)
