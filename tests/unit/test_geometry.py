import numpy as np
import pytest

from cyclist_tracker import geometry as g

pytestmark = pytest.mark.unit


def test_parse_and_format_points_round_trip():
    pts = g.parse_points("1,2, 3.5,4;5,6")
    assert pts == [(1, 2), (3.5, 4), (5, 6)]
    assert g.parse_points(g.format_points(pts)) == pts


def test_parse_points_rejects_odd_number_of_values():
    with pytest.raises(ValueError, match="even number"):
        g.parse_points("1,2,3")


@pytest.mark.parametrize(
    ("point", "line", "expected"),
    [((0, 5), ((0, 0), (10, 0)), 5.0), ((3, 4), ((0, 0), (0, 0)), 5.0), ((20, -2), ((0, 0), (10, 0)), 2.0)],
)
def test_point_to_line_distance(point, line, expected):
    # the line is infinite, so points beyond the segment ends use the perpendicular distance
    assert g.point_to_line_distance(point, line) == pytest.approx(expected)


def test_point_in_polygon_inside_edge_outside():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert g.point_in_polygon((5, 5), square)
    assert g.point_in_polygon((10, 5), square)
    assert not g.point_in_polygon((11, 5), square)


def test_bbox_helpers():
    assert g.bbox_center((0, 0, 10, 20)) == (5, 10)
    assert g.reference_point((0, 0, 10, 20), "bottom") == (5, 20)
    assert g.reference_point((0, 0, 10, 20), "center") == (5, 10)
    assert g.ltrb_to_xywh(g.xywh_to_ltrb((1, 2, 3, 4))) == (1, 2, 3, 4)


@pytest.mark.parametrize(
    ("a", "b", "iou"),
    [([0, 0, 10, 10], [0, 0, 10, 10], 1.0), ([0, 0, 10, 10], [20, 20, 5, 5], 0.0),
     ([0, 0, 10, 10], [5, 0, 10, 10], 50 / 150), ([0, 0, 0, 0], [0, 0, 0, 0], 0.0)],
)
def test_bbox_iou(a, b, iou):
    assert g.bbox_iou(a, b) == pytest.approx(iou)


def test_homography_maps_calibration_points_exactly():
    image = [(937.3, 390.2), (1100.3, 388.7), (1084.6, 506.2), (862.4, 504.0)]
    world = [(0, 0), (4, 0), (4, 10.5), (0, 10.5)]
    h = g.compute_homography(image, world)
    for p, w in zip(image, world, strict=True):
        np.testing.assert_allclose(g.pixel_to_world(p, h), w, atol=1e-4)  # 0.1 mm
    assert g.world_distance(image[0], image[1], h) == pytest.approx(4.0, abs=1e-4)
    assert g.calibration_error(image, world, h) == pytest.approx(0.0, abs=1e-4)
    np.testing.assert_allclose(g.world_to_pixel(g.pixel_to_world((1000, 450), h), h), (1000, 450), atol=1e-4)


def test_homography_least_squares_with_more_points():
    true_h = np.array([[0.02, 0.001, -1.0], [0.0005, 0.05, -3.0], [0.00001, 0.0004, 1.0]])
    image = [(100, 100), (500, 110), (520, 400), (90, 420), (300, 250), (200, 350)]
    world = [tuple(g.pixel_to_world(p, true_h)) for p in image]
    h = g.compute_homography(image, world)
    assert g.calibration_error(image, world, h) < 1e-6


def test_homography_needs_four_pairs():
    with pytest.raises(ValueError, match="at least 4"):
        g.compute_homography([(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1)])
