"""
Unit tests for geometry utilities.

Tests Point class, distance calculations, and coordinate conversions.
"""

import pytest
import numpy as np
from src.utils.geometry import (
    Point, grid_to_nautical, nautical_to_grid, path_length,
    interpolate_point, point_to_segment_distance
)


class TestPoint:
    """Test Point class operations."""

    def test_point_creation(self):
        """Test creating a Point."""
        p = Point(10.0, 20.0)
        assert p.x == 10.0
        assert p.y == 20.0

    def test_point_immutable(self):
        """Test that Point is immutable (frozen dataclass)."""
        p = Point(10.0, 20.0)
        with pytest.raises(AttributeError):
            p.x = 15.0

    def test_distance_to(self):
        """Test Euclidean distance calculation."""
        p1 = Point(0.0, 0.0)
        p2 = Point(3.0, 4.0)

        distance = p1.distance_to(p2)
        assert pytest.approx(distance, rel=0.001) == 5.0

    def test_distance_symmetric(self):
        """Test that distance is symmetric."""
        p1 = Point(10.0, 20.0)
        p2 = Point(30.0, 40.0)

        assert p1.distance_to(p2) == p2.distance_to(p1)

    def test_distance_to_self(self):
        """Test that distance to self is zero."""
        p = Point(5.0, 10.0)
        assert p.distance_to(p) == 0.0

    def test_point_addition(self):
        """Test vector addition of points."""
        p1 = Point(10.0, 20.0)
        p2 = Point(5.0, 3.0)
        p3 = p1 + p2

        assert p3.x == 15.0
        assert p3.y == 23.0

    def test_point_subtraction(self):
        """Test vector subtraction of points."""
        p1 = Point(10.0, 20.0)
        p2 = Point(3.0, 5.0)
        p3 = p1 - p2

        assert p3.x == 7.0
        assert p3.y == 15.0

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        p = Point(4.0, 6.0)
        p2 = p * 2.5

        assert p2.x == 10.0
        assert p2.y == 15.0

    def test_to_tuple(self):
        """Test conversion to tuple."""
        p = Point(12.5, 34.7)
        t = p.to_tuple()

        assert t == (12.5, 34.7)
        assert isinstance(t, tuple)


class TestCoordinateConversions:
    """Test grid <-> nautical mile conversions."""

    def test_grid_to_nautical(self):
        """Test converting grid position to nautical miles."""
        grid_pos = (5, 10)  # row=5, col=10
        cell_size = 10.0  # nm
        origin = Point(100.0, 200.0)

        point = grid_to_nautical(grid_pos, cell_size, origin)

        # Expected: x = 100 + 10*10 = 200, y = 200 + 5*10 = 250
        assert point.x == 200.0
        assert point.y == 250.0

    def test_nautical_to_grid(self):
        """Test converting nautical miles to grid position."""
        point = Point(200.0, 250.0)
        cell_size = 10.0
        origin = Point(100.0, 200.0)

        grid_pos = nautical_to_grid(point, cell_size, origin)

        # Expected: col = (200-100)/10 = 10, row = (250-200)/10 = 5
        assert grid_pos == (5, 10)

    def test_round_trip_conversion(self):
        """Test that grid -> nautical -> grid is identity (with rounding)."""
        original_grid = (7, 13)
        cell_size = 5.0
        origin = Point(0.0, 0.0)

        # Convert to nautical and back
        point = grid_to_nautical(original_grid, cell_size, origin)
        result_grid = nautical_to_grid(point, cell_size, origin)

        assert result_grid == original_grid

    def test_nautical_to_grid_with_rounding(self):
        """Test that nautical to grid rounds to nearest cell."""
        # Point between grid cells
        point = Point(103.0, 207.0)
        cell_size = 10.0
        origin = Point(100.0, 200.0)

        grid_pos = nautical_to_grid(point, cell_size, origin)

        # 103 -> col = (103-100)/10 = 0.3 -> rounds to 0
        # 207 -> row = (207-200)/10 = 0.7 -> rounds to 1
        assert grid_pos == (1, 0)


class TestPathLength:
    """Test path length calculations."""

    def test_path_length_two_points(self):
        """Test path length for straight line."""
        waypoints = [Point(0.0, 0.0), Point(3.0, 4.0)]
        length = path_length(waypoints)

        assert pytest.approx(length, rel=0.001) == 5.0

    def test_path_length_multiple_points(self):
        """Test path length for multi-segment path."""
        waypoints = [
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(10.0, 10.0)
        ]
        length = path_length(waypoints)

        # Length = 10 + 10 = 20
        assert pytest.approx(length, rel=0.001) == 20.0

    def test_path_length_single_point_error(self):
        """Test that single waypoint raises ValueError."""
        waypoints = [Point(0.0, 0.0)]

        with pytest.raises(ValueError, match="at least 2 waypoints"):
            path_length(waypoints)

    def test_path_length_empty_error(self):
        """Test that empty waypoints raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 waypoints"):
            path_length([])


class TestInterpolation:
    """Test point interpolation."""

    def test_interpolate_at_start(self):
        """Test interpolation at fraction=0 returns start point."""
        p1 = Point(10.0, 20.0)
        p2 = Point(30.0, 40.0)

        result = interpolate_point(p1, p2, 0.0)

        assert result.x == p1.x
        assert result.y == p1.y

    def test_interpolate_at_end(self):
        """Test interpolation at fraction=1 returns end point."""
        p1 = Point(10.0, 20.0)
        p2 = Point(30.0, 40.0)

        result = interpolate_point(p1, p2, 1.0)

        assert result.x == p2.x
        assert result.y == p2.y

    def test_interpolate_midpoint(self):
        """Test interpolation at fraction=0.5 returns midpoint."""
        p1 = Point(10.0, 20.0)
        p2 = Point(30.0, 60.0)

        result = interpolate_point(p1, p2, 0.5)

        assert result.x == 20.0
        assert result.y == 40.0

    def test_interpolate_arbitrary_fraction(self):
        """Test interpolation at arbitrary fraction."""
        p1 = Point(0.0, 0.0)
        p2 = Point(100.0, 200.0)

        result = interpolate_point(p1, p2, 0.25)

        assert result.x == 25.0
        assert result.y == 50.0


class TestPointToSegmentDistance:
    """Test point-to-segment distance calculations."""

    def test_distance_to_segment_perpendicular(self):
        """Test distance when point projects onto segment."""
        point = Point(5.0, 5.0)
        seg_start = Point(0.0, 0.0)
        seg_end = Point(10.0, 0.0)

        distance = point_to_segment_distance(point, seg_start, seg_end)

        # Point is 5 units above horizontal segment
        assert pytest.approx(distance, rel=0.001) == 5.0

    def test_distance_to_segment_endpoint(self):
        """Test distance when closest point is segment endpoint."""
        point = Point(-5.0, 0.0)
        seg_start = Point(0.0, 0.0)
        seg_end = Point(10.0, 0.0)

        distance = point_to_segment_distance(point, seg_start, seg_end)

        # Closest point is seg_start
        assert pytest.approx(distance, rel=0.001) == 5.0

    def test_distance_to_degenerate_segment(self):
        """Test distance when segment is a point."""
        point = Point(5.0, 5.0)
        seg_start = Point(0.0, 0.0)
        seg_end = Point(0.0, 0.0)  # Same as start

        distance = point_to_segment_distance(point, seg_start, seg_end)

        # Distance to single point
        expected = point.distance_to(seg_start)
        assert pytest.approx(distance, rel=0.001) == expected

    def test_distance_on_segment_is_zero(self):
        """Test that point on segment has zero distance."""
        point = Point(5.0, 0.0)
        seg_start = Point(0.0, 0.0)
        seg_end = Point(10.0, 0.0)

        distance = point_to_segment_distance(point, seg_start, seg_end)

        assert pytest.approx(distance, abs=0.001) == 0.0

    def test_distance_to_diagonal_segment(self):
        """Test distance to diagonal segment."""
        point = Point(5.0, 5.0)
        seg_start = Point(0.0, 0.0)
        seg_end = Point(10.0, 10.0)

        distance = point_to_segment_distance(point, seg_start, seg_end)

        # Point is ON the diagonal line
        assert pytest.approx(distance, abs=0.001) == 0.0

    def test_distance_to_diagonal_segment_off_line(self):
        """Test distance to diagonal segment when point is off line."""
        point = Point(0.0, 5.0)
        seg_start = Point(0.0, 0.0)
        seg_end = Point(10.0, 0.0)

        distance = point_to_segment_distance(point, seg_start, seg_end)

        # Point is 5 units perpendicular to horizontal segment
        # Closest point on segment is (0, 0)
        assert pytest.approx(distance, rel=0.001) == 5.0
