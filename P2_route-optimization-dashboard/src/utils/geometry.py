"""
Geometric utilities for navigation and coordinate transformations.

Provides Point class and utility functions for distance calculations,
grid conversions, and spatial operations.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class Point:
    """Immutable 2D point in nautical miles or grid coordinates.

    Attributes:
        x: X-coordinate (eastward in nautical miles or grid column)
        y: Y-coordinate (northward in nautical miles or grid row)
    """
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point.

        Args:
            other: Target point

        Returns:
            Distance in same units as coordinates (typically nautical miles)
        """
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __add__(self, other: 'Point') -> 'Point':
        """Vector addition of points."""
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        """Vector subtraction of points."""
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Point':
        """Scalar multiplication."""
        return Point(self.x * scalar, self.y * scalar)

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple (x, y)."""
        return (self.x, self.y)


def grid_to_nautical(grid_pos: Tuple[int, int],
                     cell_size: float,
                     origin: Point = Point(0, 0)) -> Point:
    """Convert grid coordinates to nautical mile coordinates.

    Args:
        grid_pos: (row, col) grid position
        cell_size: Size of each grid cell in nautical miles
        origin: Origin point in nautical miles (default: (0, 0))

    Returns:
        Point in nautical mile coordinates
    """
    row, col = grid_pos
    x = origin.x + col * cell_size
    y = origin.y + row * cell_size
    return Point(x, y)


def nautical_to_grid(point: Point,
                     cell_size: float,
                     origin: Point = Point(0, 0)) -> Tuple[int, int]:
    """Convert nautical mile coordinates to grid coordinates.

    Args:
        point: Point in nautical miles
        cell_size: Size of each grid cell in nautical miles
        origin: Origin point in nautical miles (default: (0, 0))

    Returns:
        (row, col) grid position (rounded to nearest cell)
    """
    col = round((point.x - origin.x) / cell_size)
    row = round((point.y - origin.y) / cell_size)
    return (row, col)


def path_length(waypoints: list[Point]) -> float:
    """Calculate total length of path through waypoints.

    Args:
        waypoints: Ordered list of points defining the path

    Returns:
        Total path length in same units as points

    Raises:
        ValueError: If fewer than 2 waypoints provided
    """
    if len(waypoints) < 2:
        raise ValueError("Path must have at least 2 waypoints")

    total_length = 0.0
    for i in range(1, len(waypoints)):
        total_length += waypoints[i-1].distance_to(waypoints[i])

    return total_length


def interpolate_point(p1: Point, p2: Point, fraction: float) -> Point:
    """Linearly interpolate between two points.

    Args:
        p1: Start point
        p2: End point
        fraction: Interpolation parameter (0 = p1, 1 = p2)

    Returns:
        Interpolated point
    """
    return Point(
        p1.x + fraction * (p2.x - p1.x),
        p1.y + fraction * (p2.y - p1.y)
    )


def point_to_segment_distance(point: Point,
                               seg_start: Point,
                               seg_end: Point) -> float:
    """Calculate minimum distance from point to line segment.

    Args:
        point: Point to measure from
        seg_start: Start of line segment
        seg_end: End of line segment

    Returns:
        Minimum distance from point to segment
    """
    # Vector from segment start to end
    seg_vec = seg_end - seg_start
    seg_length_sq = seg_vec.x**2 + seg_vec.y**2

    if seg_length_sq == 0:
        # Degenerate segment (point)
        return point.distance_to(seg_start)

    # Project point onto segment line
    # t = dot(point - seg_start, seg_vec) / ||seg_vec||^2
    point_vec = point - seg_start
    t = (point_vec.x * seg_vec.x + point_vec.y * seg_vec.y) / seg_length_sq

    # Clamp to segment
    t = max(0.0, min(1.0, t))

    # Find closest point on segment
    closest = seg_start + seg_vec * t

    return point.distance_to(closest)
