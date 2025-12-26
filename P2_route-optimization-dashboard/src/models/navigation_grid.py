"""
Navigation environment with grid representation and obstacles.

Provides a discrete grid for pathfinding with obstacle zones,
storm avoidance, and weather integration.
"""

from typing import Tuple, Set, List, Optional
from dataclasses import dataclass, field
import numpy as np

from src.utils.geometry import Point, nautical_to_grid, grid_to_nautical
from src.models.weather_field import WeatherField


@dataclass
class NavigationConstraints:
    """Navigation constraints and safety parameters.

    Attributes:
        min_storm_distance: Minimum safe distance from storm zones (nautical miles)
        max_wave_height: Maximum acceptable wave height (meters)
        restricted_zones: Set of (row, col) grid cells that are forbidden
    """
    min_storm_distance: float = 50.0  # nautical miles
    max_wave_height: float = 6.0  # meters
    restricted_zones: Set[Tuple[int, int]] = field(default_factory=set)


class NavigationEnvironment:
    """Discrete grid environment for ship navigation.

    Combines grid representation, weather fields, and navigation constraints
    to support pathfinding and route evaluation.
    """

    def __init__(self,
                 grid_shape: Tuple[int, int],
                 cell_size: float,
                 origin: Point = Point(0, 0),
                 weather: Optional[WeatherField] = None,
                 constraints: Optional[NavigationConstraints] = None):
        """Initialize navigation environment.

        Args:
            grid_shape: (n_rows, n_cols) grid dimensions
            cell_size: Size of each cell in nautical miles
            origin: Origin point in nautical miles
            weather: WeatherField instance (optional, creates calm if None)
            constraints: Navigation constraints (optional, creates default if None)
        """
        self.grid_shape = grid_shape
        self.cell_size = cell_size
        self.origin = origin

        if weather is None:
            from src.models.weather_field import create_calm_scenario
            weather = create_calm_scenario(grid_shape, cell_size)
        self.weather = weather

        if constraints is None:
            constraints = NavigationConstraints()
        self.constraints = constraints

        # Initialize obstacle grid (True = navigable, False = blocked)
        self.navigable = np.ones(grid_shape, dtype=bool)

        # Mark restricted zones
        for zone in constraints.restricted_zones:
            if self._is_valid_grid_pos(zone):
                self.navigable[zone] = False

    def _is_valid_grid_pos(self, pos: Tuple[int, int]) -> bool:
        """Check if grid position is within bounds.

        Args:
            pos: (row, col) grid position

        Returns:
            True if position is within grid bounds
        """
        row, col = pos
        rows, cols = self.grid_shape
        return 0 <= row < rows and 0 <= col < cols

    def is_navigable(self, point: Point) -> bool:
        """Check if a point is navigable (not in restricted or storm zones).

        Args:
            point: Query point in nautical miles

        Returns:
            True if point is safe to navigate
        """
        # Convert to grid coordinates
        grid_pos = nautical_to_grid(point, self.cell_size, self.origin)

        # Check if within bounds
        if not self._is_valid_grid_pos(grid_pos):
            return False

        # Check if marked as non-navigable
        if not self.navigable[grid_pos]:
            return False

        # Check weather conditions
        _, wave_height = self.weather.get_weather_at_point(point)
        if wave_height > self.constraints.max_wave_height:
            return False

        return True

    def is_navigable_grid(self, grid_pos: Tuple[int, int]) -> bool:
        """Check if a grid cell is navigable.

        Args:
            grid_pos: (row, col) grid position

        Returns:
            True if cell is safe to navigate
        """
        if not self._is_valid_grid_pos(grid_pos):
            return False

        point = grid_to_nautical(grid_pos, self.cell_size, self.origin)
        return self.is_navigable(point)

    def get_weather_penalty(self, point: Point) -> float:
        """Calculate weather-based cost penalty for a point.

        Higher wave heights increase the cost, making pathfinding
        prefer calmer routes.

        Args:
            point: Query point in nautical miles

        Returns:
            Cost multiplier (1.0 = no penalty, >1.0 = increased cost)
        """
        wind_speed, wave_height = self.weather.get_weather_at_point(point)

        # Penalty increases with wave height (amplified to strongly favor detours)
        # wave_height = 0: penalty = 1.0
        # wave_height = 3: penalty = 1.6
        # wave_height = 6: penalty = 3.25
        if wave_height >= self.constraints.max_wave_height:
            return 1e6  # Treat unsafe seas as effectively blocked for A*
        penalty = 1.0 + (wave_height / 4.0) ** 2

        return penalty

    def add_restricted_zone(self, center: Point, radius: float):
        """Add a circular restricted zone.

        Args:
            center: Center of restricted zone (nautical miles)
            radius: Radius of zone (nautical miles)
        """
        rows, cols = self.grid_shape

        for i in range(rows):
            for j in range(cols):
                grid_point = grid_to_nautical((i, j), self.cell_size, self.origin)
                distance = grid_point.distance_to(center)

                if distance <= radius:
                    self.navigable[i, j] = False
                    self.constraints.restricted_zones.add((i, j))

    def get_neighbors(self, grid_pos: Tuple[int, int],
                      connectivity: int = 8) -> List[Tuple[int, int]]:
        """Get navigable neighbors of a grid cell.

        Args:
            grid_pos: (row, col) grid position
            connectivity: 4 (cardinal) or 8 (include diagonals)

        Returns:
            List of navigable neighbor positions
        """
        row, col = grid_pos
        neighbors = []

        # Cardinal directions
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Add diagonals for 8-connectivity
        if connectivity == 8:
            deltas.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

        for dr, dc in deltas:
            neighbor = (row + dr, col + dc)
            if self.is_navigable_grid(neighbor):
                neighbors.append(neighbor)

        return neighbors

    def get_grid_extent_nm(self) -> Tuple[Point, Point]:
        """Get the spatial extent of the grid in nautical miles.

        Returns:
            Tuple of (bottom_left, top_right) points
        """
        bottom_left = self.origin
        top_right = Point(
            self.origin.x + self.grid_shape[1] * self.cell_size,
            self.origin.y + self.grid_shape[0] * self.cell_size
        )
        return (bottom_left, top_right)

    def compute_storm_zones(self, wave_threshold: float = 5.0) -> List[Point]:
        """Identify centers of storm zones in the environment.

        Args:
            wave_threshold: Wave height threshold for storms (meters)

        Returns:
            List of storm center points
        """
        storm_cells = []
        rows, cols = self.grid_shape

        for i in range(rows):
            for j in range(cols):
                point = grid_to_nautical((i, j), self.cell_size, self.origin)
                _, wave_height = self.weather.get_weather_at_point(point)

                if wave_height >= wave_threshold:
                    storm_cells.append(point)

        return storm_cells

    def distance_to_nearest_storm(self, point: Point,
                                   wave_threshold: float = 5.0) -> float:
        """Calculate distance from point to nearest storm zone.

        Args:
            point: Query point in nautical miles
            wave_threshold: Wave height threshold for storms

        Returns:
            Distance to nearest storm in nautical miles (inf if no storms)
        """
        storm_centers = self.compute_storm_zones(wave_threshold)

        if not storm_centers:
            return float('inf')

        min_distance = float('inf')
        for storm_center in storm_centers:
            distance = point.distance_to(storm_center)
            min_distance = min(min_distance, distance)

        return min_distance

    def get_statistics(self) -> dict:
        """Get environment statistics.

        Returns:
            Dictionary with environment info
        """
        rows, cols = self.grid_shape
        total_cells = rows * cols
        navigable_cells = self.navigable.sum()
        blocked_cells = total_cells - navigable_cells

        bottom_left, top_right = self.get_grid_extent_nm()

        weather_stats = self.weather.get_field_statistics()

        return {
            'grid': {
                'shape': self.grid_shape,
                'cell_size_nm': self.cell_size,
                'total_cells': int(total_cells),
                'navigable_cells': int(navigable_cells),
                'blocked_cells': int(blocked_cells),
                'navigable_fraction': float(navigable_cells / total_cells)
            },
            'extent_nm': {
                'x_min': bottom_left.x,
                'x_max': top_right.x,
                'y_min': bottom_left.y,
                'y_max': top_right.y
            },
            'weather': weather_stats,
            'constraints': {
                'min_storm_distance_nm': self.constraints.min_storm_distance,
                'max_wave_height_m': self.constraints.max_wave_height,
                'num_restricted_zones': len(self.constraints.restricted_zones)
            }
        }
