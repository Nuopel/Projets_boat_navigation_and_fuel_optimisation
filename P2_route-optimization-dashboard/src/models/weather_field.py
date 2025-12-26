"""
Weather field generation and interpolation.

Provides synthetic weather data (wind speed, wave height) on a 2D grid
with Gaussian smoothing for realistic spatial patterns.
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter
from src.utils.geometry import Point, nautical_to_grid, grid_to_nautical


@dataclass
class WeatherZone:
    """Defines a localized weather zone.

    Attributes:
        center: Center position in nautical miles
        radius: Effective radius in nautical miles
        wind_speed: Wind speed in knots
        wave_height: Significant wave height in meters
        intensity: Relative intensity (0-1), affects blending
    """
    center: Point
    radius: float
    wind_speed: float
    wave_height: float
    intensity: float = 1.0


class WeatherField:
    """2D weather field with wind and wave data.

    Generates synthetic weather patterns by placing weather zones
    and applying Gaussian smoothing for spatial coherence.
    """

    def __init__(self,
                 grid_shape: Tuple[int, int],
                 cell_size: float,
                 origin: Point = Point(0, 0)):
        """Initialize weather field on a grid.

        Args:
            grid_shape: (n_rows, n_cols) grid dimensions
            cell_size: Size of each cell in nautical miles
            origin: Origin point in nautical miles
        """
        self.grid_shape = grid_shape
        self.cell_size = cell_size
        self.origin = origin

        # Initialize with calm conditions
        self.wind_field = np.zeros(grid_shape, dtype=np.float32)
        self.wave_field = np.zeros(grid_shape, dtype=np.float32)

    def add_weather_zone(self, zone: WeatherZone, blend: bool = True):
        """Add a weather zone to the field.

        Args:
            zone: WeatherZone to add
            blend: If True, blend with existing weather; if False, replace
        """
        rows, cols = self.grid_shape

        for i in range(rows):
            for j in range(cols):
                # Convert grid to nautical miles
                grid_point = grid_to_nautical((i, j), self.cell_size, self.origin)

                # Calculate distance from zone center
                distance = grid_point.distance_to(zone.center)

                # Gaussian decay with distance
                sigma = zone.radius / 2.0  # Standard deviation
                weight = np.exp(-(distance ** 2) / (2 * sigma ** 2))
                weight *= zone.intensity

                if blend:
                    # Blend with existing weather (max operation for storms)
                    self.wind_field[i, j] = max(
                        self.wind_field[i, j],
                        zone.wind_speed * weight
                    )
                    self.wave_field[i, j] = max(
                        self.wave_field[i, j],
                        zone.wave_height * weight
                    )
                else:
                    # Replace
                    self.wind_field[i, j] = zone.wind_speed * weight
                    self.wave_field[i, j] = zone.wave_height * weight

    def add_base_conditions(self, wind_speed: float, wave_height: float):
        """Set baseline weather conditions across entire field.

        Args:
            wind_speed: Base wind speed in knots
            wave_height: Base wave height in meters
        """
        self.wind_field += wind_speed
        self.wave_field += wave_height

    def smooth_field(self, sigma: float = 1.0):
        """Apply Gaussian smoothing to weather fields.

        Args:
            sigma: Standard deviation for Gaussian kernel (in grid cells)
        """
        self.wind_field = gaussian_filter(self.wind_field, sigma=sigma)
        self.wave_field = gaussian_filter(self.wave_field, sigma=sigma)

    def get_weather_at_point(self, point: Point) -> Tuple[float, float]:
        """Get weather conditions at a specific point via bilinear interpolation.

        Args:
            point: Query point in nautical miles

        Returns:
            Tuple of (wind_speed_knots, wave_height_meters)
        """
        # Convert to grid coordinates (floating point)
        col_f = (point.x - self.origin.x) / self.cell_size
        row_f = (point.y - self.origin.y) / self.cell_size

        # Clamp to grid bounds
        rows, cols = self.grid_shape
        row_f = np.clip(row_f, 0, rows - 1)
        col_f = np.clip(col_f, 0, cols - 1)

        # Bilinear interpolation
        wind_speed = self._bilinear_interpolate(self.wind_field, row_f, col_f)
        wave_height = self._bilinear_interpolate(self.wave_field, row_f, col_f)

        return (wind_speed, wave_height)

    def _bilinear_interpolate(self,
                              field: np.ndarray,
                              row: float,
                              col: float) -> float:
        """Perform bilinear interpolation on a field.

        Args:
            field: 2D numpy array
            row: Row coordinate (floating point)
            col: Column coordinate (floating point)

        Returns:
            Interpolated value
        """
        rows, cols = field.shape

        # Get integer parts
        r0 = int(np.floor(row))
        r1 = min(r0 + 1, rows - 1)
        c0 = int(np.floor(col))
        c1 = min(c0 + 1, cols - 1)

        # Get fractional parts
        dr = row - r0
        dc = col - c0

        # Bilinear interpolation
        v00 = field[r0, c0]
        v01 = field[r0, c1]
        v10 = field[r1, c0]
        v11 = field[r1, c1]

        v0 = v00 * (1 - dc) + v01 * dc
        v1 = v10 * (1 - dc) + v11 * dc
        v = v0 * (1 - dr) + v1 * dr

        return float(v)

    def is_storm_zone(self, point: Point, wave_threshold: float = 5.0) -> bool:
        """Check if a point is in a storm zone.

        Args:
            point: Query point in nautical miles
            wave_threshold: Wave height threshold for storm (meters)

        Returns:
            True if wave height exceeds threshold
        """
        _, wave_height = self.get_weather_at_point(point)
        return wave_height >= wave_threshold

    def get_field_statistics(self) -> dict:
        """Calculate statistics of the weather field.

        Returns:
            Dictionary with min/max/mean/std for wind and waves
        """
        return {
            'wind': {
                'min': float(self.wind_field.min()),
                'max': float(self.wind_field.max()),
                'mean': float(self.wind_field.mean()),
                'std': float(self.wind_field.std())
            },
            'waves': {
                'min': float(self.wave_field.min()),
                'max': float(self.wave_field.max()),
                'mean': float(self.wave_field.mean()),
                'std': float(self.wave_field.std())
            }
        }


def create_calm_scenario(grid_shape: Tuple[int, int],
                         cell_size: float) -> WeatherField:
    """Create a calm weather scenario.

    Args:
        grid_shape: Grid dimensions
        cell_size: Cell size in nautical miles

    Returns:
        WeatherField with calm conditions
    """
    weather = WeatherField(grid_shape, cell_size)
    weather.add_base_conditions(wind_speed=10.0, wave_height=1.5)
    weather.smooth_field(sigma=2.0)
    return weather


def create_storm_scenario(grid_shape: Tuple[int, int],
                          cell_size: float,
                          storm_center: Point,
                          storm_radius: float = 100.0) -> WeatherField:
    """Create a scenario with a storm zone.

    Args:
        grid_shape: Grid dimensions
        cell_size: Cell size in nautical miles
        storm_center: Storm center position in nautical miles
        storm_radius: Storm effective radius in nautical miles

    Returns:
        WeatherField with storm zone
    """
    weather = WeatherField(grid_shape, cell_size)

    # Base calm conditions
    weather.add_base_conditions(wind_speed=8.0, wave_height=1.2)

    # Add storm zone
    storm = WeatherZone(
        center=storm_center,
        radius=storm_radius,
        wind_speed=35.0,  # Gale force
        wave_height=7.0,  # High seas
        intensity=1.0
    )
    weather.add_weather_zone(storm)

    # Smooth for realistic gradients
    weather.smooth_field(sigma=2.5)

    return weather


def create_multi_zone_scenario(grid_shape: Tuple[int, int],
                                cell_size: float,
                                zones: List[WeatherZone]) -> WeatherField:
    """Create a scenario with multiple weather zones.

    Args:
        grid_shape: Grid dimensions
        cell_size: Cell size in nautical miles
        zones: List of WeatherZone objects

    Returns:
        WeatherField with multiple zones
    """
    weather = WeatherField(grid_shape, cell_size)

    # Base conditions
    weather.add_base_conditions(wind_speed=12.0, wave_height=2.0)

    # Add all zones
    for zone in zones:
        weather.add_weather_zone(zone)

    # Smooth for realism
    weather.smooth_field(sigma=2.0)

    return weather
