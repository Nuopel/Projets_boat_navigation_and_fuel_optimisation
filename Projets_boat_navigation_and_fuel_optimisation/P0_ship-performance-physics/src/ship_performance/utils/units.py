"""Unit conversion utilities.

This module provides functions for converting between different unit systems
commonly used in ship performance calculations.

All conversions are bidirectional and validated for reasonable ranges.
"""

import math
from typing import Union

from .constants import (
    DEGREES_TO_RADIANS,
    GRAVITY,
    KNOTS_TO_MS,
    MS_TO_KNOTS,
    RADIANS_TO_DEGREES,
    SEAWATER_KINEMATIC_VISCOSITY,
)


def knots_to_ms(speed_knots: float) -> float:
    """Convert speed from knots to meters per second.

    Args:
        speed_knots: Speed in knots

    Returns:
        Speed in meters per second

    Raises:
        ValueError: If speed is negative

    Example:
        >>> knots_to_ms(15.0)
        7.716666666666667
        >>> knots_to_ms(0.0)
        0.0
    """
    if speed_knots < 0:
        raise ValueError(f"Speed cannot be negative, got {speed_knots} knots")
    return speed_knots * KNOTS_TO_MS


def ms_to_knots(speed_ms: float) -> float:
    """Convert speed from meters per second to knots.

    Args:
        speed_ms: Speed in meters per second

    Returns:
        Speed in knots

    Raises:
        ValueError: If speed is negative

    Example:
        >>> ms_to_knots(7.716666666666667)
        15.0
        >>> ms_to_knots(0.0)
        0.0
    """
    if speed_ms < 0:
        raise ValueError(f"Speed cannot be negative, got {speed_ms} m/s")
    return speed_ms * MS_TO_KNOTS


def degrees_to_radians(degrees: float) -> float:
    """Convert angle from degrees to radians.

    Args:
        degrees: Angle in degrees

    Returns:
        Angle in radians

    Example:
        >>> degrees_to_radians(180.0)
        3.141592653589793
        >>> degrees_to_radians(90.0)
        1.5707963267948966
    """
    return degrees * DEGREES_TO_RADIANS


def radians_to_degrees(radians: float) -> float:
    """Convert angle from radians to degrees.

    Args:
        radians: Angle in radians

    Returns:
        Angle in degrees

    Example:
        >>> radians_to_degrees(math.pi)
        180.0
        >>> radians_to_degrees(math.pi / 2)
        90.0
    """
    return radians * RADIANS_TO_DEGREES


def calculate_froude_number(speed_ms: float, length: float) -> float:
    """Calculate Froude number for a ship.

    The Froude number is a dimensionless parameter characterizing the
    relationship between ship speed and wave-making resistance.

    Fn = V / sqrt(g * L)

    Typical ranges:
    - Fn < 0.3: Low speed (displacement mode)
    - 0.3 < Fn < 0.4: Transition (resistance hump)
    - Fn > 0.4: High speed (planing possible for appropriate hulls)

    Args:
        speed_ms: Ship speed in m/s
        length: Ship length (typically LWL or LOA) in meters

    Returns:
        Froude number (dimensionless)

    Raises:
        ValueError: If speed or length is negative or length is zero

    Example:
        >>> calculate_froude_number(7.72, 150.0)
        0.20252891832373247
        >>> calculate_froude_number(10.0, 100.0)
        0.31943828249997
    """
    if speed_ms < 0:
        raise ValueError(f"Speed cannot be negative, got {speed_ms} m/s")
    if length <= 0:
        raise ValueError(f"Length must be positive, got {length} m")

    return speed_ms / math.sqrt(GRAVITY * length)


def calculate_reynolds_number(
    speed_ms: float,
    length: float,
    kinematic_viscosity: float = SEAWATER_KINEMATIC_VISCOSITY,
) -> float:
    """Calculate Reynolds number for ship flow.

    The Reynolds number characterizes the flow regime (laminar vs turbulent).
    Ship flows are always turbulent (Re > 5e5).

    Re = V * L / ν

    Typical ranges for ships:
    - Small vessels: Re ≈ 10^7
    - Medium vessels: Re ≈ 10^8 - 10^9
    - Large vessels: Re > 10^9

    Args:
        speed_ms: Ship speed in m/s
        length: Ship length in meters
        kinematic_viscosity: Kinematic viscosity in m²/s
            (default: seawater at 15°C)

    Returns:
        Reynolds number (dimensionless)

    Raises:
        ValueError: If any parameter is negative, or length/viscosity is zero

    Example:
        >>> calculate_reynolds_number(7.72, 150.0)
        973109243.6974789
        >>> calculate_reynolds_number(10.0, 100.0)
        840336134.4537815
    """
    if speed_ms < 0:
        raise ValueError(f"Speed cannot be negative, got {speed_ms} m/s")
    if length <= 0:
        raise ValueError(f"Length must be positive, got {length} m")
    if kinematic_viscosity <= 0:
        raise ValueError(
            f"Kinematic viscosity must be positive, got {kinematic_viscosity} m²/s"
        )

    return (speed_ms * length) / kinematic_viscosity


def normalize_angle(angle_degrees: float) -> float:
    """Normalize angle to range [0, 360) degrees.

    Args:
        angle_degrees: Angle in degrees

    Returns:
        Normalized angle in range [0, 360)

    Example:
        >>> normalize_angle(450.0)
        90.0
        >>> normalize_angle(-45.0)
        315.0
        >>> normalize_angle(180.0)
        180.0
    """
    return angle_degrees % 360.0


def relative_angle(
    heading_degrees: float, direction_degrees: float
) -> float:
    """Calculate relative angle between heading and direction.

    Computes the smallest angle between two directions, returned in range [0, 180].
    Useful for wind/wave angles relative to ship heading.

    Args:
        heading_degrees: Ship heading in degrees (0-360)
        direction_degrees: Wind/wave direction in degrees (0-360)

    Returns:
        Relative angle in degrees (0-180)

    Example:
        >>> relative_angle(0.0, 45.0)
        45.0
        >>> relative_angle(0.0, 315.0)
        45.0
        >>> relative_angle(90.0, 270.0)
        180.0
    """
    diff = abs(normalize_angle(direction_degrees) - normalize_angle(heading_degrees))
    return min(diff, 360.0 - diff)
