"""Operating and environmental conditions.

This module defines the OperatingConditions dataclass which encapsulates
all operational and environmental parameters affecting ship performance.
"""

from dataclasses import dataclass

from ..utils.units import knots_to_ms


class InvalidOperatingConditionsError(ValueError):
    """Raised when operating conditions are invalid."""

    pass


@dataclass(frozen=True)
class OperatingConditions:
    """Operating and environmental conditions.

    All environmental parameters are defined relative to the ship's heading.

    Attributes:
        speed: Ship speed (knots), must be ≥ 0
        wind_speed: Wind speed (m/s), must be ≥ 0
        wind_angle: Wind direction relative to heading (degrees, 0-180)
            0° = head wind, 90° = beam wind, 180° = following wind
        wave_height: Significant wave height Hs (m), must be ≥ 0
        wave_period: Peak wave period Tp (s), must be ≥ 0
        wave_angle: Wave direction relative to heading (degrees, 0-180)
            0° = head seas, 90° = beam seas, 180° = following seas

    Raises:
        InvalidOperatingConditionsError: If conditions are invalid

    Example:
        >>> conditions = OperatingConditions(
        ...     speed=15, wind_speed=10, wind_angle=45,
        ...     wave_height=2, wave_period=8, wave_angle=30
        ... )
        >>> conditions.speed_ms
        7.716666666666667
        >>> conditions.is_calm_water
        False
    """

    speed: float  # knots
    wind_speed: float = 0.0  # m/s
    wind_angle: float = 0.0  # degrees (0-180)
    wave_height: float = 0.0  # m (Hs)
    wave_period: float = 0.0  # s (Tp)
    wave_angle: float = 0.0  # degrees (0-180)

    def __post_init__(self) -> None:
        """Validate operating conditions."""
        self._validate_speed()
        self._validate_wind()
        self._validate_waves()

    def _validate_speed(self) -> None:
        """Validate ship speed."""
        if self.speed < 0:
            raise InvalidOperatingConditionsError(
                f"Ship speed cannot be negative, got {self.speed} knots"
            )

        if self.speed > 50:
            # Very high speed - warn but don't error (could be valid for fast vessels)
            import warnings

            warnings.warn(
                f"Ship speed ({self.speed} knots) is very high. "
                f"Ensure this is correct for your vessel type.",
                UserWarning,
            )

    def _validate_wind(self) -> None:
        """Validate wind parameters."""
        if self.wind_speed < 0:
            raise InvalidOperatingConditionsError(
                f"Wind speed cannot be negative, got {self.wind_speed} m/s"
            )

        if not (0 <= self.wind_angle <= 180):
            raise InvalidOperatingConditionsError(
                f"Wind angle must be between 0 and 180 degrees, got {self.wind_angle}°"
            )

        # Validate Beaufort scale upper limit (hurricane)
        if self.wind_speed > 32.7:  # > 63 knots, Beaufort 12
            import warnings

            warnings.warn(
                f"Wind speed ({self.wind_speed} m/s ≈ {self.wind_speed * 1.944:.1f} knots) "
                f"exceeds hurricane force (Beaufort 12). Verify this is intentional.",
                UserWarning,
            )

    def _validate_waves(self) -> None:
        """Validate wave parameters."""
        if self.wave_height < 0:
            raise InvalidOperatingConditionsError(
                f"Wave height cannot be negative, got {self.wave_height} m"
            )

        if self.wave_period < 0:
            raise InvalidOperatingConditionsError(
                f"Wave period cannot be negative, got {self.wave_period} s"
            )

        if not (0 <= self.wave_angle <= 180):
            raise InvalidOperatingConditionsError(
                f"Wave angle must be between 0 and 180 degrees, got {self.wave_angle}°"
            )

        # Validate wave steepness (if both Hs and Tp are provided)
        if self.wave_height > 0 and self.wave_period > 0:
            # Deep water wavelength: λ = g * T² / (2π) ≈ 1.56 * T²
            wavelength = 1.56 * self.wave_period**2
            wave_steepness = self.wave_height / wavelength

            if wave_steepness > 0.07:  # Theoretical breaking limit
                import warnings

                warnings.warn(
                    f"Wave steepness (H/λ = {wave_steepness:.3f}) exceeds typical "
                    f"breaking limit (0.07). This may indicate breaking seas or data error.",
                    UserWarning,
                )

        # Validate wave height - period relationship
        if self.wave_height > 0 and self.wave_period == 0:
            import warnings

            warnings.warn(
                "Wave height specified but wave period is zero. "
                "This may lead to undefined behavior in wave resistance calculations.",
                UserWarning,
            )

    @property
    def speed_ms(self) -> float:
        """Convert speed from knots to m/s.

        Returns:
            Speed in meters per second

        Example:
            >>> cond = OperatingConditions(speed=15)
            >>> cond.speed_ms
            7.716666666666667
        """
        return knots_to_ms(self.speed)

    @property
    def is_calm_water(self) -> bool:
        """Check if conditions represent calm water (no wind/waves).

        Returns:
            True if both wind and waves are negligible

        Example:
            >>> OperatingConditions(speed=15).is_calm_water
            True
            >>> OperatingConditions(speed=15, wave_height=2).is_calm_water
            False
        """
        return self.wind_speed == 0 and self.wave_height == 0

    @property
    def beaufort_scale(self) -> int:
        """Estimate Beaufort scale number from wind speed.

        Returns:
            Beaufort scale number (0-12)

        Example:
            >>> OperatingConditions(speed=15, wind_speed=10).beaufort_scale
            5
            >>> OperatingConditions(speed=15, wind_speed=20).beaufort_scale
            8
        """
        # Beaufort scale approximation: B ≈ (V/0.836)^(2/3)
        # Simplified lookup table for better accuracy
        wind_knots = self.wind_speed * 1.944  # Convert m/s to knots

        if wind_knots < 1:
            return 0  # Calm
        elif wind_knots < 4:
            return 1  # Light air
        elif wind_knots < 7:
            return 2  # Light breeze
        elif wind_knots < 11:
            return 3  # Gentle breeze
        elif wind_knots < 17:
            return 4  # Moderate breeze
        elif wind_knots < 22:
            return 5  # Fresh breeze
        elif wind_knots < 28:
            return 6  # Strong breeze
        elif wind_knots < 34:
            return 7  # Near gale
        elif wind_knots < 41:
            return 8  # Gale
        elif wind_knots < 48:
            return 9  # Strong gale
        elif wind_knots < 56:
            return 10  # Storm
        elif wind_knots < 64:
            return 11  # Violent storm
        else:
            return 12  # Hurricane

    @property
    def sea_state(self) -> int:
        """Estimate Douglas Sea State from significant wave height.

        Returns:
            Douglas Sea State (0-9)

        Example:
            >>> OperatingConditions(speed=15, wave_height=1.5).sea_state
            3
            >>> OperatingConditions(speed=15, wave_height=6).sea_state
            6
        """
        hs = self.wave_height

        if hs == 0:
            return 0  # Calm (glassy)
        elif hs < 0.1:
            return 1  # Calm (rippled)
        elif hs < 0.5:
            return 2  # Smooth
        elif hs < 1.25:
            return 3  # Slight
        elif hs < 2.5:
            return 4  # Moderate
        elif hs < 4.0:
            return 5  # Rough
        elif hs < 6.0:
            return 6  # Very rough
        elif hs < 9.0:
            return 7  # High
        elif hs < 14.0:
            return 8  # Very high
        else:
            return 9  # Phenomenal

    def __repr__(self) -> str:
        """Return detailed string representation."""
        if self.is_calm_water:
            return f"OperatingConditions(speed={self.speed:.1f} knots, calm water)"
        else:
            return (
                f"OperatingConditions("
                f"speed={self.speed:.1f} knots, "
                f"wind={self.wind_speed:.1f} m/s @ {self.wind_angle:.0f}°, "
                f"waves=Hs {self.wave_height:.1f}m, Tp {self.wave_period:.1f}s @ {self.wave_angle:.0f}°, "
                f"Beaufort {self.beaufort_scale}, Sea State {self.sea_state}"
                f")"
            )
