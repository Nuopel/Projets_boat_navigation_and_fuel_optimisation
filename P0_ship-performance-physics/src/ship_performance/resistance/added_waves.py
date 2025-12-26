"""Added resistance in waves using simplified empirical method.

This module implements a simplified approach to calculate additional resistance
experienced by ships in waves. The resistance depends on wave height, period,
heading angle, and ship geometry.

The implementation uses an empirical method inspired by Kwon (2008) and similar
simplified approaches suitable for preliminary performance prediction.

References:
    Kwon, Y. J. (2008). "Speed loss due to added resistance in wind and waves."
    The Naval Architect, 14-16.
"""

import math
from typing import Optional

from ..core import OperatingConditions, ShipParameters
from ..utils.constants import GRAVITY


class AddedWaveResistance:
    """Calculate added resistance in waves using simplified empirical method.

    Added resistance in waves is the additional resistance experienced due to
    ship motions and wave reflection. It depends primarily on:
    - Wave height (Hs) - quadratic relationship
    - Wave period (Tp) - affects encounter frequency
    - Wave heading angle - maximum in head seas
    - Ship geometry - L/B ratio, block coefficient

    The resistance shows characteristic behavior:
    - Increases with wave height squared (∝ Hs²)
    - Maximum in head seas (0°), minimum in following seas (180°)
    - Peak when encounter frequency matches ship natural frequency
    - Can equal or exceed calm water resistance in severe seas

    Example:
        >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
        >>> conditions = OperatingConditions(speed=15, wave_height=3, wave_period=10, wave_angle=30)
        >>> calc = AddedWaveResistance()
        >>> R_aw = calc.calculate(ship, conditions)
        >>> print(f"Added wave resistance: {R_aw/1000:.1f} kN")
    """

    def __init__(
        self,
        water_density: float = 1025.0,
        gravity: float = GRAVITY,
        method: str = "simplified_kwon",
    ):
        """Initialize added wave resistance calculator.

        Args:
            water_density: Water density in kg/m³ (default: seawater)
            gravity: Gravitational acceleration in m/s² (default: 9.81)
            method: Calculation method (default: "simplified_kwon")
                Options: "simplified_kwon", "empirical"
        """
        self._water_density = water_density
        self._gravity = gravity
        self._method = method

    @property
    def name(self) -> str:
        """Component name for reporting."""
        return "Added Resistance in Waves"

    def calculate(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> float:
        """Calculate added resistance in waves in Newtons.

        Args:
            ship: Ship parameters
            conditions: Operating conditions including wave parameters

        Returns:
            Added wave resistance in Newtons (N)

        Example:
            >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
            >>> conditions = OperatingConditions(speed=15, wave_height=3, wave_period=10, wave_angle=30)
            >>> calc = AddedWaveResistance()
            >>> R_aw = calc.calculate(ship, conditions)
        """
        # No waves → no added resistance
        if conditions.wave_height == 0:
            return 0.0

        # Calculate based on selected method
        if self._method == "simplified_kwon":
            return self._calculate_simplified_kwon(ship, conditions)
        else:
            return self._calculate_empirical(ship, conditions)

    def _calculate_simplified_kwon(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> float:
        """Calculate using simplified Kwon-inspired method.

        Based on empirical correlations for merchant ships.

        Args:
            ship: Ship parameters
            conditions: Operating conditions

        Returns:
            Added wave resistance in Newtons
        """
        Hs = conditions.wave_height  # Significant wave height (m)
        Tp = conditions.wave_period  # Peak period (s)
        heading_deg = conditions.wave_angle  # Wave heading (degrees, 0 = head seas)

        # Deep water wavelength: λ = g × Tp² / (2π)
        lambda_wave = (self._gravity * Tp**2) / (2 * math.pi)

        # Non-dimensional wave height relative to ship length
        Hs_L = Hs / ship.length

        # Wavelength to ship length ratio
        lambda_L = lambda_wave / ship.length

        # Heading factor (cosine law, max at head seas)
        # 0° = head seas (factor = 1.0)
        # 90° = beam seas (factor = 0.0)
        # 180° = following seas (factor = -1.0, but we take max(0, ...))
        heading_rad = math.radians(heading_deg)
        heading_factor = max(0.0, math.cos(heading_rad))

        # Wavelength encounter factor (peak when λ ≈ L)
        # Ship experiences maximum added resistance when wavelength is similar to ship length
        if lambda_L > 0.1:  # Avoid very short waves
            wavelength_factor = math.exp(-abs(math.log(lambda_L)) / 2)
        else:
            wavelength_factor = 0.1  # Minimal effect for very short waves

        # Block coefficient effect (fuller ships have higher added resistance)
        cb = ship.block_coefficient
        cb_factor = 0.5 + cb  # Range: ~1.05 to 1.35

        # Empirical coefficient (calibrated to match published data)
        # Adjusted to give realistic resistance values
        # Typical added resistance: 20-100 kN in moderate seas (Hs=3m)
        C_aw = 0.015 * cb_factor * wavelength_factor

        # Added resistance formula (simplified Kwon approach)
        # R_aw ≈ C_aw × ρ × g × B² × Hs² / L × heading_factor
        # Dimension check: [kg/m³] × [m/s²] × [m²] × [m²] / [m] = [kg⋅m/s²] = [N] ✓

        resistance = (
            C_aw
            * self._water_density
            * self._gravity
            * ship.beam**2
            * Hs**2
            / ship.length
            * heading_factor
        )

        return max(0.0, resistance)

    def _calculate_empirical(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> float:
        """Calculate using simple empirical formula.

        Very simplified approach based on wave height only.

        Args:
            ship: Ship parameters
            conditions: Operating conditions

        Returns:
            Added wave resistance in Newtons
        """
        Hs = conditions.wave_height
        heading_deg = conditions.wave_angle

        # Heading factor
        heading_factor = max(0.0, math.cos(math.radians(heading_deg)))

        # Simple empirical formula: R_aw ∝ Displacement × Hs² × heading
        # Coefficient adjusted to give realistic values
        displacement_kg = ship.displacement * 1000  # tonnes to kg
        coefficient = 0.001  # Empirical tuning

        resistance = (
            coefficient * displacement_kg * self._gravity * Hs**2 * heading_factor
        )

        return max(0.0, resistance)

    def breakdown(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> dict[str, float]:
        """Get detailed breakdown of added wave resistance components.

        Args:
            ship: Ship parameters
            conditions: Operating conditions

        Returns:
            Dictionary with components:
                - wave_height: Significant wave height Hs (m)
                - wave_period: Peak period Tp (s)
                - wave_angle: Heading angle (degrees)
                - wavelength: Deep water wavelength (m)
                - wavelength_to_length_ratio: λ/L
                - heading_factor: Directional coefficient (0-1)
                - wavelength_factor: Encounter frequency effect (0-1)
                - resistance: Total added wave resistance (N)
        """
        Hs = conditions.wave_height
        Tp = conditions.wave_period
        heading_deg = conditions.wave_angle

        # Calculate wavelength
        lambda_wave = (self._gravity * Tp**2) / (2 * math.pi)
        lambda_L = lambda_wave / ship.length

        # Heading factor
        heading_factor = max(0.0, math.cos(math.radians(heading_deg)))

        # Wavelength factor
        if lambda_L > 0.1:
            wavelength_factor = math.exp(-abs(math.log(lambda_L)) / 2)
        else:
            wavelength_factor = 0.1

        # Total resistance
        resistance = self.calculate(ship, conditions)

        return {
            "wave_height": Hs,
            "wave_period": Tp,
            "wave_angle": heading_deg,
            "wavelength": lambda_wave,
            "wavelength_to_length_ratio": lambda_L,
            "heading_factor": heading_factor,
            "wavelength_factor": wavelength_factor,
            "resistance": resistance,
        }
