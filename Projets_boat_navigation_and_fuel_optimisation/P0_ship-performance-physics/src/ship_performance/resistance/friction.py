"""Frictional resistance calculation using ITTC 1957 correlation line.

This module implements the ITTC (International Towing Tank Conference) 1957
friction line formula for calculating the frictional resistance of ship hulls.

References:
    ITTC (1957). "Report of Resistance Committee."
    ITTC Recommended Procedures: https://ittc.info/media/2021/75-02-02-02.pdf
"""

import math
from typing import Optional

from ..core import OperatingConditions, ShipParameters
from ..utils.constants import SEAWATER_KINEMATIC_VISCOSITY
from ..utils.units import calculate_reynolds_number


class FrictionResistance:
    """Calculate frictional resistance using ITTC 1957 correlation line.

    The frictional resistance is calculated using the ITTC 1957 formula:
        Cf = 0.075 / (log10(Re) - 2)²
        Rf = 0.5 × ρ × Cf × S × V²

    where:
        - Cf: Friction coefficient (dimensionless)
        - Re: Reynolds number
        - ρ: Water density (kg/m³)
        - S: Wetted surface area (m²)
        - V: Speed (m/s)

    This formula includes a form factor correction for ship-shaped hulls
    and is valid for Reynolds numbers typical of ship flows (Re > 10⁶).

    Example:
        >>> from ship_performance.core import ShipParameters, OperatingConditions
        >>> from ship_performance.resistance import FrictionResistance
        >>>
        >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
        >>> conditions = OperatingConditions(speed=15)  # knots
        >>> calc = FrictionResistance()
        >>> R_f = calc.calculate(ship, conditions)
        >>> print(f"Friction resistance: {R_f/1000:.1f} kN")
    """

    def __init__(
        self,
        form_factor: Optional[float] = None,
        water_density: float = 1025.0,
        kinematic_viscosity: Optional[float] = None,
    ):
        """Initialize friction resistance calculator.

        Args:
            form_factor: Form factor k (1 + k multiplies Cf). If None, estimated.
            water_density: Water density in kg/m³ (default: seawater 1025)
            kinematic_viscosity: Kinematic viscosity in m²/s (default: 15°C seawater)
        """
        self._form_factor = form_factor
        self._water_density = water_density
        self._kinematic_viscosity = kinematic_viscosity or SEAWATER_KINEMATIC_VISCOSITY

    @property
    def name(self) -> str:
        """Component name for reporting."""
        return "Frictional Resistance (ITTC 1957)"

    def calculate(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> float:
        """Calculate frictional resistance in Newtons.

        Args:
            ship: Ship parameters including wetted surface
            conditions: Operating conditions including speed

        Returns:
            Frictional resistance in Newtons (N)

        Raises:
            ValueError: If Reynolds number is too low (< 10⁵)

        Example:
            >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
            >>> conditions = OperatingConditions(speed=15)
            >>> calc = FrictionResistance()
            >>> R_f = calc.calculate(ship, conditions)
        """
        speed_ms = conditions.speed_ms

        # Handle zero or very low speed
        if speed_ms < 0.01:  # Below ~0.02 knots
            return 0.0

        reynolds_number = calculate_reynolds_number(
            speed_ms, ship.length, self._kinematic_viscosity
        )

        # ITTC 1957 correlation line
        cf = self._ittc_friction_coefficient(reynolds_number)

        # Form factor correction (1 + k)
        form_factor = self._form_factor or self._estimate_form_factor(ship)
        cf_ship = (1 + form_factor) * cf

        # Frictional resistance: Rf = 0.5 * ρ * Cf * S * V²
        resistance = (
            0.5
            * self._water_density
            * cf_ship
            * ship.wetted_surface
            * speed_ms**2
        )

        return resistance

    def _ittc_friction_coefficient(self, reynolds_number: float) -> float:
        """Calculate ITTC 1957 friction coefficient.

        Cf = 0.075 / (log10(Re) - 2)²

        Args:
            reynolds_number: Reynolds number (dimensionless)

        Returns:
            Friction coefficient Cf (dimensionless)

        Raises:
            ValueError: If Reynolds number is too low
        """
        if reynolds_number < 1e5:
            raise ValueError(
                f"Reynolds number too low for ITTC formula: {reynolds_number:.2e}. "
                f"Must be > 1e5 for turbulent flow."
            )

        log_re = math.log10(reynolds_number)
        cf = 0.075 / ((log_re - 2.0) ** 2)

        return cf

    def _estimate_form_factor(self, ship: ShipParameters) -> float:
        """Estimate form factor k using simplified Holtrop method.

        Form factor accounts for pressure resistance due to 3D flow around hull.
        Typical values: 0.05-0.25 (fuller hulls have higher k).

        Uses correlation:
            k ≈ f(L/B, B/T, Cb, Cp)

        Args:
            ship: Ship parameters

        Returns:
            Estimated form factor k (dimensionless)
        """
        # Simplified Holtrop form factor estimation
        lb_ratio = ship.length_beam_ratio
        bt_ratio = ship.beam_draft_ratio
        cb = ship.block_coefficient
        cp = ship.prismatic_coefficient

        # Holtrop correlation (simplified)
        # k ≈ c × (B/L)^a × (T/L)^b × Cp^c
        # Typical range: 0.05-0.25

        # Simplified empirical formula based on fullness
        # Fuller ships (higher Cb) have higher form factors
        k = 0.095 + 0.35 * (cb - 0.65)

        # Adjust for slenderness
        if lb_ratio > 8:
            k *= 0.9  # Slender ships have lower form factors
        elif lb_ratio < 6:
            k *= 1.1  # Full ships have higher form factors

        # Ensure reasonable bounds
        k = max(0.05, min(0.30, k))

        return k

    def breakdown(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> dict[str, float]:
        """Get detailed breakdown of friction resistance components.

        Args:
            ship: Ship parameters
            conditions: Operating conditions

        Returns:
            Dictionary with components:
                - reynolds_number: Re
                - friction_coefficient: Cf (ITTC)
                - form_factor: k
                - effective_cf: (1+k)×Cf
                - resistance: Total friction resistance (N)
        """
        speed_ms = conditions.speed_ms
        re = calculate_reynolds_number(speed_ms, ship.length, self._kinematic_viscosity)
        cf = self._ittc_friction_coefficient(re)
        k = self._form_factor or self._estimate_form_factor(ship)
        cf_eff = (1 + k) * cf
        resistance = self.calculate(ship, conditions)

        return {
            "reynolds_number": re,
            "friction_coefficient": cf,
            "form_factor": k,
            "effective_cf": cf_eff,
            "resistance": resistance,
        }
