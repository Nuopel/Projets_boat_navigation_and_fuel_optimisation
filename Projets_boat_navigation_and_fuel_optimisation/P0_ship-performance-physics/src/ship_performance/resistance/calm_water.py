"""Composite calm water resistance calculator.

This module combines frictional and wave-making resistance to provide
total calm water resistance for ship performance prediction.
"""

from typing import Optional

from ..core import OperatingConditions, ShipParameters
from .friction import FrictionResistance
from .wave_making import WaveMakingResistance


class CalmWaterResistance:
    """Calculate total calm water resistance (friction + wave-making).

    This class composes frictional and wave-making resistance calculators
    to provide the total resistance in calm water conditions (no wind/waves).

    The total resistance is:
        R_total = R_friction + R_wave

    where:
        - R_friction: ITTC 1957 frictional resistance with form factor
        - R_wave: Holtrop-Mennen wave-making resistance

    This implements the ResistanceCalculator protocol from interfaces.

    Example:
        >>> from ship_performance.core import ShipParameters, OperatingConditions
        >>> from ship_performance.resistance import CalmWaterResistance
        >>>
        >>> ship = ShipParameters(
        ...     length=150, beam=25, draft=8,
        ...     displacement=15000, block_coefficient=0.7
        ... )
        >>> conditions = OperatingConditions(speed=15)
        >>>
        >>> calc = CalmWaterResistance()
        >>> R_total = calc.calculate(ship, conditions)
        >>> print(f"Total resistance: {R_total/1000:.1f} kN")
        >>>
        >>> # Get detailed breakdown
        >>> breakdown = calc.breakdown(ship, conditions)
        >>> print(f"Friction: {breakdown['friction']/1000:.1f} kN")
        >>> print(f"Wave-making: {breakdown['wave_making']/1000:.1f} kN")
    """

    def __init__(
        self,
        friction_calculator: Optional[FrictionResistance] = None,
        wave_calculator: Optional[WaveMakingResistance] = None,
    ):
        """Initialize calm water resistance calculator.

        Args:
            friction_calculator: Friction resistance calculator (default: FrictionResistance())
            wave_calculator: Wave-making resistance calculator (default: WaveMakingResistance())
        """
        self.friction = friction_calculator or FrictionResistance()
        self.wave_making = wave_calculator or WaveMakingResistance()

    @property
    def name(self) -> str:
        """Component name for reporting."""
        return "Calm Water Resistance (Total)"

    def calculate(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> float:
        """Calculate total calm water resistance in Newtons.

        Args:
            ship: Ship parameters including wetted surface
            conditions: Operating conditions (speed, no wind/waves)

        Returns:
            Total calm water resistance in Newtons (N)

        Example:
            >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
            >>> conditions = OperatingConditions(speed=15)
            >>> calc = CalmWaterResistance()
            >>> R_total = calc.calculate(ship, conditions)
        """
        R_friction = self.friction.calculate(ship, conditions)
        R_wave = self.wave_making.calculate(ship, conditions)

        return R_friction + R_wave

    def breakdown(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> dict[str, float]:
        """Get detailed breakdown of resistance components.

        Args:
            ship: Ship parameters
            conditions: Operating conditions

        Returns:
            Dictionary with resistance components:
                - friction: Frictional resistance (N)
                - wave_making: Wave-making resistance (N)
                - total: Total resistance (N)
                - friction_percent: Percentage of total
                - wave_percent: Percentage of total

        Example:
            >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
            >>> conditions = OperatingConditions(speed=15)
            >>> calc = CalmWaterResistance()
            >>> breakdown = calc.breakdown(ship, conditions)
            >>> for component, value in breakdown.items():
            ...     print(f"{component}: {value}")
        """
        R_friction = self.friction.calculate(ship, conditions)
        R_wave = self.wave_making.calculate(ship, conditions)
        R_total = R_friction + R_wave

        # Calculate percentages
        if R_total > 0:
            friction_percent = 100 * R_friction / R_total
            wave_percent = 100 * R_wave / R_total
        else:
            friction_percent = 0.0
            wave_percent = 0.0

        return {
            "friction": R_friction,
            "wave_making": R_wave,
            "total": R_total,
            "friction_percent": friction_percent,
            "wave_percent": wave_percent,
        }

    def detailed_breakdown(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> dict[str, dict]:
        """Get very detailed breakdown including sub-components.

        Args:
            ship: Ship parameters
            conditions: Operating conditions

        Returns:
            Nested dictionary with detailed information:
                - friction: Dict with Re, Cf, form factor, etc.
                - wave_making: Dict with Fn, c_w, coefficients, etc.
                - total: Summary dict

        Example:
            >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
            >>> conditions = OperatingConditions(speed=15)
            >>> calc = CalmWaterResistance()
            >>> details = calc.detailed_breakdown(ship, conditions)
            >>> print(f"Reynolds number: {details['friction']['reynolds_number']:.2e}")
            >>> print(f"Froude number: {details['wave_making']['froude_number']:.3f}")
        """
        friction_breakdown = self.friction.breakdown(ship, conditions)
        wave_breakdown = self.wave_making.breakdown(ship, conditions)
        basic_breakdown = self.breakdown(ship, conditions)

        return {
            "friction": friction_breakdown,
            "wave_making": wave_breakdown,
            "total": basic_breakdown,
        }
