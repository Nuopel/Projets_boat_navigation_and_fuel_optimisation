"""Interfaces for resistance calculation components.

This module defines protocols for resistance calculators, enabling
polymorphic resistance calculation through composition.
"""

from typing import Protocol, runtime_checkable

from ..core import OperatingConditions, ShipParameters


@runtime_checkable
class ResistanceCalculator(Protocol):
    """Protocol for resistance calculation components.

    All resistance calculators must implement this interface to be compatible
    with the ship performance model.

    This uses Python's Protocol (PEP 544) for structural subtyping, allowing
    any class with matching methods to be used without explicit inheritance.
    """

    def calculate(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> float:
        """Calculate resistance component in Newtons.

        Args:
            ship: Ship parameters including dimensions and coefficients
            conditions: Operating conditions including speed and environment

        Returns:
            Resistance force in Newtons (N)

        Raises:
            ValueError: If inputs are invalid or calculation cannot proceed

        Example:
            >>> calculator = SomeResistanceCalculator()
            >>> ship = ShipParameters(length=150, beam=25, draft=8, ...)
            >>> conditions = OperatingConditions(speed=15)
            >>> resistance = calculator.calculate(ship, conditions)
            >>> print(f"Resistance: {resistance:.0f} N")
        """
        ...

    @property
    def name(self) -> str:
        """Component name for reporting and identification.

        Returns:
            Human-readable name of the resistance component

        Example:
            >>> calculator = CalmWaterResistance()
            >>> calculator.name
            'Calm Water Resistance'
        """
        ...


@runtime_checkable
class ResistanceBreakdown(Protocol):
    """Protocol for resistance calculators that provide component breakdown.

    This is an optional extended interface for calculators that can provide
    detailed breakdown of resistance components.
    """

    def breakdown(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> dict[str, float]:
        """Calculate resistance breakdown by sub-components.

        Args:
            ship: Ship parameters
            conditions: Operating conditions

        Returns:
            Dictionary mapping component names to resistance values (N)

        Example:
            >>> calculator = CalmWaterResistance()
            >>> breakdown = calculator.breakdown(ship, conditions)
            >>> print(breakdown)
            {'friction': 125000.0, 'wave_making': 85000.0, 'appendage': 5000.0}
        """
        ...
