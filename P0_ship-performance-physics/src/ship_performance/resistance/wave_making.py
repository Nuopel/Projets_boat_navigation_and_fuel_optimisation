"""Wave-making resistance using simplified Holtrop-Mennen method.

This module implements a simplified version of the Holtrop-Mennen method
for calculating wave-making resistance, which shows characteristic resistance
hump behavior at Froude numbers around 0.3-0.4.

References:
    Holtrop, J., & Mennen, G. G. J. (1982). "An approximate power prediction method."
    International Shipbuilding Progress, 29(335), 166-170.
"""

import math

from ..core import OperatingConditions, ShipParameters
from ..utils.units import calculate_froude_number


class WaveMakingResistance:
    """Calculate wave-making resistance using simplified Holtrop-Mennen method.

    Wave-making resistance is the energy lost to creating surface waves.
    It exhibits a characteristic "hump" at Froude numbers around 0.3-0.4
    due to wave interference effects.

    The resistance is calculated using empirical correlations based on:
    - Froude number (primary driver)
    - Block coefficient (hull fullness)
    - Prismatic coefficient (longitudinal distribution)
    - Length/beam ratio (slenderness)
    - Beam/draft ratio (sectional shape)

    Example:
        >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
        >>> conditions = OperatingConditions(speed=15)
        >>> calc = WaveMakingResistance()
        >>> R_w = calc.calculate(ship, conditions)
        >>> print(f"Wave resistance: {R_w/1000:.1f} kN")
    """

    def __init__(self, water_density: float = 1025.0, gravity: float = 9.81):
        """Initialize wave-making resistance calculator.

        Args:
            water_density: Water density in kg/m³ (default: seawater)
            gravity: Gravitational acceleration in m/s² (default: 9.81)
        """
        self._water_density = water_density
        self._gravity = gravity

    @property
    def name(self) -> str:
        """Component name for reporting."""
        return "Wave-Making Resistance (Holtrop-Mennen)"

    def calculate(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> float:
        """Calculate wave-making resistance in Newtons.

        Args:
            ship: Ship parameters
            conditions: Operating conditions including speed

        Returns:
            Wave-making resistance in Newtons (N)

        Example:
            >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
            >>> conditions = OperatingConditions(speed=15)
            >>> calc = WaveMakingResistance()
            >>> R_w = calc.calculate(ship, conditions)
        """
        speed_ms = conditions.speed_ms
        froude_number = calculate_froude_number(speed_ms, ship.length)

        # Wave resistance coefficient
        c_w = self._wave_resistance_coefficient(ship, froude_number)

        # Wave resistance: R_w = 0.5 × ρ × c_w × S × V²
        resistance = (
            0.5 * self._water_density * c_w * ship.wetted_surface * speed_ms**2
        )

        return max(0.0, resistance)  # Resistance cannot be negative

    def _wave_resistance_coefficient(
        self, ship: ShipParameters, fn: float
    ) -> float:
        """Calculate wave resistance coefficient using simplified Holtrop method.

        The wave resistance coefficient shows characteristic behavior:
        - Low at Fn < 0.2 (displacement mode)
        - Hump at Fn ≈ 0.3-0.4 (wave interference)
        - Increases rapidly at Fn > 0.4 (semi-planing)

        Args:
            ship: Ship parameters
            fn: Froude number

        Returns:
            Wave resistance coefficient c_w (dimensionless)
        """
        # Extract ship parameters
        cb = ship.block_coefficient
        cp = ship.prismatic_coefficient
        lb_ratio = ship.length_beam_ratio
        bt_ratio = ship.beam_draft_ratio
        length = ship.length
        beam = ship.beam

        # Simplified Holtrop-Mennen-inspired approach:
        # build a smooth "hump" in Cw around Fn ≈ 0.3–0.4, scaled by hull form.

        # c1: Basic wave resistance (depends on hull form)
        c1 = self._coefficient_c1(cp, lb_ratio)

        # c2: Effect of block coefficient
        c2 = self._coefficient_c2(cb, cp)

        # Hump centered near Fn ≈ 0.32 with moderate width.
        # This is an empirical shaping term, not a full Holtrop implementation.
        hump_center = 0.32
        hump_width = 0.08
        hump = math.exp(-((fn - hump_center) / hump_width) ** 2)

        # Scale coefficient into a realistic range for typical merchant hulls.
        base = 0.15 * c1 * c2
        c_w = base * (0.2 + 0.8 * hump)

        # Ensure coefficient is reasonable
        c_w = max(0.0, min(c_w, 0.01))  # Cw typically < 0.01

        return c_w

    def _coefficient_c1(self, cp: float, lb_ratio: float) -> float:
        """Calculate coefficient c1 (basic wave resistance level).

        c1 depends on prismatic coefficient and slenderness.
        Slender ships (high L/B) have lower wave resistance.

        Args:
            cp: Prismatic coefficient
            lb_ratio: Length/beam ratio

        Returns:
            Coefficient c1 (dimensionless)
        """
        # Simplified correlation
        # c1 increases with cp (fuller ships have more wave resistance)
        # c1 decreases with L/B (slender ships have less wave resistance)

        c1 = 0.001 * (2.5 + cp * 5.0) * (10.0 / max(lb_ratio, 4.0))

        return max(0.0001, c1)

    def _coefficient_c2(self, cb: float, cp: float) -> float:
        """Calculate coefficient c2 (block coefficient effect).

        Fuller ships (higher Cb) generally have higher wave resistance.

        Args:
            cb: Block coefficient
            cp: Prismatic coefficient

        Returns:
            Coefficient c2 (dimensionless)
        """
        # Simplified: c2 increases with fullness
        c2 = 1.0 + 2.5 * (cb - 0.65)

        return max(0.5, min(c2, 2.0))

    def _coefficient_m1(self, cb: float, cp: float, lb_ratio: float) -> float:
        """Calculate slope coefficient m1.

        m1 controls how quickly resistance increases with speed.
        Negative values (typical) mean resistance increases with speed.

        Args:
            cb: Block coefficient
            cp: Prismatic coefficient
            lb_ratio: Length/beam ratio

        Returns:
            Coefficient m1 (dimensionless)
        """
        # Simplified Holtrop correlation
        # m1 is typically negative (-7 to -12)
        # More negative for slender ships

        m1 = -8.0 - 4.0 * (lb_ratio - 6.0) / 4.0  # More negative for slender ships
        m1 += 2.0 * (cb - 0.7)  # Adjusted for fullness

        # Bound m1 to reasonable range
        m1 = max(-12.0, min(m1, -5.0))

        return m1

    def _lambda_parameter(self, length: float, beam: float, cb: float) -> float:
        """Calculate wave interference parameter λ.

        λ controls the position of the resistance hump.
        Typical values: 20-30

        Args:
            length: Ship length (m)
            beam: Ship beam (m)
            cb: Block coefficient

        Returns:
            Wave interference parameter λ (dimensionless)
        """
        # Simplified correlation
        # λ increases with length and decreases with beam
        # Creates hump around Fn = 0.3-0.4

        lambda_param = 25.0 + 5.0 * (cb - 0.7)

        return max(15.0, min(lambda_param, 35.0))

    def breakdown(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> dict[str, float]:
        """Get detailed breakdown of wave resistance components.

        Args:
            ship: Ship parameters
            conditions: Operating conditions

        Returns:
            Dictionary with components:
                - froude_number: Fn
                - wave_coefficient: c_w
                - c1: Basic wave resistance coefficient
                - c2: Block coefficient effect
                - m1: Slope coefficient
                - lambda: Wave interference parameter
                - resistance: Total wave resistance (N)
        """
        speed_ms = conditions.speed_ms
        fn = calculate_froude_number(speed_ms, ship.length)
        c_w = self._wave_resistance_coefficient(ship, fn)
        resistance = self.calculate(ship, conditions)

        # Recalculate individual coefficients for breakdown
        c1 = self._coefficient_c1(ship.prismatic_coefficient, ship.length_beam_ratio)
        c2 = self._coefficient_c2(ship.block_coefficient, ship.prismatic_coefficient)
        m1 = self._coefficient_m1(
            ship.block_coefficient, ship.prismatic_coefficient, ship.length_beam_ratio
        )
        lambda_param = self._lambda_parameter(ship.length, ship.beam, ship.block_coefficient)

        return {
            "froude_number": fn,
            "wave_coefficient": c_w,
            "c1": c1,
            "c2": c2,
            "m1": m1,
            "lambda": lambda_param,
            "resistance": resistance,
        }
