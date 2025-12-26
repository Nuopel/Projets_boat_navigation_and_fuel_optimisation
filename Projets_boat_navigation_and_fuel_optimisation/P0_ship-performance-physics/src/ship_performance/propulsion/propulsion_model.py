"""Propulsion model for power requirements calculation.

This module implements the propulsion chain from effective power (resistance × speed)
to brake power (engine output), accounting for propeller and shaft efficiencies.

Power Chain:
  P_effective (towing power) → P_delivered (to propeller) → P_brake (from engine)

Efficiencies:
  η_propeller: Propeller efficiency (0.50-0.75 typical)
  η_shaft: Shaft/gearbox efficiency (0.96-0.99 typical)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PowerBreakdown:
    """Complete power breakdown through propulsion chain.

    Attributes:
        effective_power: Power to overcome resistance (kW)
        delivered_power: Power delivered to propeller (kW)
        brake_power: Power from main engine (kW)
        propeller_efficiency: Propeller efficiency (η_P)
        shaft_efficiency: Shaft/transmission efficiency (η_S)
        overall_efficiency: Overall propulsive efficiency (η_D)
        speed_ms: Ship speed (m/s)
        total_resistance: Total resistance (N)
    """

    effective_power: float
    delivered_power: float
    brake_power: float
    propeller_efficiency: float
    shaft_efficiency: float
    overall_efficiency: float
    speed_ms: float
    total_resistance: float


class PropulsionModel:
    """Propulsion model for calculating power requirements.

    This class implements the propulsion chain from effective power to brake power,
    accounting for propeller and shaft efficiencies.

    The model uses simplified assumptions:
    - Constant propeller efficiency (can be speed-dependent)
    - Constant shaft efficiency
    - No cavitation or off-design effects

    Example:
        >>> propulsion = PropulsionModel(propeller_efficiency=0.65, shaft_efficiency=0.98)
        >>> breakdown = propulsion.calculate_power(resistance=150000, speed_ms=7.72)
        >>> print(f"Brake power: {breakdown.brake_power:.1f} kW")
    """

    def __init__(
        self,
        propeller_efficiency: float = 0.65,
        shaft_efficiency: float = 0.98,
    ):
        """Initialize propulsion model.

        Args:
            propeller_efficiency: Propeller open-water efficiency η_P (0-1)
                Typical values:
                - Slow merchant ships: 0.60-0.70
                - Fast ships: 0.55-0.65
                - Very slow (bulk): 0.65-0.75
            shaft_efficiency: Shaft and bearing efficiency η_S (0-1)
                Typical values:
                - Direct drive: 0.98-0.99
                - Geared drive: 0.96-0.98

        Raises:
            ValueError: If efficiencies are not in valid range (0-1)
        """
        if not 0 < propeller_efficiency <= 1.0:
            raise ValueError(
                f"Propeller efficiency must be in (0, 1], got {propeller_efficiency}"
            )
        if not 0 < shaft_efficiency <= 1.0:
            raise ValueError(
                f"Shaft efficiency must be in (0, 1], got {shaft_efficiency}"
            )

        self._eta_propeller = propeller_efficiency
        self._eta_shaft = shaft_efficiency
        self._eta_overall = propeller_efficiency * shaft_efficiency

    @property
    def propeller_efficiency(self) -> float:
        """Get propeller efficiency."""
        return self._eta_propeller

    @property
    def shaft_efficiency(self) -> float:
        """Get shaft efficiency."""
        return self._eta_shaft

    @property
    def overall_efficiency(self) -> float:
        """Get overall propulsive efficiency (η_D = η_P × η_S)."""
        return self._eta_overall

    def calculate_power(self, total_resistance: float, speed_ms: float) -> PowerBreakdown:
        """Calculate complete power breakdown.

        Args:
            total_resistance: Total ship resistance (N)
            speed_ms: Ship speed (m/s)

        Returns:
            PowerBreakdown with all power components

        Example:
            >>> propulsion = PropulsionModel()
            >>> breakdown = propulsion.calculate_power(resistance=150000, speed_ms=7.72)
            >>> print(f"Effective: {breakdown.effective_power:.1f} kW")
            >>> print(f"Brake: {breakdown.brake_power:.1f} kW")
        """
        # Effective power (power to overcome resistance)
        # P_E = R × V
        P_effective = (total_resistance * speed_ms) / 1000  # Convert W to kW

        # Delivered power (power delivered to propeller)
        # P_D = P_E / η_P
        P_delivered = P_effective / self._eta_propeller

        # Brake power (power from main engine)
        # P_B = P_D / η_S = P_E / (η_P × η_S) = P_E / η_D
        P_brake = P_delivered / self._eta_shaft

        return PowerBreakdown(
            effective_power=P_effective,
            delivered_power=P_delivered,
            brake_power=P_brake,
            propeller_efficiency=self._eta_propeller,
            shaft_efficiency=self._eta_shaft,
            overall_efficiency=self._eta_overall,
            speed_ms=speed_ms,
            total_resistance=total_resistance,
        )

    def calculate_effective_power(self, total_resistance: float, speed_ms: float) -> float:
        """Calculate effective power only (convenience method).

        Args:
            total_resistance: Total ship resistance (N)
            speed_ms: Ship speed (m/s)

        Returns:
            Effective power in kW

        Example:
            >>> propulsion = PropulsionModel()
            >>> P_E = propulsion.calculate_effective_power(150000, 7.72)
        """
        return (total_resistance * speed_ms) / 1000

    def calculate_brake_power(self, total_resistance: float, speed_ms: float) -> float:
        """Calculate brake power only (convenience method).

        Args:
            total_resistance: Total ship resistance (N)
            speed_ms: Ship speed (m/s)

        Returns:
            Brake power in kW

        Example:
            >>> propulsion = PropulsionModel()
            >>> P_B = propulsion.calculate_brake_power(150000, 7.72)
        """
        P_effective = (total_resistance * speed_ms) / 1000
        return P_effective / self._eta_overall

    def power_breakdown_string(self, breakdown: PowerBreakdown) -> str:
        """Generate formatted string representation of power breakdown.

        Args:
            breakdown: PowerBreakdown instance

        Returns:
            Multi-line formatted string

        Example:
            >>> propulsion = PropulsionModel()
            >>> breakdown = propulsion.calculate_power(150000, 7.72)
            >>> print(propulsion.power_breakdown_string(breakdown))
        """
        lines = [
            "Power Chain Breakdown:",
            f"  Effective Power (P_E):  {breakdown.effective_power:>8.1f} kW",
            f"  Delivered Power (P_D):  {breakdown.delivered_power:>8.1f} kW  (÷ η_P = {breakdown.propeller_efficiency:.3f})",
            f"  Brake Power (P_B):      {breakdown.brake_power:>8.1f} kW  (÷ η_S = {breakdown.shaft_efficiency:.3f})",
            "",
            f"  Overall Efficiency (η_D): {breakdown.overall_efficiency:.3f}",
            f"  Ship Speed:               {breakdown.speed_ms:.2f} m/s ({breakdown.speed_ms * 1.944:.1f} knots)",
            f"  Total Resistance:         {breakdown.total_resistance/1000:.1f} kN",
        ]
        return "\n".join(lines)


def estimate_propeller_efficiency(
    ship_speed_knots: float,
    ship_type: str = "cargo"
) -> float:
    """Estimate propeller efficiency based on ship speed and type.

    This is a simplified estimation. In reality, propeller efficiency depends on:
    - Propeller design (diameter, pitch, blade area ratio)
    - Advance coefficient J = V_A / (n × D)
    - Hull wake fraction
    - Thrust loading coefficient

    Args:
        ship_speed_knots: Service speed in knots
        ship_type: Type of ship ("cargo", "tanker", "container", "bulk")

    Returns:
        Estimated propeller efficiency (0-1)

    Example:
        >>> eta_p = estimate_propeller_efficiency(15, "cargo")
        >>> print(f"Estimated η_P: {eta_p:.3f}")
    """
    # Base efficiency by ship type
    base_efficiency = {
        "cargo": 0.65,
        "tanker": 0.68,  # Slower, larger propellers
        "bulk": 0.70,    # Very slow, optimized propellers
        "container": 0.62,  # Faster, higher loading
        "ferry": 0.60,   # High speed
        "cruise": 0.63,
    }

    eta_base = base_efficiency.get(ship_type.lower(), 0.65)

    # Speed correction (higher speed generally reduces efficiency slightly)
    # This is a simplification
    if ship_speed_knots < 12:
        speed_factor = 1.02  # Slow ships benefit from lower loading
    elif ship_speed_knots < 18:
        speed_factor = 1.00  # Normal range
    else:
        speed_factor = 0.97  # Fast ships have higher loading

    estimated_efficiency = eta_base * speed_factor

    # Ensure within reasonable bounds
    return max(0.50, min(0.75, estimated_efficiency))
