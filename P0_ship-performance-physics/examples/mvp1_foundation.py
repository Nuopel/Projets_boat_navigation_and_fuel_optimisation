#!/usr/bin/env python
"""MVP-1 Foundation Demo.

Demonstrates creation and usage of core data models from the ship performance
prediction package.

This script shows:
- Creating different ship types with ShipParameters
- Defining various operating conditions
- Accessing derived properties and unit conversions
- Calculating dimensionless numbers (Froude, Reynolds)
"""

from ship_performance.core import OperatingConditions, ShipParameters
from ship_performance.utils.units import calculate_froude_number, calculate_reynolds_number


def print_separator(title: str) -> None:
    """Print a formatted section separator."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demonstrate_cargo_ship() -> ShipParameters:
    """Demonstrate creating and inspecting a cargo ship."""
    print_separator("Cargo Ship Example")

    ship = ShipParameters(
        length=150,  # meters
        beam=25,  # meters
        draft=8,  # meters
        displacement=21525,  # tonnes
        block_coefficient=0.7,  # typical for cargo ships
    )

    print(f"\n{ship}")
    print(f"\nDimensions:")
    print(f"  Length:        {ship.length:.1f} m")
    print(f"  Beam:          {ship.beam:.1f} m")
    print(f"  Draft:         {ship.draft:.1f} m")
    print(f"  Displacement:  {ship.displacement:.0f} tonnes")

    print(f"\nCoefficients:")
    print(f"  Block (Cb):      {ship.block_coefficient:.3f}")
    print(f"  Prismatic (Cp):  {ship.prismatic_coefficient:.3f}")
    print(f"  Waterplane (Cwp): {ship.waterplane_coefficient:.3f}")

    print(f"\nDimensional Ratios:")
    print(f"  L/B ratio:  {ship.length_beam_ratio:.2f}")
    print(f"  B/T ratio:  {ship.beam_draft_ratio:.2f}")
    print(f"  L/T ratio:  {ship.length_draft_ratio:.2f}")

    print(f"\nEstimated Values:")
    print(f"  Wetted surface:      {ship.wetted_surface:.1f} m²")
    print(f"  Frontal area:        {ship.frontal_area:.1f} m²")
    print(f"  Volumetric disp.:    {ship.volumetric_displacement:.1f} m³")

    return ship


def demonstrate_tanker() -> ShipParameters:
    """Demonstrate creating a large tanker."""
    print_separator("Tanker Example")

    ship = ShipParameters(
        length=250,
        beam=42,
        draft=14,
        displacement=123554,
        block_coefficient=0.82,  # higher for full-bodied tankers
    )

    print(f"\n{ship}")
    print(f"\nDimensions: {ship.length}m × {ship.beam}m × {ship.draft}m")
    print(f"Displacement: {ship.displacement:,.0f} tonnes")
    print(f"Block coefficient: {ship.block_coefficient:.3f} (full-bodied hull)")
    print(f"L/B ratio: {ship.length_beam_ratio:.2f} (typical for tankers)")

    return ship


def demonstrate_container_ship() -> ShipParameters:
    """Demonstrate creating a container ship."""
    print_separator("Container Ship Example")

    ship = ShipParameters(
        length=200,
        beam=30,
        draft=10,
        displacement=39975,
        block_coefficient=0.65,  # lower for finer hull lines
        wetted_surface=5500,  # provided explicitly
        frontal_area=800,  # provided explicitly
    )

    print(f"\n{ship}")
    print(f"\nContainer ship characteristics:")
    print(f"  Finer hull lines: Cb = {ship.block_coefficient:.3f}")
    print(f"  Higher L/B ratio: {ship.length_beam_ratio:.2f} (for speed)")
    print(f"  Provided wetted surface: {ship.wetted_surface:.0f} m²")
    print(f"  Provided frontal area: {ship.frontal_area:.0f} m²")

    return ship


def demonstrate_operating_conditions() -> None:
    """Demonstrate various operating conditions."""
    print_separator("Operating Conditions Examples")

    # Calm water
    calm = OperatingConditions(speed=15)
    print(f"\n1. Calm Water Conditions:")
    print(f"   {calm}")
    print(f"   Speed: {calm.speed} knots ({calm.speed_ms:.2f} m/s)")
    print(f"   Is calm: {calm.is_calm_water}")

    # Moderate weather
    moderate = OperatingConditions(
        speed=15, wind_speed=10, wind_angle=45, wave_height=2, wave_period=8, wave_angle=30
    )
    print(f"\n2. Moderate Weather:")
    print(f"   {moderate}")
    print(f"   Wind: {moderate.wind_speed} m/s @ {moderate.wind_angle}° (Beaufort {moderate.beaufort_scale})")
    print(f"   Waves: Hs={moderate.wave_height}m, Tp={moderate.wave_period}s @ {moderate.wave_angle}° (Sea State {moderate.sea_state})")

    # Storm conditions
    print(f"\n3. Storm Conditions:")
    storm = OperatingConditions(
        speed=8,  # reduced speed
        wind_speed=20,  # ~39 knots
        wind_angle=0,  # head wind
        wave_height=6,  # very rough seas
        wave_period=11,
        wave_angle=0,  # head seas
    )
    print(f"   Speed reduced to: {storm.speed} knots")
    print(f"   Wind: {storm.wind_speed} m/s head wind (Beaufort {storm.beaufort_scale} - Gale)")
    print(f"   Waves: Hs={storm.wave_height}m head seas (Sea State {storm.sea_state} - Very Rough)")

    # High speed operation
    print(f"\n4. High Speed Operation:")
    fast = OperatingConditions(speed=25, wind_speed=5, wind_angle=90)
    print(f"   Speed: {fast.speed} knots ({fast.speed_ms:.2f} m/s)")
    print(f"   Wind: {fast.wind_speed} m/s beam wind (Beaufort {fast.beaufort_scale})")


def demonstrate_dimensionless_numbers(ship: ShipParameters) -> None:
    """Demonstrate calculating dimensionless numbers."""
    print_separator(f"Dimensionless Numbers for {ship.length}m Ship")

    speeds_knots = [5, 10, 15, 20, 25]

    print(f"\n{'Speed (knots)':>14} | {'Speed (m/s)':>12} | {'Froude':>8} | {'Reynolds':>12} | {'Regime'}")
    print("-" * 70)

    for speed_knots in speeds_knots:
        conditions = OperatingConditions(speed=speed_knots)
        speed_ms = conditions.speed_ms

        fn = calculate_froude_number(speed_ms, ship.length)
        re = calculate_reynolds_number(speed_ms, ship.length)

        # Classify regime
        if fn < 0.3:
            regime = "Displacement"
        elif fn < 0.4:
            regime = "Transition"
        else:
            regime = "Semi-planing"

        print(
            f"{speed_knots:>14.1f} | {speed_ms:>12.2f} | {fn:>8.3f} | {re:>12.2e} | {regime}"
        )

    print(f"\nNote:")
    print(f"  - Fn < 0.3:  Displacement mode (typical merchant ships)")
    print(f"  - 0.3 < Fn < 0.4: Transition (resistance hump)")
    print(f"  - Fn > 0.4:  Semi-planing/planing regime (fast vessels)")
    print(f"  - All Re > 5e5: Fully turbulent flow")


def demonstrate_immutability() -> None:
    """Demonstrate that data models are immutable."""
    print_separator("Immutability Demonstration")

    ship = ShipParameters(length=150, beam=25, draft=8, displacement=21525, block_coefficient=0.7)

    print(f"\nCreated ship: {ship}")
    print(f"\nAttempting to modify length...")

    try:
        ship.length = 200  # type: ignore
        print("  ERROR: Modification succeeded (this shouldn't happen!)")
    except (AttributeError, TypeError) as e:
        print(f"  ✓ Modification blocked: {type(e).__name__}")
        print(f"    Ship parameters are immutable (frozen dataclass)")

    conditions = OperatingConditions(speed=15)
    print(f"\n\nAttempting to modify speed...")

    try:
        conditions.speed = 20  # type: ignore
        print("  ERROR: Modification succeeded (this shouldn't happen!)")
    except (AttributeError, TypeError) as e:
        print(f"  ✓ Modification blocked: {type(e).__name__}")
        print(f"    Operating conditions are immutable (frozen dataclass)")

    print(f"\n  Why immutability?")
    print(f"    - Prevents accidental modifications")
    print(f"    - Enables safe sharing between functions")
    print(f"    - Makes code more predictable and testable")


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  MVP-1: Foundation & Core Abstractions Demo")
    print("  Ship Performance Prediction Package")
    print("=" * 70)

    # Demonstrate different ship types
    cargo = demonstrate_cargo_ship()
    tanker = demonstrate_tanker()
    container = demonstrate_container_ship()

    # Demonstrate operating conditions
    demonstrate_operating_conditions()

    # Demonstrate dimensionless numbers
    demonstrate_dimensionless_numbers(cargo)

    # Demonstrate immutability
    demonstrate_immutability()

    # Summary
    print_separator("Summary")
    print(f"\n✓ Successfully demonstrated:")
    print(f"  - Creating ship parameters with validation")
    print(f"  - Automatic estimation of wetted surface and frontal area")
    print(f"  - Derived properties (ratios, coefficients)")
    print(f"  - Operating conditions with environmental parameters")
    print(f"  - Unit conversions (knots ↔ m/s)")
    print(f"  - Dimensionless numbers (Froude, Reynolds)")
    print(f"  - Beaufort scale and Sea State classification")
    print(f"  - Immutability guarantees")

    print(f"\n✓ MVP-1 Foundation is working correctly!")
    print(f"\nNext: MVP-2 will implement calm water resistance calculations")
    print(f"      using these core data models.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
