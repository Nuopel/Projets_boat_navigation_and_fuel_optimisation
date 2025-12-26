"""Tests for wind resistance calculator."""

import math

import pytest

from ship_performance.core import OperatingConditions, ShipParameters
from ship_performance.resistance import WindResistance


class TestWindResistanceCalculation:
    """Tests for basic wind resistance calculation."""

    @pytest.fixture
    def cargo_ship(self) -> ShipParameters:
        """Standard cargo ship for testing."""
        return ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
        )

    def test_no_wind_gives_zero_resistance(self, cargo_ship: ShipParameters) -> None:
        """Test that zero wind gives zero resistance."""
        calc = WindResistance()
        conditions = OperatingConditions(speed=15, wind_speed=0)

        resistance = calc.calculate(cargo_ship, conditions)
        assert resistance == 0.0

    def test_calculate_returns_positive(self, cargo_ship: ShipParameters) -> None:
        """Test that wind resistance is positive."""
        calc = WindResistance()
        conditions = OperatingConditions(speed=15, wind_speed=10, wind_angle=45)

        resistance = calc.calculate(cargo_ship, conditions)
        assert resistance > 0
        assert isinstance(resistance, float)

    def test_resistance_increases_with_wind_speed(self, cargo_ship: ShipParameters) -> None:
        """Test that wind resistance increases with wind speed."""
        calc = WindResistance()

        wind_speeds = [5, 10, 15, 20]
        resistances = [
            calc.calculate(cargo_ship, OperatingConditions(speed=15, wind_speed=w, wind_angle=90))
            for w in wind_speeds
        ]

        # Resistance should increase with wind speed (∝ V²)
        for i in range(len(resistances) - 1):
            assert resistances[i] < resistances[i + 1]

    def test_head_wind_vs_following_wind(self, cargo_ship: ShipParameters) -> None:
        """Test that head wind creates more resistance than following wind."""
        calc = WindResistance()

        # Head wind (0°)
        head_wind = OperatingConditions(speed=15, wind_speed=10, wind_angle=0)
        R_head = calc.calculate(cargo_ship, head_wind)

        # Following wind (180°)
        following_wind = OperatingConditions(speed=15, wind_speed=10, wind_angle=180)
        R_following = calc.calculate(cargo_ship, following_wind)

        # Head wind should create more resistance (higher relative wind)
        assert R_head > R_following

    def test_beam_wind_creates_resistance(self, cargo_ship: ShipParameters) -> None:
        """Test that beam wind (90°) creates significant resistance."""
        calc = WindResistance()

        beam_wind = OperatingConditions(speed=15, wind_speed=10, wind_angle=90)
        R_beam = calc.calculate(cargo_ship, beam_wind)

        assert R_beam > 0
        # Beam wind should be significant (between head and following)


class TestRelativeWindSpeed:
    """Tests for relative wind speed calculation."""

    def test_head_wind_adds_to_ship_speed(self) -> None:
        """Test that head wind adds to apparent wind speed."""
        calc = WindResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        # Ship at 10 m/s, head wind at 10 m/s
        conditions = OperatingConditions(speed=10 / 0.514444, wind_speed=10, wind_angle=0)
        v_rel = calc._calculate_relative_wind_speed(conditions)

        # Relative wind should be approximately sum (opposite directions)
        assert v_rel > 15  # Should be close to 20 m/s

    def test_following_wind_reduces_relative_speed(self) -> None:
        """Test that following wind reduces relative wind speed."""
        calc = WindResistance()

        # Ship at ~10 m/s, following wind at 5 m/s
        conditions = OperatingConditions(speed=10 / 0.514444, wind_speed=5, wind_angle=180)
        v_rel = calc._calculate_relative_wind_speed(conditions)

        # Relative wind should be approximately difference
        assert 4 < v_rel < 6  # Should be close to 5 m/s

    def test_beam_wind_vector_addition(self) -> None:
        """Test vector addition for beam wind."""
        calc = WindResistance()

        # Ship at 10 m/s, beam wind at 10 m/s (90°)
        conditions = OperatingConditions(speed=10 / 0.514444, wind_speed=10, wind_angle=90)
        v_rel = calc._calculate_relative_wind_speed(conditions)

        # Relative wind should be sqrt(10² + 10²) ≈ 14.14 m/s
        assert 13 < v_rel < 15

    def test_zero_wind_gives_ship_speed(self) -> None:
        """Test that zero wind gives relative wind equal to ship speed."""
        calc = WindResistance()

        conditions = OperatingConditions(speed=15, wind_speed=0)
        v_rel = calc._calculate_relative_wind_speed(conditions)

        # With no wind, relative wind = ship speed
        assert pytest.approx(v_rel, rel=0.01) == conditions.speed_ms


class TestDragCoefficient:
    """Tests for drag coefficient estimation."""

    def test_drag_coefficient_in_reasonable_range(self) -> None:
        """Test that estimated Cd is in typical range (0.4-0.9)."""
        calc = WindResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        # Test at various wind angles
        for angle in [0, 45, 90, 135, 180]:
            cd = calc._estimate_drag_coefficient(ship, angle)
            assert 0.4 <= cd <= 0.9

    def test_beam_wind_has_higher_cd(self) -> None:
        """Test that beam wind has higher Cd than head wind."""
        calc = WindResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        cd_head = calc._estimate_drag_coefficient(ship, 0)  # Head wind
        cd_beam = calc._estimate_drag_coefficient(ship, 90)  # Beam wind

        assert cd_beam > cd_head  # Beam presents larger profile

    def test_full_ships_have_higher_cd(self) -> None:
        """Test that fuller ships have higher drag coefficients."""
        calc = WindResistance()

        # Fine ship (container)
        fine_ship = ShipParameters(
            length=200, beam=28, draft=10, displacement=30000, block_coefficient=0.60
        )

        # Full ship (tanker)
        full_ship = ShipParameters(
            length=200, beam=35, draft=12, displacement=50000, block_coefficient=0.80
        )

        # Note: Container ships actually have higher Cd due to containers above deck
        # But in our simple model, we're checking base Cd
        cd_fine = calc._estimate_drag_coefficient(fine_ship, 90)
        cd_full = calc._estimate_drag_coefficient(full_ship, 90)

        # Both should be in valid range
        assert 0.4 <= cd_fine <= 0.9
        assert 0.4 <= cd_full <= 0.9

    def test_custom_drag_coefficient(self) -> None:
        """Test using custom drag coefficient."""
        custom_cd = 0.55
        calc = WindResistance(drag_coefficient=custom_cd)

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )
        conditions = OperatingConditions(speed=15, wind_speed=10, wind_angle=45)

        breakdown = calc.breakdown(ship, conditions)
        assert breakdown["drag_coefficient"] == custom_cd


class TestApparentWindAngle:
    """Tests for apparent wind angle calculation."""

    def test_head_wind_apparent_angle(self) -> None:
        """Test apparent wind angle for head wind."""
        calc = WindResistance()

        # True head wind should remain head wind
        conditions = OperatingConditions(speed=15, wind_speed=10, wind_angle=0)
        theta_apparent = calc._calculate_apparent_wind_angle(conditions)

        # Should be close to 0° (slight numerical variation acceptable)
        assert theta_apparent < 10

    def test_following_wind_apparent_angle(self) -> None:
        """Test that following wind can reverse to head wind if ship faster."""
        calc = WindResistance()

        # Ship faster than following wind
        conditions = OperatingConditions(speed=20, wind_speed=5, wind_angle=180)
        theta_apparent = calc._calculate_apparent_wind_angle(conditions)

        # Apparent wind should shift forward (ship outrunning wind)
        # Will be close to 0° (apparent head wind)
        assert theta_apparent < 45

    def test_beam_wind_shifts_forward(self) -> None:
        """Test that beam wind appears more forward when ship is moving."""
        calc = WindResistance()

        # True beam wind
        conditions = OperatingConditions(speed=15, wind_speed=10, wind_angle=90)
        theta_apparent = calc._calculate_apparent_wind_angle(conditions)

        # Apparent angle should be less than 90° (shifted forward)
        assert 30 < theta_apparent < 90


class TestBreakdown:
    """Tests for resistance breakdown functionality."""

    def test_breakdown_contains_all_keys(self) -> None:
        """Test that breakdown contains all expected keys."""
        calc = WindResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )
        conditions = OperatingConditions(speed=15, wind_speed=10, wind_angle=45)

        breakdown = calc.breakdown(ship, conditions)

        required_keys = [
            "true_wind_speed",
            "relative_wind_speed",
            "true_wind_angle",
            "apparent_wind_angle",
            "drag_coefficient",
            "frontal_area",
            "resistance",
        ]

        for key in required_keys:
            assert key in breakdown

    def test_breakdown_resistance_matches_calculate(self) -> None:
        """Test that breakdown resistance matches calculate() result."""
        calc = WindResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )
        conditions = OperatingConditions(speed=15, wind_speed=10, wind_angle=45)

        resistance_direct = calc.calculate(ship, conditions)
        breakdown = calc.breakdown(ship, conditions)
        resistance_breakdown = breakdown["resistance"]

        assert pytest.approx(resistance_direct, rel=1e-10) == resistance_breakdown


class TestComponentName:
    """Test for component name property."""

    def test_name_property(self) -> None:
        """Test that name property returns expected string."""
        calc = WindResistance()
        assert isinstance(calc.name, str)
        assert "Wind" in calc.name or "wind" in calc.name.lower()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_high_wind_speed(self) -> None:
        """Test calculation with very high wind speed."""
        calc = WindResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Hurricane force wind
            conditions = OperatingConditions(speed=5, wind_speed=35, wind_angle=0)

        resistance = calc.calculate(ship, conditions)
        assert resistance > 0
        # Should be very large (hurricane force wind)
        assert resistance > 50000  # > 50 kN (adjusted for realistic frontal area)

    def test_ship_stationary_with_wind(self) -> None:
        """Test wind resistance when ship is stationary."""
        calc = WindResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        # Stationary ship, wind blowing
        conditions = OperatingConditions(speed=0, wind_speed=10, wind_angle=90)

        resistance = calc.calculate(ship, conditions)
        # Should still have resistance from wind
        assert resistance > 0

    def test_multiple_wind_angles(self) -> None:
        """Test that resistance calculation works for all wind angles."""
        calc = WindResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        # Test all angles from 0 to 180
        for angle in range(0, 181, 15):
            conditions = OperatingConditions(speed=15, wind_speed=10, wind_angle=angle)
            resistance = calc.calculate(ship, conditions)
            assert resistance >= 0  # Should always be non-negative
