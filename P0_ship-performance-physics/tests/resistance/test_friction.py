"""Tests for friction resistance calculator."""

import pytest

from ship_performance.core import OperatingConditions, ShipParameters
from ship_performance.resistance import FrictionResistance


class TestFrictionResistanceCalculation:
    """Tests for basic friction resistance calculation."""

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

    @pytest.fixture
    def moderate_speed(self) -> OperatingConditions:
        """Moderate speed conditions."""
        return OperatingConditions(speed=15)  # knots

    def test_calculate_returns_positive(self, cargo_ship: ShipParameters, moderate_speed: OperatingConditions) -> None:
        """Test that friction resistance is positive."""
        calc = FrictionResistance()
        resistance = calc.calculate(cargo_ship, moderate_speed)

        assert resistance > 0
        assert isinstance(resistance, float)

    def test_resistance_increases_with_speed(self, cargo_ship: ShipParameters) -> None:
        """Test that friction resistance increases with speed."""
        calc = FrictionResistance()

        speeds = [5, 10, 15, 20]
        resistances = [
            calc.calculate(cargo_ship, OperatingConditions(speed=s))
            for s in speeds
        ]

        # Resistance should increase monotonically with speed
        for i in range(len(resistances) - 1):
            assert resistances[i] < resistances[i + 1]

    def test_resistance_scales_with_wetted_surface(self) -> None:
        """Test that resistance scales with wetted surface area."""
        calc = FrictionResistance()
        conditions = OperatingConditions(speed=15)

        # Small ship
        small_ship = ShipParameters(
            length=100, beam=15, draft=5, displacement=5000, block_coefficient=0.7
        )

        # Large ship (roughly 2x dimensions)
        large_ship = ShipParameters(
            length=200, beam=30, draft=10, displacement=40000, block_coefficient=0.7
        )

        R_small = calc.calculate(small_ship, conditions)
        R_large = calc.calculate(large_ship, conditions)

        # Larger ship should have significantly more resistance
        assert R_large > R_small * 2  # More than 2x due to wetted surface scaling

    def test_zero_speed_gives_zero_resistance(self, cargo_ship: ShipParameters) -> None:
        """Test that zero speed gives zero resistance."""
        calc = FrictionResistance()
        conditions = OperatingConditions(speed=0)

        resistance = calc.calculate(cargo_ship, conditions)
        assert resistance == 0.0


class TestITTCFrictionCoefficient:
    """Tests for ITTC 1957 friction coefficient calculation."""

    def test_friction_coefficient_typical_reynolds(self) -> None:
        """Test Cf for typical ship Reynolds numbers."""
        calc = FrictionResistance()

        # Typical Re for medium ship at moderate speed: ~1e9
        # Cf should be around 0.0015-0.0025
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )
        conditions = OperatingConditions(speed=15)

        breakdown = calc.breakdown(ship, conditions)
        cf = breakdown["friction_coefficient"]

        assert 0.001 < cf < 0.003  # Typical range
        assert isinstance(cf, float)

    def test_reynolds_number_increases_with_speed(self) -> None:
        """Test that Reynolds number increases with speed."""
        calc = FrictionResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        speeds = [5, 10, 15, 20]
        reynolds_numbers = [
            calc.breakdown(ship, OperatingConditions(speed=s))["reynolds_number"]
            for s in speeds
        ]

        # Reynolds number should increase with speed
        for i in range(len(reynolds_numbers) - 1):
            assert reynolds_numbers[i] < reynolds_numbers[i + 1]

    def test_friction_coefficient_decreases_with_reynolds(self) -> None:
        """Test that Cf decreases as Re increases (turbulent flow)."""
        calc = FrictionResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        speeds = [5, 15, 25]
        friction_coeffs = [
            calc.breakdown(ship, OperatingConditions(speed=s))["friction_coefficient"]
            for s in speeds
        ]

        # Cf decreases with increasing Re (logarithmic relationship)
        assert friction_coeffs[0] > friction_coeffs[1] > friction_coeffs[2]


class TestFormFactor:
    """Tests for form factor estimation."""

    def test_form_factor_in_reasonable_range(self) -> None:
        """Test that estimated form factor is in typical range (0.05-0.30)."""
        calc = FrictionResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )
        conditions = OperatingConditions(speed=15)

        breakdown = calc.breakdown(ship, conditions)
        k = breakdown["form_factor"]

        assert 0.05 <= k <= 0.30  # Typical range for merchant ships

    def test_form_factor_higher_for_full_ships(self) -> None:
        """Test that fuller ships have higher form factors."""
        calc = FrictionResistance()
        conditions = OperatingConditions(speed=15)

        # Fine ship (low Cb)
        fine_ship = ShipParameters(
            length=150, beam=20, draft=7, displacement=12000, block_coefficient=0.60
        )

        # Full ship (high Cb)
        full_ship = ShipParameters(
            length=150, beam=30, draft=10, displacement=25000, block_coefficient=0.80
        )

        k_fine = calc.breakdown(fine_ship, conditions)["form_factor"]
        k_full = calc.breakdown(full_ship, conditions)["form_factor"]

        assert k_full > k_fine  # Fuller ships have higher form factors

    def test_custom_form_factor(self) -> None:
        """Test using custom form factor."""
        custom_k = 0.15
        calc = FrictionResistance(form_factor=custom_k)

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )
        conditions = OperatingConditions(speed=15)

        breakdown = calc.breakdown(ship, conditions)
        assert breakdown["form_factor"] == custom_k


class TestBreakdown:
    """Tests for resistance breakdown functionality."""

    def test_breakdown_contains_all_keys(self) -> None:
        """Test that breakdown contains all expected keys."""
        calc = FrictionResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )
        conditions = OperatingConditions(speed=15)

        breakdown = calc.breakdown(ship, conditions)

        required_keys = [
            "reynolds_number",
            "friction_coefficient",
            "form_factor",
            "effective_cf",
            "resistance",
        ]

        for key in required_keys:
            assert key in breakdown

    def test_effective_cf_calculation(self) -> None:
        """Test that effective Cf = (1 + k) × Cf."""
        calc = FrictionResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )
        conditions = OperatingConditions(speed=15)

        breakdown = calc.breakdown(ship, conditions)

        cf = breakdown["friction_coefficient"]
        k = breakdown["form_factor"]
        cf_eff = breakdown["effective_cf"]

        assert pytest.approx(cf_eff, rel=1e-6) == (1 + k) * cf

    def test_breakdown_resistance_matches_calculate(self) -> None:
        """Test that breakdown resistance matches calculate() result."""
        calc = FrictionResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )
        conditions = OperatingConditions(speed=15)

        resistance_direct = calc.calculate(ship, conditions)
        breakdown = calc.breakdown(ship, conditions)
        resistance_breakdown = breakdown["resistance"]

        assert pytest.approx(resistance_direct, rel=1e-10) == resistance_breakdown


class TestComponentName:
    """Test for component name property."""

    def test_name_property(self) -> None:
        """Test that name property returns expected string."""
        calc = FrictionResistance()
        assert isinstance(calc.name, str)
        assert "Friction" in calc.name or "friction" in calc.name.lower()
        assert "ITTC" in calc.name


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_slow_speed(self) -> None:
        """Test calculation at very slow speed."""
        calc = FrictionResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )
        conditions = OperatingConditions(speed=0.1)  # Very slow

        # Should not raise error and should give small resistance
        resistance = calc.calculate(ship, conditions)
        assert resistance >= 0
        assert resistance < 1000  # Very small

    def test_very_high_speed(self) -> None:
        """Test calculation at very high speed."""
        calc = FrictionResistance()
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore high speed warning
            conditions = OperatingConditions(speed=40)  # Very fast

        # Should not raise error and should give large resistance
        resistance = calc.calculate(ship, conditions)
        assert resistance > 0
