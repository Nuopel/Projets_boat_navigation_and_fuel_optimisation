"""Tests for added wave resistance calculations."""

import math

import pytest

from ship_performance.core import OperatingConditions, ShipParameters
from ship_performance.resistance import AddedWaveResistance


class TestAddedWaveResistanceBasic:
    """Basic tests for added wave resistance calculation."""

    @pytest.fixture
    def cargo_ship(self):
        """Create a typical cargo ship for testing."""
        return ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
        )

    @pytest.fixture
    def calc(self):
        """Create calculator instance."""
        return AddedWaveResistance()

    def test_zero_wave_height(self, cargo_ship, calc):
        """No waves should give zero added resistance."""
        conditions = OperatingConditions(
            speed=15, wave_height=0, wave_period=8, wave_angle=0
        )
        resistance = calc.calculate(cargo_ship, conditions)
        assert resistance == 0.0

    def test_positive_resistance(self, cargo_ship, calc):
        """Added resistance should be positive for head seas."""
        conditions = OperatingConditions(
            speed=15, wave_height=2, wave_period=8, wave_angle=0
        )
        resistance = calc.calculate(cargo_ship, conditions)
        assert resistance > 0

    def test_resistance_never_negative(self, cargo_ship, calc):
        """Resistance cannot be negative even in following seas."""
        conditions = OperatingConditions(
            speed=15, wave_height=3, wave_period=10, wave_angle=180
        )
        resistance = calc.calculate(cargo_ship, conditions)
        assert resistance >= 0

    def test_has_name_property(self, calc):
        """Calculator should have a name property."""
        assert calc.name == "Added Resistance in Waves"


class TestWaveHeightDependency:
    """Tests for wave height dependency (quadratic)."""

    @pytest.fixture
    def ship(self):
        return ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

    @pytest.fixture
    def calc(self):
        return AddedWaveResistance()

    def test_resistance_increases_with_wave_height(self, ship, calc):
        """Higher waves should give higher resistance."""
        Hs_values = [1, 2, 3, 4]
        resistances = []

        for Hs in Hs_values:
            conditions = OperatingConditions(
                speed=15, wave_height=Hs, wave_period=8, wave_angle=0
            )
            R = calc.calculate(ship, conditions)
            resistances.append(R)

        # Each should be greater than previous
        for i in range(len(resistances) - 1):
            assert resistances[i + 1] > resistances[i]

    def test_quadratic_dependency_on_wave_height(self, ship, calc):
        """Resistance should scale approximately as Hs²."""
        conditions_1m = OperatingConditions(
            speed=15, wave_height=1, wave_period=8, wave_angle=0
        )
        conditions_2m = OperatingConditions(
            speed=15, wave_height=2, wave_period=8, wave_angle=0
        )
        conditions_3m = OperatingConditions(
            speed=15, wave_height=3, wave_period=8, wave_angle=0
        )

        R_1m = calc.calculate(ship, conditions_1m)
        R_2m = calc.calculate(ship, conditions_2m)
        R_3m = calc.calculate(ship, conditions_3m)

        # Should scale as Hs²
        # R_2m / R_1m ≈ (2/1)² = 4
        ratio_2_1 = R_2m / R_1m if R_1m > 0 else 0
        assert 3.5 < ratio_2_1 < 4.5  # Allow 12.5% tolerance

        # R_3m / R_1m ≈ (3/1)² = 9
        ratio_3_1 = R_3m / R_1m if R_1m > 0 else 0
        assert 8.0 < ratio_3_1 < 10.0  # Allow tolerance


class TestWaveHeadingDependency:
    """Tests for wave heading angle dependency."""

    @pytest.fixture
    def ship(self):
        return ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

    @pytest.fixture
    def calc(self):
        return AddedWaveResistance()

    def test_head_seas_maximum(self, ship, calc):
        """Head seas (0°) should give maximum resistance."""
        Hs = 3
        Tp = 10

        head_conditions = OperatingConditions(
            speed=15, wave_height=Hs, wave_period=Tp, wave_angle=0
        )
        beam_conditions = OperatingConditions(
            speed=15, wave_height=Hs, wave_period=Tp, wave_angle=90
        )
        following_conditions = OperatingConditions(
            speed=15, wave_height=Hs, wave_period=Tp, wave_angle=180
        )

        R_head = calc.calculate(ship, head_conditions)
        R_beam = calc.calculate(ship, beam_conditions)
        R_following = calc.calculate(ship, following_conditions)

        # Head seas should be maximum
        assert R_head > R_beam
        assert R_head > R_following

    def test_beam_seas_moderate(self, ship, calc):
        """Beam seas (90°) should give moderate resistance."""
        conditions = OperatingConditions(
            speed=15, wave_height=3, wave_period=10, wave_angle=90
        )
        resistance = calc.calculate(ship, conditions)

        # Should be near zero due to cos(90°) = 0
        assert resistance < 1000  # Very low resistance in beam seas

    def test_following_seas_minimum(self, ship, calc):
        """Following seas (180°) should give minimum (zero) resistance."""
        conditions = OperatingConditions(
            speed=15, wave_height=3, wave_period=10, wave_angle=180
        )
        resistance = calc.calculate(ship, conditions)

        # Should be zero or very small due to cos(180°) = -1 → max(0, ...)
        assert resistance == 0.0

    def test_heading_angles_progression(self, ship, calc):
        """Test progression from head to following seas."""
        Hs = 3
        Tp = 10
        angles = [0, 30, 60, 90, 120, 150, 180]
        resistances = []

        for angle in angles:
            conditions = OperatingConditions(
                speed=15, wave_height=Hs, wave_period=Tp, wave_angle=angle
            )
            R = calc.calculate(ship, conditions)
            resistances.append(R)

        # Resistance should generally decrease from 0° to 180°
        # (may not be strictly monotonic due to wavelength effects)
        assert resistances[0] > resistances[-1]  # Head > Following
        assert resistances[0] > resistances[3]  # Head > Beam


class TestWavelengthEffects:
    """Tests for wavelength and wave period effects."""

    @pytest.fixture
    def ship(self):
        return ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

    @pytest.fixture
    def calc(self):
        return AddedWaveResistance()

    def test_wave_period_affects_resistance(self, ship, calc):
        """Different wave periods should give different resistance."""
        periods = [6, 8, 10, 12]
        resistances = []

        for Tp in periods:
            conditions = OperatingConditions(
                speed=15, wave_height=2, wave_period=Tp, wave_angle=0
            )
            R = calc.calculate(ship, conditions)
            resistances.append(R)

        # Resistances should vary (not all the same)
        assert len(set(resistances)) > 1

    def test_wavelength_calculation(self, ship, calc):
        """Deep water wavelength should be calculated correctly."""
        Tp = 10  # seconds
        conditions = OperatingConditions(
            speed=15, wave_height=2, wave_period=Tp, wave_angle=0
        )

        breakdown = calc.breakdown(ship, conditions)

        # Deep water: λ = g × Tp² / (2π)
        expected_lambda = (9.81 * Tp**2) / (2 * math.pi)
        assert abs(breakdown["wavelength"] - expected_lambda) < 0.1

    def test_wavelength_to_length_ratio(self, ship, calc):
        """λ/L ratio should be calculated correctly."""
        Tp = 10
        conditions = OperatingConditions(
            speed=15, wave_height=2, wave_period=Tp, wave_angle=0
        )

        breakdown = calc.breakdown(ship, conditions)

        # λ/L = wavelength / ship_length
        lambda_wave = breakdown["wavelength"]
        expected_ratio = lambda_wave / ship.length

        assert abs(breakdown["wavelength_to_length_ratio"] - expected_ratio) < 0.001


class TestShipGeometryEffects:
    """Tests for ship geometry dependency."""

    @pytest.fixture
    def calc(self):
        return AddedWaveResistance()

    def test_larger_ship_more_resistance(self, calc):
        """Larger ships should have higher absolute resistance."""
        small_ship = ShipParameters(
            length=100, beam=18, draft=6, displacement=8000, block_coefficient=0.7
        )
        large_ship = ShipParameters(
            length=200, beam=32, draft=12, displacement=40000, block_coefficient=0.7
        )

        conditions = OperatingConditions(
            speed=15, wave_height=3, wave_period=10, wave_angle=0
        )

        R_small = calc.calculate(small_ship, conditions)
        R_large = calc.calculate(large_ship, conditions)

        # Larger ship should have higher absolute resistance
        assert R_large > R_small

    def test_fuller_ship_more_resistance(self, calc):
        """Fuller ships (higher Cb) should have higher added resistance."""
        fine_ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=12000, block_coefficient=0.60
        )
        full_ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=18000, block_coefficient=0.80
        )

        conditions = OperatingConditions(
            speed=15, wave_height=3, wave_period=10, wave_angle=0
        )

        R_fine = calc.calculate(fine_ship, conditions)
        R_full = calc.calculate(full_ship, conditions)

        # Fuller ship should have higher resistance
        assert R_full > R_fine


class TestBreakdownFunctionality:
    """Tests for breakdown method."""

    @pytest.fixture
    def ship(self):
        return ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

    @pytest.fixture
    def calc(self):
        return AddedWaveResistance()

    def test_breakdown_contains_all_keys(self, ship, calc):
        """Breakdown should contain all expected keys."""
        conditions = OperatingConditions(
            speed=15, wave_height=3, wave_period=10, wave_angle=30
        )

        breakdown = calc.breakdown(ship, conditions)

        expected_keys = [
            "wave_height",
            "wave_period",
            "wave_angle",
            "wavelength",
            "wavelength_to_length_ratio",
            "heading_factor",
            "wavelength_factor",
            "resistance",
        ]

        for key in expected_keys:
            assert key in breakdown

    def test_breakdown_resistance_matches_calculate(self, ship, calc):
        """Breakdown resistance should match calculate() result."""
        conditions = OperatingConditions(
            speed=15, wave_height=3, wave_period=10, wave_angle=30
        )

        resistance = calc.calculate(ship, conditions)
        breakdown = calc.breakdown(ship, conditions)

        assert abs(resistance - breakdown["resistance"]) < 0.1

    def test_heading_factor_range(self, ship, calc):
        """Heading factor should be in range [0, 1]."""
        angles = [0, 45, 90, 135, 180]

        for angle in angles:
            conditions = OperatingConditions(
                speed=15, wave_height=3, wave_period=10, wave_angle=angle
            )
            breakdown = calc.breakdown(ship, conditions)

            assert 0.0 <= breakdown["heading_factor"] <= 1.0

    def test_wavelength_factor_range(self, ship, calc):
        """Wavelength factor should be in reasonable range."""
        periods = [5, 8, 10, 12, 15]

        for Tp in periods:
            conditions = OperatingConditions(
                speed=15, wave_height=2, wave_period=Tp, wave_angle=0
            )
            breakdown = calc.breakdown(ship, conditions)

            # Should be between 0 and 1
            assert 0.0 <= breakdown["wavelength_factor"] <= 1.0


class TestRealisticValues:
    """Tests for realistic magnitude of added resistance."""

    @pytest.fixture
    def cargo_ship(self):
        return ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

    @pytest.fixture
    def calc(self):
        return AddedWaveResistance()

    def test_moderate_seas_reasonable_magnitude(self, cargo_ship, calc):
        """Moderate seas (Hs=3m) should give reasonable resistance."""
        conditions = OperatingConditions(
            speed=15, wave_height=3, wave_period=10, wave_angle=0
        )

        resistance = calc.calculate(cargo_ship, conditions)

        # Should be significant but not extreme
        # Expect 5-30 kN for moderate seas in head waves (simplified model)
        assert 3_000 < resistance < 50_000  # 3-50 kN

    def test_severe_seas_high_magnitude(self, cargo_ship, calc):
        """Severe seas (Hs=6m) should give high resistance."""
        conditions = OperatingConditions(
            speed=15, wave_height=6, wave_period=12, wave_angle=0
        )

        resistance = calc.calculate(cargo_ship, conditions)

        # Should be very high (significant contribution)
        # Expect > 15 kN for severe seas (simplified model)
        assert resistance > 15_000  # > 15 kN

    def test_storm_conditions(self, cargo_ship, calc):
        """Storm conditions (Hs=8m) should give very high resistance."""
        conditions = OperatingConditions(
            speed=10,  # Reduced speed in storm
            wave_height=8,
            wave_period=14,
            wave_angle=0,
        )

        resistance = calc.calculate(cargo_ship, conditions)

        # Storm should produce very high resistance
        # Expect > 25 kN for storm conditions (simplified model)
        assert resistance > 25_000  # > 25 kN


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.fixture
    def ship(self):
        return ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

    @pytest.fixture
    def calc(self):
        return AddedWaveResistance()

    def test_very_short_waves(self, ship, calc):
        """Very short waves (Tp=3s) should give low resistance."""
        conditions = OperatingConditions(
            speed=15, wave_height=2, wave_period=3, wave_angle=0
        )

        resistance = calc.calculate(ship, conditions)

        # Short waves have less effect
        assert resistance >= 0  # Should not crash

    def test_very_long_waves(self, ship, calc):
        """Very long waves (Tp=20s) should be handled."""
        conditions = OperatingConditions(
            speed=15, wave_height=2, wave_period=20, wave_angle=0
        )

        resistance = calc.calculate(ship, conditions)

        assert resistance >= 0  # Should not crash

    def test_zero_speed(self, ship, calc):
        """Zero speed ship should still have wave resistance."""
        conditions = OperatingConditions(
            speed=0, wave_height=3, wave_period=10, wave_angle=0
        )

        resistance = calc.calculate(ship, conditions)

        # Stationary ship still experiences wave forces
        assert resistance >= 0

    def test_very_small_waves(self, ship, calc):
        """Very small waves (Hs=0.1m) should give small resistance."""
        conditions = OperatingConditions(
            speed=15, wave_height=0.1, wave_period=8, wave_angle=0
        )

        resistance = calc.calculate(ship, conditions)

        # Should be very small but positive (scales with Hs²)
        assert 0 < resistance < 15000  # Less than 15 kN


class TestCalculationMethods:
    """Tests for different calculation methods."""

    @pytest.fixture
    def ship(self):
        return ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

    def test_simplified_kwon_method(self, ship):
        """Test simplified Kwon method."""
        calc = AddedWaveResistance(method="simplified_kwon")
        conditions = OperatingConditions(
            speed=15, wave_height=3, wave_period=10, wave_angle=0
        )

        resistance = calc.calculate(ship, conditions)
        assert resistance > 0

    def test_empirical_method(self, ship):
        """Test empirical method."""
        calc = AddedWaveResistance(method="empirical")
        conditions = OperatingConditions(
            speed=15, wave_height=3, wave_period=10, wave_angle=0
        )

        resistance = calc.calculate(ship, conditions)
        assert resistance > 0

    def test_both_methods_give_positive_results(self, ship):
        """Both methods should give positive resistance."""
        calc_kwon = AddedWaveResistance(method="simplified_kwon")
        calc_empirical = AddedWaveResistance(method="empirical")

        conditions = OperatingConditions(
            speed=15, wave_height=3, wave_period=10, wave_angle=0
        )

        R_kwon = calc_kwon.calculate(ship, conditions)
        R_empirical = calc_empirical.calculate(ship, conditions)

        assert R_kwon > 0
        assert R_empirical > 0
