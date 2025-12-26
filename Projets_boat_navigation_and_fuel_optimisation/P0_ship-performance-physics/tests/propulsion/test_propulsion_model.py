"""Tests for propulsion model (power chain calculations)."""

import pytest

from ship_performance.propulsion import PropulsionModel


class TestPropulsionModel:
    """Test suite for PropulsionModel class."""

    def test_initialization_with_defaults(self):
        """Test model initialization with default parameters."""
        model = PropulsionModel()

        assert model.propeller_efficiency == 0.65
        assert model.shaft_efficiency == 0.98
        assert abs(model.overall_efficiency - 0.65 * 0.98) < 0.001

    def test_initialization_with_custom_values(self):
        """Test model initialization with custom efficiency values."""
        model = PropulsionModel(propeller_efficiency=0.70, shaft_efficiency=0.99)

        assert model.propeller_efficiency == 0.70
        assert model.shaft_efficiency == 0.99
        assert abs(model.overall_efficiency - 0.70 * 0.99) < 0.001

    def test_initialization_validates_efficiency_range(self):
        """Test that efficiency values must be in valid range (0-1)."""
        # Should work
        PropulsionModel(propeller_efficiency=0.50, shaft_efficiency=0.95)
        PropulsionModel(propeller_efficiency=0.75, shaft_efficiency=1.0)

        # Should raise ValueError for out-of-range values
        with pytest.raises(ValueError, match="must be in"):
            PropulsionModel(propeller_efficiency=-0.1)

        with pytest.raises(ValueError, match="must be in"):
            PropulsionModel(propeller_efficiency=1.5)

        with pytest.raises(ValueError, match="must be in"):
            PropulsionModel(shaft_efficiency=-0.05)

        with pytest.raises(ValueError, match="must be in"):
            PropulsionModel(shaft_efficiency=1.2)

    def test_calculate_power_basic(self):
        """Test basic power calculation."""
        model = PropulsionModel(propeller_efficiency=0.65, shaft_efficiency=0.98)

        # 100 kN resistance at 5 m/s
        result = model.calculate_power(total_resistance=100_000, speed_ms=5.0)

        # P_E = R × V / 1000 = 100,000 N × 5 m/s / 1000 = 500 kW
        assert abs(result.effective_power - 500.0) < 0.1

        # P_D = P_E / η_P = 500 / 0.65 ≈ 769.2 kW
        assert abs(result.delivered_power - 769.2) < 0.5

        # P_B = P_D / η_S = 769.2 / 0.98 ≈ 784.9 kW
        assert abs(result.brake_power - 784.9) < 0.5

    def test_calculate_power_realistic_ship(self):
        """Test power calculation for realistic cargo ship."""
        model = PropulsionModel(propeller_efficiency=0.65, shaft_efficiency=0.98)

        # 150m cargo ship at 15 knots (~7.7 m/s)
        # Total resistance: ~140 kN
        result = model.calculate_power(total_resistance=140_000, speed_ms=7.72)

        # P_E = 140,000 × 7.72 / 1000 ≈ 1,080 kW
        assert 1_000 < result.effective_power < 1_200

        # P_D ≈ 1,662 kW
        assert 1_500 < result.delivered_power < 1_900

        # P_B ≈ 1,696 kW
        assert 1_600 < result.brake_power < 2_000

    def test_calculate_power_zero_resistance(self):
        """Test power calculation with zero resistance."""
        model = PropulsionModel()

        result = model.calculate_power(total_resistance=0.0, speed_ms=10.0)

        assert result.effective_power == 0.0
        assert result.delivered_power == 0.0
        assert result.brake_power == 0.0

    def test_calculate_power_zero_speed(self):
        """Test power calculation with zero speed."""
        model = PropulsionModel()

        result = model.calculate_power(total_resistance=100_000, speed_ms=0.0)

        assert result.effective_power == 0.0
        assert result.delivered_power == 0.0
        assert result.brake_power == 0.0

    def test_power_breakdown_structure(self):
        """Test that PowerBreakdown dataclass has correct structure."""
        model = PropulsionModel()

        result = model.calculate_power(total_resistance=100_000, speed_ms=5.0)

        # Check all required fields exist
        assert hasattr(result, "effective_power")
        assert hasattr(result, "delivered_power")
        assert hasattr(result, "brake_power")
        assert hasattr(result, "propeller_efficiency")
        assert hasattr(result, "shaft_efficiency")
        assert hasattr(result, "overall_efficiency")
        assert hasattr(result, "speed_ms")
        assert hasattr(result, "total_resistance")

    def test_power_chain_relationships(self):
        """Test relationships in power chain: P_E < P_D < P_B."""
        model = PropulsionModel(propeller_efficiency=0.65, shaft_efficiency=0.98)

        result = model.calculate_power(total_resistance=100_000, speed_ms=5.0)

        # Power should increase through chain (losses)
        assert result.effective_power < result.delivered_power
        assert result.delivered_power < result.brake_power

        # Check relationships with tolerances
        assert abs(result.delivered_power - result.effective_power / 0.65) < 0.5
        assert abs(result.brake_power - result.delivered_power / 0.98) < 0.5

    def test_efficiency_values_in_result(self):
        """Test that efficiency values are correctly stored in result."""
        model = PropulsionModel(propeller_efficiency=0.70, shaft_efficiency=0.99)

        result = model.calculate_power(total_resistance=100_000, speed_ms=5.0)

        assert result.propeller_efficiency == 0.70
        assert result.shaft_efficiency == 0.99
        assert abs(result.overall_efficiency - 0.70 * 0.99) < 0.001

    def test_high_efficiency_propulsion(self):
        """Test propulsion with high efficiency (modern ship)."""
        model = PropulsionModel(propeller_efficiency=0.75, shaft_efficiency=0.99)

        result = model.calculate_power(total_resistance=100_000, speed_ms=5.0)

        # P_E = 500 kW
        assert abs(result.effective_power - 500.0) < 0.1

        # P_B should be lower with higher efficiency
        # P_B = 500 / (0.75 × 0.99) ≈ 673.4 kW
        assert 670 < result.brake_power < 680

    def test_low_efficiency_propulsion(self):
        """Test propulsion with low efficiency (older ship)."""
        model = PropulsionModel(propeller_efficiency=0.55, shaft_efficiency=0.96)

        result = model.calculate_power(total_resistance=100_000, speed_ms=5.0)

        # P_E = 500 kW
        assert abs(result.effective_power - 500.0) < 0.1

        # P_B should be higher with lower efficiency
        # P_B = 500 / (0.55 × 0.96) ≈ 946.9 kW
        assert 940 < result.brake_power < 955

    def test_power_scaling_with_speed(self):
        """Test that power scales approximately as V³."""
        model = PropulsionModel()

        # At constant resistance, power scales linearly with speed
        # But in reality, resistance scales as V², so power as V³

        # Assuming resistance ∝ V² (simplified)
        v1, v2 = 5.0, 10.0
        R1 = 100_000
        R2 = R1 * (v2 / v1) ** 2  # R ∝ V²

        result1 = model.calculate_power(total_resistance=R1, speed_ms=v1)
        result2 = model.calculate_power(total_resistance=R2, speed_ms=v2)

        # P ∝ R × V ∝ V² × V = V³
        # So P2 / P1 should ≈ (v2/v1)³ = 8
        power_ratio = result2.effective_power / result1.effective_power
        expected_ratio = (v2 / v1) ** 3

        assert abs(power_ratio - expected_ratio) < 0.1

    def test_large_ship_power_requirements(self):
        """Test power calculation for large ship (tanker/container)."""
        model = PropulsionModel()

        # Large ship: 200,000 tonnes displacement, 20 knots (~10.3 m/s)
        # Resistance: ~800 kN (estimated)
        result = model.calculate_power(total_resistance=800_000, speed_ms=10.3)

        # P_E = 800,000 × 10.3 / 1000 = 8,240 kW
        assert 8_000 < result.effective_power < 8_500

        # P_B ≈ 12,926 kW
        assert 12_000 < result.brake_power < 14_000

    def test_power_breakdown_immutability(self):
        """Test that PowerBreakdown is immutable (frozen dataclass)."""
        model = PropulsionModel()
        result = model.calculate_power(total_resistance=100_000, speed_ms=5.0)

        # Should not be able to modify fields
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            result.brake_power = 9999

    def test_result_contains_input_parameters(self):
        """Test that result contains the input parameters."""
        model = PropulsionModel()

        resistance = 125_000
        speed = 6.5

        result = model.calculate_power(total_resistance=resistance, speed_ms=speed)

        assert result.total_resistance == resistance
        assert result.speed_ms == speed

    def test_typical_propulsive_efficiency_range(self):
        """Test that overall efficiency is in typical range (0.55-0.75)."""
        # Typical modern ship
        model1 = PropulsionModel(propeller_efficiency=0.70, shaft_efficiency=0.99)
        assert 0.65 < model1.overall_efficiency < 0.75

        # Typical older ship
        model2 = PropulsionModel(propeller_efficiency=0.60, shaft_efficiency=0.96)
        assert 0.55 < model2.overall_efficiency < 0.65

    def test_power_calculation_consistency(self):
        """Test that repeated calculations give same results."""
        model = PropulsionModel()

        result1 = model.calculate_power(total_resistance=100_000, speed_ms=5.0)
        result2 = model.calculate_power(total_resistance=100_000, speed_ms=5.0)

        assert result1.effective_power == result2.effective_power
        assert result1.delivered_power == result2.delivered_power
        assert result1.brake_power == result2.brake_power


class TestPowerBreakdown:
    """Tests for PowerBreakdown dataclass."""

    def test_power_breakdown_creation(self):
        """Test creating PowerBreakdown directly."""
        from ship_performance.propulsion import PowerBreakdown

        breakdown = PowerBreakdown(
            effective_power=500.0,
            delivered_power=769.2,
            brake_power=784.9,
            propeller_efficiency=0.65,
            shaft_efficiency=0.98,
            overall_efficiency=0.637,
            speed_ms=5.0,
            total_resistance=100_000,
        )

        assert breakdown.effective_power == 500.0
        assert breakdown.brake_power == 784.9

    def test_power_breakdown_frozen(self):
        """Test that PowerBreakdown is immutable."""
        from ship_performance.propulsion import PowerBreakdown

        breakdown = PowerBreakdown(
            effective_power=500.0,
            delivered_power=769.2,
            brake_power=784.9,
            propeller_efficiency=0.65,
            shaft_efficiency=0.98,
            overall_efficiency=0.637,
            speed_ms=5.0,
            total_resistance=100_000,
        )

        with pytest.raises(Exception):
            breakdown.brake_power = 1000.0
