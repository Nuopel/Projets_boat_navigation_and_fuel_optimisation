"""Tests for fuel consumption model."""

import pytest

from ship_performance.propulsion import FuelConsumptionModel, FuelType


class TestFuelConsumptionModel:
    """Test suite for FuelConsumptionModel class."""

    def test_initialization_with_defaults(self):
        """Test model initialization with default parameters."""
        model = FuelConsumptionModel()

        assert model.fuel_type == FuelType.HFO
        assert model.base_sfoc == 185.0  # Typical HFO SFOC

    def test_initialization_with_custom_fuel_type(self):
        """Test initialization with different fuel types."""
        # MDO
        model_mdo = FuelConsumptionModel(fuel_type=FuelType.MDO)
        assert model_mdo.fuel_type == FuelType.MDO
        assert model_mdo.base_sfoc == 180.0

        # LNG
        model_lng = FuelConsumptionModel(fuel_type=FuelType.LNG)
        assert model_lng.fuel_type == FuelType.LNG
        assert model_lng.base_sfoc == 155.0

    def test_initialization_with_custom_sfoc(self):
        """Test initialization with custom SFOC value."""
        model = FuelConsumptionModel(fuel_type=FuelType.HFO, base_sfoc=175.0)

        assert model.base_sfoc == 175.0

    def test_calculate_consumption_basic(self):
        """Test basic fuel consumption calculation."""
        model = FuelConsumptionModel(fuel_type=FuelType.HFO, base_sfoc=185.0)

        # 1000 kW brake power, no rated power (assumes 80% load)
        result = model.calculate_consumption(brake_power=1000.0)

        # FC = P_B × SFOC / 1000
        # At 80% load, SFOC ≈ 185 g/kWh (in optimal range)
        # FC ≈ 1000 × 185 / 1000 = 185 kg/h
        assert 180 < result.fuel_rate < 190

        # Daily consumption ≈ 185 × 24 / 1000 = 4.44 t/day
        assert 4.3 < result.fuel_rate_per_day < 4.6

    def test_calculate_consumption_with_rated_power(self):
        """Test consumption calculation with known rated power."""
        model = FuelConsumptionModel(fuel_type=FuelType.HFO, base_sfoc=185.0)

        # 6000 kW brake power, 8000 kW rated → 75% load (optimal)
        result = model.calculate_consumption(brake_power=6000.0, rated_power=8000.0)

        assert result.engine_load_factor == 0.75

        # At 75% load, SFOC should be at base value (optimal)
        assert abs(result.specific_consumption - 185.0) < 1.0

        # FC = 6000 × 185 / 1000 = 1,110 kg/h
        assert 1_100 < result.fuel_rate < 1_120

    def test_load_factor_optimal_range(self):
        """Test that SFOC is optimal at 75-85% load."""
        model = FuelConsumptionModel(fuel_type=FuelType.HFO, base_sfoc=180.0)

        # 75% load
        result_75 = model.calculate_consumption(brake_power=7500.0, rated_power=10000.0)
        # 80% load
        result_80 = model.calculate_consumption(brake_power=8000.0, rated_power=10000.0)
        # 85% load
        result_85 = model.calculate_consumption(brake_power=8500.0, rated_power=10000.0)

        # All should be at or very close to base SFOC
        assert abs(result_75.specific_consumption - 180.0) < 1.0
        assert abs(result_80.specific_consumption - 180.0) < 1.0
        assert abs(result_85.specific_consumption - 180.0) < 1.0

    def test_load_factor_low_load_penalty(self):
        """Test that SFOC increases at low loads."""
        model = FuelConsumptionModel(fuel_type=FuelType.HFO, base_sfoc=180.0)

        # 50% load - should have penalty
        result_50 = model.calculate_consumption(brake_power=5000.0, rated_power=10000.0)

        # 80% load - optimal
        result_80 = model.calculate_consumption(brake_power=8000.0, rated_power=10000.0)

        # Low load should have higher SFOC
        assert result_50.specific_consumption > result_80.specific_consumption
        assert result_50.specific_consumption > 180.0

        # At 50% load, correction ≈ 1 + (0.75 - 0.50) × 0.4 = 1.10
        # SFOC ≈ 180 × 1.10 = 198 g/kWh
        assert 195 < result_50.specific_consumption < 200

    def test_load_factor_high_load_penalty(self):
        """Test that SFOC increases at high loads."""
        model = FuelConsumptionModel(fuel_type=FuelType.HFO, base_sfoc=180.0)

        # 100% load - should have slight penalty
        result_100 = model.calculate_consumption(brake_power=10000.0, rated_power=10000.0)

        # 80% load - optimal
        result_80 = model.calculate_consumption(brake_power=8000.0, rated_power=10000.0)

        # High load should have higher SFOC
        assert result_100.specific_consumption > result_80.specific_consumption

        # At 100% load, correction ≈ 1 + (1.00 - 0.85) × 0.35 = 1.0525
        # SFOC ≈ 180 × 1.0525 ≈ 189.5 g/kWh
        assert 188 < result_100.specific_consumption < 195

    def test_fuel_type_hfo(self):
        """Test HFO fuel properties and consumption."""
        model = FuelConsumptionModel(fuel_type=FuelType.HFO)

        result = model.calculate_consumption(brake_power=5000.0, rated_power=6000.0)

        assert result.fuel_type == FuelType.HFO
        # HFO typical SFOC: 185 g/kWh
        assert 180 < result.specific_consumption < 195

    def test_fuel_type_mdo(self):
        """Test MDO fuel properties and consumption."""
        model = FuelConsumptionModel(fuel_type=FuelType.MDO)

        result = model.calculate_consumption(brake_power=5000.0, rated_power=6000.0)

        assert result.fuel_type == FuelType.MDO
        # MDO typical SFOC: 180 g/kWh (slightly better than HFO)
        assert 175 < result.specific_consumption < 190

    def test_fuel_type_lng(self):
        """Test LNG fuel properties and consumption."""
        model = FuelConsumptionModel(fuel_type=FuelType.LNG)

        result = model.calculate_consumption(brake_power=5000.0, rated_power=6000.0)

        assert result.fuel_type == FuelType.LNG
        # LNG typical SFOC: 155 g/kWh (best efficiency)
        assert 150 < result.specific_consumption < 165

    def test_co2_emissions_calculated(self):
        """Test that CO2 emissions are calculated."""
        model = FuelConsumptionModel(fuel_type=FuelType.HFO)

        result = model.calculate_consumption(brake_power=5000.0)

        # CO2 should be calculated
        assert result.co2_rate is not None
        assert result.co2_rate > 0

        # CO2 = fuel_rate × carbon_content × (44/12)
        # For HFO: carbon_content = 0.85
        # CO2 ≈ fuel_rate × 0.85 × 3.667 ≈ fuel_rate × 3.12
        expected_co2 = result.fuel_rate * 0.85 * (44 / 12)
        assert abs(result.co2_rate - expected_co2) < 1.0

    def test_co2_lower_for_lng(self):
        """Test that LNG produces less CO2 than HFO."""
        # HFO
        model_hfo = FuelConsumptionModel(fuel_type=FuelType.HFO)
        result_hfo = model_hfo.calculate_consumption(brake_power=5000.0)

        # LNG
        model_lng = FuelConsumptionModel(fuel_type=FuelType.LNG)
        result_lng = model_lng.calculate_consumption(brake_power=5000.0)

        # LNG should produce less CO2 (lower carbon content + better efficiency)
        assert result_lng.co2_rate < result_hfo.co2_rate

    def test_daily_fuel_consumption(self):
        """Test daily fuel consumption calculation."""
        model = FuelConsumptionModel(fuel_type=FuelType.HFO, base_sfoc=185.0)

        result = model.calculate_consumption(brake_power=5000.0)

        # Daily = hourly × 24 / 1000 (to tonnes)
        expected_daily = result.fuel_rate * 24 / 1000
        assert abs(result.fuel_rate_per_day - expected_daily) < 0.01

    def test_realistic_cargo_ship_consumption(self):
        """Test realistic fuel consumption for cargo ship."""
        model = FuelConsumptionModel(fuel_type=FuelType.HFO)

        # 150m cargo ship: ~1700 kW brake power at 15 knots
        result = model.calculate_consumption(brake_power=1700.0, rated_power=3000.0)

        # Should consume ~7-9 tonnes/day
        assert 6 < result.fuel_rate_per_day < 10

        # Fuel rate: ~300-350 kg/h
        assert 280 < result.fuel_rate < 380

    def test_realistic_container_ship_consumption(self):
        """Test realistic fuel consumption for large container ship."""
        model = FuelConsumptionModel(fuel_type=FuelType.HFO)

        # Large container ship: ~40,000 kW at full speed
        result = model.calculate_consumption(brake_power=40000.0, rated_power=50000.0)

        # Should consume ~180-220 tonnes/day
        assert 170 < result.fuel_rate_per_day < 230

    def test_voyage_consumption_calculation(self):
        """Test voyage total fuel calculation."""
        model = FuelConsumptionModel(fuel_type=FuelType.HFO)

        # 5-day voyage at 5000 kW
        voyage = model.calculate_voyage_consumption(
            brake_power=5000.0, duration_hours=120, rated_power=6000.0
        )

        assert "total_fuel" in voyage
        assert "fuel_rate" in voyage
        assert "co2_total" in voyage

        # Total fuel should be reasonable for 5 days at 5000 kW
        # FC ≈ 5000 × 185 / 1000 ≈ 925 kg/h → 22.2 t/day → 111 t for 5 days
        assert 100 < voyage["total_fuel"] < 130  # tonnes

    def test_zero_brake_power(self):
        """Test consumption with zero power (idle/stopped)."""
        model = FuelConsumptionModel()

        result = model.calculate_consumption(brake_power=0.0)

        assert result.fuel_rate == 0.0
        assert result.fuel_rate_per_day == 0.0

    def test_result_structure(self):
        """Test that FuelConsumptionResult has correct structure."""
        model = FuelConsumptionModel()

        result = model.calculate_consumption(brake_power=5000.0)

        # Check all required fields
        assert hasattr(result, "fuel_rate")
        assert hasattr(result, "fuel_rate_per_day")
        assert hasattr(result, "specific_consumption")
        assert hasattr(result, "brake_power")
        assert hasattr(result, "engine_load_factor")
        assert hasattr(result, "fuel_type")
        assert hasattr(result, "co2_rate")

    def test_result_immutability(self):
        """Test that FuelConsumptionResult is immutable."""
        model = FuelConsumptionModel()
        result = model.calculate_consumption(brake_power=5000.0)

        # Should not be able to modify
        with pytest.raises(Exception):
            result.fuel_rate = 9999

    def test_consumption_consistency(self):
        """Test that repeated calculations give same results."""
        model = FuelConsumptionModel(fuel_type=FuelType.HFO)

        result1 = model.calculate_consumption(brake_power=5000.0, rated_power=6000.0)
        result2 = model.calculate_consumption(brake_power=5000.0, rated_power=6000.0)

        assert result1.fuel_rate == result2.fuel_rate
        assert result1.specific_consumption == result2.specific_consumption

    def test_sfoc_range_realistic(self):
        """Test that SFOC values are in realistic range."""
        # Modern engine with HFO
        model = FuelConsumptionModel(fuel_type=FuelType.HFO, base_sfoc=175.0)
        result = model.calculate_consumption(brake_power=5000.0, rated_power=6000.0)

        # Should be in range 165-195 g/kWh (typical for modern engines)
        assert 165 < result.specific_consumption < 195

    def test_fuel_comparison_same_power(self):
        """Test fuel consumption comparison across fuel types."""
        brake_power = 5000.0
        rated_power = 6000.0

        results = {}
        for fuel_type in [FuelType.HFO, FuelType.MDO, FuelType.MGO, FuelType.LNG]:
            model = FuelConsumptionModel(fuel_type=fuel_type)
            results[fuel_type] = model.calculate_consumption(brake_power, rated_power)

        # LNG should have best (lowest) SFOC
        assert results[FuelType.LNG].specific_consumption < results[FuelType.HFO].specific_consumption

        # LNG should consume least fuel by mass
        assert results[FuelType.LNG].fuel_rate < results[FuelType.HFO].fuel_rate


class TestFuelProperties:
    """Tests for fuel properties constants."""

    def test_fuel_properties_exist(self):
        """Test that fuel properties are defined for all fuel types."""
        from ship_performance.propulsion.fuel_consumption import FUEL_PROPERTIES

        assert FuelType.HFO in FUEL_PROPERTIES
        assert FuelType.MDO in FUEL_PROPERTIES
        assert FuelType.MGO in FUEL_PROPERTIES
        assert FuelType.LNG in FUEL_PROPERTIES

    def test_hfo_properties(self):
        """Test HFO fuel properties."""
        from ship_performance.propulsion.fuel_consumption import FUEL_PROPERTIES

        hfo = FUEL_PROPERTIES[FuelType.HFO]

        assert hfo.fuel_type == FuelType.HFO
        assert hfo.lower_heating_value == 40200  # kJ/kg
        assert hfo.density == 991  # kg/m³
        assert hfo.carbon_content == 0.85
        assert hfo.typical_sfoc == 185  # g/kWh

    def test_lng_properties(self):
        """Test LNG fuel properties."""
        from ship_performance.propulsion.fuel_consumption import FUEL_PROPERTIES

        lng = FUEL_PROPERTIES[FuelType.LNG]

        assert lng.fuel_type == FuelType.LNG
        assert lng.lower_heating_value == 50000  # kJ/kg (higher than HFO)
        assert lng.carbon_content == 0.75  # Lower than HFO
        assert lng.typical_sfoc == 155  # g/kWh (better than HFO)

    def test_carbon_content_ordering(self):
        """Test that LNG has lower carbon content than fossil fuels."""
        from ship_performance.propulsion.fuel_consumption import FUEL_PROPERTIES

        lng_carbon = FUEL_PROPERTIES[FuelType.LNG].carbon_content
        hfo_carbon = FUEL_PROPERTIES[FuelType.HFO].carbon_content

        assert lng_carbon < hfo_carbon


class TestLoadFactorCorrection:
    """Tests for SFOC load factor correction."""

    def test_optimal_load_no_correction(self):
        """Test that optimal load range has no correction."""
        model = FuelConsumptionModel(base_sfoc=180.0)

        # Test within optimal range (75-85%)
        for load_pct in [75, 78, 80, 82, 85]:
            brake_power = load_pct * 100  # Scale to get desired load factor
            rated_power = 10000

            result = model.calculate_consumption(brake_power, rated_power)

            # SFOC should be at or very close to base value
            assert abs(result.specific_consumption - 180.0) < 1.0

    def test_low_load_correction(self):
        """Test SFOC correction at low loads."""
        model = FuelConsumptionModel(base_sfoc=180.0)

        # 50% load
        result_50 = model.calculate_consumption(brake_power=5000.0, rated_power=10000.0)
        # 25% load
        result_25 = model.calculate_consumption(brake_power=2500.0, rated_power=10000.0)

        # Both should have penalties
        assert result_50.specific_consumption > 180.0
        assert result_25.specific_consumption > result_50.specific_consumption

    def test_high_load_correction(self):
        """Test SFOC correction at high loads."""
        model = FuelConsumptionModel(base_sfoc=180.0)

        # 95% load
        result_95 = model.calculate_consumption(brake_power=9500.0, rated_power=10000.0)
        # 100% load
        result_100 = model.calculate_consumption(brake_power=10000.0, rated_power=10000.0)

        # Both should have penalties
        assert result_95.specific_consumption > 180.0
        assert result_100.specific_consumption > result_95.specific_consumption
