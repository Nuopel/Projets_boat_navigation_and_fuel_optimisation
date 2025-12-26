"""Tests for complete ship performance model (integration tests)."""

import pytest

from ship_performance.core import OperatingConditions, ShipParameters
from ship_performance.models import PerformanceResult, ShipPerformanceModel
from ship_performance.propulsion import FuelType, PropulsionModel


class TestShipPerformanceModel:
    """Test suite for ShipPerformanceModel (complete integration)."""

    def test_initialization_with_defaults(self):
        """Test model initialization with default sub-models."""
        model = ShipPerformanceModel()

        assert model.resistance is not None
        assert model.propulsion is not None
        assert model.fuel is not None

    def test_initialization_with_custom_propulsion(self):
        """Test initialization with custom propulsion model."""
        custom_propulsion = PropulsionModel(propeller_efficiency=0.70, shaft_efficiency=0.99)

        model = ShipPerformanceModel(propulsion_model=custom_propulsion)

        assert model.propulsion.propeller_efficiency == 0.70
        assert model.propulsion.shaft_efficiency == 0.99

    def test_predict_calm_water_basic(self):
        """Test complete prediction in calm water."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        conditions = OperatingConditions(speed=15)  # Calm water

        result = model.predict(ship, conditions)

        # Should return PerformanceResult
        assert isinstance(result, PerformanceResult)

        # Check resistance components
        assert result.resistance_calm_water > 0
        assert result.resistance_wind == 0  # No wind
        assert result.resistance_waves == 0  # No waves
        assert result.resistance_total == result.resistance_calm_water

        # Check power chain
        assert result.effective_power > 0
        assert result.delivered_power > result.effective_power
        assert result.brake_power > result.delivered_power

        # Check fuel consumption
        assert result.fuel_rate > 0
        assert result.fuel_rate_per_day > 0

    def test_predict_with_weather(self):
        """Test prediction with wind and waves."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        conditions = OperatingConditions(
            speed=15, wind_speed=10, wind_angle=30, wave_height=2.5, wave_period=9, wave_angle=30
        )

        result = model.predict(ship, conditions)

        # All resistance components should be present
        assert result.resistance_calm_water > 0
        assert result.resistance_wind > 0
        assert result.resistance_waves > 0

        # Total should be sum
        total_calculated = (
            result.resistance_calm_water + result.resistance_wind + result.resistance_waves
        )
        assert abs(result.resistance_total - total_calculated) < 1.0

        # Fuel should be higher than calm water
        assert result.fuel_rate > 0

    def test_realistic_cargo_ship_consumption(self):
        """Test realistic fuel consumption for 150m cargo ship."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        conditions = OperatingConditions(speed=15)  # 15 knots

        result = model.predict(ship, conditions)

        # Realistic ranges for 150m cargo ship at 15 knots
        # Total resistance: 120-180 kN
        assert 120_000 < result.resistance_total < 180_000

        # Brake power: 1,500-2,500 kW
        assert 1_500 < result.brake_power < 2_500

        # Daily fuel: 6-10 tonnes/day
        assert 6 < result.fuel_rate_per_day < 10

    def test_weather_increases_fuel_consumption(self):
        """Test that weather conditions increase fuel consumption."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        # Calm
        calm = OperatingConditions(speed=15)

        # Weather
        weather = OperatingConditions(
            speed=15, wind_speed=12, wind_angle=30, wave_height=3, wave_period=10, wave_angle=30
        )

        result_calm = model.predict(ship, calm)
        result_weather = model.predict(ship, weather)

        # Weather should increase resistance and fuel
        assert result_weather.resistance_total > result_calm.resistance_total
        assert result_weather.brake_power > result_calm.brake_power
        assert result_weather.fuel_rate > result_calm.fuel_rate

        # Typical increase: 10-30%
        fuel_increase_pct = 100 * (result_weather.fuel_rate - result_calm.fuel_rate) / result_calm.fuel_rate
        assert 5 < fuel_increase_pct < 40

    def test_speed_consumption_curve(self):
        """Test generating speed-consumption curve."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        template = OperatingConditions(speed=15)  # Calm water
        speeds = [10, 12, 14, 16, 18, 20]

        curve = model.speed_consumption_curve(ship, template, speeds)

        # Should return list of results
        assert len(curve) == len(speeds)

        # Check that each result corresponds to correct speed
        for i, result in enumerate(curve):
            assert result.speed_knots == speeds[i]

        # Fuel consumption should increase with speed
        for i in range(len(curve) - 1):
            assert curve[i + 1].fuel_rate > curve[i].fuel_rate

        # Power should scale approximately as V³
        # Check ratio between first and last
        power_ratio = curve[-1].brake_power / curve[0].brake_power
        speed_ratio = speeds[-1] / speeds[0]
        expected_ratio = speed_ratio**3

        # Allow some tolerance (resistance doesn't scale exactly as V²)
        assert abs(power_ratio - expected_ratio) / expected_ratio < 0.3

    def test_voyage_simulation(self):
        """Test complete voyage simulation."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        conditions = OperatingConditions(
            speed=14, wind_speed=8, wind_angle=30, wave_height=2, wave_period=9, wave_angle=30
        )

        # 5-day voyage
        duration_hours = 120

        voyage = model.voyage_simulation(ship, conditions, duration_hours)

        # Check required fields
        assert "fuel_consumed" in voyage
        assert "average_fuel_rate" in voyage
        assert "average_power" in voyage
        assert "distance" in voyage
        assert "duration_hours" in voyage

        # Check values
        assert voyage["duration_hours"] == 120
        assert voyage["average_speed"] == 14

        # Distance = speed × time = 14 knots × 120 hours = 1,680 NM
        assert abs(voyage["distance"] - 1680) < 1

        # Fuel consumed should be reasonable (30-50 tonnes for 5 days)
        assert 25 < voyage["fuel_consumed"] < 60

        # CO2 should be calculated
        assert voyage["co2_emitted"] is not None
        assert voyage["co2_emitted"] > 0

    def test_different_fuel_types(self):
        """Test performance with different fuel types."""
        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        conditions = OperatingConditions(speed=15)

        # Test each fuel type
        results = {}
        for fuel_type in [FuelType.HFO, FuelType.MDO, FuelType.LNG]:
            from ship_performance.propulsion import FuelConsumptionModel

            fuel_model = FuelConsumptionModel(fuel_type=fuel_type)
            model = ShipPerformanceModel(fuel_model=fuel_model)

            results[fuel_type] = model.predict(ship, conditions)

        # Resistance and power should be same (independent of fuel)
        assert abs(results[FuelType.HFO].resistance_total - results[FuelType.LNG].resistance_total) < 1
        assert abs(results[FuelType.HFO].brake_power - results[FuelType.LNG].brake_power) < 1

        # LNG should have lower SFOC
        assert results[FuelType.LNG].specific_consumption < results[FuelType.HFO].specific_consumption

        # LNG should consume less fuel by mass
        assert results[FuelType.LNG].fuel_rate < results[FuelType.HFO].fuel_rate

    def test_result_summary_string(self):
        """Test that result summary string is generated."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        conditions = OperatingConditions(speed=15)

        result = model.predict(ship, conditions)

        summary = result.summary_string()

        # Should contain key information
        assert "Ship Performance Prediction Results" in summary
        assert "RESISTANCE BREAKDOWN" in summary
        assert "POWER REQUIREMENTS" in summary
        assert "FUEL CONSUMPTION" in summary

        # Should contain numerical values
        assert "kN" in summary
        assert "kW" in summary
        assert "kg/h" in summary

    def test_performance_result_structure(self):
        """Test that PerformanceResult has all required fields."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        conditions = OperatingConditions(speed=15, wind_speed=10, wave_height=2)

        result = model.predict(ship, conditions)

        # Resistance fields
        assert hasattr(result, "resistance_calm_water")
        assert hasattr(result, "resistance_wind")
        assert hasattr(result, "resistance_waves")
        assert hasattr(result, "resistance_total")

        # Power fields
        assert hasattr(result, "effective_power")
        assert hasattr(result, "delivered_power")
        assert hasattr(result, "brake_power")

        # Efficiency fields
        assert hasattr(result, "propeller_efficiency")
        assert hasattr(result, "shaft_efficiency")
        assert hasattr(result, "overall_efficiency")

        # Fuel fields
        assert hasattr(result, "fuel_rate")
        assert hasattr(result, "fuel_rate_per_day")
        assert hasattr(result, "specific_consumption")
        assert hasattr(result, "fuel_type")

        # Operating conditions
        assert hasattr(result, "speed_knots")
        assert hasattr(result, "speed_ms")

        # Optional
        assert hasattr(result, "co2_rate")
        assert hasattr(result, "engine_load_factor")

    def test_result_immutability(self):
        """Test that PerformanceResult is immutable."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        conditions = OperatingConditions(speed=15)

        result = model.predict(ship, conditions)

        # Should not be able to modify
        with pytest.raises(Exception):
            result.fuel_rate = 9999

    def test_large_container_ship(self):
        """Test performance for large container ship."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=300, beam=48, draft=14, displacement=100000, block_coefficient=0.65
        )

        conditions = OperatingConditions(speed=22)  # Fast service speed

        result = model.predict(ship, conditions)

        # Large ship at high speed should have high power requirements
        # Brake power should be > 10 MW for this ship at 22 knots
        assert result.brake_power > 10_000  # > 10 MW

        # Should consume significant fuel
        assert result.fuel_rate_per_day > 50

    def test_slow_steaming(self):
        """Test slow steaming scenario (fuel optimization)."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=200, beam=32, draft=11, displacement=40000, block_coefficient=0.68
        )

        # Normal speed
        normal = OperatingConditions(speed=20)

        # Slow steaming
        slow = OperatingConditions(speed=14)

        result_normal = model.predict(ship, normal)
        result_slow = model.predict(ship, slow)

        # Slow steaming should significantly reduce fuel consumption
        fuel_reduction_pct = 100 * (result_normal.fuel_rate - result_slow.fuel_rate) / result_normal.fuel_rate

        # Typical reduction: 40-60% (power scales as V³)
        assert fuel_reduction_pct > 30

    def test_storm_conditions_impact(self):
        """Test performance in storm conditions."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        # Calm
        calm = OperatingConditions(speed=15)

        # Storm
        storm = OperatingConditions(
            speed=15, wind_speed=20, wind_angle=0, wave_height=6, wave_period=12, wave_angle=0
        )

        result_calm = model.predict(ship, calm)
        result_storm = model.predict(ship, storm)

        # Storm should significantly increase resistance
        resistance_increase = 100 * (result_storm.resistance_total - result_calm.resistance_total) / result_calm.resistance_total

        # Typical increase: 30-50% in severe weather
        assert resistance_increase > 20

        # Fuel increase should be similar
        fuel_increase = 100 * (result_storm.fuel_rate - result_calm.fuel_rate) / result_calm.fuel_rate
        assert fuel_increase > 20

    def test_head_vs_following_seas(self):
        """Test difference between head seas and following seas."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        # Head seas (waves from ahead)
        head_seas = OperatingConditions(
            speed=15, wave_height=3, wave_period=10, wave_angle=0  # 0° = head seas
        )

        # Following seas (waves from behind)
        following_seas = OperatingConditions(
            speed=15, wave_height=3, wave_period=10, wave_angle=180  # 180° = following
        )

        result_head = model.predict(ship, head_seas)
        result_following = model.predict(ship, following_seas)

        # Head seas should have higher resistance than following seas
        # (though with simplified model, following might be zero)
        assert result_head.resistance_waves >= result_following.resistance_waves

    def test_consistency_across_predictions(self):
        """Test that repeated predictions give consistent results."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        conditions = OperatingConditions(speed=15, wind_speed=10, wave_height=2)

        result1 = model.predict(ship, conditions)
        result2 = model.predict(ship, conditions)

        # Should be exactly the same
        assert result1.resistance_total == result2.resistance_total
        assert result1.brake_power == result2.brake_power
        assert result1.fuel_rate == result2.fuel_rate

    def test_power_chain_relationships(self):
        """Test that power chain follows correct relationships."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        conditions = OperatingConditions(speed=15)

        result = model.predict(ship, conditions)

        # P_E < P_D < P_B (due to losses)
        assert result.effective_power < result.delivered_power
        assert result.delivered_power < result.brake_power

        # Check efficiency relationships
        assert abs(result.delivered_power - result.effective_power / result.propeller_efficiency) < 1
        assert abs(result.brake_power - result.delivered_power / result.shaft_efficiency) < 1

        # Overall efficiency
        expected_overall = result.propeller_efficiency * result.shaft_efficiency
        assert abs(result.overall_efficiency - expected_overall) < 0.001

    def test_efficiencies_in_realistic_range(self):
        """Test that calculated efficiencies are realistic."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        conditions = OperatingConditions(speed=15)

        result = model.predict(ship, conditions)

        # Propeller efficiency: typically 0.50-0.75
        assert 0.50 < result.propeller_efficiency < 0.80

        # Shaft efficiency: typically 0.96-0.99
        assert 0.95 < result.shaft_efficiency < 1.0

        # Overall propulsive efficiency: typically 0.55-0.75
        assert 0.50 < result.overall_efficiency < 0.80

    def test_co2_emissions_calculated(self):
        """Test that CO2 emissions are calculated."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        conditions = OperatingConditions(speed=15)

        result = model.predict(ship, conditions)

        # CO2 should be calculated
        assert result.co2_rate is not None
        assert result.co2_rate > 0

        # CO2 should be roughly 3x fuel rate (carbon conversion factor)
        # For HFO: ~3.1-3.2 kg CO2 per kg fuel
        co2_fuel_ratio = result.co2_rate / result.fuel_rate
        assert 2.8 < co2_fuel_ratio < 3.5

    def test_voyage_co2_emissions(self):
        """Test CO2 emissions in voyage simulation."""
        model = ShipPerformanceModel()

        ship = ShipParameters(
            length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7
        )

        conditions = OperatingConditions(speed=15)

        voyage = model.voyage_simulation(ship, conditions, duration_hours=120)

        # CO2 emissions should be calculated
        assert voyage["co2_emitted"] is not None
        assert voyage["co2_emitted"] > 0

        # Should be ~3x fuel consumed
        co2_fuel_ratio = voyage["co2_emitted"] / voyage["fuel_consumed"]
        assert 2.8 < co2_fuel_ratio < 3.5


class TestPerformanceResultDataclass:
    """Tests for PerformanceResult dataclass."""

    def test_performance_result_creation(self):
        """Test creating PerformanceResult directly."""
        result = PerformanceResult(
            resistance_calm_water=130000,
            resistance_wind=5000,
            resistance_waves=3000,
            resistance_total=138000,
            effective_power=1065,
            delivered_power=1638,
            brake_power=1672,
            propeller_efficiency=0.65,
            shaft_efficiency=0.98,
            overall_efficiency=0.637,
            fuel_rate=309,
            fuel_rate_per_day=7.4,
            specific_consumption=185,
            fuel_type=FuelType.HFO,
            speed_knots=15,
            speed_ms=7.72,
            co2_rate=962,
            engine_load_factor=0.80,
        )

        assert result.resistance_total == 138000
        assert result.fuel_rate == 309

    def test_performance_result_frozen(self):
        """Test that PerformanceResult is immutable."""
        result = PerformanceResult(
            resistance_calm_water=130000,
            resistance_wind=0,
            resistance_waves=0,
            resistance_total=130000,
            effective_power=1000,
            delivered_power=1538,
            brake_power=1570,
            propeller_efficiency=0.65,
            shaft_efficiency=0.98,
            overall_efficiency=0.637,
            fuel_rate=290,
            fuel_rate_per_day=7.0,
            specific_consumption=185,
            fuel_type=FuelType.HFO,
            speed_knots=15,
            speed_ms=7.72,
        )

        with pytest.raises(Exception):
            result.fuel_rate = 500
