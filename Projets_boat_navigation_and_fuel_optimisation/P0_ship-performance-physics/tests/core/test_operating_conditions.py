"""Tests for operating conditions dataclass."""

import warnings

import pytest

from ship_performance.core import InvalidOperatingConditionsError, OperatingConditions


class TestOperatingConditionsConstruction:
    """Tests for creating OperatingConditions instances."""

    def test_create_calm_water(self) -> None:
        """Test creating calm water conditions."""
        conditions = OperatingConditions(speed=15)

        assert conditions.speed == 15
        assert conditions.wind_speed == 0.0
        assert conditions.wave_height == 0.0
        assert conditions.is_calm_water

    def test_create_moderate_weather(self) -> None:
        """Test creating moderate weather conditions."""
        conditions = OperatingConditions(
            speed=15, wind_speed=10, wind_angle=45, wave_height=2, wave_period=8, wave_angle=30
        )

        assert conditions.speed == 15
        assert conditions.wind_speed == 10
        assert conditions.wind_angle == 45
        assert conditions.wave_height == 2
        assert conditions.wave_period == 8
        assert conditions.wave_angle == 30
        assert not conditions.is_calm_water

    def test_create_storm_conditions(self) -> None:
        """Test creating storm conditions."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore high wind/wave warnings
            conditions = OperatingConditions(
                speed=10,
                wind_speed=25,
                wind_angle=0,
                wave_height=8,
                wave_period=12,
                wave_angle=0,
            )

            assert conditions.speed == 10
            assert conditions.wind_speed == 25
            assert conditions.wave_height == 8


class TestSpeedValidation:
    """Tests for ship speed validation."""

    def test_valid_speed(self) -> None:
        """Test valid ship speeds."""
        for speed in [0, 5, 10, 15, 20, 25]:
            conditions = OperatingConditions(speed=speed)
            assert conditions.speed == speed

    def test_negative_speed(self) -> None:
        """Test that negative speed raises error."""
        with pytest.raises(InvalidOperatingConditionsError, match="cannot be negative"):
            OperatingConditions(speed=-10)

    def test_very_high_speed_warning(self) -> None:
        """Test warning for very high speed."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OperatingConditions(speed=60)  # 60 knots is very fast
            assert len(w) >= 1
            assert "very high" in str(w[0].message).lower()


class TestWindValidation:
    """Tests for wind parameter validation."""

    def test_valid_wind_speed(self) -> None:
        """Test valid wind speeds."""
        for wind_speed in [0, 5, 10, 15, 20]:
            conditions = OperatingConditions(speed=15, wind_speed=wind_speed)
            assert conditions.wind_speed == wind_speed

    def test_negative_wind_speed(self) -> None:
        """Test that negative wind speed raises error."""
        with pytest.raises(InvalidOperatingConditionsError, match="cannot be negative"):
            OperatingConditions(speed=15, wind_speed=-10)

    def test_hurricane_wind_warning(self) -> None:
        """Test warning for hurricane-force winds."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OperatingConditions(speed=15, wind_speed=35)  # >33 m/s is hurricane
            assert len(w) >= 1
            assert "hurricane" in str(w[0].message).lower()

    def test_valid_wind_angles(self) -> None:
        """Test valid wind angles."""
        for angle in [0, 45, 90, 135, 180]:
            conditions = OperatingConditions(speed=15, wind_speed=10, wind_angle=angle)
            assert conditions.wind_angle == angle

    def test_negative_wind_angle(self) -> None:
        """Test that negative wind angle raises error."""
        with pytest.raises(InvalidOperatingConditionsError, match="must be between 0 and 180"):
            OperatingConditions(speed=15, wind_speed=10, wind_angle=-45)

    def test_wind_angle_above_180(self) -> None:
        """Test that wind angle > 180 raises error."""
        with pytest.raises(InvalidOperatingConditionsError, match="must be between 0 and 180"):
            OperatingConditions(speed=15, wind_speed=10, wind_angle=270)


class TestWaveValidation:
    """Tests for wave parameter validation."""

    def test_valid_wave_height(self) -> None:
        """Test valid wave heights."""
        for hs in [0, 1, 2, 4, 6]:
            conditions = OperatingConditions(speed=15, wave_height=hs, wave_period=8)
            assert conditions.wave_height == hs

    def test_negative_wave_height(self) -> None:
        """Test that negative wave height raises error."""
        with pytest.raises(InvalidOperatingConditionsError, match="cannot be negative"):
            OperatingConditions(speed=15, wave_height=-2)

    def test_valid_wave_period(self) -> None:
        """Test valid wave periods."""
        for tp in [0, 5, 8, 10, 12]:
            conditions = OperatingConditions(speed=15, wave_height=2, wave_period=tp)
            assert conditions.wave_period == tp

    def test_negative_wave_period(self) -> None:
        """Test that negative wave period raises error."""
        with pytest.raises(InvalidOperatingConditionsError, match="cannot be negative"):
            OperatingConditions(speed=15, wave_height=2, wave_period=-8)

    def test_valid_wave_angles(self) -> None:
        """Test valid wave angles."""
        for angle in [0, 45, 90, 135, 180]:
            conditions = OperatingConditions(
                speed=15, wave_height=2, wave_period=8, wave_angle=angle
            )
            assert conditions.wave_angle == angle

    def test_negative_wave_angle(self) -> None:
        """Test that negative wave angle raises error."""
        with pytest.raises(InvalidOperatingConditionsError, match="must be between 0 and 180"):
            OperatingConditions(speed=15, wave_height=2, wave_period=8, wave_angle=-30)

    def test_wave_angle_above_180(self) -> None:
        """Test that wave angle > 180 raises error."""
        with pytest.raises(InvalidOperatingConditionsError, match="must be between 0 and 180"):
            OperatingConditions(speed=15, wave_height=2, wave_period=8, wave_angle=270)

    def test_steep_wave_warning(self) -> None:
        """Test warning for very steep waves."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Very steep: Hs=5m, Tp=5s → H/λ > 0.07
            OperatingConditions(speed=15, wave_height=5, wave_period=5)
            assert len(w) >= 1
            assert "steepness" in str(w[0].message).lower()

    def test_wave_height_without_period_warning(self) -> None:
        """Test warning when wave height specified but period is zero."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OperatingConditions(speed=15, wave_height=2, wave_period=0)
            assert len(w) >= 1
            assert "period is zero" in str(w[0].message).lower()


class TestDerivedProperties:
    """Tests for computed properties."""

    def test_speed_ms_conversion(self) -> None:
        """Test speed conversion to m/s."""
        conditions = OperatingConditions(speed=15)
        assert pytest.approx(conditions.speed_ms, rel=1e-6) == 7.716666666666667

    def test_is_calm_water_true(self) -> None:
        """Test calm water detection."""
        conditions = OperatingConditions(speed=15)
        assert conditions.is_calm_water

        conditions2 = OperatingConditions(speed=15, wind_speed=0, wave_height=0)
        assert conditions2.is_calm_water

    def test_is_calm_water_false_with_wind(self) -> None:
        """Test that wind prevents calm water status."""
        conditions = OperatingConditions(speed=15, wind_speed=10)
        assert not conditions.is_calm_water

    def test_is_calm_water_false_with_waves(self) -> None:
        """Test that waves prevent calm water status."""
        conditions = OperatingConditions(speed=15, wave_height=2)
        assert not conditions.is_calm_water

    def test_beaufort_scale_calm(self) -> None:
        """Test Beaufort scale for calm conditions."""
        conditions = OperatingConditions(speed=15, wind_speed=0)
        assert conditions.beaufort_scale == 0

    def test_beaufort_scale_moderate(self) -> None:
        """Test Beaufort scale for moderate breeze."""
        conditions = OperatingConditions(speed=15, wind_speed=8)  # ~15 knots
        assert conditions.beaufort_scale == 4

    def test_beaufort_scale_gale(self) -> None:
        """Test Beaufort scale for gale."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            conditions = OperatingConditions(speed=15, wind_speed=20)  # ~39 knots
            assert conditions.beaufort_scale == 8

    def test_sea_state_calm(self) -> None:
        """Test sea state for calm conditions."""
        conditions = OperatingConditions(speed=15, wave_height=0)
        assert conditions.sea_state == 0

    def test_sea_state_slight(self) -> None:
        """Test sea state for slight seas."""
        conditions = OperatingConditions(speed=15, wave_height=1.0)
        assert conditions.sea_state == 3

    def test_sea_state_moderate(self) -> None:
        """Test sea state for moderate seas."""
        conditions = OperatingConditions(speed=15, wave_height=2.0)
        assert conditions.sea_state == 4

    def test_sea_state_rough(self) -> None:
        """Test sea state for rough seas."""
        conditions = OperatingConditions(speed=15, wave_height=3.5)
        assert conditions.sea_state == 5

    def test_sea_state_very_rough(self) -> None:
        """Test sea state for very rough seas."""
        conditions = OperatingConditions(speed=15, wave_height=5.5)
        assert conditions.sea_state == 6


class TestImmutability:
    """Tests that OperatingConditions is immutable (frozen dataclass)."""

    def test_cannot_modify_speed(self) -> None:
        """Test that speed cannot be modified after creation."""
        conditions = OperatingConditions(speed=15)
        with pytest.raises((AttributeError, TypeError)):
            conditions.speed = 20  # type: ignore

    def test_cannot_modify_wind_speed(self) -> None:
        """Test that wind speed cannot be modified after creation."""
        conditions = OperatingConditions(speed=15, wind_speed=10)
        with pytest.raises((AttributeError, TypeError)):
            conditions.wind_speed = 15  # type: ignore


class TestStringRepresentation:
    """Tests for string representation."""

    def test_repr_calm_water(self) -> None:
        """Test __repr__ for calm water conditions."""
        conditions = OperatingConditions(speed=15)
        repr_str = repr(conditions)

        assert "15" in repr_str  # speed
        assert "calm" in repr_str.lower()

    def test_repr_moderate_weather(self) -> None:
        """Test __repr__ for moderate weather conditions."""
        conditions = OperatingConditions(
            speed=15, wind_speed=10, wind_angle=45, wave_height=2, wave_period=8, wave_angle=30
        )
        repr_str = repr(conditions)

        assert "15" in repr_str  # speed
        assert "10" in repr_str  # wind speed
        assert "2" in repr_str  # wave height or sea state
        assert "Beaufort" in repr_str or "beaufort" in repr_str.lower()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_speed(self) -> None:
        """Test zero speed (ship at rest)."""
        conditions = OperatingConditions(speed=0)
        assert conditions.speed == 0
        assert conditions.speed_ms == 0.0

    def test_boundary_wind_angles(self) -> None:
        """Test boundary wind angles (0 and 180)."""
        head_wind = OperatingConditions(speed=15, wind_speed=10, wind_angle=0)
        assert head_wind.wind_angle == 0

        following_wind = OperatingConditions(speed=15, wind_speed=10, wind_angle=180)
        assert following_wind.wind_angle == 180

    def test_boundary_wave_angles(self) -> None:
        """Test boundary wave angles (0 and 180)."""
        head_seas = OperatingConditions(speed=15, wave_height=2, wave_period=8, wave_angle=0)
        assert head_seas.wave_angle == 0

        following_seas = OperatingConditions(speed=15, wave_height=2, wave_period=8, wave_angle=180)
        assert following_seas.wave_angle == 180
