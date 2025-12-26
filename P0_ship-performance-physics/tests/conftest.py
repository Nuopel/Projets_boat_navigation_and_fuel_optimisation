"""Pytest configuration and shared fixtures.

This module provides common fixtures used across the test suite.
"""

import pytest


# Fixtures will be added as MVPs are developed
# Example structure for future reference:

# @pytest.fixture
# def cargo_ship():
#     """Standard cargo ship for testing."""
#     from ship_performance.core import ShipParameters
#     return ShipParameters(
#         length=150,
#         beam=25,
#         draft=8,
#         displacement=15000,
#         block_coefficient=0.7,
#         wetted_surface=4000,
#         frontal_area=600
#     )
#
# @pytest.fixture
# def calm_conditions():
#     """Calm weather operating conditions."""
#     from ship_performance.core import OperatingConditions
#     return OperatingConditions(
#         speed=15,
#         wind_speed=0,
#         wind_angle=0,
#         wave_height=0,
#         wave_period=0,
#         wave_angle=0
#     )
