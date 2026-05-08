"""Tests du validateur runtime local du projet."""

from __future__ import annotations

import pytest

from scripts.validate_runtime import (
    RUNTIME_REQUIRED_ARTIFACTS,
    build_smoke_user_conditions,
    pick_area_and_crop,
)


class _FakeService:
    """Double de test minimal pour les helpers de validation runtime."""

    available_areas = ["France", "Kenya"]
    crops_by_area = {
        "France": ["Barley", "Wheat"],
        "Kenya": ["Cassava"],
    }
    simulation_options = {
        "regions": ["North", "South"],
        "soil_types": ["Clay", "Sandy"],
        "weather_conditions": ["Sunny", "Rainy"],
    }


def test_pick_area_and_crop_uses_first_available_pair_by_default() -> None:
    area, crop = pick_area_and_crop(_FakeService())
    assert area == "France"
    assert crop == "Barley"


def test_pick_area_and_crop_rejects_unknown_crop_for_country() -> None:
    with pytest.raises(ValueError, match="not available"):
        pick_area_and_crop(_FakeService(), country="France", crop="Cassava")


def test_build_smoke_user_conditions_switches_values_when_alternatives_exist() -> None:
    reference_profile = {
        "region": "North",
        "soil_type": "Clay",
        "rainfall_mm": 500.0,
        "temperature_celsius": 20.0,
        "fertilizer_used": True,
        "irrigation_used": False,
        "weather_condition": "Sunny",
        "days_to_harvest": 120.0,
    }

    conditions = build_smoke_user_conditions(
        _FakeService(),
        reference_profile=reference_profile,
    )

    assert conditions["region"] == "South"
    assert conditions["soil_type"] == "Sandy"
    assert conditions["weather_condition"] == "Rainy"
    assert conditions["rainfall_mm"] == 525.0
    assert conditions["temperature_celsius"] == 21.5
    assert conditions["fertilizer_used"] is False
    assert conditions["irrigation_used"] is True
    assert conditions["days_to_harvest"] == 127.0


def test_runtime_validation_artifacts_do_not_reference_best_pipeline() -> None:
    assert all("best_pipeline" not in str(path) for path in RUNTIME_REQUIRED_ARTIFACTS)
