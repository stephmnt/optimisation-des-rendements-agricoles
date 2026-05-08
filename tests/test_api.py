"""Tests des endpoints FastAPI exposes par l'application finale."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

import main
from tests.support import FakeAdjustedYieldService


def test_v2_health_endpoint_reports_strategy(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_adjusted_yield_service", lambda: FakeAdjustedYieldService())
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "strategy": "2_models_3_predictions_combined",
        "historical_model_name": "random_forest_search_01",
        "simulation_model_name": "linear_regression",
    }


def test_v2_metadata_endpoint_returns_country_specific_catalog(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_adjusted_yield_service", lambda: FakeAdjustedYieldService())
    client = TestClient(main.app)

    response = client.get("/metadata", params={"country": "France"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["countries"] == ["France", "Kenya"]
    assert payload["available_crops"] == ["Maize", "Wheat"]
    assert payload["target_year"] == 2016
    assert payload["global_reference_profile"]["region"] == "North"
    assert payload["inferred_region"] == "North"


def test_v2_baseline_endpoint_returns_p1_and_reference_profile(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_adjusted_yield_service", lambda: FakeAdjustedYieldService())
    client = TestClient(main.app)

    response = client.post("/baseline", json={"country": "France", "crop": "Wheat"})

    assert response.status_code == 200
    assert response.json()["p1_historical_prediction"] == 5.25
    assert response.json()["reference_profile"]["rainfall_mm"] == 620.0


def test_v2_predict_endpoint_returns_adjusted_breakdown(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_adjusted_yield_service", lambda: FakeAdjustedYieldService())
    client = TestClient(main.app)

    response = client.post(
        "/predict",
        json={
            "country": "France",
            "crop": "Wheat",
            "user_conditions": {
                "region": "South",
                "soil_type": "Clay",
                "rainfall_mm": 540.0,
                "temperature_celsius": 24.0,
                "fertilizer_used": True,
                "irrigation_used": True,
                "weather_condition": "Sunny",
                "days_to_harvest": 95.0,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["final_prediction"] == 6.1
    assert payload["local_adjustment"] == 0.85
    assert payload["user_profile"]["region"] == "South"
    assert payload["explanation"]["historical_shap"]["available"] is True
    assert payload["explanation"]["local_adjustment"]["total_adjustment"] == 0.85


def test_v2_recommend_endpoint_returns_ranked_table(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_adjusted_yield_service", lambda: FakeAdjustedYieldService())
    client = TestClient(main.app)

    response = client.post(
        "/recommend",
        json={
            "country": "Kenya",
            "user_conditions": {
                "rainfall_mm": 580.0,
                "temperature_celsius": 23.0,
            },
            "candidate_crops": ["Rice, paddy", "Maize"],
            "top_n": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["country"] == "Kenya"
    assert payload["best_crop"] == payload["recommendations"][0]["crop"]
    assert len(payload["recommendations"]) == 2
    assert payload["recommendations"][0]["final_prediction"] >= payload["recommendations"][1]["final_prediction"]
