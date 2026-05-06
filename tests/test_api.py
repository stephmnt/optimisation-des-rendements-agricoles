"""Tests des endpoints FastAPI exposes par l'application finale."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

import main


class FakeAdjustedYieldService:
    """Double de test du service metier pour isoler l'API."""

    available_areas = ["France", "Kenya"]
    available_crops = ["Maize", "Rice, paddy", "Wheat"]
    crops_by_area = {
        "France": ["Maize", "Wheat"],
        "Kenya": ["Maize", "Rice, paddy"],
    }
    target_year = 2016
    historical_metadata = {"model_name": "random_forest_search_01"}
    simulation_metadata = {"model_name": "linear_regression"}
    simulation_global_reference = {
        "region": "North",
        "soil_type": "Sandy",
        "rainfall_mm": 620.0,
        "temperature_celsius": 21.0,
        "fertilizer_used": True,
        "irrigation_used": False,
        "weather_condition": "Sunny",
        "days_to_harvest": 110.0,
    }
    simulation_options = {
        "regions": ["North", "South"],
        "soil_types": ["Clay", "Sandy"],
        "weather_conditions": ["Cloudy", "Sunny"],
    }

    def get_baseline(self, area: str, crop: str, *, reference_overrides=None) -> dict[str, object]:
        """Retourne un baseline deterministic pour les tests API."""
        assert area
        assert crop
        return {
            "country": area,
            "crop": crop,
            "target_year": 2016,
            "p1_historical_prediction": 5.25,
            "reference_profile": {
                **self.simulation_global_reference,
                **(reference_overrides or {}),
            },
            "rainfall_reference_source": "row_latest_history",
            "temperature_reference_source": "crop_median",
        }

    def predict_adjusted_yield(self, area: str, crop: str, user_conditions: dict[str, object], *, reference_overrides=None) -> dict[str, object]:
        """Retourne une prediction ajustee deterministic pour les tests API."""
        assert area
        assert crop
        assert "rainfall_mm" in user_conditions
        reference_profile = {
            **self.simulation_global_reference,
            **(reference_overrides or {}),
        }
        return {
            "country": area,
            "crop": crop,
            "p1_historical_prediction": 5.25,
            "p2_reference_simulation": 6.00,
            "p3_user_simulation": 6.85,
            "local_adjustment": 0.85,
            "gap_vs_historical_pct": 16.19,
            "final_prediction": 6.10,
            "reference_profile": reference_profile,
            "user_profile": {
                **reference_profile,
                **user_conditions,
            },
            "explanation": {
                "historical_shap": {
                    "available": True,
                    "status": "ok",
                    "message": None,
                    "model_prediction": 5.25,
                    "base_value": 4.20,
                    "prediction_from_shap": 5.25,
                    "top_contributions": [
                        {
                            "feature": "target_yield_t_ha_2015",
                            "raw_value": 5.4,
                            "contribution": 0.7,
                            "abs_contribution": 0.7,
                        }
                    ],
                },
                "local_adjustment": {
                    "method": "exact_linear_delta_decomposition",
                    "reference_prediction": 6.00,
                    "user_prediction": 6.85,
                    "total_adjustment": 0.85,
                    "top_contributions": [
                        {
                            "feature": "rainfall_mm",
                            "reference_value": 620.0,
                            "user_value": 540.0,
                            "contribution_delta": -0.25,
                            "abs_contribution_delta": 0.25,
                        }
                    ],
                },
            },
            "rainfall_reference_source": "row_latest_history",
            "temperature_reference_source": "crop_median",
        }

    def recommend_crops(
        self,
        area: str,
        user_conditions: dict[str, object],
        candidate_crops: list[str] | None = None,
        *,
        reference_overrides=None,
    ) -> pd.DataFrame:
        """Retourne un classement deterministic pour les tests API."""
        del user_conditions
        del reference_overrides
        crops = candidate_crops or self.crops_by_area[area]
        rows = [
            {
                "country": area,
                "crop": crop,
                "p1_historical_prediction": 5.0 + index,
                "p2_reference_simulation": 6.0,
                "p3_user_simulation": 6.5 + index * 0.1,
                "local_adjustment": 0.5 + index * 0.1,
                "gap_vs_historical_pct": 10.0 + index,
                "final_prediction": 6.2 + (len(crops) - index),
                "recommendation_rank": index + 1,
                "rainfall_reference_source": "row_latest_history",
                "temperature_reference_source": "crop_median",
            }
            for index, crop in enumerate(crops)
        ]
        return pd.DataFrame(rows).sort_values("final_prediction", ascending=False).reset_index(drop=True)


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
