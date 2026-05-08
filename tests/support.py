"""Doubles et helpers partages par les tests du projet."""

from __future__ import annotations

from urllib.parse import urlsplit

import pandas as pd
import requests


class FakeAdjustedYieldService:
    """Double de test commun pour isoler l'API et le client Streamlit."""

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
        """Retourne un baseline deterministic pour les tests."""
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

    def predict_adjusted_yield(
        self,
        area: str,
        crop: str,
        user_conditions: dict[str, object],
        *,
        reference_overrides=None,
    ) -> dict[str, object]:
        """Retourne une prediction ajustee deterministic pour les tests."""
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
        """Retourne un classement deterministic pour les tests."""
        del user_conditions
        del reference_overrides
        crops = candidate_crops or self.crops_by_area[area]
        rows = [
            {
                "country": area,
                "crop": crop,
                "p1_historical_prediction": 5.0 + index,
                "p2_reference_simulation": 6.0,
                "p3_user_simulation": 6.4 + index * 0.2,
                "local_adjustment": 0.4 + index * 0.2,
                "gap_vs_historical_pct": 8.0 + index,
                "final_prediction": 7.5 - index * 0.5,
                "recommendation_rank": index + 1,
                "rainfall_reference_source": "row_latest_history",
                "temperature_reference_source": "crop_median",
            }
            for index, crop in enumerate(crops)
        ]
        return pd.DataFrame(rows)


class RequestsCompatibleResponse:
    """Adaptateur minimal pour simuler `requests.Response` dans les tests."""

    def __init__(self, response) -> None:
        self._response = response

    def raise_for_status(self) -> None:
        """Releve les erreurs HTTP au format attendu par `requests`."""
        try:
            self._response.raise_for_status()
        except Exception as exc:
            raise requests.HTTPError(str(exc)) from exc

    def json(self):
        """Retourne le corps JSON de la reponse proxifiee."""
        return self._response.json()


def bridge_request(client):
    """Redirige les appels `requests` du client Streamlit vers le TestClient FastAPI."""

    def request(method: str, url: str, timeout: int = 15, **kwargs):
        del timeout
        parsed = urlsplit(url)
        response = client.request(
            method,
            parsed.path,
            params=kwargs.get("params"),
            json=kwargs.get("json"),
        )
        return RequestsCompatibleResponse(response)

    return request
