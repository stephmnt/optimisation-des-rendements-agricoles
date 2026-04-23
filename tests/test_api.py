from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

import main
from main import PredictRequest, PredictionService, RecommendRequest, RecommendationItem


class FakePredictionService:
    model_source = "unit-test"
    available_areas = ["France", "Kenya"]
    available_crops = ["Wheat", "Maize", "Rice, paddy"]

    def default_context(self, area: str | None = None) -> dict[str, float]:
        if area == "France":
            return {
                "average_rain_fall_mm_per_year": 720.0,
                "pesticides_tonnes": 15.0,
                "avg_temp": 14.0,
            }
        return {
            "average_rain_fall_mm_per_year": 600.0,
            "pesticides_tonnes": 12.0,
            "avg_temp": 18.0,
        }

    def predict(self, payload: PredictRequest) -> float:
        assert payload.crop
        return 7.6543

    def recommend(self, payload: RecommendRequest) -> list[RecommendationItem]:
        return [
            RecommendationItem(crop="Wheat", predicted_yield_t_ha=8.2),
            RecommendationItem(crop="Maize", predicted_yield_t_ha=7.4),
            RecommendationItem(crop="Rice, paddy", predicted_yield_t_ha=6.1),
        ]


def _sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "area": "France",
                "crop": "Wheat",
                "year": 2022,
                "average_rain_fall_mm_per_year": 700.0,
                "pesticides_tonnes": 20.0,
                "avg_temp": 13.0,
                "target_yield_t_ha": 5.4,
            },
            {
                "area": "France",
                "crop": "Maize",
                "year": 2022,
                "average_rain_fall_mm_per_year": 680.0,
                "pesticides_tonnes": 18.0,
                "avg_temp": 15.0,
                "target_yield_t_ha": 6.2,
            },
            {
                "area": "Kenya",
                "crop": "Maize",
                "year": 2022,
                "average_rain_fall_mm_per_year": 900.0,
                "pesticides_tonnes": 6.0,
                "avg_temp": 22.0,
                "target_yield_t_ha": 4.7,
            },
            {
                "area": "Kenya",
                "crop": "Rice, paddy",
                "year": 2022,
                "average_rain_fall_mm_per_year": 980.0,
                "pesticides_tonnes": 8.0,
                "avg_temp": 24.0,
                "target_yield_t_ha": 5.8,
            },
        ]
    )


def test_prediction_service_trains_fallback_model_when_artifact_is_missing(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    _sample_dataset().to_csv(dataset_path, index=False)

    service = PredictionService(
        dataset_path=dataset_path,
        model_path=tmp_path / "missing_model.joblib",
    )

    prediction = service.predict(
        PredictRequest(
            area="France",
            crop="Wheat",
            year=2026,
            average_rain_fall_mm_per_year=710.0,
            pesticides_tonnes=16.0,
            avg_temp=14.0,
        )
    )

    assert service.model_source == "fallback-trained"
    assert prediction >= 0.0
    assert service.available_crops == ["Maize", "Rice, paddy", "Wheat"]


def test_default_context_falls_back_to_global_median_when_area_values_are_missing(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        [
            {
                "area": "Area A",
                "crop": "Wheat",
                "year": 2022,
                "average_rain_fall_mm_per_year": 700.0,
                "pesticides_tonnes": None,
                "avg_temp": 13.0,
                "target_yield_t_ha": 5.4,
            },
            {
                "area": "Area A",
                "crop": "Maize",
                "year": 2022,
                "average_rain_fall_mm_per_year": 720.0,
                "pesticides_tonnes": None,
                "avg_temp": 14.0,
                "target_yield_t_ha": 6.2,
            },
            {
                "area": "Area B",
                "crop": "Maize",
                "year": 2022,
                "average_rain_fall_mm_per_year": 900.0,
                "pesticides_tonnes": 8.0,
                "avg_temp": 22.0,
                "target_yield_t_ha": 4.7,
            },
        ]
    )
    dataset_path = tmp_path / "dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    service = PredictionService(
        dataset_path=dataset_path,
        model_path=tmp_path / "missing_model.joblib",
    )

    defaults = service.default_context("Area A")

    assert defaults["average_rain_fall_mm_per_year"] == 710.0
    assert defaults["pesticides_tonnes"] == 8.0
    assert defaults["avg_temp"] == 13.5


def test_health_endpoint_reports_runtime_state(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_prediction_service", lambda: FakePredictionService())
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_source": "unit-test"}


def test_metadata_endpoint_returns_catalog_and_area_defaults(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_prediction_service", lambda: FakePredictionService())
    client = TestClient(main.app)

    response = client.get("/metadata", params={"area": "France"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["areas"] == ["France", "Kenya"]
    assert payload["crops"] == ["Wheat", "Maize", "Rice, paddy"]
    assert payload["default_context"] == {
        "average_rain_fall_mm_per_year": 720.0,
        "pesticides_tonnes": 15.0,
        "avg_temp": 14.0,
    }


def test_predict_endpoint_returns_prediction(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_prediction_service", lambda: FakePredictionService())
    client = TestClient(main.app)

    response = client.post(
        "/predict",
        json={
            "area": "France",
            "crop": "Wheat",
            "year": 2026,
            "average_rain_fall_mm_per_year": 700.0,
            "pesticides_tonnes": 12.0,
            "avg_temp": 14.0,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"predicted_yield_t_ha": 7.6543}


def test_recommend_endpoint_returns_ranked_predictions(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_prediction_service", lambda: FakePredictionService())
    client = TestClient(main.app)

    response = client.post(
        "/recommend",
        json={
            "area": "Kenya",
            "year": 2026,
            "average_rain_fall_mm_per_year": 850.0,
            "pesticides_tonnes": 7.0,
            "avg_temp": 23.0,
            "candidate_crops": ["Wheat", "Maize", "Rice, paddy"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "recommendations": [
            {"crop": "Wheat", "predicted_yield_t_ha": 8.2},
            {"crop": "Maize", "predicted_yield_t_ha": 7.4},
            {"crop": "Rice, paddy", "predicted_yield_t_ha": 6.1},
        ]
    }
