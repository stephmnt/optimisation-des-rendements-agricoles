from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import requests
from fastapi.testclient import TestClient
from PIL import Image

import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_SRC = PROJECT_ROOT / "streamlit" / "src"
if str(STREAMLIT_SRC) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_SRC))

import streamlit_app


class FakeAdjustedYieldService:
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
    ):
        del user_conditions
        del reference_overrides
        crops = candidate_crops or self.crops_by_area[area]
        return __import__("pandas").DataFrame(
            [
                {
                    "country": area,
                    "crop": crop,
                    "p1_historical_prediction": 5.0 + idx,
                    "p2_reference_simulation": 6.0,
                    "p3_user_simulation": 6.4 + idx * 0.2,
                    "local_adjustment": 0.4 + idx * 0.2,
                    "gap_vs_historical_pct": 8.0 + idx,
                    "final_prediction": 7.5 - idx * 0.5,
                    "recommendation_rank": idx + 1,
                    "rainfall_reference_source": "row_latest_history",
                    "temperature_reference_source": "crop_median",
                }
                for idx, crop in enumerate(crops)
            ]
        )


class RequestsCompatibleResponse:
    def __init__(self, response) -> None:
        self._response = response

    def raise_for_status(self) -> None:
        try:
            self._response.raise_for_status()
        except Exception as exc:
            raise requests.HTTPError(str(exc)) from exc

    def json(self):
        return self._response.json()


def _bridge_request(client: TestClient):
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


def test_fetch_metadata_and_baseline_read_v2_catalog(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_adjusted_yield_service", lambda: FakeAdjustedYieldService())
    client = TestClient(main.app)
    monkeypatch.setattr(streamlit_app.requests, "request", _bridge_request(client))

    metadata = streamlit_app.fetch_metadata(country="France")
    baseline = streamlit_app.fetch_baseline(country="France", crop="Wheat")

    assert streamlit_app.list_countries(metadata) == ["France", "Kenya"]
    assert streamlit_app.list_crops(metadata) == ["Maize", "Wheat"]
    assert metadata["inferred_region"] == "North"
    assert baseline["p1_historical_prediction"] == 5.25
    assert baseline["reference_profile"]["region"] == "North"


def test_predict_and_recommend_adjusted_call_v2_api(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_adjusted_yield_service", lambda: FakeAdjustedYieldService())
    client = TestClient(main.app)
    monkeypatch.setattr(streamlit_app.requests, "request", _bridge_request(client))

    user_conditions = streamlit_app.build_user_conditions(
        region="South",
        soil_type="Clay",
        rainfall_mm=540.0,
        temperature_celsius=24.0,
        fertilizer_used=True,
        irrigation_used=True,
        weather_condition="Sunny",
        days_to_harvest=95.0,
    )
    prediction = streamlit_app.predict_adjusted(
        country="France",
        crop="Wheat",
        user_conditions=user_conditions,
    )
    recommendations = streamlit_app.recommend_adjusted(
        country="France",
        user_conditions=user_conditions,
        candidate_crops=["Wheat", "Maize"],
        top_n=2,
    )

    recommendation_df = streamlit_app.recommendation_records_to_frame(recommendations)

    assert prediction["final_prediction"] == 6.1
    assert prediction["local_adjustment"] == 0.85
    assert prediction["explanation"]["historical_shap"]["available"] is True
    assert recommendation_df["crop"].tolist() == ["Wheat", "Maize"]
    assert recommendation_df["final_prediction"].tolist() == [7.5, 7.0]


def test_format_recommendations_for_display_exposes_new_columns() -> None:
    raw_df = __import__("pandas").DataFrame(
        [
            {
                "country": "France",
                "crop": "Wheat",
                "p1_historical_prediction": 5.25,
                "p2_reference_simulation": 6.00,
                "p3_user_simulation": 6.85,
                "local_adjustment": 0.85,
                "gap_vs_historical_pct": 16.19,
                "final_prediction": 6.10,
                "recommendation_rank": 1,
            }
        ]
    )

    display_df = streamlit_app.format_recommendations_for_display(raw_df)

    assert display_df.columns.tolist() == [
        "Rang",
        "Culture",
        "P1 historique (t/ha)",
        "P2 référence",
        "P3 utilisateur",
        "Ajustement local",
        "Écart vs historique (%)",
        "Rendement final (t/ha)",
    ]


def test_crop_icon_path_uses_top_level_icones_directory() -> None:
    maize_icon = streamlit_app.crop_icon_path("Maize")
    wheat_icon = streamlit_app.crop_icon_path("Wheat")

    assert maize_icon is not None
    assert wheat_icon is not None
    assert maize_icon.parent.name == "icones"
    assert wheat_icon.parent.name == "icones"
    assert maize_icon.name == "corn.png"


def test_load_image_for_display_bounds_app_image_to_128_pixels(tmp_path: Path) -> None:
    image_path = tmp_path / "agriculture.png"
    Image.new("RGB", (640, 360), color="green").save(image_path)

    payload = streamlit_app.load_image_for_display(
        str(image_path),
        max_size=streamlit_app.APP_IMAGE_MAX_SIZE,
    )

    with Image.open(BytesIO(payload.getvalue())) as resized:
        assert resized.width <= 128
        assert resized.height <= 128


def test_predict_adjusted_raises_api_error_when_backend_is_unreachable(monkeypatch) -> None:
    def raising_request(*args, **kwargs):
        del args, kwargs
        raise requests.ConnectionError("backend down")

    monkeypatch.setattr(streamlit_app.requests, "request", raising_request)

    try:
        streamlit_app.predict_adjusted(
            country="France",
            crop="Wheat",
            user_conditions={
                "rainfall_mm": 540.0,
            },
        )
    except streamlit_app.ApiError as exc:
        assert "Impossible de joindre l'API v2 FastAPI" in str(exc)
    else:
        raise AssertionError("Une ApiError était attendue")
