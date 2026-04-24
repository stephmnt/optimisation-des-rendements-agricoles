from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import requests
from fastapi.testclient import TestClient
from PIL import Image

import main
from main import PredictRequest, RecommendRequest, RecommendationItem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_SRC = PROJECT_ROOT / "streamlit" / "src"
if str(STREAMLIT_SRC) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_SRC))

import streamlit_app


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
            "average_rain_fall_mm_per_year": 650.0,
            "pesticides_tonnes": 10.0,
            "avg_temp": 19.0,
        }

    def predict(self, payload: PredictRequest) -> float:
        assert payload.area
        return 6.4321

    def recommend(self, payload: RecommendRequest) -> list[RecommendationItem]:
        return [
            RecommendationItem(crop="Rice, paddy", predicted_yield_t_ha=6.9),
            RecommendationItem(crop="Wheat", predicted_yield_t_ha=6.3),
            RecommendationItem(crop="Maize", predicted_yield_t_ha=5.5),
        ]


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


def test_fetch_metadata_reads_catalog_from_fastapi(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_prediction_service", lambda: FakePredictionService())
    client = TestClient(main.app)
    monkeypatch.setattr(streamlit_app.requests, "request", _bridge_request(client))

    metadata = streamlit_app.fetch_metadata(area="France")

    assert streamlit_app.list_areas(metadata) == ["France", "Kenya"]
    assert streamlit_app.list_crops(metadata) == ["Wheat", "Maize", "Rice, paddy"]
    assert streamlit_app.build_default_context(metadata) == {
        "average_rain_fall_mm_per_year": 720.0,
        "pesticides_tonnes": 15.0,
        "avg_temp": 14.0,
    }


def test_predict_and_recommend_crops_call_fastapi(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_prediction_service", lambda: FakePredictionService())
    client = TestClient(main.app)
    monkeypatch.setattr(streamlit_app.requests, "request", _bridge_request(client))

    prediction = streamlit_app.predict_yield(
        area="France",
        crop="Wheat",
        average_rain_fall_mm_per_year=700.0,
        pesticides_tonnes=12.0,
        avg_temp=14.0,
        year=2026,
    )
    recommendations = streamlit_app.recommend_crops(
        area="France",
        hectares=10.0,
        average_rain_fall_mm_per_year=700.0,
        pesticides_tonnes=12.0,
        avg_temp=14.0,
        candidate_crops=["Rice, paddy", "Wheat", "Maize"],
        year=2026,
    )

    assert prediction == 6.4321
    assert recommendations["crop"].tolist() == ["Rice, paddy", "Wheat", "Maize"]
    assert recommendations["predicted_total_production_tons"].tolist() == [69.0, 63.0, 55.0]


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


def test_predict_yield_raises_api_error_when_backend_is_unreachable(monkeypatch) -> None:
    def raising_request(*args, **kwargs):
        del args, kwargs
        raise requests.ConnectionError("backend down")

    monkeypatch.setattr(streamlit_app.requests, "request", raising_request)

    try:
        streamlit_app.predict_yield(
            area="France",
            crop="Wheat",
            average_rain_fall_mm_per_year=700.0,
            pesticides_tonnes=12.0,
            avg_temp=14.0,
            year=2026,
        )
    except streamlit_app.ApiError as exc:
        assert "Impossible de joindre l'API FastAPI" in str(exc)
    else:
        raise AssertionError("Une ApiError était attendue")


def test_build_default_context_accepts_null_values() -> None:
    defaults = streamlit_app.build_default_context(
        {
            "default_context": {
                "average_rain_fall_mm_per_year": 700.0,
                "pesticides_tonnes": None,
                "avg_temp": 14.0,
            }
        }
    )

    assert defaults == {
        "average_rain_fall_mm_per_year": 700.0,
        "pesticides_tonnes": 0.0,
        "avg_temp": 14.0,
    }
