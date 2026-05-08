"""Tests du client Streamlit et de ses helpers de presentation."""

from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import requests

import main
import streamlit_app
from tests.support import FakeAdjustedYieldService, bridge_request


def test_fetch_metadata_and_baseline_read_v2_catalog(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_adjusted_yield_service", lambda: FakeAdjustedYieldService())
    client = TestClient(main.app)
    monkeypatch.setattr(streamlit_app.requests, "request", bridge_request(client))

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
    monkeypatch.setattr(streamlit_app.requests, "request", bridge_request(client))

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
        "Rendement historique estimé (t/ha)",
        "Rendement de référence (t/ha)",
        "Rendement avec vos conditions (t/ha)",
        "Impact des conditions locales (t/ha)",
        "Écart vs historique (%)",
        "Rendement final estimé (t/ha)",
    ]


def test_business_labels_translate_soil_weather_and_region() -> None:
    assert streamlit_app.translate_region_name("North") == "Nord"
    assert streamlit_app.translate_soil_type("Clay") == "Argileux"
    assert streamlit_app.translate_weather_condition("Sunny") == "Ensoleillé"


def test_condition_profiles_comparison_frame_formats_values_for_business_users() -> None:
    frame = streamlit_app.condition_profiles_comparison_frame(
        {
            "region": "North",
            "soil_type": "Sandy",
            "rainfall_mm": 620.0,
            "temperature_celsius": 21.0,
            "fertilizer_used": True,
            "irrigation_used": False,
            "weather_condition": "Sunny",
            "days_to_harvest": 110.0,
        },
        {
            "region": "North",
            "soil_type": "Clay",
            "rainfall_mm": 540.0,
            "temperature_celsius": 24.0,
            "fertilizer_used": True,
            "irrigation_used": True,
            "weather_condition": "Rainy",
            "days_to_harvest": 95.0,
        },
    )

    soil_row = frame.loc[frame["Paramètre"] == "Type de sol"].iloc[0]
    weather_row = frame.loc[frame["Paramètre"] == "Météo dominante"].iloc[0]
    irrigation_row = frame.loc[frame["Paramètre"] == "Irrigation"].iloc[0]

    assert soil_row["Référence"] == "Sableux"
    assert soil_row["Votre parcelle"] == "Argileux"
    assert weather_row["Votre parcelle"] == "Pluvieux"
    assert irrigation_row["Référence"] == "Non"
    assert irrigation_row["Votre parcelle"] == "Oui"


def test_prediction_breakdown_frame_uses_business_friendly_labels() -> None:
    breakdown_df = streamlit_app.prediction_breakdown_frame(
        {
            "p1_historical_prediction": 5.25,
            "p2_reference_simulation": 6.00,
            "p3_user_simulation": 6.85,
            "local_adjustment": 0.85,
            "final_prediction": 6.10,
        }
    )

    assert breakdown_df["composant"].tolist() == [
        "Rendement historique estimé",
        "Rendement de référence",
        "Rendement avec vos conditions",
        "Impact de vos conditions",
        "Rendement final estimé",
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
