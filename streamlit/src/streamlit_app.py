from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from PIL import Image


SPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SPACE_ROOT.parent
DEFAULT_API_TIMEOUT_SECONDS = 15
ICON_MAX_SIZE = (64, 64)


def _default_image_path() -> Path:
    deployed_path = SPACE_ROOT / "agriculture.png"
    local_path = PROJECT_ROOT / "agriculture.png"
    return deployed_path if deployed_path.exists() else local_path


def _default_icon_dir() -> Path:
    deployed_path = SPACE_ROOT / "icones"
    local_path = PROJECT_ROOT / "streamlit" / "icones"
    return deployed_path if deployed_path.exists() else local_path


DEFAULT_IMAGE_PATH = _default_image_path()
DEFAULT_ICON_DIR = _default_icon_dir()

CROP_LABELS = {
    "Cassava": "Manioc",
    "Maize": "Maïs",
    "Plantains and others": "Plantain",
    "Potatoes": "Pommes de terre",
    "Rice, paddy": "Riz",
    "Sorghum": "Sorgho",
    "Soybeans": "Soja",
    "Sweet potatoes": "Patates douces",
    "Wheat": "Blé",
    "Yams": "Ignames",
}

CROP_ICON_FILES = {
    "Cassava": "cassava.png",
    "Maize": "corn.png",
    "Plantains and others": "plantain.png",
    "Potatoes": "potato.png",
    "Rice, paddy": "rice.png",
    "Sorghum": "sorghum.png",
    "Soybeans": "soybean.png",
    "Sweet potatoes": "sweet-potato.png",
    "Wheat": "wheat.png",
    "Yams": "yam.png",
}


class ApiError(RuntimeError):
    pass


def default_api_base_url() -> str:
    return os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _request_json(
    method: str,
    path: str,
    *,
    api_base_url: str | None = None,
    timeout: int = DEFAULT_API_TIMEOUT_SECONDS,
    **request_kwargs: Any,
) -> dict[str, Any]:
    base_url = (api_base_url or default_api_base_url()).rstrip("/")
    url = f"{base_url}{path}"
    try:
        response = requests.request(method, url, timeout=timeout, **request_kwargs)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ApiError(f"Impossible de joindre l'API FastAPI : {exc}") from exc
    return response.json()


def fetch_metadata(*, area: str | None = None, api_base_url: str | None = None) -> dict[str, Any]:
    params = {"area": area} if area else None
    return _request_json("GET", "/metadata", api_base_url=api_base_url, params=params)


def list_areas(metadata: dict[str, Any]) -> list[str]:
    return list(metadata.get("areas", []))


def list_crops(metadata: dict[str, Any]) -> list[str]:
    return list(metadata.get("crops", []))


def current_year(metadata: dict[str, Any]) -> int:
    return int(metadata["current_year"])


def build_default_context(metadata: dict[str, Any]) -> dict[str, float]:
    defaults = metadata["default_context"]

    def to_float_or_default(value: Any, fallback: float) -> float:
        if value is None:
            return fallback
        return float(value)

    return {
        "average_rain_fall_mm_per_year": to_float_or_default(defaults.get("average_rain_fall_mm_per_year"), 0.0),
        "pesticides_tonnes": to_float_or_default(defaults.get("pesticides_tonnes"), 0.0),
        "avg_temp": to_float_or_default(defaults.get("avg_temp"), 0.0),
    }


def translate_crop_name(crop: str) -> str:
    return CROP_LABELS.get(crop, crop)


def crop_display_label(crop: str) -> str:
    translated = translate_crop_name(crop)
    return f"{translated} ({crop})" if translated != crop else crop


def crop_icon_path(crop: str, icon_dir: Path | str = DEFAULT_ICON_DIR) -> Path | None:
    filename = CROP_ICON_FILES.get(crop)
    if not filename:
        return None

    path = Path(icon_dir) / filename
    if path.exists():
        return path
    return None


def predict_yield(
    *,
    area: str,
    crop: str,
    average_rain_fall_mm_per_year: float,
    pesticides_tonnes: float,
    avg_temp: float,
    year: int,
    api_base_url: str | None = None,
) -> float:
    payload = {
        "area": area,
        "crop": crop,
        "year": int(year),
        "average_rain_fall_mm_per_year": float(average_rain_fall_mm_per_year),
        "pesticides_tonnes": float(pesticides_tonnes),
        "avg_temp": float(avg_temp),
    }
    response = _request_json("POST", "/predict", api_base_url=api_base_url, json=payload)
    return float(response["predicted_yield_t_ha"])


def recommend_crops(
    *,
    area: str,
    hectares: float,
    average_rain_fall_mm_per_year: float,
    pesticides_tonnes: float,
    avg_temp: float,
    candidate_crops: list[str],
    year: int,
    api_base_url: str | None = None,
) -> pd.DataFrame:
    payload = {
        "area": area,
        "year": int(year),
        "average_rain_fall_mm_per_year": float(average_rain_fall_mm_per_year),
        "pesticides_tonnes": float(pesticides_tonnes),
        "avg_temp": float(avg_temp),
        "candidate_crops": candidate_crops,
    }
    response = _request_json("POST", "/recommend", api_base_url=api_base_url, json=payload)
    recommendations = response["recommendations"]
    hectares_value = float(hectares)
    records = [
        {
            "crop": item["crop"],
            "predicted_yield_t_ha": float(item["predicted_yield_t_ha"]),
            "predicted_total_production_tons": float(item["predicted_yield_t_ha"]) * hectares_value,
        }
        for item in recommendations
    ]
    return pd.DataFrame(records)


def format_recommendations_for_display(recommendations: pd.DataFrame) -> pd.DataFrame:
    display_df = recommendations.copy()
    display_df["Culture"] = display_df["crop"].map(crop_display_label)
    display_df["Rendement prédit (t/ha)"] = display_df["predicted_yield_t_ha"]
    display_df["Production totale prédite (tonnes)"] = display_df["predicted_total_production_tons"]
    return display_df[["Culture", "Rendement prédit (t/ha)", "Production totale prédite (tonnes)"]]


def display_stretch_image(image_path: str) -> None:
    try:
        st.image(image_path, width="stretch")
    except TypeError:
        try:
            st.image(image_path, use_container_width=True)
        except TypeError:
            st.image(image_path, use_column_width=True)


def display_stretch_dataframe(data) -> None:
    try:
        st.dataframe(data, width="stretch")
    except TypeError:
        st.dataframe(data, use_container_width=True)


def display_stretch_bar_chart(data) -> None:
    try:
        st.bar_chart(data, width="container")
    except TypeError:
        st.bar_chart(data, use_container_width=True)


def load_icon_for_display(image_path: str, max_size: tuple[int, int] = ICON_MAX_SIZE) -> BytesIO:
    with Image.open(image_path) as image:
        prepared = image.copy()
    prepared.thumbnail(max_size)
    buffer = BytesIO()
    prepared.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


@st.cache_data(show_spinner=False)
def get_metadata(area: str | None = None):
    return fetch_metadata(area=area)


def main() -> None:
    st.set_page_config(
        page_title="Rendement Agricole",
        page_icon="🌾",
        layout="wide",
    )

    try:
        base_metadata = get_metadata()
    except ApiError as exc:
        st.title("Rendement Agricole")
        st.error(str(exc))
        st.stop()

    areas = list_areas(base_metadata)
    crops = list_crops(base_metadata)
    year_used = current_year(base_metadata)

    if not areas or not crops:
        st.title("Rendement Agricole")
        st.error("L'API n'a retourné aucune zone ou culture exploitable.")
        st.stop()

    st.title("Rendement Agricole")
    st.caption(
        "Interface Streamlit découplée : les formulaires interrogent l'API FastAPI "
        "interne du conteneur Docker."
    )
    st.info(
        f"Année utilisée dans les scénarios : {year_used}. "
        f"Source du modèle côté API : {base_metadata.get('model_source', 'inconnue')}."
    )

    if DEFAULT_IMAGE_PATH.exists():
        display_stretch_image(str(DEFAULT_IMAGE_PATH))

    st.sidebar.header("Contexte de simulation")
    selected_area = st.sidebar.selectbox("Zone", areas)
    try:
        selected_area_metadata = get_metadata(selected_area)
    except ApiError:
        selected_area_metadata = base_metadata
    defaults = build_default_context(selected_area_metadata)

    hectares = st.sidebar.number_input("Surface (hectares)", min_value=0.1, value=10.0, step=0.5)
    rainfall = st.sidebar.number_input(
        "Pluie annuelle moyenne (mm)",
        min_value=0.0,
        value=round(defaults["average_rain_fall_mm_per_year"], 2),
        step=10.0,
    )
    pesticides = st.sidebar.number_input(
        "Pesticides (tonnes)",
        min_value=0.0,
        value=round(defaults["pesticides_tonnes"], 2),
        step=1.0,
    )
    avg_temp = st.sidebar.number_input(
        "Température moyenne (°C)",
        value=round(defaults["avg_temp"], 2),
        step=0.5,
    )

    predict_tab, recommend_tab = st.tabs(["Prédire un rendement", "Recommander une culture"])

    with predict_tab:
        selected_crop = st.selectbox(
            "Culture",
            crops,
            format_func=crop_display_label,
        )
        selected_crop_icon = crop_icon_path(selected_crop)
        if selected_crop_icon is not None:
            icon_col, text_col = st.columns([1, 5])
            icon_col.image(load_icon_for_display(str(selected_crop_icon)))
            text_col.markdown(f"**Culture sélectionnée :** {crop_display_label(selected_crop)}")

        if st.button("Calculer le rendement", type="primary"):
            try:
                predicted_yield = predict_yield(
                    area=selected_area,
                    crop=selected_crop,
                    average_rain_fall_mm_per_year=rainfall,
                    pesticides_tonnes=pesticides,
                    avg_temp=avg_temp,
                    year=year_used,
                )
            except ApiError as exc:
                st.error(str(exc))
                st.stop()

            total_production = predicted_yield * float(hectares)

            metric_1, metric_2 = st.columns(2)
            metric_1.metric("Rendement prédit", f"{predicted_yield:.2f} t/ha")
            metric_2.metric("Production totale", f"{total_production:.2f} tonnes")

            st.write(
                {
                    "zone": selected_area,
                    "culture": translate_crop_name(selected_crop),
                    "année utilisée": year_used,
                    "surface (hectares)": hectares,
                    "pluie annuelle moyenne (mm)": rainfall,
                    "pesticides (tonnes)": pesticides,
                    "température moyenne (°C)": avg_temp,
                }
            )

    with recommend_tab:
        default_selection = crops[: min(5, len(crops))]
        candidate_crops = st.multiselect(
            "Cultures à comparer",
            options=crops,
            default=default_selection,
            format_func=crop_display_label,
        )
        top_n = st.slider(
            "Nombre de recommandations à afficher",
            min_value=1,
            max_value=max(1, len(crops)),
            value=min(5, len(crops)),
        )

        if st.button("Lancer la recommandation"):
            if not candidate_crops:
                st.warning("Sélectionne au moins une culture.")
            else:
                try:
                    recommendations = recommend_crops(
                        area=selected_area,
                        hectares=hectares,
                        average_rain_fall_mm_per_year=rainfall,
                        pesticides_tonnes=pesticides,
                        avg_temp=avg_temp,
                        candidate_crops=candidate_crops,
                        year=year_used,
                    ).head(top_n)
                except ApiError as exc:
                    st.error(str(exc))
                    st.stop()

                display_recommendations = format_recommendations_for_display(recommendations)

                display_stretch_dataframe(
                    display_recommendations.style.format(
                        {
                            "Rendement prédit (t/ha)": "{:.2f}",
                            "Production totale prédite (tonnes)": "{:.2f}",
                        }
                    )
                )
                st.markdown("#### Aperçu visuel des cultures recommandées")
                preview_columns = st.columns(min(top_n, 5))
                for column, (_, row) in zip(preview_columns, recommendations.iterrows()):
                    icon_path = crop_icon_path(row["crop"])
                    if icon_path is not None:
                        column.image(load_icon_for_display(str(icon_path)))
                    column.caption(translate_crop_name(row["crop"]))
                    column.write(f"{row['predicted_total_production_tons']:.2f} t")
                display_stretch_bar_chart(
                    display_recommendations.set_index("Culture")["Production totale prédite (tonnes)"]
                )


if __name__ == "__main__":
    main()
