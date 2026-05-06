"""Interface Streamlit metier pour la prediction et la recommandation.

Le front reste leger: il recupere le catalogue et les predictions via l'API
FastAPI, puis reformule les resultats dans un vocabulaire compréhensible pour
un profil agricole non technique.
"""

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
APP_IMAGE_MAX_SIZE = (128, 128)
ICON_MAX_SIZE = (64, 64)
APP_BADGES_MARKDOWN = (
    "[![GitHub](https://img.shields.io/badge/github-repo?logo=github&labelColor=black&color=blue)]"
    "(https://github.com/stephmnt/optimisation-des-rendements-agricoles) "
    "![Python](https://img.shields.io/badge/code-python?logo=python&logoColor=green&label=python&color=green) "
    "[![GitHub Release Date](https://img.shields.io/github/release-date/stephmnt/optimisation-des-rendements-agricoles?display_date=published_at&style=flat-square)]"
    "(https://github.com/stephmnt/optimisation-des-rendements-agricoles/releases) "
    "[![project_license](https://img.shields.io/github/license/stephmnt/optimisation-des-rendements-agricoles.svg)]"
    "(https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/LICENSE)"
)


def _default_image_path() -> Path:
    """Selectionne l'image principale selon le contexte local ou deploye."""
    deployed_path = SPACE_ROOT / "agriculture.png"
    local_path = PROJECT_ROOT / "agriculture.png"
    return deployed_path if deployed_path.exists() else local_path


def _default_icon_dir() -> Path:
    """Selectionne le dossier d'icones selon le contexte local ou deploye."""
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

REGION_LABELS = {
    "North": "Nord",
    "South": "Sud",
    "East": "Est",
    "West": "Ouest",
}

SOIL_TYPE_LABELS = {
    "Chalky": "Calcaire",
    "Clay": "Argileux",
    "Loam": "Terre franche",
    "Peaty": "Tourbeux",
    "Sandy": "Sableux",
    "Silt": "Limoneux",
}

WEATHER_CONDITION_LABELS = {
    "Cloudy": "Nuageux",
    "Rainy": "Pluvieux",
    "Sunny": "Ensoleillé",
}

REFERENCE_SOURCE_LABELS = {
    "row_latest_history": "Dernière valeur historique disponible",
    "crop_median": "Médiane observée pour cette culture",
    "simulation_global_default": "Valeur moyenne du jeu de simulation",
}

CONDITION_FIELD_LABELS = {
    "region": "Zone agro-climatique",
    "soil_type": "Type de sol",
    "rainfall_mm": "Pluviométrie (mm)",
    "temperature_celsius": "Température moyenne (degC)",
    "fertilizer_used": "Apport d'engrais",
    "irrigation_used": "Irrigation",
    "weather_condition": "Météo dominante",
    "days_to_harvest": "Durée avant récolte (jours)",
}


class ApiError(RuntimeError):
    """Erreur levee quand le client Streamlit ne peut pas joindre l'API."""

    pass


def default_api_base_url() -> str:
    """Retourne l'URL de base de l'API ciblee par l'interface."""
    return (
        os.getenv("API_V2_BASE_URL")
        or os.getenv("API_BASE_URL_V2")
        or "http://127.0.0.1:8001"
    ).rstrip("/")


def _request_json(
    method: str,
    path: str,
    *,
    api_base_url: str | None = None,
    timeout: int = DEFAULT_API_TIMEOUT_SECONDS,
    **request_kwargs: Any,
) -> dict[str, Any]:
    """Execute une requete HTTP vers l'API et retourne son JSON.

    Args:
        method: Verbe HTTP utilise.
        path: Chemin relatif de l'endpoint.
        api_base_url: URL de base optionnelle de l'API.
        timeout: Timeout HTTP en secondes.
        **request_kwargs: Arguments transmis a `requests.request`.

    Returns:
        dict[str, Any]: Corps JSON de la reponse.
    """
    base_url = (api_base_url or default_api_base_url()).rstrip("/")
    url = f"{base_url}{path}"
    try:
        response = requests.request(method, url, timeout=timeout, **request_kwargs)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ApiError(f"Impossible de joindre l'API v2 FastAPI : {exc}") from exc
    return response.json()


def fetch_metadata(*, country: str | None = None, api_base_url: str | None = None) -> dict[str, Any]:
    """Recupere le catalogue de l'API, eventuellement filtre par pays."""
    params = {"country": country} if country else None
    return _request_json("GET", "/metadata", api_base_url=api_base_url, params=params)


def fetch_baseline(*, country: str, crop: str, api_base_url: str | None = None) -> dict[str, Any]:
    """Recupere le baseline historique pour un couple pays/culture."""
    return _request_json(
        "POST",
        "/baseline",
        api_base_url=api_base_url,
        json={"country": country, "crop": crop},
    )


def predict_adjusted(
    *,
    country: str,
    crop: str,
    user_conditions: dict[str, Any],
    api_base_url: str | None = None,
) -> dict[str, Any]:
    """Demande une prediction ajustee a l'API FastAPI."""
    return _request_json(
        "POST",
        "/predict",
        api_base_url=api_base_url,
        json={
            "country": country,
            "crop": crop,
            "user_conditions": user_conditions,
        },
    )


def recommend_adjusted(
    *,
    country: str,
    user_conditions: dict[str, Any],
    candidate_crops: list[str] | None = None,
    top_n: int | None = None,
    api_base_url: str | None = None,
) -> dict[str, Any]:
    """Demande un classement multi-cultures a l'API FastAPI."""
    payload: dict[str, Any] = {
        "country": country,
        "user_conditions": user_conditions,
    }
    if candidate_crops:
        payload["candidate_crops"] = candidate_crops
    if top_n is not None:
        payload["top_n"] = int(top_n)
    return _request_json("POST", "/recommend", api_base_url=api_base_url, json=payload)


def list_countries(metadata: dict[str, Any]) -> list[str]:
    """Extrait la liste des pays du payload metadata."""
    return list(metadata.get("countries", []))


def list_crops(metadata: dict[str, Any]) -> list[str]:
    """Extrait la liste des cultures disponibles du payload metadata."""
    return list(metadata.get("available_crops", []))


def global_reference_profile(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extrait le profil global de reference depuis le payload metadata."""
    return dict(metadata["global_reference_profile"])


def translate_crop_name(crop: str) -> str:
    """Traduit un nom de culture brut vers un libelle metier francise."""
    return CROP_LABELS.get(crop, crop)


def crop_display_label(crop: str) -> str:
    """Construit le libelle complet d'affichage d'une culture."""
    translated = translate_crop_name(crop)
    return f"{translated} ({crop})" if translated != crop else crop


def _translated_value(raw_value: str, translations: dict[str, str]) -> str:
    """Applique une table de traduction a une valeur categorielle."""
    return translations.get(raw_value, raw_value)


def translate_region_name(region: str) -> str:
    """Traduit une region technique en libelle metier francais."""
    return _translated_value(region, REGION_LABELS)


def translate_soil_type(soil_type: str) -> str:
    """Traduit un type de sol technique en libelle metier francais."""
    return _translated_value(soil_type, SOIL_TYPE_LABELS)


def translate_weather_condition(weather_condition: str) -> str:
    """Traduit une condition meteo technique en libelle metier francais."""
    return _translated_value(weather_condition, WEATHER_CONDITION_LABELS)


def condition_field_label(field_name: str) -> str:
    """Traduit un nom de variable interne en libelle d'interface."""
    return CONDITION_FIELD_LABELS.get(field_name, field_name)


def option_display_label(option_kind: str, raw_value: str) -> str:
    """Traduit une option de formulaire selon son type."""
    if option_kind == "region":
        return translate_region_name(raw_value)
    if option_kind == "soil_type":
        return translate_soil_type(raw_value)
    if option_kind == "weather_condition":
        return translate_weather_condition(raw_value)
    return raw_value


def condition_value_for_display(field_name: str, value: Any) -> Any:
    """Normalise une valeur de condition pour l'affichage utilisateur."""
    if value is None:
        return None
    if field_name == "region":
        return translate_region_name(str(value))
    if field_name == "soil_type":
        return translate_soil_type(str(value))
    if field_name == "weather_condition":
        return translate_weather_condition(str(value))
    if field_name in {"fertilizer_used", "irrigation_used"}:
        return "Oui" if bool(value) else "Non"
    return value


def condition_profile_for_display(profile: dict[str, Any]) -> dict[str, Any]:
    """Remet en forme un profil de conditions pour un affichage lisible."""
    ordered_fields = [
        "region",
        "soil_type",
        "rainfall_mm",
        "temperature_celsius",
        "fertilizer_used",
        "irrigation_used",
        "weather_condition",
        "days_to_harvest",
    ]
    return {
        condition_field_label(field_name): condition_value_for_display(field_name, profile.get(field_name))
        for field_name in ordered_fields
        if field_name in profile
    }


def condition_profiles_comparison_frame(
    reference_profile: dict[str, Any],
    user_profile: dict[str, Any],
) -> pd.DataFrame:
    """Construit un tableau comparatif entre profil de reference et saisie utilisateur."""
    ordered_fields = [
        "region",
        "soil_type",
        "rainfall_mm",
        "temperature_celsius",
        "fertilizer_used",
        "irrigation_used",
        "weather_condition",
        "days_to_harvest",
    ]
    return pd.DataFrame(
        [
            {
                "Paramètre": condition_field_label(field_name),
                "Référence": condition_value_for_display(field_name, reference_profile.get(field_name)),
                "Votre parcelle": condition_value_for_display(field_name, user_profile.get(field_name)),
            }
            for field_name in ordered_fields
            if field_name in reference_profile or field_name in user_profile
        ]
    )


def reference_source_display_label(source_name: str) -> str:
    """Traduit la provenance d'une reference metier en libelle lisible."""
    return REFERENCE_SOURCE_LABELS.get(source_name, source_name)


def translate_feature_name(feature_name: str) -> str:
    """Traduit un nom de feature modele en libelle compréhensible."""
    if feature_name in CONDITION_FIELD_LABELS:
        return condition_field_label(feature_name)
    if feature_name.startswith("target_yield_t_ha_"):
        year = feature_name.rsplit("_", 1)[-1]
        return f"Rendement historique ({year})"
    if feature_name.startswith("avg_temp_"):
        year = feature_name.rsplit("_", 1)[-1]
        return f"Température moyenne ({year})"
    if feature_name.startswith("average_rain_fall_mm_per_year_"):
        year = feature_name.rsplit("_", 1)[-1]
        return f"Pluviométrie ({year})"
    if feature_name.startswith("pesticides_tonnes_"):
        year = feature_name.rsplit("_", 1)[-1]
        return f"Pesticides ({year})"
    if feature_name.startswith("yield_t_ha_"):
        year = feature_name.rsplit("_", 1)[-1]
        return f"Rendement ({year})"
    return feature_name.replace("_", " ")


def translate_feature_value(feature_name: str, value: Any) -> Any:
    """Traduit ou reformate la valeur d'une feature pour l'analyse detaillee."""
    if value is None:
        return None
    if feature_name in {"region", "soil_type", "weather_condition"}:
        return condition_value_for_display(feature_name, value)
    if feature_name in {"fertilizer_used", "irrigation_used"}:
        return condition_value_for_display(feature_name, value)
    return value


def crop_icon_path(crop: str, icon_dir: Path | str = DEFAULT_ICON_DIR) -> Path | None:
    """Retourne l'icone associee a une culture si elle existe."""
    filename = CROP_ICON_FILES.get(crop)
    if not filename:
        return None
    path = Path(icon_dir) / filename
    return path if path.exists() else None


def load_image_for_display(image_path: str, max_size: tuple[int, int]) -> BytesIO:
    """Charge une image et la redimensionne pour Streamlit."""
    with Image.open(image_path) as image:
        prepared = image.copy()
    prepared.thumbnail(max_size)
    buffer = BytesIO()
    prepared.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def load_icon_for_display(image_path: str, max_size: tuple[int, int] = ICON_MAX_SIZE) -> BytesIO:
    """Charge une icone en appliquant la taille standard de l'interface."""
    return load_image_for_display(image_path, max_size=max_size)


def display_bounded_image(image_path: str, max_size: tuple[int, int] = APP_IMAGE_MAX_SIZE) -> None:
    """Affiche une image dans Streamlit avec une taille bornee."""
    st.image(load_image_for_display(image_path, max_size=max_size))


def display_stretch_dataframe(data) -> None:
    """Affiche un dataframe en tenant compte des variantes de l'API Streamlit."""
    try:
        st.dataframe(data, width="stretch")
    except TypeError:
        st.dataframe(data, use_container_width=True)


@st.cache_data(show_spinner=False)
def get_metadata(country: str | None = None) -> dict[str, Any]:
    """Version cachee du chargement de metadata pour l'interface."""
    return fetch_metadata(country=country)


@st.cache_data(show_spinner=False)
def get_baseline(country: str, crop: str) -> dict[str, Any]:
    """Version cachee du chargement du baseline pour l'interface."""
    return fetch_baseline(country=country, crop=crop)


def build_user_conditions(
    *,
    region: str,
    soil_type: str,
    rainfall_mm: float,
    temperature_celsius: float,
    fertilizer_used: bool,
    irrigation_used: bool,
    weather_condition: str,
    days_to_harvest: float,
) -> dict[str, Any]:
    """Construit le payload `user_conditions` attendu par l'API."""
    return {
        "region": region,
        "soil_type": soil_type,
        "rainfall_mm": float(rainfall_mm),
        "temperature_celsius": float(temperature_celsius),
        "fertilizer_used": bool(fertilizer_used),
        "irrigation_used": bool(irrigation_used),
        "weather_condition": weather_condition,
        "days_to_harvest": float(days_to_harvest),
    }


def recommendation_records_to_frame(response_payload: dict[str, Any]) -> pd.DataFrame:
    """Convertit la reponse de recommandation en dataframe pandas."""
    return pd.DataFrame(response_payload.get("recommendations", []))


def format_recommendations_for_display(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Prepare le tableau final de recommandation pour l'affichage metier."""
    display_df = recommendations.copy()
    display_df["Culture"] = display_df["crop"].map(crop_display_label)
    display_df["Rendement historique estimé (t/ha)"] = display_df["p1_historical_prediction"]
    display_df["Rendement de référence (t/ha)"] = display_df["p2_reference_simulation"]
    display_df["Rendement avec vos conditions (t/ha)"] = display_df["p3_user_simulation"]
    display_df["Impact des conditions locales (t/ha)"] = display_df["local_adjustment"]
    display_df["Écart vs historique (%)"] = display_df["gap_vs_historical_pct"]
    display_df["Rendement final estimé (t/ha)"] = display_df["final_prediction"]
    display_df["Rang"] = display_df["recommendation_rank"]
    return display_df[
        [
            "Rang",
            "Culture",
            "Rendement historique estimé (t/ha)",
            "Rendement de référence (t/ha)",
            "Rendement avec vos conditions (t/ha)",
            "Impact des conditions locales (t/ha)",
            "Écart vs historique (%)",
            "Rendement final estimé (t/ha)",
        ]
    ]


def prediction_breakdown_frame(prediction_payload: dict[str, Any]) -> pd.DataFrame:
    """Construit la decomposition metier de la prediction finale."""
    return pd.DataFrame(
        [
            {"composant": "Rendement historique estimé", "valeur": prediction_payload["p1_historical_prediction"]},
            {"composant": "Rendement de référence", "valeur": prediction_payload["p2_reference_simulation"]},
            {"composant": "Rendement avec vos conditions", "valeur": prediction_payload["p3_user_simulation"]},
            {"composant": "Impact de vos conditions", "valeur": prediction_payload["local_adjustment"]},
            {"composant": "Rendement final estimé", "valeur": prediction_payload["final_prediction"]},
        ]
    )


def historical_shap_frame(prediction_payload: dict[str, Any]) -> pd.DataFrame:
    """Extrait les contributions SHAP historiques en dataframe."""
    explanation = prediction_payload["explanation"]["historical_shap"]
    return pd.DataFrame(explanation.get("top_contributions", []))


def local_adjustment_explanation_frame(prediction_payload: dict[str, Any]) -> pd.DataFrame:
    """Extrait la decomposition de l'ajustement local en dataframe."""
    explanation = prediction_payload["explanation"]["local_adjustment"]
    return pd.DataFrame(explanation.get("top_contributions", []))


def _inject_page_style() -> None:
    """Injecte le style CSS principal de l'interface Streamlit."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }
        .ux-banner {
            background: linear-gradient(135deg, #f5f0e4 0%, #eef4e8 45%, #f9fbf3 100%);
            border: 1px solid #d7e3d0;
            border-radius: 20px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
        }
        .ux-card {
            background: transparent;
            border: 1px solid #d9e7d5;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            min-height: 120px;
        }
        [data-testid="stMetric"] {
            background: transparent;
            border: 1px solid #d9e7d5;
            border-radius: 16px;
            padding: 0.85rem 1rem;
            box-shadow: none;
        }
        .ux-kicker {
            color: #2f6a3d;
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
        }
        .ux-title {
            font-size: 2rem;
            line-height: 1.1;
            font-weight: 700;
            color: #183b24;
            margin: 0.2rem 0 0.5rem 0;
        }
        .ux-formula {
            background: #1f4b2f;
            color: #f8fff4;
            border-radius: 14px;
            padding: 0.8rem 1rem;
            font-weight: 600;
            text-align: center;
            margin: 0.75rem 0 0.25rem 0;
        }
        .ux-step {
            font-size: 1rem;
            font-weight: 700;
            color: #224232;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_workflow_banner() -> None:
    """Affiche le titre principal de l'application."""
    st.markdown(
        """# Rendement Agricole : Prédiction et Recommandation 
        """,
        unsafe_allow_html=True,
    )


def _render_mode_cards() -> None:
    """Affiche les deux cartes de cas d'usage principaux."""
    col_prediction, col_recommendation = st.columns(2)
    with col_prediction:
        st.markdown(
            """
            <div class="ux-card">
                <div class="ux-step">Estimer une culture</div>
                <div>Choisissez un pays et une culture pour estimer le rendement attendu sur votre parcelle en tenant compte de vos conditions locales.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_recommendation:
        st.markdown(
            """
            <div class="ux-card">
                <div class="ux-step">Comparer plusieurs cultures</div>
                <div>Renseignez les conditions de votre parcelle pour comparer les cultures disponibles et identifier les options les plus prometteuses.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _condition_form(
    *,
    key_prefix: str,
    defaults: dict[str, Any],
    simulation_options: dict[str, list[str]],
    inferred_region: str | None,
) -> dict[str, Any]:
    """Affiche le formulaire de conditions et retourne le payload saisi."""
    soil_type_options = simulation_options["soil_types"]
    weather_options = simulation_options["weather_conditions"]
    resolved_region = inferred_region or str(defaults["region"])

    col_1, col_2 = st.columns(2)
    with col_1:
        st.text_input(
            "Zone agro-climatique",
            value=option_display_label("region", resolved_region),
            disabled=True,
            key=f"{key_prefix}_region_display",
        )
        rainfall_mm = st.number_input(
            "Pluviométrie (mm)",
            min_value=0.0,
            value=float(defaults["rainfall_mm"]),
            step=10.0,
            key=f"{key_prefix}_rainfall",
        )
        weather_condition = st.selectbox(
            "Météo dominante",
            weather_options,
            index=weather_options.index(defaults["weather_condition"]) if defaults["weather_condition"] in weather_options else 0,
            format_func=lambda value: option_display_label("weather_condition", value),
            key=f"{key_prefix}_weather",
        )
        fertilizer_used = st.toggle(
            "Apport d'engrais",
            value=bool(defaults["fertilizer_used"]),
            key=f"{key_prefix}_fertilizer",
        )
    with col_2:
        soil_type = st.selectbox(
            "Type de sol",
            soil_type_options,
            index=soil_type_options.index(defaults["soil_type"]) if defaults["soil_type"] in soil_type_options else 0,
            format_func=lambda value: option_display_label("soil_type", value),
            key=f"{key_prefix}_soil",
        )
        temperature_celsius = st.number_input(
            "Température moyenne (degC)",
            value=float(defaults["temperature_celsius"]),
            step=0.5,
            key=f"{key_prefix}_temperature",
        )
        days_to_harvest = st.number_input(
            "Durée avant récolte (jours)",
            min_value=1.0,
            value=float(defaults["days_to_harvest"]),
            step=1.0,
            key=f"{key_prefix}_harvest",
        )
        irrigation_used = st.toggle(
            "Irrigation",
            value=bool(defaults["irrigation_used"]),
            key=f"{key_prefix}_irrigation",
        )

    return build_user_conditions(
        region=resolved_region,
        soil_type=soil_type,
        rainfall_mm=rainfall_mm,
        temperature_celsius=temperature_celsius,
        fertilizer_used=fertilizer_used,
        irrigation_used=irrigation_used,
        weather_condition=weather_condition,
        days_to_harvest=days_to_harvest,
    )


def main() -> None:
    """Point d'entree principal de l'application Streamlit."""
    st.set_page_config(
        page_title="Rendement Agricole - UX v2",
        page_icon="🌾",
        layout="wide",
    )
    _inject_page_style()

    try:
        base_metadata = get_metadata()
    except ApiError as exc:
        st.title("Rendement Agricole")
        st.error(str(exc))
        st.stop()

    countries = list_countries(base_metadata)
    if not countries:
        st.title("Rendement Agricole")
        st.error("L'API v2 n'a retourné aucun pays exploitable.")
        st.stop()

    st.title("Rendement Agricole")
    st.caption("Estimez le rendement d'une culture et comparez plusieurs options selon les conditions de votre parcelle.")
    st.markdown(APP_BADGES_MARKDOWN)
    _render_workflow_banner()

    top_col, image_col = st.columns([3, 1])
    with top_col:
        _render_mode_cards()
    with image_col:
        display_bounded_image(str(DEFAULT_IMAGE_PATH))

    mode = st.radio(
        "Votre besoin",
        ["Estimer une culture", "Comparer plusieurs cultures"],
        horizontal=True,
    )

    selected_country = st.selectbox("Pays", countries)
    try:
        country_metadata = get_metadata(selected_country)
    except ApiError:
        country_metadata = base_metadata

    area_crops = list_crops(country_metadata)
    simulation_options = dict(country_metadata["simulation_options"])
    inferred_region = country_metadata.get("inferred_region")

    if mode == "Estimer une culture":
        if not area_crops:
            st.error("Aucune culture disponible pour ce pays dans l'API v2.")
            st.stop()

        selected_crop = st.selectbox(
            "Culture",
            area_crops,
            format_func=crop_display_label,
        )

        try:
            baseline_payload = get_baseline(selected_country, selected_crop)
        except ApiError as exc:
            st.error(str(exc))
            st.stop()

        baseline_cols = st.columns([1, 1, 1])
        baseline_cols[0].metric("Rendement historique estimé", f"{baseline_payload['p1_historical_prediction']:.2f} t/ha")
        baseline_cols[1].metric("Année de référence", str(baseline_payload["target_year"]))
        baseline_cols[2].write(
            {
                "Repère pluie": reference_source_display_label(baseline_payload["rainfall_reference_source"]),
                "Repère température": reference_source_display_label(baseline_payload["temperature_reference_source"]),
            }
        )

        selected_crop_icon = crop_icon_path(selected_crop)
        if selected_crop_icon is not None:
            icon_col, text_col = st.columns([1, 6])
            icon_col.image(load_icon_for_display(str(selected_crop_icon)))
            text_col.markdown(f"**Culture sélectionnée :** {crop_display_label(selected_crop)}")

        st.markdown("### Conditions de votre parcelle")
        with st.form("prediction_form"):
            prediction_conditions = _condition_form(
                key_prefix="prediction",
                defaults=baseline_payload["reference_profile"],
                simulation_options=simulation_options,
                inferred_region=inferred_region,
            )
            submitted_prediction = st.form_submit_button("Estimer le rendement", type="primary")

        if submitted_prediction:
            try:
                prediction_payload = predict_adjusted(
                    country=selected_country,
                    crop=selected_crop,
                    user_conditions=prediction_conditions,
                )
            except ApiError as exc:
                st.error(str(exc))
                st.stop()

            metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
            metric_col_1.metric("Rendement final estimé", f"{prediction_payload['final_prediction']:.2f} t/ha")
            metric_col_2.metric("Impact de vos conditions", f"{prediction_payload['local_adjustment']:+.2f} t/ha")
            metric_col_3.metric("Écart vs historique", f"{prediction_payload['gap_vs_historical_pct']:+.1f} %")

            st.markdown("### Comment l'estimation est construite")
            st.caption("Cette vue montre le point de départ historique, l'effet de vos conditions de parcelle et le rendement final estimé. La valeur est indiquée en tonnes par hectare (t/ha).")
            display_stretch_dataframe(
                prediction_breakdown_frame(prediction_payload).style.format({"valeur": "{:.2f}"})
            )

            breakdown_plot_df = prediction_breakdown_frame(prediction_payload)
            st.bar_chart(breakdown_plot_df.set_index("composant")["valeur"])

            with st.expander("Voir l'analyse détaillée du modèle"):
                st.caption("Cette partie donne une lecture plus technique des facteurs qui expliquent l'estimation.")
                shap_col, delta_col = st.columns(2)

                with shap_col:
                    st.markdown("#### Facteurs historiques les plus influents")
                    shap_explanation = prediction_payload["explanation"]["historical_shap"]
                    if shap_explanation["available"]:
                        st.caption(
                            f"Point de départ du modèle : {shap_explanation['base_value']:.2f} | "
                            f"Reconstruction : {shap_explanation['prediction_from_shap']:.2f}"
                        )
                        shap_df = historical_shap_frame(prediction_payload)
                        if not shap_df.empty:
                            shap_display_df = shap_df.copy()
                            shap_display_df["raw_feature"] = shap_display_df["feature"]
                            shap_display_df["raw_value"] = shap_display_df.apply(
                                lambda row: translate_feature_value(str(row["raw_feature"]), row["raw_value"]),
                                axis=1,
                            )
                            shap_display_df["feature"] = shap_display_df["raw_feature"].map(translate_feature_name)
                            shap_display_df = shap_display_df.drop(columns=["raw_feature"])
                            shap_display_df = shap_display_df.rename(
                                columns={
                                    "feature": "Facteur",
                                    "raw_value": "Valeur",
                                    "contribution": "Impact sur l'estimation",
                                    "abs_contribution": "Impact absolu",
                                }
                            )
                            display_stretch_dataframe(
                                shap_display_df.style.format(
                                    {
                                        "Impact sur l'estimation": "{:+.3f}",
                                        "Impact absolu": "{:.3f}",
                                    }
                                )
                            )
                            shap_chart_df = historical_shap_frame(prediction_payload).copy()
                            shap_chart_df["feature"] = shap_chart_df["feature"].map(translate_feature_name)
                            st.bar_chart(shap_chart_df.set_index("feature")["contribution"])
                    else:
                        st.info(
                            shap_explanation.get("message")
                            or "L'analyse détaillée n'est pas disponible dans cet environnement."
                        )

                with delta_col:
                    st.markdown("#### Effet de vos conditions de parcelle")
                    delta_explanation = prediction_payload["explanation"]["local_adjustment"]
                    st.caption(
                        f"Méthode : {delta_explanation['method']} | "
                        f"Impact total : {delta_explanation['total_adjustment']:+.2f}"
                    )
                    delta_df = local_adjustment_explanation_frame(prediction_payload)
                    if not delta_df.empty:
                        delta_display_df = delta_df.copy()
                        delta_display_df["reference_value"] = delta_display_df.apply(
                            lambda row: translate_feature_value(str(row["feature"]), row["reference_value"]),
                            axis=1,
                        )
                        delta_display_df["user_value"] = delta_display_df.apply(
                            lambda row: translate_feature_value(str(row["feature"]), row["user_value"]),
                            axis=1,
                        )
                        delta_display_df["feature"] = delta_display_df["feature"].map(translate_feature_name)
                        delta_display_df = delta_display_df.rename(
                            columns={
                                "feature": "Facteur",
                                "reference_value": "Valeur de référence",
                                "user_value": "Votre valeur",
                                "contribution_delta": "Impact sur l'écart",
                                "abs_contribution_delta": "Impact absolu",
                            }
                        )
                        display_stretch_dataframe(
                            delta_display_df.style.format(
                                {
                                    "Impact sur l'écart": "{:+.3f}",
                                    "Impact absolu": "{:.3f}",
                                }
                            )
                        )
                        delta_chart_df = local_adjustment_explanation_frame(prediction_payload).copy()
                        delta_chart_df["feature"] = delta_chart_df["feature"].map(translate_feature_name)
                        st.bar_chart(delta_chart_df.set_index("feature")["contribution_delta"])

            with st.expander("Voir les profils utilisés pour la comparaison"):
                display_stretch_dataframe(
                    condition_profiles_comparison_frame(
                        prediction_payload["reference_profile"],
                        prediction_payload["user_profile"],
                    )
                )

    else:
        st.markdown("### Conditions de votre parcelle")
        recommendation_defaults = global_reference_profile(country_metadata)
        with st.form("recommendation_form"):
            recommendation_conditions = _condition_form(
                key_prefix="recommendation",
                defaults=recommendation_defaults,
                simulation_options=simulation_options,
                inferred_region=inferred_region,
            )
            candidate_crops = st.multiselect(
                "Cultures à comparer",
                options=area_crops,
                default=area_crops,
                format_func=crop_display_label,
            )
            top_n = st.slider(
                "Nombre de cultures affichées",
                min_value=1,
                max_value=max(1, len(area_crops)),
                value=min(5, max(1, len(area_crops))),
            )
            submitted_recommendation = st.form_submit_button("Comparer les cultures", type="primary")

        if submitted_recommendation:
            if not candidate_crops:
                st.warning("Sélectionne au moins une culture.")
            else:
                try:
                    recommendation_payload = recommend_adjusted(
                        country=selected_country,
                        user_conditions=recommendation_conditions,
                        candidate_crops=candidate_crops,
                        top_n=top_n,
                    )
                except ApiError as exc:
                    st.error(str(exc))
                    st.stop()

                recommendation_df = recommendation_records_to_frame(recommendation_payload)
                display_df = format_recommendations_for_display(recommendation_df)

                head_col_1, head_col_2 = st.columns([2, 1])
                head_col_1.metric(
                    "Culture la plus adaptée",
                    crop_display_label(recommendation_payload["best_crop"]),
                )
                head_col_2.metric(
                    "Rendement estimé",
                    f"{recommendation_payload['best_final_prediction']:.2f} t/ha",
                )

                display_stretch_dataframe(
                    display_df.style.format(
                        {
                            "Rendement historique estimé (t/ha)": "{:.2f}",
                            "Rendement de référence (t/ha)": "{:.2f}",
                            "Rendement avec vos conditions (t/ha)": "{:.2f}",
                            "Impact des conditions locales (t/ha)": "{:+.2f}",
                            "Écart vs historique (%)": "{:+.1f}",
                            "Rendement final estimé (t/ha)": "{:.2f}",
                        }
                    )
                )

                st.markdown("### Classement visuel")
                chart_df = recommendation_df.copy()
                chart_df["Culture"] = chart_df["crop"].map(crop_display_label)
                st.bar_chart(chart_df.set_index("Culture")["final_prediction"])

                st.markdown("### Aperçu des cultures recommandées")
                preview_columns = st.columns(min(len(recommendation_df), 5))
                for column, (_, row) in zip(preview_columns, recommendation_df.iterrows()):
                    icon_path = crop_icon_path(row["crop"])
                    if icon_path is not None:
                        column.image(load_icon_for_display(str(icon_path)))
                    column.caption(crop_display_label(row["crop"]))
                    column.write(f"{row['final_prediction']:.2f} t/ha")


if __name__ == "__main__":
    main()
