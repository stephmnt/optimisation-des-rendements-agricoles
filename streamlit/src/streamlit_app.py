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
    base_url = (api_base_url or default_api_base_url()).rstrip("/")
    url = f"{base_url}{path}"
    try:
        response = requests.request(method, url, timeout=timeout, **request_kwargs)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ApiError(f"Impossible de joindre l'API v2 FastAPI : {exc}") from exc
    return response.json()


def fetch_metadata(*, country: str | None = None, api_base_url: str | None = None) -> dict[str, Any]:
    params = {"country": country} if country else None
    return _request_json("GET", "/metadata", api_base_url=api_base_url, params=params)


def fetch_baseline(*, country: str, crop: str, api_base_url: str | None = None) -> dict[str, Any]:
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
    return list(metadata.get("countries", []))


def list_crops(metadata: dict[str, Any]) -> list[str]:
    return list(metadata.get("available_crops", []))


def global_reference_profile(metadata: dict[str, Any]) -> dict[str, Any]:
    return dict(metadata["global_reference_profile"])


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
    return path if path.exists() else None


def load_image_for_display(image_path: str, max_size: tuple[int, int]) -> BytesIO:
    with Image.open(image_path) as image:
        prepared = image.copy()
    prepared.thumbnail(max_size)
    buffer = BytesIO()
    prepared.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def load_icon_for_display(image_path: str, max_size: tuple[int, int] = ICON_MAX_SIZE) -> BytesIO:
    return load_image_for_display(image_path, max_size=max_size)


def display_bounded_image(image_path: str, max_size: tuple[int, int] = APP_IMAGE_MAX_SIZE) -> None:
    st.image(load_image_for_display(image_path, max_size=max_size))


def display_stretch_dataframe(data) -> None:
    try:
        st.dataframe(data, width="stretch")
    except TypeError:
        st.dataframe(data, use_container_width=True)


@st.cache_data(show_spinner=False)
def get_metadata(country: str | None = None) -> dict[str, Any]:
    return fetch_metadata(country=country)


@st.cache_data(show_spinner=False)
def get_baseline(country: str, crop: str) -> dict[str, Any]:
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
    return pd.DataFrame(response_payload.get("recommendations", []))


def format_recommendations_for_display(recommendations: pd.DataFrame) -> pd.DataFrame:
    display_df = recommendations.copy()
    display_df["Culture"] = display_df["crop"].map(crop_display_label)
    display_df["P1 historique (t/ha)"] = display_df["p1_historical_prediction"]
    display_df["P2 référence"] = display_df["p2_reference_simulation"]
    display_df["P3 utilisateur"] = display_df["p3_user_simulation"]
    display_df["Ajustement local"] = display_df["local_adjustment"]
    display_df["Écart vs historique (%)"] = display_df["gap_vs_historical_pct"]
    display_df["Rendement final (t/ha)"] = display_df["final_prediction"]
    display_df["Rang"] = display_df["recommendation_rank"]
    return display_df[
        [
            "Rang",
            "Culture",
            "P1 historique (t/ha)",
            "P2 référence",
            "P3 utilisateur",
            "Ajustement local",
            "Écart vs historique (%)",
            "Rendement final (t/ha)",
        ]
    ]


def prediction_breakdown_frame(prediction_payload: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"composant": "P1 historique", "valeur": prediction_payload["p1_historical_prediction"]},
            {"composant": "P2 référence", "valeur": prediction_payload["p2_reference_simulation"]},
            {"composant": "P3 utilisateur", "valeur": prediction_payload["p3_user_simulation"]},
            {"composant": "P3 - P2", "valeur": prediction_payload["local_adjustment"]},
            {"composant": "Prédiction finale", "valeur": prediction_payload["final_prediction"]},
        ]
    )


def historical_shap_frame(prediction_payload: dict[str, Any]) -> pd.DataFrame:
    explanation = prediction_payload["explanation"]["historical_shap"]
    return pd.DataFrame(explanation.get("top_contributions", []))


def local_adjustment_explanation_frame(prediction_payload: dict[str, Any]) -> pd.DataFrame:
    explanation = prediction_payload["explanation"]["local_adjustment"]
    return pd.DataFrame(explanation.get("top_contributions", []))


def _inject_page_style() -> None:
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
    st.markdown(
        """# Rendement Agricole : Prédiction et Recommandation 
        """,
        unsafe_allow_html=True,
    )


def _render_mode_cards() -> None:
    col_prediction, col_recommendation = st.columns(2)
    with col_prediction:
        st.markdown(
            """
            <div class="ux-card">
                <div class="ux-step">Mode Prédiction</div>
                <div>Sélection d’un pays et d’une culture, affichage du rendement moyen historique, puis ajustement avec les conditions locales de la parcelle.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_recommendation:
        st.markdown(
            """
            <div class="ux-card">
                <div class="ux-step">Mode Recommandation</div>
                <div>Un seul contexte local utilisateur, puis un classement de toutes les cultures disponibles pour le pays choisi, trié par rendement final.</div>
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
    soil_type_options = simulation_options["soil_types"]
    weather_options = simulation_options["weather_conditions"]
    resolved_region = inferred_region or str(defaults["region"])

    col_1, col_2 = st.columns(2)
    with col_1:
        st.text_input(
            "Région simulée",
            value=resolved_region,
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
            "Condition météo",
            weather_options,
            index=weather_options.index(defaults["weather_condition"]) if defaults["weather_condition"] in weather_options else 0,
            key=f"{key_prefix}_weather",
        )
        fertilizer_used = st.toggle(
            "Engrais utilisés",
            value=bool(defaults["fertilizer_used"]),
            key=f"{key_prefix}_fertilizer",
        )
    with col_2:
        soil_type = st.selectbox(
            "Type de sol",
            soil_type_options,
            index=soil_type_options.index(defaults["soil_type"]) if defaults["soil_type"] in soil_type_options else 0,
            key=f"{key_prefix}_soil",
        )
        temperature_celsius = st.number_input(
            "Température moyenne (°C)",
            value=float(defaults["temperature_celsius"]),
            step=0.5,
            key=f"{key_prefix}_temperature",
        )
        days_to_harvest = st.number_input(
            "Jours jusqu'à récolte",
            min_value=1.0,
            value=float(defaults["days_to_harvest"]),
            step=1.0,
            key=f"{key_prefix}_harvest",
        )
        irrigation_used = st.toggle(
            "Irrigation utilisée",
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
    st.caption("Expérience utilisateur alignée sur la stratégie P1 + (P3 - P2).")
    st.markdown(APP_BADGES_MARKDOWN)
    _render_workflow_banner()

    top_col, image_col = st.columns([3, 1])
    with top_col:
        _render_mode_cards()
    with image_col:
        display_bounded_image(str(DEFAULT_IMAGE_PATH))

    mode = st.radio(
        "Mode",
        ["Prédiction", "Recommandation"],
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

    if mode == "Prédiction":
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
        baseline_cols[0].metric("P1 historique", f"{baseline_payload['p1_historical_prediction']:.2f} t/ha")
        baseline_cols[1].metric("Année cible", str(baseline_payload["target_year"]))
        baseline_cols[2].write(
            {
                "source pluie": baseline_payload["rainfall_reference_source"],
                "source température": baseline_payload["temperature_reference_source"],
            }
        )

        selected_crop_icon = crop_icon_path(selected_crop)
        if selected_crop_icon is not None:
            icon_col, text_col = st.columns([1, 6])
            icon_col.image(load_icon_for_display(str(selected_crop_icon)))
            text_col.markdown(f"**Culture sélectionnée :** {crop_display_label(selected_crop)}")

        st.markdown("### Conditions spécifiques de la parcelle")
        with st.form("prediction_form"):
            prediction_conditions = _condition_form(
                key_prefix="prediction",
                defaults=baseline_payload["reference_profile"],
                simulation_options=simulation_options,
                inferred_region=inferred_region,
            )
            submitted_prediction = st.form_submit_button("Calculer la prédiction ajustée", type="primary")

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
            metric_col_1.metric("Rendement final ajusté", f"{prediction_payload['final_prediction']:.2f} t/ha")
            metric_col_2.metric("Ajustement local", f"{prediction_payload['local_adjustment']:+.2f} t/ha")
            metric_col_3.metric("Écart vs historique", f"{prediction_payload['gap_vs_historical_pct']:+.1f} %")

            st.markdown("### Décomposition du calcul")
            display_stretch_dataframe(
                prediction_breakdown_frame(prediction_payload).style.format({"valeur": "{:.2f}"})
            )

            breakdown_plot_df = prediction_breakdown_frame(prediction_payload)
            st.bar_chart(breakdown_plot_df.set_index("composant")["valeur"])

            st.markdown("### Pourquoi ce résultat ?")
            shap_col, delta_col = st.columns(2)

            with shap_col:
                st.markdown("#### SHAP sur `P1`")
                shap_explanation = prediction_payload["explanation"]["historical_shap"]
                if shap_explanation["available"]:
                    st.caption(
                        f"Base SHAP : {shap_explanation['base_value']:.2f} | "
                        f"Reconstruction : {shap_explanation['prediction_from_shap']:.2f}"
                    )
                    shap_df = historical_shap_frame(prediction_payload)
                    if not shap_df.empty:
                        shap_display_df = shap_df.rename(
                            columns={
                                "feature": "Feature",
                                "raw_value": "Valeur",
                                "contribution": "Contribution SHAP",
                                "abs_contribution": "Contribution absolue",
                            }
                        )
                        display_stretch_dataframe(
                            shap_display_df.style.format(
                                {
                                    "Contribution SHAP": "{:+.3f}",
                                    "Contribution absolue": "{:.3f}",
                                }
                            )
                        )
                        st.bar_chart(shap_df.set_index("feature")["contribution"])
                else:
                    st.info(
                        shap_explanation.get("message")
                        or "SHAP n'est pas disponible dans cet environnement."
                    )

            with delta_col:
                st.markdown("#### Décomposition de `P3 - P2`")
                delta_explanation = prediction_payload["explanation"]["local_adjustment"]
                st.caption(
                    f"Méthode : {delta_explanation['method']} | "
                    f"Delta total : {delta_explanation['total_adjustment']:+.2f}"
                )
                delta_df = local_adjustment_explanation_frame(prediction_payload)
                if not delta_df.empty:
                    delta_display_df = delta_df.rename(
                        columns={
                            "feature": "Feature",
                            "reference_value": "Valeur P2",
                            "user_value": "Valeur P3",
                            "contribution_delta": "Contribution au delta",
                            "abs_contribution_delta": "Contribution absolue",
                        }
                    )
                    display_stretch_dataframe(
                        delta_display_df.style.format(
                            {
                                "Contribution au delta": "{:+.3f}",
                                "Contribution absolue": "{:.3f}",
                            }
                        )
                    )
                    st.bar_chart(delta_df.set_index("feature")["contribution_delta"])

            with st.expander("Profils utilisés pour P2 et P3"):
                profile_col_1, profile_col_2 = st.columns(2)
                profile_col_1.write({"profil de référence P2": prediction_payload["reference_profile"]})
                profile_col_2.write({"profil utilisateur P3": prediction_payload["user_profile"]})

    else:
        st.markdown("### Conditions de la parcelle")
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
                "Nombre de recommandations affichées",
                min_value=1,
                max_value=max(1, len(area_crops)),
                value=min(5, max(1, len(area_crops))),
            )
            submitted_recommendation = st.form_submit_button("Lancer la recommandation", type="primary")

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
                    "Culture recommandée",
                    crop_display_label(recommendation_payload["best_crop"]),
                )
                head_col_2.metric(
                    "Meilleur rendement final",
                    f"{recommendation_payload['best_final_prediction']:.2f} t/ha",
                )

                display_stretch_dataframe(
                    display_df.style.format(
                        {
                            "P1 historique (t/ha)": "{:.2f}",
                            "P2 référence": "{:.2f}",
                            "P3 utilisateur": "{:.2f}",
                            "Ajustement local": "{:+.2f}",
                            "Écart vs historique (%)": "{:+.1f}",
                            "Rendement final (t/ha)": "{:.2f}",
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
