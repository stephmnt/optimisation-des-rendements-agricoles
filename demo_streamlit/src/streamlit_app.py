from __future__ import annotations

from pathlib import Path

import streamlit as st

from app_logic import (
    DEFAULT_IMAGE_PATH,
    build_default_context,
    current_year,
    fit_demo_model,
    list_areas,
    list_crops,
    predict_yield,
    recommend_crops,
)


st.set_page_config(
    page_title="Rendement Agricole",
    page_icon="🌾",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_model_and_dataset():
    return fit_demo_model()


model, dataset = get_model_and_dataset()
areas = list_areas(dataset)
crops = list_crops(dataset)
year_used = current_year()

st.title("Rendement Agricole")
st.caption(
    "Démo Streamlit autonome : prédiction de rendement et recommandation de cultures "
    "à partir du dataset consolidé du projet."
)
st.info(
    f"Année utilisée dans les scénarios : {year_used}. "
    "Le modèle est léger et réentraîné au démarrage à partir des données historiques du projet."
)

if DEFAULT_IMAGE_PATH.exists():
    st.image(str(DEFAULT_IMAGE_PATH), use_container_width=True)

st.sidebar.header("Contexte de simulation")
selected_area = st.sidebar.selectbox("Zone", areas)
defaults = build_default_context(dataset, selected_area)

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
    selected_crop = st.selectbox("Culture", crops)
    if st.button("Calculer le rendement", type="primary"):
        predicted_yield = predict_yield(
            model,
            area=selected_area,
            crop=selected_crop,
            average_rain_fall_mm_per_year=rainfall,
            pesticides_tonnes=pesticides,
            avg_temp=avg_temp,
            year=year_used,
        )
        total_production = predicted_yield * float(hectares)

        metric_1, metric_2 = st.columns(2)
        metric_1.metric("Rendement prédit", f"{predicted_yield:.2f} t/ha")
        metric_2.metric("Production totale", f"{total_production:.2f} tonnes")

        st.write(
            {
                "zone": selected_area,
                "culture": selected_crop,
                "annee_utilisee": year_used,
                "surface_hectares": hectares,
                "pluie_mm": rainfall,
                "pesticides_tonnes": pesticides,
                "temperature_moyenne_c": avg_temp,
            }
        )

with recommend_tab:
    default_selection = crops[: min(5, len(crops))]
    candidate_crops = st.multiselect(
        "Cultures à comparer",
        options=crops,
        default=default_selection,
    )
    top_n = st.slider("Nombre de recommandations à afficher", min_value=1, max_value=max(1, len(crops)), value=min(5, len(crops)))

    if st.button("Lancer la recommandation"):
        if not candidate_crops:
            st.warning("Sélectionne au moins une culture.")
        else:
            recommendations = recommend_crops(
                model,
                area=selected_area,
                hectares=hectares,
                average_rain_fall_mm_per_year=rainfall,
                pesticides_tonnes=pesticides,
                avg_temp=avg_temp,
                candidate_crops=candidate_crops,
                year=year_used,
            ).head(top_n)

            st.dataframe(
                recommendations.style.format(
                    {
                        "predicted_yield_t_ha": "{:.2f}",
                        "predicted_total_production_tons": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )
            st.bar_chart(
                recommendations.set_index("crop")["predicted_total_production_tons"],
                use_container_width=True,
            )
