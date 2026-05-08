"""Tests unitaires legers pour `scripts/experience_1.py`."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.experience_1 import XGBOOST_AVAILABLE, build_candidate_models, build_modeling_context, pivot_history


def test_pivot_history_keeps_requested_years_and_column_names() -> None:
    long_df = pd.DataFrame(
        [
            {"area": "France", "crop": "Wheat", "year": 2014, "target_yield_t_ha": 5.1},
            {"area": "France", "crop": "Wheat", "year": 2015, "target_yield_t_ha": 5.4},
            {"area": "Kenya", "crop": "Maize", "year": 2015, "target_yield_t_ha": 3.2},
        ]
    )

    wide_df = pivot_history(long_df, ["area", "crop"], "target_yield_t_ha", [2014, 2015, 2016])

    assert wide_df.columns.tolist() == [
        "area",
        "crop",
        "target_yield_t_ha_2014",
        "target_yield_t_ha_2015",
        "target_yield_t_ha_2016",
    ]
    france_row = wide_df.loc[(wide_df["area"] == "France") & (wide_df["crop"] == "Wheat")].iloc[0]
    assert france_row["target_yield_t_ha_2014"] == 5.1
    assert france_row["target_yield_t_ha_2015"] == 5.4
    assert pd.isna(france_row["target_yield_t_ha_2016"])


def test_build_modeling_context_excludes_area_and_drops_empty_train_numeric_features() -> None:
    experience_dataset = pd.DataFrame(
        [
            {
                "area": "A",
                "crop": "Wheat",
                "target_yield_t_ha_2013": 4.5,
                "target_yield_t_ha_2014": 4.7,
                "target_yield_t_ha_2015": 4.9,
                "target_yield_t_ha_2016": 5.0,
                "average_rain_fall_mm_per_year_2013": 500.0,
                "average_rain_fall_mm_per_year_2014": 520.0,
                "average_rain_fall_mm_per_year_2015": 510.0,
                "pesticides_tonnes_2013": 10.0,
                "pesticides_tonnes_2014": 10.5,
                "pesticides_tonnes_2015": 11.0,
                "avg_temp_2013": 20.0,
                "avg_temp_2014": 20.5,
                "avg_temp_2015": None,
            },
            {
                "area": "B",
                "crop": "Maize",
                "target_yield_t_ha_2013": 3.5,
                "target_yield_t_ha_2014": 3.7,
                "target_yield_t_ha_2015": 3.9,
                "target_yield_t_ha_2016": 4.1,
                "average_rain_fall_mm_per_year_2013": 610.0,
                "average_rain_fall_mm_per_year_2014": 600.0,
                "average_rain_fall_mm_per_year_2015": 590.0,
                "pesticides_tonnes_2013": 7.0,
                "pesticides_tonnes_2014": 7.5,
                "pesticides_tonnes_2015": 8.0,
                "avg_temp_2013": 24.0,
                "avg_temp_2014": 23.5,
                "avg_temp_2015": None,
            },
            {
                "area": "C",
                "crop": "Wheat",
                "target_yield_t_ha_2013": 5.5,
                "target_yield_t_ha_2014": 5.7,
                "target_yield_t_ha_2015": 5.9,
                "target_yield_t_ha_2016": 6.0,
                "average_rain_fall_mm_per_year_2013": 550.0,
                "average_rain_fall_mm_per_year_2014": 540.0,
                "average_rain_fall_mm_per_year_2015": 560.0,
                "pesticides_tonnes_2013": 9.0,
                "pesticides_tonnes_2014": 9.2,
                "pesticides_tonnes_2015": 9.4,
                "avg_temp_2013": 19.5,
                "avg_temp_2014": 19.0,
                "avg_temp_2015": None,
            },
            {
                "area": "D",
                "crop": "Maize",
                "target_yield_t_ha_2013": 2.5,
                "target_yield_t_ha_2014": 2.7,
                "target_yield_t_ha_2015": 2.9,
                "target_yield_t_ha_2016": 3.0,
                "average_rain_fall_mm_per_year_2013": 720.0,
                "average_rain_fall_mm_per_year_2014": 700.0,
                "average_rain_fall_mm_per_year_2015": 710.0,
                "pesticides_tonnes_2013": 6.0,
                "pesticides_tonnes_2014": 6.2,
                "pesticides_tonnes_2015": 6.4,
                "avg_temp_2013": 25.0,
                "avg_temp_2014": 24.5,
                "avg_temp_2015": None,
            },
            {
                "area": "E",
                "crop": "Wheat",
                "target_yield_t_ha_2013": 4.1,
                "target_yield_t_ha_2014": 4.3,
                "target_yield_t_ha_2015": 4.5,
                "target_yield_t_ha_2016": 4.7,
                "average_rain_fall_mm_per_year_2013": 480.0,
                "average_rain_fall_mm_per_year_2014": 490.0,
                "average_rain_fall_mm_per_year_2015": 500.0,
                "pesticides_tonnes_2013": 8.0,
                "pesticides_tonnes_2014": 8.1,
                "pesticides_tonnes_2015": 8.3,
                "avg_temp_2013": 18.5,
                "avg_temp_2014": 18.7,
                "avg_temp_2015": None,
            },
        ]
    )

    context = build_modeling_context(
        experience_dataset,
        target_year=2016,
        feature_years=[2013, 2014, 2015],
        selected_yield_years=[2013, 2014, 2015],
        seed=42,
    )

    assert context.target_col == "target_yield_t_ha_2016"
    assert "area" not in context.feature_cols
    assert context.categorical_features == ["crop"]
    assert "avg_temp_2015" in context.train_empty_numeric_features
    assert "avg_temp_2015" not in context.numeric_features
    assert context.encoded_feature_count >= len(context.feature_cols)


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="xgboost n'est pas installe dans cet environnement de test.")
def test_build_candidate_models_registers_expected_search_space() -> None:
    candidate_models = build_candidate_models(seed=42)

    assert len(candidate_models) == 13
    assert "random_forest_search_01" in candidate_models
    assert "xgboost_regularized" in candidate_models
    assert "xgboost_random_forest_search_04" in candidate_models
    assert {spec["search_method"] for spec in candidate_models.values()} == {"parameter_grid"}
    assert all(int(spec["parameter_grid_index"]) >= 1 for spec in candidate_models.values())
