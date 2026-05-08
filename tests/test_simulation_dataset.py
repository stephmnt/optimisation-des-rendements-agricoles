"""Tests du nettoyage partage du dataset de simulation locale."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.simulation_dataset import (
    SIMULATION_ACP_NUMERIC_COLUMNS,
    load_normalized_simulation_dataset,
)


def test_load_normalized_simulation_dataset_standardizes_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "crop_yield.csv"
    pd.DataFrame(
        [
            {
                "Region": " North ",
                "Soil_Type": " Clay ",
                "Crop": " Wheat ",
                "Rainfall_mm": "540.5",
                "Temperature_Celsius": "23.0",
                "Fertilizer_Used": "true",
                "Irrigation_Used": "false",
                "Weather_Condition": " Sunny ",
                "Days_to_Harvest": "110",
                "Yield_tons_per_hectare": "6.4",
            },
            {
                "Region": "South",
                "Soil_Type": "Sandy",
                "Crop": "Maize",
                "Rainfall_mm": "500",
                "Temperature_Celsius": "21",
                "Fertilizer_Used": "false",
                "Irrigation_Used": "true",
                "Weather_Condition": "Rainy",
                "Days_to_Harvest": "95",
                "Yield_tons_per_hectare": "-1",
            },
        ]
    ).to_csv(csv_path, index=False)

    dataset = load_normalized_simulation_dataset(csv_path, boolean_dtype="boolean")

    assert dataset.columns.tolist() == [
        "region",
        "soil_type",
        "crop",
        "rainfall_mm",
        "temperature_celsius",
        "fertilizer_used",
        "irrigation_used",
        "weather_condition",
        "days_to_harvest",
        "yield_tons_per_hectare",
    ]
    assert len(dataset) == 1
    assert dataset.iloc[0]["region"] == "North"
    assert dataset.iloc[0]["soil_type"] == "Clay"
    assert dataset.iloc[0]["weather_condition"] == "Sunny"
    assert str(dataset["fertilizer_used"].dtype) == "boolean"
    assert SIMULATION_ACP_NUMERIC_COLUMNS == ["rainfall_mm", "temperature_celsius", "days_to_harvest"]


def test_load_normalized_simulation_dataset_can_return_python_bool_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "crop_yield.csv"
    pd.DataFrame(
        [
            {
                "Region": "North",
                "Soil_Type": "Clay",
                "Crop": "Wheat",
                "Rainfall_mm": 540.5,
                "Temperature_Celsius": 23.0,
                "Fertilizer_Used": True,
                "Irrigation_Used": False,
                "Weather_Condition": "Sunny",
                "Days_to_Harvest": 110,
                "Yield_tons_per_hectare": 6.4,
            }
        ]
    ).to_csv(csv_path, index=False)

    dataset = load_normalized_simulation_dataset(csv_path, boolean_dtype="bool")

    assert str(dataset["fertilizer_used"].dtype) == "bool"
    assert str(dataset["irrigation_used"].dtype) == "bool"
