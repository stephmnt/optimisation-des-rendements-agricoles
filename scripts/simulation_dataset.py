"""Briques communes autour du dataset de simulation locale.

Ce module centralise le renommage et le nettoyage de
`data/simulation/crop_yield.csv` afin d'eviter que l'ACP et le moteur runtime
fassent diverger leurs hypotheses de preparation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd


SIMULATION_COLUMN_RENAMES = {
    "Region": "region",
    "Soil_Type": "soil_type",
    "Crop": "crop",
    "Rainfall_mm": "rainfall_mm",
    "Temperature_Celsius": "temperature_celsius",
    "Fertilizer_Used": "fertilizer_used",
    "Irrigation_Used": "irrigation_used",
    "Weather_Condition": "weather_condition",
    "Days_to_Harvest": "days_to_harvest",
    "Yield_tons_per_hectare": "yield_tons_per_hectare",
}

SIMULATION_CATEGORICAL_COLUMNS = [
    "region",
    "soil_type",
    "crop",
    "weather_condition",
]

SIMULATION_BOOLEAN_COLUMNS = [
    "fertilizer_used",
    "irrigation_used",
]

SIMULATION_NUMERIC_COLUMNS = [
    "rainfall_mm",
    "temperature_celsius",
    "days_to_harvest",
    "yield_tons_per_hectare",
]

SIMULATION_ACP_NUMERIC_COLUMNS = [
    "rainfall_mm",
    "temperature_celsius",
    "days_to_harvest",
]


def normalize_simulation_label(value: Any) -> str:
    """Nettoie une etiquette textuelle issue du dataset de simulation."""
    return str(value).strip()


def _coerce_boolean_value(value: Any) -> bool | pd._libs.missing.NAType:
    """Convertit defensivement une valeur vers un booleen pandas-compatible."""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "oui"}:
        return True
    if normalized in {"false", "0", "no", "n", "non"}:
        return False
    return bool(value)


def load_normalized_simulation_dataset(
    csv_path: str | Path,
    *,
    boolean_dtype: Literal["boolean", "bool"] = "bool",
) -> pd.DataFrame:
    """Charge et normalise le dataset de simulation locale.

    Args:
        csv_path: Fichier CSV source a charger.
        boolean_dtype: Type a utiliser pour les colonnes booleennes.

    Returns:
        pd.DataFrame: Dataset nettoye avec schema homogenise.
    """
    simulation_df = pd.read_csv(Path(csv_path)).rename(columns=SIMULATION_COLUMN_RENAMES)

    simulation_df[SIMULATION_CATEGORICAL_COLUMNS] = simulation_df[SIMULATION_CATEGORICAL_COLUMNS].apply(
        lambda column: column.map(normalize_simulation_label)
    )
    simulation_df[SIMULATION_NUMERIC_COLUMNS] = simulation_df[SIMULATION_NUMERIC_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    for column in SIMULATION_BOOLEAN_COLUMNS:
        normalized_series = simulation_df[column].map(_coerce_boolean_value).astype("boolean")
        if boolean_dtype == "bool":
            if normalized_series.isna().any():
                raise ValueError(
                    f"Column {column!r} contains missing values and cannot be coerced to bool."
                )
            simulation_df[column] = normalized_series.astype(bool)
        else:
            simulation_df[column] = normalized_series

    simulation_df = simulation_df.loc[simulation_df["yield_tons_per_hectare"] >= 0].reset_index(drop=True)
    return simulation_df
