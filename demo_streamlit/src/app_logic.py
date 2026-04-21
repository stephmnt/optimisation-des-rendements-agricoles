from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SPACE_ROOT.parent


def _default_data_path() -> Path:
    deployed_path = SPACE_ROOT / "data" / "dataset_consolide.csv"
    local_path = PROJECT_ROOT / "data" / "dataset_consolide.csv"
    return deployed_path if deployed_path.exists() else local_path


def _default_image_path() -> Path:
    deployed_path = SPACE_ROOT / "agriculture.png"
    local_path = PROJECT_ROOT / "agriculture.png"
    return deployed_path if deployed_path.exists() else local_path


DEFAULT_DATA_PATH = _default_data_path()
DEFAULT_IMAGE_PATH = _default_image_path()

FEATURE_COLUMNS = [
    "area",
    "crop",
    "year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]
TARGET_COLUMN = "target_yield_t_ha"
NUMERIC_COLUMNS = [
    "year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]
CATEGORICAL_COLUMNS = ["area", "crop"]


def current_year() -> int:
    return date.today().year


def load_dataset(data_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")

    dataset = pd.read_csv(path)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise KeyError(f"Colonnes manquantes dans le dataset : {missing_columns}")

    data = dataset[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    data[["year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", TARGET_COLUMN]] = data[
        ["year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", TARGET_COLUMN]
    ].apply(pd.to_numeric, errors="coerce")
    data["area"] = data["area"].astype("string").str.strip()
    data["crop"] = data["crop"].astype("string").str.strip()

    data = data.dropna(subset=["area", "crop", TARGET_COLUMN]).reset_index(drop=True)
    return data


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def fit_demo_model(data_path: Path | str = DEFAULT_DATA_PATH) -> tuple[Pipeline, pd.DataFrame]:
    dataset = load_dataset(data_path)
    model = build_pipeline()
    model.fit(dataset[FEATURE_COLUMNS], dataset[TARGET_COLUMN])
    return model, dataset


def list_areas(dataset: pd.DataFrame) -> list[str]:
    return sorted(dataset["area"].dropna().unique().tolist())


def list_crops(dataset: pd.DataFrame) -> list[str]:
    return sorted(dataset["crop"].dropna().unique().tolist())


def build_default_context(dataset: pd.DataFrame, area: str | None = None) -> dict[str, float]:
    reference = dataset
    if area:
        area_slice = dataset.loc[dataset["area"] == area]
        if not area_slice.empty:
            reference = area_slice

    medians = reference[["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]].median()
    return {
        "average_rain_fall_mm_per_year": float(medians["average_rain_fall_mm_per_year"]),
        "pesticides_tonnes": float(medians["pesticides_tonnes"]),
        "avg_temp": float(medians["avg_temp"]),
    }


def _sanitize_prediction(value: float) -> float:
    return max(float(value), 0.0)


def predict_yield(
    model: Pipeline,
    *,
    area: str,
    crop: str,
    average_rain_fall_mm_per_year: float,
    pesticides_tonnes: float,
    avg_temp: float,
    year: int | None = None,
) -> float:
    prediction_year = year or current_year()
    row = pd.DataFrame(
        [
            {
                "area": area,
                "crop": crop,
                "year": int(prediction_year),
                "average_rain_fall_mm_per_year": float(average_rain_fall_mm_per_year),
                "pesticides_tonnes": float(pesticides_tonnes),
                "avg_temp": float(avg_temp),
            }
        ]
    )
    prediction = model.predict(row)[0]
    return _sanitize_prediction(prediction)


def recommend_crops(
    model: Pipeline,
    *,
    area: str,
    hectares: float,
    average_rain_fall_mm_per_year: float,
    pesticides_tonnes: float,
    avg_temp: float,
    candidate_crops: list[str],
    year: int | None = None,
) -> pd.DataFrame:
    prediction_year = year or current_year()
    rows = pd.DataFrame(
        [
            {
                "area": area,
                "crop": crop,
                "year": int(prediction_year),
                "average_rain_fall_mm_per_year": float(average_rain_fall_mm_per_year),
                "pesticides_tonnes": float(pesticides_tonnes),
                "avg_temp": float(avg_temp),
            }
            for crop in candidate_crops
        ]
    )
    predictions = [_sanitize_prediction(value) for value in model.predict(rows)]
    hectares_value = float(hectares)

    recommendations = pd.DataFrame(
        {
            "crop": candidate_crops,
            "predicted_yield_t_ha": predictions,
            "predicted_total_production_tons": [value * hectares_value for value in predictions],
        }
    )
    recommendations = recommendations.sort_values(
        by="predicted_total_production_tons",
        ascending=False,
        ignore_index=True,
    )
    return recommendations
