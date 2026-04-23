from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "dataset_consolide.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "best_pipeline.joblib"

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


def load_dataset(data_path: Path | str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")

    dataset = pd.read_csv(path)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise KeyError(f"Colonnes manquantes dans le dataset : {missing_columns}")

    data = dataset[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    numeric_columns = NUMERIC_COLUMNS + [TARGET_COLUMN]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    data["area"] = data["area"].astype("string").str.strip()
    data["crop"] = data["crop"].astype("string").str.strip()
    return data.dropna(subset=["area", "crop", TARGET_COLUMN]).reset_index(drop=True)


def build_fallback_pipeline() -> Pipeline:
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


class PredictRequest(BaseModel):
    area: str = Field(..., min_length=1)
    crop: str = Field(..., min_length=1)
    year: int = Field(..., ge=1900, le=2100)
    average_rain_fall_mm_per_year: float | None = None
    pesticides_tonnes: float | None = None
    avg_temp: float | None = None


class PredictResponse(BaseModel):
    predicted_yield_t_ha: float


class RecommendRequest(BaseModel):
    area: str = Field(..., min_length=1)
    year: int = Field(..., ge=1900, le=2100)
    average_rain_fall_mm_per_year: float | None = None
    pesticides_tonnes: float | None = None
    avg_temp: float | None = None
    candidate_crops: list[str] | None = None


class RecommendationItem(BaseModel):
    crop: str
    predicted_yield_t_ha: float


class RecommendResponse(BaseModel):
    recommendations: list[RecommendationItem]


class DefaultContextResponse(BaseModel):
    average_rain_fall_mm_per_year: float
    pesticides_tonnes: float
    avg_temp: float


class MetadataResponse(BaseModel):
    areas: list[str]
    crops: list[str]
    current_year: int
    default_context: DefaultContextResponse
    model_source: str


class HealthResponse(BaseModel):
    status: str
    model_source: str


class PredictionService:
    def __init__(
        self,
        *,
        dataset_path: Path | str | None = None,
        model_path: Path | str | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path or os.getenv("DATASET_PATH", DEFAULT_DATASET_PATH))
        self.model_path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
        self.dataset = load_dataset(self.dataset_path)
        self.model, self.model_source = self._load_model()
        self.feature_columns = list(getattr(self.model, "feature_names_in_", FEATURE_COLUMNS))
        self.available_areas = sorted(self.dataset["area"].dropna().unique().tolist())
        self.available_crops = sorted(self.dataset["crop"].dropna().unique().tolist())

    def _load_model(self) -> tuple[Pipeline, str]:
        if self.model_path.exists():
            return joblib.load(self.model_path), "artifact"

        model = build_fallback_pipeline()
        model.fit(self.dataset[FEATURE_COLUMNS], self.dataset[TARGET_COLUMN])
        return model, "fallback-trained"

    def default_context(self, area: str | None = None) -> dict[str, float]:
        reference = self.dataset
        if area:
            area_slice = self.dataset.loc[self.dataset["area"] == area]
            if not area_slice.empty:
                reference = area_slice

        columns = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
        medians = reference[columns].median()
        global_medians = self.dataset[columns].median()

        def resolve_default(column: str) -> float:
            area_value = medians[column]
            if pd.notna(area_value):
                return float(area_value)

            global_value = global_medians[column]
            if pd.notna(global_value):
                return float(global_value)

            return 0.0

        return {
            "average_rain_fall_mm_per_year": resolve_default("average_rain_fall_mm_per_year"),
            "pesticides_tonnes": resolve_default("pesticides_tonnes"),
            "avg_temp": resolve_default("avg_temp"),
        }

    def _predict_rows(self, rows: list[dict[str, object]]) -> list[float]:
        feature_frame = pd.DataFrame(rows).reindex(columns=self.feature_columns)
        predictions = self.model.predict(feature_frame)
        return [max(float(value), 0.0) for value in predictions]

    def predict(self, payload: PredictRequest) -> float:
        return self._predict_rows([payload.model_dump()])[0]

    def recommend(self, payload: RecommendRequest) -> list[RecommendationItem]:
        requested_crops = payload.candidate_crops or self.available_crops
        crops = [crop.strip() for crop in requested_crops if crop and crop.strip()]
        crops = list(dict.fromkeys(crops))
        if not crops:
            raise HTTPException(status_code=400, detail="Aucune culture disponible pour la recommandation.")

        base_context = payload.model_dump(exclude={"candidate_crops"})
        rows = [{**base_context, "crop": crop} for crop in crops]
        predictions = self._predict_rows(rows)
        recommendations = [
            RecommendationItem(crop=crop, predicted_yield_t_ha=round(prediction, 4))
            for crop, prediction in zip(crops, predictions)
        ]
        return sorted(recommendations, key=lambda item: item.predicted_yield_t_ha, reverse=True)


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    return PredictionService()


app = FastAPI(
    title="Optimisation des rendements agricoles",
    description="API de prédiction et de recommandation de rendement agricole.",
    version="2.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = get_prediction_service()
    return HealthResponse(status="ok", model_source=service.model_source)


@app.get("/metadata", response_model=MetadataResponse)
def metadata(area: str | None = Query(default=None)) -> MetadataResponse:
    service = get_prediction_service()
    return MetadataResponse(
        areas=service.available_areas,
        crops=service.available_crops,
        current_year=current_year(),
        default_context=DefaultContextResponse(**service.default_context(area)),
        model_source=service.model_source,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    prediction = get_prediction_service().predict(payload)
    return PredictResponse(predicted_yield_t_ha=round(prediction, 4))


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    recommendations = get_prediction_service().recommend(payload)
    return RecommendResponse(recommendations=recommendations)
