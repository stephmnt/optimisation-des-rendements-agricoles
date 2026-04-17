from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from scripts.project_config import load_preparation_config


config = load_preparation_config()
artifacts_dir = config["ARTIFACTS_DIR"]
dataset_path = config["DATASET_PATH"]
model_path = artifacts_dir / "models" / "best_pipeline.joblib"

if not model_path.exists():
    raise FileNotFoundError(f"Modèle introuvable : {model_path}")
if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset consolidé introuvable : {dataset_path}")

model = joblib.load(model_path)
feature_columns = list(model.feature_names_in_)
available_crops = sorted(pd.read_csv(dataset_path)["crop"].dropna().unique().tolist())

app = FastAPI(
    title="Optimisation des rendements agricoles",
    description="API minimale de prédiction et de recommandation de rendement.",
    version="1.0.0",
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


def _predict_rows(rows: list[dict]) -> list[float]:
    feature_frame = pd.DataFrame(rows).reindex(columns=feature_columns)
    predictions = model.predict(feature_frame)
    return [float(value) for value in predictions]


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    prediction = _predict_rows([payload.model_dump()])[0]
    return PredictResponse(predicted_yield_t_ha=round(prediction, 4))


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    crops = payload.candidate_crops or available_crops
    if not crops:
        raise HTTPException(status_code=400, detail="Aucune culture disponible pour la recommandation.")

    base_context = payload.model_dump(exclude={"candidate_crops"})
    rows = [{**base_context, "crop": crop} for crop in crops]
    predictions = _predict_rows(rows)

    recommendations = sorted(
        [
            RecommendationItem(crop=crop, predicted_yield_t_ha=round(prediction, 4))
            for crop, prediction in zip(crops, predictions)
        ],
        key=lambda item: item.predicted_yield_t_ha,
        reverse=True,
    )
    return RecommendResponse(recommendations=recommendations)
