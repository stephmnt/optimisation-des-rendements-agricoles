from datetime import date
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
model = joblib.load(model_path)
feature_columns = list(model.feature_names_in_)
available_crops = sorted(pd.read_csv(dataset_path)["crop"].dropna().unique().tolist())
CURRENT_YEAR = date.today().year

app = FastAPI(
    title="Optimisation des rendements agricoles",
    description="API de prédiction et de recommandation de rendement.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    area: str = Field(..., min_length=1)
    crop: str = Field(..., min_length=1)
    hectares: float = Field(..., gt=0)
    average_rain_fall_mm_per_year: float | None = None
    pesticides_tonnes: float | None = None
    avg_temp: float | None = None


class PredictResponse(BaseModel):
    year_used: int
    hectares: float
    predicted_yield_t_ha: float
    predicted_total_production_tons: float


class RecommendRequest(BaseModel):
    area: str = Field(..., min_length=1)
    hectares: float = Field(..., gt=0)
    average_rain_fall_mm_per_year: float | None = None
    pesticides_tonnes: float | None = None
    avg_temp: float | None = None
    candidate_crops: list[str] | None = None


class RecommendationItem(BaseModel):
    crop: str
    predicted_yield_t_ha: float
    predicted_total_production_tons: float


class RecommendResponse(BaseModel):
    year_used: int
    hectares: float
    recommendations: list[RecommendationItem]


def _predict_rows(rows: list[dict]) -> list[float]:
    feature_frame = pd.DataFrame(rows).reindex(columns=feature_columns)
    predictions = model.predict(feature_frame)
    return [float(value) for value in predictions]


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    payload_dict = payload.model_dump()
    hectares = float(payload_dict.pop("hectares"))
    prediction = _predict_rows([{**payload_dict, "year": CURRENT_YEAR}])[0]
    total_production = prediction * hectares
    return PredictResponse(
        year_used=CURRENT_YEAR,
        hectares=round(hectares, 4),
        predicted_yield_t_ha=round(prediction, 4),
        predicted_total_production_tons=round(total_production, 4),
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    crops = payload.candidate_crops or available_crops

    payload_dict = payload.model_dump(exclude={"candidate_crops"})
    hectares = float(payload_dict.pop("hectares"))
    base_context = {**payload_dict, "year": CURRENT_YEAR}
    rows = [{**base_context, "crop": crop} for crop in crops]
    predictions = _predict_rows(rows)

    recommendations = sorted(
        [
            RecommendationItem(
                crop=crop,
                predicted_yield_t_ha=round(prediction, 4),
                predicted_total_production_tons=round(prediction * hectares, 4),
            )
            for crop, prediction in zip(crops, predictions)
        ],
        key=lambda item: item.predicted_total_production_tons,
        reverse=True,
    )
    return RecommendResponse(
        year_used=CURRENT_YEAR,
        hectares=round(hectares, 4),
        recommendations=recommendations,
    )
