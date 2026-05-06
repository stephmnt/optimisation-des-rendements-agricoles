from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from scripts.prediction_adjustment import AdjustedYieldService


def _round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


COUNTRY_TO_SIMULATION_REGION = {
    'Afghanistan': 'East',
    'Albania': 'North',
    'Algeria': 'North',
    'American Samoa': 'West',
    'Angola': 'East',
    'Antigua and Barbuda': 'West',
    'Argentina': 'West',
    'Armenia': 'East',
    'Australia': 'East',
    'Austria': 'North',
    'Azerbaijan': 'East',
    'Bahamas': 'West',
    'Bahrain': 'East',
    'Bangladesh': 'East',
    'Barbados': 'West',
    'Belarus': 'North',
    'Belgium': 'North',
    'Belgium-Luxembourg': 'North',
    'Belize': 'West',
    'Benin': 'North',
    'Bermuda': 'West',
    'Bhutan': 'East',
    'Bolivia (Plurinational State of)': 'West',
    'Bosnia and Herzegovina': 'North',
    'Botswana': 'East',
    'Brazil': 'West',
    'Brunei Darussalam': 'East',
    'Bulgaria': 'North',
    'Burkina Faso': 'North',
    'Burundi': 'East',
    'Cabo Verde': 'West',
    'Cambodia': 'East',
    'Cameroon': 'East',
    'Canada': 'West',
    'Cayman Islands': 'West',
    'Central African Republic': 'East',
    'Chad': 'East',
    'Chile': 'West',
    'China': 'East',
    'China, Hong Kong SAR': 'East',
    'China, Taiwan Province of': 'East',
    'China, mainland': 'East',
    'Colombia': 'West',
    'Comoros': 'East',
    'Congo': 'East',
    'Cook Islands': 'West',
    'Costa Rica': 'West',
    'Croatia': 'North',
    'Cuba': 'West',
    'Cyprus': 'North',
    'Czechia': 'North',
    'Czechoslovakia': 'North',
    "Côte d'Ivoire": 'North',
    "Democratic People's Republic of Korea": 'East',
    'Democratic Republic of the Congo': 'East',
    'Denmark': 'North',
    'Djibouti': 'East',
    'Dominica': 'West',
    'Dominican Republic': 'West',
    'Ecuador': 'West',
    'Egypt': 'East',
    'El Salvador': 'West',
    'Equatorial Guinea': 'East',
    'Eritrea': 'East',
    'Estonia': 'North',
    'Eswatini': 'East',
    'Ethiopia': 'East',
    'Ethiopia PDR': 'East',
    'Faroe Islands': 'North',
    'Fiji': 'East',
    'Finland': 'North',
    'France': 'North',
    'French Guiana': 'West',
    'French Polynesia': 'West',
    'Gabon': 'East',
    'Gambia': 'West',
    'Georgia': 'East',
    'Germany': 'North',
    'Ghana': 'North',
    'Greece': 'North',
    'Grenada': 'West',
    'Guadeloupe': 'West',
    'Guam': 'East',
    'Guatemala': 'West',
    'Guinea': 'West',
    'Guinea-Bissau': 'West',
    'Guyana': 'West',
    'Haiti': 'West',
    'Honduras': 'West',
    'Hungary': 'North',
    'Iceland': 'North',
    'India': 'East',
    'Indonesia': 'East',
    'Iran (Islamic Republic of)': 'East',
    'Iraq': 'East',
    'Ireland': 'North',
    'Israel': 'East',
    'Italy': 'North',
    'Jamaica': 'West',
    'Japan': 'East',
    'Jordan': 'East',
    'Kazakhstan': 'East',
    'Kenya': 'East',
    'Kuwait': 'East',
    'Kyrgyzstan': 'East',
    "Lao People's Democratic Republic": 'East',
    'Latvia': 'North',
    'Lebanon': 'East',
    'Lesotho': 'South',
    'Liberia': 'West',
    'Libya': 'North',
    'Lithuania': 'North',
    'Luxembourg': 'North',
    'Madagascar': 'East',
    'Malawi': 'East',
    'Malaysia': 'East',
    'Maldives': 'East',
    'Mali': 'North',
    'Malta': 'North',
    'Martinique': 'West',
    'Mauritania': 'North',
    'Mauritius': 'East',
    'Mexico': 'West',
    'Micronesia (Federated States of)': 'East',
    'Mongolia': 'East',
    'Montenegro': 'North',
    'Montserrat': 'West',
    'Morocco': 'North',
    'Mozambique': 'East',
    'Myanmar': 'East',
    'Namibia': 'South',
    'Nepal': 'East',
    'Netherlands': 'North',
    'New Caledonia': 'East',
    'New Zealand': 'East',
    'Nicaragua': 'West',
    'Niger': 'North',
    'Nigeria': 'North',
    'Niue': 'West',
    'Norway': 'North',
    'Occupied Palestinian Territory': 'East',
    'Oman': 'East',
    'Pacific Islands Trust Territory': 'East',
    'Pakistan': 'East',
    'Panama': 'West',
    'Papua New Guinea': 'East',
    'Paraguay': 'West',
    'Peru': 'West',
    'Philippines': 'East',
    'Poland': 'North',
    'Portugal': 'North',
    'Puerto Rico': 'West',
    'Qatar': 'East',
    'Republic of Korea': 'East',
    'Republic of Moldova': 'North',
    'Romania': 'North',
    'Russian Federation': 'East',
    'Rwanda': 'East',
    'Réunion': 'East',
    'Saint Kitts and Nevis': 'West',
    'Saint Lucia': 'West',
    'Saint Vincent and the Grenadines': 'West',
    'Samoa': 'West',
    'Sao Tome and Principe': 'East',
    'Saudi Arabia': 'East',
    'Senegal': 'West',
    'Serbia': 'North',
    'Serbia and Montenegro': 'North',
    'Seychelles': 'East',
    'Sierra Leone': 'West',
    'Singapore': 'East',
    'Slovakia': 'North',
    'Slovenia': 'North',
    'Solomon Islands': 'East',
    'Somalia': 'East',
    'South Africa': 'South',
    'South Sudan': 'East',
    'Spain': 'North',
    'Sri Lanka': 'East',
    'Sudan': 'East',
    'Sudan (former)': 'North',
    'Suriname': 'West',
    'Sweden': 'North',
    'Switzerland': 'North',
    'Syrian Arab Republic': 'East',
    'Tajikistan': 'East',
    'Thailand': 'East',
    'The former Yugoslav Republic of Macedonia': 'North',
    'Timor-Leste': 'East',
    'Togo': 'North',
    'Tonga': 'West',
    'Trinidad and Tobago': 'West',
    'Tunisia': 'North',
    'Turkey': 'North',
    'Turkmenistan': 'East',
    'USSR': 'North',
    'Uganda': 'East',
    'Ukraine': 'North',
    'United Arab Emirates': 'East',
    'United Kingdom': 'North',
    'United Republic of Tanzania': 'East',
    'United States of America': 'West',
    'Uruguay': 'West',
    'Uzbekistan': 'East',
    'Vanuatu': 'East',
    'Venezuela (Bolivarian Republic of)': 'West',
    'Viet Nam': 'East',
    'Wallis and Futuna Islands': 'West',
    'Yemen': 'East',
    'Yugoslav SFR': 'North',
    'Zambia': 'East',
    'Zimbabwe': 'East',
}


def _normalize_country(country: str) -> str:
    return country.strip()


def _infer_region_from_country(country: str | None) -> str | None:
    if not country:
        return None
    return COUNTRY_TO_SIMULATION_REGION.get(_normalize_country(country))


def _reference_overrides_from_country(country: str | None) -> dict[str, str]:
    inferred_region = _infer_region_from_country(country)
    if inferred_region is None:
        return {}
    return {"region": inferred_region}


class UserConditionsRequest(BaseModel):
    region: str | None = None
    soil_type: str | None = None
    rainfall_mm: float | None = Field(default=None, ge=0.0)
    temperature_celsius: float | None = None
    fertilizer_used: bool | None = None
    irrigation_used: bool | None = None
    weather_condition: str | None = None
    days_to_harvest: float | None = Field(default=None, ge=0.0)


class ConditionProfileResponse(BaseModel):
    region: str
    soil_type: str
    rainfall_mm: float
    temperature_celsius: float
    fertilizer_used: bool
    irrigation_used: bool
    weather_condition: str
    days_to_harvest: float


class SimulationOptionsResponse(BaseModel):
    regions: list[str]
    soil_types: list[str]
    weather_conditions: list[str]


class HealthV2Response(BaseModel):
    status: str
    strategy: str
    historical_model_name: str
    simulation_model_name: str


class MetadataV2Response(BaseModel):
    countries: list[str]
    available_crops: list[str]
    target_year: int
    strategy: str
    historical_model_name: str
    simulation_model_name: str
    global_reference_profile: ConditionProfileResponse
    simulation_options: SimulationOptionsResponse
    inferred_region: str | None = None


class BaselineRequest(BaseModel):
    country: str = Field(..., min_length=1)
    crop: str = Field(..., min_length=1)


class BaselineResponse(BaseModel):
    country: str
    crop: str
    target_year: int
    p1_historical_prediction: float
    reference_profile: ConditionProfileResponse
    rainfall_reference_source: str
    temperature_reference_source: str


class PredictAdjustedRequest(BaseModel):
    country: str = Field(..., min_length=1)
    crop: str = Field(..., min_length=1)
    user_conditions: UserConditionsRequest


class HistoricalShapContributionResponse(BaseModel):
    feature: str
    raw_value: str | float | bool | None
    contribution: float
    abs_contribution: float


class HistoricalShapExplanationResponse(BaseModel):
    available: bool
    status: str
    message: str | None = None
    model_prediction: float
    base_value: float | None = None
    prediction_from_shap: float | None = None
    top_contributions: list[HistoricalShapContributionResponse] = Field(default_factory=list)


class LocalAdjustmentContributionResponse(BaseModel):
    feature: str
    reference_value: str | float | bool | None
    user_value: str | float | bool | None
    contribution_delta: float
    abs_contribution_delta: float


class LocalAdjustmentExplanationResponse(BaseModel):
    method: str
    reference_prediction: float
    user_prediction: float
    total_adjustment: float
    top_contributions: list[LocalAdjustmentContributionResponse] = Field(default_factory=list)


class PredictionExplanationResponse(BaseModel):
    historical_shap: HistoricalShapExplanationResponse
    local_adjustment: LocalAdjustmentExplanationResponse


class PredictAdjustedResponse(BaseModel):
    country: str
    crop: str
    p1_historical_prediction: float
    p2_reference_simulation: float
    p3_user_simulation: float
    local_adjustment: float
    gap_vs_historical_pct: float
    final_prediction: float
    reference_profile: ConditionProfileResponse
    user_profile: ConditionProfileResponse
    rainfall_reference_source: str
    temperature_reference_source: str
    explanation: PredictionExplanationResponse


class RecommendationItemResponse(BaseModel):
    country: str
    crop: str
    p1_historical_prediction: float
    p2_reference_simulation: float
    p3_user_simulation: float
    local_adjustment: float
    gap_vs_historical_pct: float
    final_prediction: float
    recommendation_rank: int
    rainfall_reference_source: str
    temperature_reference_source: str


class RecommendAdjustedRequest(BaseModel):
    country: str = Field(..., min_length=1)
    user_conditions: UserConditionsRequest
    candidate_crops: list[str] | None = None
    top_n: int | None = Field(default=None, ge=1)


class RecommendAdjustedResponse(BaseModel):
    country: str
    best_crop: str
    best_final_prediction: float
    recommendations: list[RecommendationItemResponse]


def _to_condition_profile(profile: dict[str, object]) -> ConditionProfileResponse:
    return ConditionProfileResponse(
        region=str(profile["region"]),
        soil_type=str(profile["soil_type"]),
        rainfall_mm=_round_float(float(profile["rainfall_mm"])),
        temperature_celsius=_round_float(float(profile["temperature_celsius"])),
        fertilizer_used=bool(profile["fertilizer_used"]),
        irrigation_used=bool(profile["irrigation_used"]),
        weather_condition=str(profile["weather_condition"]),
        days_to_harvest=_round_float(float(profile["days_to_harvest"])),
    )


def _to_prediction_explanation(explanation_payload: dict[str, object]) -> PredictionExplanationResponse:
    historical_payload = explanation_payload["historical_shap"]
    local_payload = explanation_payload["local_adjustment"]

    return PredictionExplanationResponse(
        historical_shap=HistoricalShapExplanationResponse(
            available=bool(historical_payload["available"]),
            status=str(historical_payload["status"]),
            message=historical_payload.get("message"),
            model_prediction=_round_float(historical_payload["model_prediction"]),
            base_value=(
                _round_float(historical_payload["base_value"])
                if historical_payload.get("base_value") is not None
                else None
            ),
            prediction_from_shap=(
                _round_float(historical_payload["prediction_from_shap"])
                if historical_payload.get("prediction_from_shap") is not None
                else None
            ),
            top_contributions=[
                HistoricalShapContributionResponse(
                    feature=str(item["feature"]),
                    raw_value=item.get("raw_value"),
                    contribution=_round_float(item["contribution"]),
                    abs_contribution=_round_float(item["abs_contribution"]),
                )
                for item in historical_payload.get("top_contributions", [])
            ],
        ),
        local_adjustment=LocalAdjustmentExplanationResponse(
            method=str(local_payload["method"]),
            reference_prediction=_round_float(local_payload["reference_prediction"]),
            user_prediction=_round_float(local_payload["user_prediction"]),
            total_adjustment=_round_float(local_payload["total_adjustment"]),
            top_contributions=[
                LocalAdjustmentContributionResponse(
                    feature=str(item["feature"]),
                    reference_value=item.get("reference_value"),
                    user_value=item.get("user_value"),
                    contribution_delta=_round_float(item["contribution_delta"]),
                    abs_contribution_delta=_round_float(item["abs_contribution_delta"]),
                )
                for item in local_payload.get("top_contributions", [])
            ],
        ),
    )


@lru_cache(maxsize=1)
def get_adjusted_yield_service() -> AdjustedYieldService:
    return AdjustedYieldService()


app = FastAPI(
    title="Optimisation des rendements agricoles - API v2",
    description="API v2 basée sur 2 modèles et 3 prédictions combinées.",
    version="2.0.0",
)


@app.get("/health", response_model=HealthV2Response)
def health() -> HealthV2Response:
    service = get_adjusted_yield_service()
    return HealthV2Response(
        status="ok",
        strategy="2_models_3_predictions_combined",
        historical_model_name=str(service.historical_metadata["model_name"]),
        simulation_model_name=str(service.simulation_metadata["model_name"]),
    )


@app.get("/metadata", response_model=MetadataV2Response)
def metadata(country: str | None = Query(default=None)) -> MetadataV2Response:
    service = get_adjusted_yield_service()
    available_crops = service.available_crops
    inferred_region = _infer_region_from_country(country)
    if country:
        available_crops = service.crops_by_area.get(country.strip(), [])
    reference_profile = dict(service.simulation_global_reference)
    if inferred_region is not None:
        reference_profile["region"] = inferred_region

    return MetadataV2Response(
        countries=service.available_areas,
        available_crops=available_crops,
        target_year=service.target_year,
        strategy="2_models_3_predictions_combined",
        historical_model_name=str(service.historical_metadata["model_name"]),
        simulation_model_name=str(service.simulation_metadata["model_name"]),
        global_reference_profile=_to_condition_profile(reference_profile),
        simulation_options=SimulationOptionsResponse(**service.simulation_options),
        inferred_region=inferred_region,
    )


@app.post("/baseline", response_model=BaselineResponse)
def baseline(payload: BaselineRequest) -> BaselineResponse:
    service = get_adjusted_yield_service()
    try:
        baseline_payload = service.get_baseline(
            payload.country,
            payload.crop,
            reference_overrides=_reference_overrides_from_country(payload.country),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return BaselineResponse(
        country=baseline_payload["country"],
        crop=baseline_payload["crop"],
        target_year=int(baseline_payload["target_year"]),
        p1_historical_prediction=_round_float(baseline_payload["p1_historical_prediction"]),
        reference_profile=_to_condition_profile(baseline_payload["reference_profile"]),
        rainfall_reference_source=str(baseline_payload["rainfall_reference_source"]),
        temperature_reference_source=str(baseline_payload["temperature_reference_source"]),
    )


@app.post("/predict", response_model=PredictAdjustedResponse)
def predict(payload: PredictAdjustedRequest) -> PredictAdjustedResponse:
    service = get_adjusted_yield_service()
    try:
        prediction_payload = service.predict_adjusted_yield(
            area=payload.country,
            crop=payload.crop,
            user_conditions=payload.user_conditions.model_dump(),
            reference_overrides=_reference_overrides_from_country(payload.country),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictAdjustedResponse(
        country=prediction_payload["country"],
        crop=prediction_payload["crop"],
        p1_historical_prediction=_round_float(prediction_payload["p1_historical_prediction"]),
        p2_reference_simulation=_round_float(prediction_payload["p2_reference_simulation"]),
        p3_user_simulation=_round_float(prediction_payload["p3_user_simulation"]),
        local_adjustment=_round_float(prediction_payload["local_adjustment"]),
        gap_vs_historical_pct=_round_float(prediction_payload["gap_vs_historical_pct"]),
        final_prediction=_round_float(prediction_payload["final_prediction"]),
        reference_profile=_to_condition_profile(prediction_payload["reference_profile"]),
        user_profile=_to_condition_profile(prediction_payload["user_profile"]),
        rainfall_reference_source=str(prediction_payload["rainfall_reference_source"]),
        temperature_reference_source=str(prediction_payload["temperature_reference_source"]),
        explanation=_to_prediction_explanation(prediction_payload["explanation"]),
    )


@app.post("/recommend", response_model=RecommendAdjustedResponse)
def recommend(payload: RecommendAdjustedRequest) -> RecommendAdjustedResponse:
    service = get_adjusted_yield_service()
    try:
        recommendation_df = service.recommend_crops(
            area=payload.country,
            user_conditions=payload.user_conditions.model_dump(),
            candidate_crops=payload.candidate_crops,
            reference_overrides=_reference_overrides_from_country(payload.country),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if payload.top_n is not None:
        recommendation_df = recommendation_df.head(payload.top_n).reset_index(drop=True)

    recommendations = [
        RecommendationItemResponse(
            country=str(row["country"]),
            crop=str(row["crop"]),
            p1_historical_prediction=_round_float(row["p1_historical_prediction"]),
            p2_reference_simulation=_round_float(row["p2_reference_simulation"]),
            p3_user_simulation=_round_float(row["p3_user_simulation"]),
            local_adjustment=_round_float(row["local_adjustment"]),
            gap_vs_historical_pct=_round_float(row["gap_vs_historical_pct"]),
            final_prediction=_round_float(row["final_prediction"]),
            recommendation_rank=int(row["recommendation_rank"]),
            rainfall_reference_source=str(row["rainfall_reference_source"]),
            temperature_reference_source=str(row["temperature_reference_source"]),
        )
        for _, row in recommendation_df.iterrows()
    ]

    if not recommendations:
        raise HTTPException(status_code=400, detail="No recommendation available for the provided request.")

    return RecommendAdjustedResponse(
        country=payload.country.strip(),
        best_crop=recommendations[0].crop,
        best_final_prediction=recommendations[0].final_prediction,
        recommendations=recommendations,
    )
