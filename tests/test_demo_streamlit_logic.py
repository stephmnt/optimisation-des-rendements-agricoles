from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_SRC = PROJECT_ROOT / "demo_streamlit" / "src"
if str(STREAMLIT_SRC) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_SRC))

from app_logic import current_year, fit_demo_model, list_crops, load_dataset, predict_yield, recommend_crops


DATASET_PATH = PROJECT_ROOT / "data" / "dataset_consolide.csv"


def test_load_dataset_has_required_columns() -> None:
    dataset = load_dataset(DATASET_PATH)

    assert not dataset.empty
    assert {"area", "crop", "year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "target_yield_t_ha"} <= set(dataset.columns)


def test_predict_yield_returns_non_negative_float(tmp_path: Path) -> None:
    sample = load_dataset(DATASET_PATH).head(500).copy()
    sample_path = tmp_path / "dataset_sample.csv"
    sample.to_csv(sample_path, index=False)

    model, dataset = fit_demo_model(sample_path)
    row = dataset.iloc[0]

    prediction = predict_yield(
        model,
        area=row["area"],
        crop=row["crop"],
        average_rain_fall_mm_per_year=float(row["average_rain_fall_mm_per_year"]),
        pesticides_tonnes=float(row["pesticides_tonnes"]),
        avg_temp=float(row["avg_temp"]),
        year=current_year(),
    )

    assert isinstance(prediction, float)
    assert prediction >= 0.0


def test_recommend_crops_is_sorted_by_total_production(tmp_path: Path) -> None:
    sample = load_dataset(DATASET_PATH).head(1000).copy()
    sample_path = tmp_path / "dataset_sample.csv"
    sample.to_csv(sample_path, index=False)

    model, dataset = fit_demo_model(sample_path)
    row = dataset.iloc[0]
    candidate_crops = list_crops(dataset)[:3]

    recommendations = recommend_crops(
        model,
        area=row["area"],
        hectares=12.0,
        average_rain_fall_mm_per_year=float(row["average_rain_fall_mm_per_year"]),
        pesticides_tonnes=float(row["pesticides_tonnes"]),
        avg_temp=float(row["avg_temp"]),
        candidate_crops=candidate_crops,
        year=current_year(),
    )

    assert recommendations["crop"].tolist()
    assert recommendations["predicted_total_production_tons"].is_monotonic_decreasing
