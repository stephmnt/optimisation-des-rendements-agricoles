"""Tests des helpers de journalisation MLflow du projet."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.mlflow_logging import EvaluationPredictionLookupModel, sanitize_logged_model_name


class _FakeContext:
    """Contexte pyfunc minimal utilise pour simuler les artefacts MLflow."""

    def __init__(self, *, predictions: Path, specification: Path) -> None:
        self.artifacts = {
            "predictions": str(predictions),
            "specification": str(specification),
        }


def test_sanitize_logged_model_name_uses_run_suffix() -> None:
    assert sanitize_logged_model_name("experience_1__xgboost_random_forest_search_04") == "xgboost_random_forest_search_04"
    assert sanitize_logged_model_name("best_model_summary::random_forest") == "random_forest"
    assert sanitize_logged_model_name(" model ") == "model"


def test_evaluation_prediction_lookup_model_returns_matching_rows(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    specification_path = tmp_path / "specification.json"

    pd.DataFrame(
        [
            {
                "area": "France",
                "crop": "Wheat",
                "year": 2016,
                "actual": 6.4,
                "prediction": 6.1,
                "split": "test",
                "forecast_horizon": 1,
            },
            {
                "area": "Kenya",
                "crop": "Maize",
                "year": 2016,
                "actual": 5.2,
                "prediction": 5.5,
                "split": "test",
                "forecast_horizon": 1,
            },
        ]
    ).to_csv(predictions_path, index=False)
    specification_path.write_text(json.dumps({"order": [1, 0, 0]}), encoding="utf-8")

    model = EvaluationPredictionLookupModel()
    model.load_context(_FakeContext(predictions=predictions_path, specification=specification_path))

    result = model.predict(
        None,
        pd.DataFrame(
            [
                {"area": "France", "crop": "Wheat", "year": 2016},
                {"area": "Spain", "crop": "Barley", "year": 2016},
            ]
        ),
    )

    assert result.loc[0, "prediction"] == 6.1
    assert result.loc[0, "actual"] == 6.4
    assert pd.isna(result.loc[1, "prediction"])
