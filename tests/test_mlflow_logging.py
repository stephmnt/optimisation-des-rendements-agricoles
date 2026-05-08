"""Tests des helpers de journalisation MLflow du projet."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.mlflow_logging import (
    EvaluationPredictionLookupModel,
    log_and_register_sklearn_model,
    sanitize_logged_model_name,
)


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


def test_log_and_register_sklearn_model_returns_registry_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.mlflow_logging.mlflow.active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="run-123")),
    )
    monkeypatch.setattr("scripts.mlflow_logging.mlflow.get_tracking_uri", lambda: "sqlite:////tmp/mlflow.db")
    monkeypatch.setattr(
        "scripts.mlflow_logging.mlflow.sklearn.log_model",
        lambda estimator, **kwargs: SimpleNamespace(model_uri=f"runs:/run-123/{kwargs['name']}"),
    )
    monkeypatch.setattr(
        "scripts.mlflow_logging.resolve_registered_model_version_for_run",
        lambda **kwargs: SimpleNamespace(version="5"),
    )

    summary = log_and_register_sklearn_model(
        estimator={"model": "fake"},
        artifact_name="runtime::historical",
        registered_model_name="p1_historical_pipeline",
        model_metadata={"runtime_model_role": "historical"},
    )

    assert summary["logged_model_name"] == "historical"
    assert summary["registered_model_name"] == "p1_historical_pipeline"
    assert summary["registered_model_version"] == "5"
    assert summary["run_id"] == "run-123"
    assert summary["model_uri"] == "models:/p1_historical_pipeline/5"
