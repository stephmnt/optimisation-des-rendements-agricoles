from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.promote_registered_model import (
    build_export_metadata,
    resolve_model_version,
    resolve_registered_model_name,
)


def test_resolve_registered_model_name_accepts_single_model_without_explicit_selection() -> None:
    assert resolve_registered_model_name(["xgboost_random_forest"]) == "xgboost_random_forest"


def test_resolve_registered_model_name_rejects_missing_registry_entry() -> None:
    with pytest.raises(ValueError) as exc_info:
        resolve_registered_model_name([])
    message = str(exc_info.value)
    assert "Aucun registered model MLflow" in message
    assert "--registered-model" in message


def test_resolve_registered_model_name_rejects_multiple_models_without_explicit_selection() -> None:
    with pytest.raises(ValueError) as exc_info:
        resolve_registered_model_name(["Random Forest", "xgboost_random_forest"])
    message = str(exc_info.value)
    assert "Plusieurs registered models MLflow" in message
    assert "--registered-model" in message
    assert "Random Forest" in message
    assert "xgboost_random_forest" in message


def test_resolve_model_version_defaults_to_latest_numeric_version() -> None:
    versions = [
        SimpleNamespace(version="1"),
        SimpleNamespace(version="3"),
        SimpleNamespace(version="2"),
    ]

    selected = resolve_model_version(versions)

    assert selected.version == "3"


def test_build_export_metadata_preserves_existing_fields_and_adds_registry_context() -> None:
    source_run = SimpleNamespace(
        info=SimpleNamespace(run_name="run-champion", experiment_id="7"),
        data=SimpleNamespace(metrics={"test_rmse": 1.23}, params={"model_name": "xgboost_random_forest"}),
    )
    model_version = SimpleNamespace(
        version="4",
        current_stage="Production",
        source="models:/m-123456",
        run_id="run-123",
    )

    metadata = build_export_metadata(
        existing_metadata={"custom_field": "kept"},
        registered_model_name="xgboost_random_forest",
        model_version=model_version,
        tracking_uri="sqlite:////tmp/mlflow.db",
        model_output_path=Path("/tmp/best_pipeline.joblib"),
        source_run=source_run,
    )

    assert metadata["custom_field"] == "kept"
    assert metadata["registered_model_name"] == "xgboost_random_forest"
    assert metadata["registered_model_version"] == "4"
    assert metadata["registered_model_run_id"] == "run-123"
    assert metadata["used_by_final_api"] is False
    assert metadata["source_run_metrics"]["test_rmse"] == 1.23
