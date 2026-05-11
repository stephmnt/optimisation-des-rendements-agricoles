"""Tests de la configuration MLflow commune du projet."""

from __future__ import annotations

from pathlib import Path

from scripts.mlflow_config import (
    DEFAULT_MLFLOW_TRACKING_URI,
    FULL_PIPELINE_EXPERIMENT_NAME,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_DB_PATH,
    PROJECT_ROOT,
    experiment_artifact_location,
    mlflow_artifacts_dir_for_tracking_uri,
    normalize_tracking_uri,
)


def test_default_tracking_uri_targets_project_database() -> None:
    assert MLFLOW_DB_PATH == (PROJECT_ROOT / "artifacts" / "mlflow.db").resolve()
    assert DEFAULT_MLFLOW_TRACKING_URI == f"sqlite:///{MLFLOW_DB_PATH}"


def test_relative_sqlite_tracking_uri_is_resolved_from_project_root() -> None:
    assert normalize_tracking_uri("sqlite:///artifacts/mlflow.db") == DEFAULT_MLFLOW_TRACKING_URI


def test_absolute_sqlite_tracking_uri_is_preserved(tmp_path: Path) -> None:
    db_path = tmp_path / "mlflow.db"
    assert normalize_tracking_uri(f"sqlite:///{db_path}") == f"sqlite:///{db_path}"


def test_full_pipeline_artifact_location_uses_project_mlruns() -> None:
    expected_dir = MLFLOW_ARTIFACTS_DIR / FULL_PIPELINE_EXPERIMENT_NAME
    assert experiment_artifact_location(FULL_PIPELINE_EXPERIMENT_NAME) == expected_dir.resolve().as_uri()


def test_temporary_sqlite_tracking_uri_uses_local_artifact_root(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    assert mlflow_artifacts_dir_for_tracking_uri(tracking_uri) == tmp_path / "mlruns"
    assert (
        experiment_artifact_location(FULL_PIPELINE_EXPERIMENT_NAME, tracking_uri=tracking_uri)
        == (tmp_path / "mlruns" / FULL_PIPELINE_EXPERIMENT_NAME).resolve().as_uri()
    )
