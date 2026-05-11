"""Tests legers pour la trace MLflow du pipeline complet."""

from __future__ import annotations

from pathlib import Path

from mlflow.tracking import MlflowClient

from scripts.mlflow_config import FULL_PIPELINE_EXPERIMENT_NAME
from scripts.run_full_pipeline import log_pipeline_summary_to_mlflow, run_full_pipeline


def test_log_pipeline_summary_creates_full_pipeline_experiment(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    summary = {
        "historical_model": {
            "registered_model_name": "p1_historical_pipeline",
            "registered_model_version": "1",
            "metrics": {"test_rmse": 2.0, "test_r2": 0.9},
        },
        "simulation_model": {
            "registered_model_name": "p23_simulation_pipeline",
            "registered_model_version": "3",
            "metrics": {"test_rmse": 0.5, "test_r2": 0.91},
        },
        "runtime_validation": {"status": "ok"},
    }

    result = log_pipeline_summary_to_mlflow(
        summary,
        tracking_uri=tracking_uri,
        skip_preparation=True,
        skip_runtime_validation=False,
        reuse_simulation_artifact=False,
        simulation_sample_size=10_000,
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(FULL_PIPELINE_EXPERIMENT_NAME)
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert result["run_id"] == runs[0].info.run_id
    assert runs[0].data.params["historical_registered_model"] == "p1_historical_pipeline"
    assert runs[0].data.params["simulation_registered_model_version"] == "3"
    assert runs[0].data.metrics["historical_test_rmse"] == 2.0


def test_run_full_pipeline_promotes_latest_versions_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    promotion_kwargs = {}

    monkeypatch.setattr("scripts.run_full_pipeline.train_historical_model", lambda **kwargs: {"metrics": {}})
    monkeypatch.setattr("scripts.run_full_pipeline.train_simulation_model", lambda **kwargs: {"metrics": {}})
    monkeypatch.setattr("scripts.run_full_pipeline.validate_runtime", lambda: {"status": "ok"})
    monkeypatch.setattr(
        "scripts.run_full_pipeline.promote_registered_models",
        lambda **kwargs: promotion_kwargs.update(kwargs) or {"historical": {}, "simulation": {}},
    )

    run_full_pipeline(
        skip_preparation=True,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        historical_version="5",
        simulation_version="3",
    )

    assert promotion_kwargs["historical_version"] == "5"
    assert promotion_kwargs["simulation_version"] == "3"
    assert promotion_kwargs["allow_latest_version"] is True
