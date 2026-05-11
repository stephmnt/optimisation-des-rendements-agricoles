"""Tests du script de promotion des registered models runtime MLflow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.promote_registered_model import (
    build_export_metadata,
    promote_registered_models,
    resolve_model_version_for_role,
    resolve_registered_model_name_for_role,
)
from scripts.runtime_model_specs import (
    HISTORICAL_RUNTIME_MODEL_SPEC,
    SIMULATION_RUNTIME_MODEL_SPEC,
)


class _FakeMlflowClient:
    """Double minimal du client MLflow pour les tests de promotion."""

    def __init__(self, *, models: list[str], versions_by_name: dict[str, list[SimpleNamespace]]) -> None:
        self._models = models
        self._versions_by_name = versions_by_name

    def search_registered_models(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name) for name in self._models]

    def search_model_versions(self, filter_string: str) -> list[SimpleNamespace]:
        model_name = filter_string.split("'", maxsplit=2)[1]
        return list(self._versions_by_name.get(model_name, []))

    def get_run(self, run_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            info=SimpleNamespace(run_name=f"run-{run_id}", experiment_id="7"),
            data=SimpleNamespace(metrics={"test_rmse": 1.23}, params={"source": "unit-test"}),
        )


def test_resolve_registered_model_name_for_role_rejects_missing_historical_model() -> None:
    with pytest.raises(ValueError) as exc_info:
        resolve_registered_model_name_for_role(
            role_spec=HISTORICAL_RUNTIME_MODEL_SPEC,
            available_names=[],
        )

    assert "No MLflow registered model found for role 'historical'" in str(exc_info.value)


def test_resolve_registered_model_name_for_role_rejects_ambiguous_simulation_candidates() -> None:
    with pytest.raises(ValueError) as exc_info:
        resolve_registered_model_name_for_role(
            role_spec=SIMULATION_RUNTIME_MODEL_SPEC,
            available_names=[
                SIMULATION_RUNTIME_MODEL_SPEC.registered_model_name,
                SIMULATION_RUNTIME_MODEL_SPEC.registered_model_name,
            ],
        )

    assert "Multiple candidate registered models found for role 'simulation'" in str(exc_info.value)
    assert "--simulation-registered-model" in str(exc_info.value)


def test_resolve_model_version_for_role_requires_explicit_version_when_multiple_versions_exist() -> None:
    versions = [
        SimpleNamespace(version="1"),
        SimpleNamespace(version="2"),
    ]

    with pytest.raises(ValueError) as exc_info:
        resolve_model_version_for_role(
            versions,
            role_spec=HISTORICAL_RUNTIME_MODEL_SPEC,
            registered_model_name=HISTORICAL_RUNTIME_MODEL_SPEC.registered_model_name,
        )

    assert "--historical-version" in str(exc_info.value)


def test_resolve_model_version_for_role_can_select_latest_version() -> None:
    versions = [
        SimpleNamespace(version="4"),
        SimpleNamespace(version="5"),
        SimpleNamespace(version="2"),
    ]

    selected_version = resolve_model_version_for_role(
        versions,
        role_spec=HISTORICAL_RUNTIME_MODEL_SPEC,
        registered_model_name=HISTORICAL_RUNTIME_MODEL_SPEC.registered_model_name,
        allow_latest_version=True,
    )

    assert selected_version.version == "5"


def test_build_export_metadata_preserves_existing_fields_and_adds_runtime_registry_context() -> None:
    source_run = SimpleNamespace(
        info=SimpleNamespace(run_name="run-champion", experiment_id="7"),
        data=SimpleNamespace(metrics={"test_rmse": 1.23}, params={"model_name": "p1_historical_pipeline"}),
    )
    model_version = SimpleNamespace(
        version="4",
        current_stage="Production",
        source="models:/m-123456",
        run_id="run-123",
    )

    metadata = build_export_metadata(
        existing_metadata={"custom_field": "kept"},
        role_spec=HISTORICAL_RUNTIME_MODEL_SPEC,
        registered_model_name=HISTORICAL_RUNTIME_MODEL_SPEC.registered_model_name,
        model_version=model_version,
        tracking_uri="sqlite:////tmp/mlflow.db",
        model_output_path=Path("/tmp/p1_historical_pipeline.joblib"),
        metadata_output_path=Path("/tmp/p1_historical_metadata.json"),
        source_run=source_run,
    )

    assert metadata["custom_field"] == "kept"
    assert metadata["runtime_model_role"] == "historical"
    assert metadata["registered_model_name"] == HISTORICAL_RUNTIME_MODEL_SPEC.registered_model_name
    assert metadata["registered_model_version"] == "4"
    assert metadata["registered_model_run_id"] == "run-123"
    assert metadata["source_run_metrics"]["test_rmse"] == 1.23


def test_promote_registered_models_exports_both_runtime_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions_by_name = {
        HISTORICAL_RUNTIME_MODEL_SPEC.registered_model_name: [
            SimpleNamespace(version="7", current_stage="None", source="models:/hist/7", run_id="hist-run")
        ],
        SIMULATION_RUNTIME_MODEL_SPEC.registered_model_name: [
            SimpleNamespace(version="3", current_stage="None", source="models:/sim/3", run_id="sim-run")
        ],
    }
    fake_client = _FakeMlflowClient(
        models=[
            HISTORICAL_RUNTIME_MODEL_SPEC.registered_model_name,
            SIMULATION_RUNTIME_MODEL_SPEC.registered_model_name,
        ],
        versions_by_name=versions_by_name,
    )

    monkeypatch.setattr("scripts.promote_registered_model.MlflowClient", lambda tracking_uri=None: fake_client)
    monkeypatch.setattr(
        "scripts.promote_registered_model.mlflow.sklearn.load_model",
        lambda model_uri: {"loaded_from": model_uri},
    )

    historical_metadata_path = tmp_path / HISTORICAL_RUNTIME_MODEL_SPEC.output_metadata_path.name
    historical_metadata_path.write_text(json.dumps({"custom_field": "kept"}), encoding="utf-8")

    summary = promote_registered_models(
        tracking_uri="sqlite:////tmp/mlflow.db",
        models_dir=tmp_path,
    )

    historical_artifact = tmp_path / HISTORICAL_RUNTIME_MODEL_SPEC.output_model_path.name
    simulation_artifact = tmp_path / SIMULATION_RUNTIME_MODEL_SPEC.output_model_path.name
    simulation_metadata_path = tmp_path / SIMULATION_RUNTIME_MODEL_SPEC.output_metadata_path.name

    assert summary["historical"]["registered_model_version"] == "7"
    assert summary["simulation"]["registered_model_version"] == "3"
    assert historical_artifact.exists()
    assert simulation_artifact.exists()

    historical_metadata = json.loads(historical_metadata_path.read_text(encoding="utf-8"))
    simulation_metadata = json.loads(simulation_metadata_path.read_text(encoding="utf-8"))

    assert historical_metadata["custom_field"] == "kept"
    assert historical_metadata["runtime_model_role"] == "historical"
    assert historical_metadata["registered_model_run_id"] == "hist-run"
    assert simulation_metadata["runtime_model_role"] == "simulation"
    assert simulation_metadata["registered_model_run_id"] == "sim-run"
    assert simulation_metadata["model_uri"] == "models:/p23_simulation_pipeline/3"
