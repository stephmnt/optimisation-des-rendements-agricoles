"""Centralise les contrats des deux modeles runtime du projet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.mlflow_config import DEFAULT_MLFLOW_TRACKING_URI, PROJECT_ROOT


DEFAULT_MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"


@dataclass(frozen=True)
class RuntimeModelSpec:
    """Decrit un modele runtime attendu par l'API finale."""

    role: str
    registered_model_name: str
    output_model_path: Path
    output_metadata_path: Path


HISTORICAL_RUNTIME_MODEL_SPEC = RuntimeModelSpec(
    role="historical",
    registered_model_name="p1_historical_pipeline",
    output_model_path=DEFAULT_MODELS_DIR / "p1_historical_pipeline.joblib",
    output_metadata_path=DEFAULT_MODELS_DIR / "p1_historical_metadata.json",
)

SIMULATION_RUNTIME_MODEL_SPEC = RuntimeModelSpec(
    role="simulation",
    registered_model_name="p23_simulation_pipeline",
    output_model_path=DEFAULT_MODELS_DIR / "p23_simulation_pipeline.joblib",
    output_metadata_path=DEFAULT_MODELS_DIR / "p23_simulation_metadata.json",
)

RUNTIME_MODEL_SPECS = {
    HISTORICAL_RUNTIME_MODEL_SPEC.role: HISTORICAL_RUNTIME_MODEL_SPEC,
    SIMULATION_RUNTIME_MODEL_SPEC.role: SIMULATION_RUNTIME_MODEL_SPEC,
}
