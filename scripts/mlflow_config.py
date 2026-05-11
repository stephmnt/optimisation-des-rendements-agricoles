"""Configuration MLflow commune aux scripts et a l'interface locale."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB_PATH = (PROJECT_ROOT / "artifacts" / "mlflow.db").resolve()
MLFLOW_ARTIFACTS_DIR = (PROJECT_ROOT / "artifacts" / "mlruns").resolve()
DEFAULT_MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"

EXPERIENCE_1_EXPERIMENT_NAME = "experience_1"
SIMULATION_RUNTIME_EXPERIMENT_NAME = "simulation_runtime"
FULL_PIPELINE_EXPERIMENT_NAME = "run_full_pipeline"


def normalize_tracking_uri(tracking_uri: str | None = None) -> str:
    """Retourne un tracking URI MLflow stable depuis la racine du projet."""
    resolved_uri = tracking_uri or DEFAULT_MLFLOW_TRACKING_URI
    if not resolved_uri.startswith("sqlite:///"):
        return resolved_uri

    db_path = Path(resolved_uri.removeprefix("sqlite:///"))
    if not db_path.is_absolute():
        db_path = (PROJECT_ROOT / db_path).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def ensure_mlflow_directories() -> None:
    """Cree les dossiers MLflow attendus par le projet."""
    MLFLOW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def mlflow_artifacts_dir_for_tracking_uri(tracking_uri: str | None = None) -> Path:
    """Retourne la racine d'artefacts adaptee au tracking URI fourni."""
    resolved_uri = normalize_tracking_uri(tracking_uri)
    if resolved_uri == DEFAULT_MLFLOW_TRACKING_URI:
        artifact_root = MLFLOW_ARTIFACTS_DIR
    elif resolved_uri.startswith("sqlite:///"):
        artifact_root = Path(resolved_uri.removeprefix("sqlite:///")).resolve().parent / "mlruns"
    else:
        artifact_root = MLFLOW_ARTIFACTS_DIR

    artifact_root.mkdir(parents=True, exist_ok=True)
    return artifact_root


def experiment_artifact_location(experiment_name: str, tracking_uri: str | None = None) -> str:
    """Retourne l'emplacement d'artefacts standard d'une experience MLflow."""
    artifact_dir = mlflow_artifacts_dir_for_tracking_uri(tracking_uri) / experiment_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir.resolve().as_uri()
