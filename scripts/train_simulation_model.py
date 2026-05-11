"""Point d'entree CLI pour la brique de simulation locale P2/P3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlflow_logging import log_and_register_sklearn_model
from scripts.mlflow_config import (
    SIMULATION_RUNTIME_EXPERIMENT_NAME,
    experiment_artifact_location,
    normalize_tracking_uri,
)
from scripts.pipeline_utils import ensure_paths_exist, relative_to_project
from scripts.prediction_adjustment import (
    SIMULATION_METADATA_PATH,
    SIMULATION_MODEL_PATH,
    load_or_train_simulation_model,
)
from scripts.runtime_model_specs import (
    DEFAULT_MLFLOW_TRACKING_URI,
    SIMULATION_RUNTIME_MODEL_SPEC,
)


SIMULATION_OUTPUTS = [
    SIMULATION_MODEL_PATH,
    SIMULATION_METADATA_PATH,
]
SIMULATION_MLFLOW_EXPERIMENT_NAME = SIMULATION_RUNTIME_EXPERIMENT_NAME


def parse_args() -> argparse.Namespace:
    """Construit l'interface en ligne de commande du script."""
    parser = argparse.ArgumentParser(
        description="Load or retrain the local simulation model used for the P2/P3 adjustment.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain the simulation model even if artifacts already exist.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200_000,
        help="Maximum number of rows sampled during training.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Train in memory without rewriting the model artifacts.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_MLFLOW_TRACKING_URI,
        help="Tracking URI MLflow utilise pour journaliser et enregistrer le modele.",
    )
    return parser.parse_args()


def _ensure_simulation_mlflow_experiment(tracking_uri: str) -> None:
    """Initialise l'experiment MLflow utilise par la brique de simulation."""
    tracking_uri = normalize_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(SIMULATION_MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(
            SIMULATION_MLFLOW_EXPERIMENT_NAME,
            artifact_location=experiment_artifact_location(
                SIMULATION_MLFLOW_EXPERIMENT_NAME,
                tracking_uri=tracking_uri,
            ),
        )
    mlflow.set_experiment(SIMULATION_MLFLOW_EXPERIMENT_NAME)


def _register_simulation_runtime_model(
    *,
    loaded_model,
    tracking_uri: str,
) -> dict[str, str]:
    """Journalise et enregistre le modele local comme registered model MLflow."""
    _ensure_simulation_mlflow_experiment(tracking_uri)
    metrics = loaded_model.metadata.get("metrics", {})
    with mlflow.start_run(run_name=f"{SIMULATION_MLFLOW_EXPERIMENT_NAME}__runtime_model"):
        mlflow.log_param("runtime_model_role", SIMULATION_RUNTIME_MODEL_SPEC.role)
        mlflow.log_param("registered_model_name", SIMULATION_RUNTIME_MODEL_SPEC.registered_model_name)
        mlflow.log_param("training_entrypoint", "scripts/train_simulation_model.py")
        mlflow.log_param("model_name", loaded_model.metadata.get("model_name"))
        mlflow.log_param("dataset_source", loaded_model.metadata.get("dataset_source"))
        mlflow.log_param("sample_size", loaded_model.metadata.get("sample_size"))
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                mlflow.log_metric(metric_name, float(metric_value))
        return log_and_register_sklearn_model(
            loaded_model.pipeline,
            artifact_name=SIMULATION_RUNTIME_MODEL_SPEC.registered_model_name,
            registered_model_name=SIMULATION_RUNTIME_MODEL_SPEC.registered_model_name,
            model_metadata={
                "runtime_model_role": SIMULATION_RUNTIME_MODEL_SPEC.role,
                "training_entrypoint": "scripts/train_simulation_model.py",
            },
        )


def train_simulation_model(
    *,
    force_retrain: bool = False,
    save_artifact: bool = True,
    sample_size: int = 200_000,
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
) -> dict[str, object]:
    """Charge ou reentraine le modele local de simulation.

    Args:
        force_retrain: Force le reentrainement meme si les artefacts existent.
        save_artifact: Ecrit les artefacts sur disque si `True`.
        sample_size: Nombre maximal de lignes echantillonnees pour l'entrainement.
        tracking_uri: Tracking URI MLflow utilise pour le registry.

    Returns:
        dict[str, object]: Resume du dataset utilise, des metriques et des sorties.
    """
    tracking_uri = normalize_tracking_uri(tracking_uri)
    reused_existing_artifact = (
        not force_retrain
        and SIMULATION_MODEL_PATH.exists()
        and SIMULATION_METADATA_PATH.exists()
    )
    loaded_model, simulation_df = load_or_train_simulation_model(
        force_retrain=force_retrain,
        save_artifact=save_artifact,
        sample_size=sample_size,
    )
    registration = _register_simulation_runtime_model(
        loaded_model=loaded_model,
        tracking_uri=tracking_uri,
    )
    loaded_model.metadata.update(
        {
            "runtime_model_role": SIMULATION_RUNTIME_MODEL_SPEC.role,
            "registered_model_name": registration["registered_model_name"],
            "registered_model_version": registration["registered_model_version"],
            "registered_model_run_id": registration["run_id"],
            "model_uri": registration["model_uri"],
        }
    )

    output_paths: list[str] = []
    if save_artifact:
        SIMULATION_METADATA_PATH.write_text(
            json.dumps(loaded_model.metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        resolved_outputs = ensure_paths_exist(SIMULATION_OUTPUTS, label="simulation model outputs")
        output_paths = [relative_to_project(path) for path in resolved_outputs]

    metrics = loaded_model.metadata.get("metrics", {})
    print(
        "[simulation] Model ready "
        f"(sample_size={loaded_model.metadata.get('sample_size')}, "
        f"test_rmse={metrics.get('test_rmse')}, test_r2={metrics.get('test_r2')})"
    )
    return {
        "dataset_rows": int(len(simulation_df)),
        "sample_size": loaded_model.metadata.get("sample_size"),
        "artifact_source": "reused_existing" if reused_existing_artifact else "retrained",
        "registered_model_name": registration["registered_model_name"],
        "registered_model_version": registration["registered_model_version"],
        "registered_model_run_id": registration["run_id"],
        "model_uri": registration["model_uri"],
        "metrics": metrics,
        "outputs": output_paths,
    }


def main() -> None:
    """Execute le script de simulation depuis la CLI."""
    args = parse_args()
    train_simulation_model(
        force_retrain=args.force_retrain,
        save_artifact=not args.no_save,
        sample_size=args.sample_size,
        tracking_uri=args.tracking_uri,
    )


if __name__ == "__main__":
    main()
