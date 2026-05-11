"""Orchestre la chaine locale officielle, de la preparation a la validation."""

from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlflow_config import (
    DEFAULT_MLFLOW_TRACKING_URI,
    FULL_PIPELINE_EXPERIMENT_NAME,
    experiment_artifact_location,
    normalize_tracking_uri,
)
from scripts.promote_registered_model import promote_registered_models
from scripts.run_preparation import run_preparation
from scripts.train_historical_model import train_historical_model
from scripts.train_simulation_model import train_simulation_model
from scripts.validate_runtime import validate_runtime


def _ensure_full_pipeline_experiment(tracking_uri: str) -> None:
    """Prepare l'experience MLflow qui trace les executions du pipeline complet."""
    mlflow.set_tracking_uri(tracking_uri)
    while mlflow.active_run() is not None:
        mlflow.end_run()

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(FULL_PIPELINE_EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(
            FULL_PIPELINE_EXPERIMENT_NAME,
            artifact_location=experiment_artifact_location(
                FULL_PIPELINE_EXPERIMENT_NAME,
                tracking_uri=tracking_uri,
            ),
        )
    mlflow.set_experiment(FULL_PIPELINE_EXPERIMENT_NAME)


def _log_numeric_metrics(prefix: str, metrics: object) -> None:
    """Journalise les metriques numeriques disponibles dans un dictionnaire."""
    if not isinstance(metrics, dict):
        return

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, bool) or not isinstance(metric_value, Real):
            continue
        numeric_value = float(metric_value)
        if math.isfinite(numeric_value):
            mlflow.log_metric(f"{prefix}_{metric_name}", numeric_value)


def _log_param_if_present(name: str, value: object) -> None:
    """Journalise un parametre MLflow seulement s'il est renseigne."""
    if value is not None:
        mlflow.log_param(name, value)


def log_pipeline_summary_to_mlflow(
    summary: dict[str, object],
    *,
    tracking_uri: str,
    skip_preparation: bool,
    skip_runtime_validation: bool,
    reuse_simulation_artifact: bool,
    simulation_sample_size: int,
) -> dict[str, str]:
    """Ajoute une trace MLflow lisible pour une execution de `run_full_pipeline.py`."""
    resolved_tracking_uri = normalize_tracking_uri(tracking_uri)
    _ensure_full_pipeline_experiment(resolved_tracking_uri)
    serializable_summary = json.loads(json.dumps(summary, ensure_ascii=True, default=str))

    with mlflow.start_run(run_name=FULL_PIPELINE_EXPERIMENT_NAME) as run:
        mlflow.log_param("entrypoint", "scripts/run_full_pipeline.py")
        mlflow.log_param("skip_preparation", bool(skip_preparation))
        mlflow.log_param("skip_runtime_validation", bool(skip_runtime_validation))
        mlflow.log_param("reuse_simulation_artifact", bool(reuse_simulation_artifact))
        mlflow.log_param("simulation_sample_size", int(simulation_sample_size))

        historical_model = serializable_summary.get("historical_model", {})
        simulation_model = serializable_summary.get("simulation_model", {})
        runtime_validation = serializable_summary.get("runtime_validation", {})

        if isinstance(historical_model, dict):
            _log_param_if_present("historical_registered_model", historical_model.get("registered_model_name"))
            _log_param_if_present(
                "historical_registered_model_version",
                historical_model.get("registered_model_version"),
            )
            _log_numeric_metrics("historical", historical_model.get("metrics"))
        if isinstance(simulation_model, dict):
            _log_param_if_present("simulation_registered_model", simulation_model.get("registered_model_name"))
            _log_param_if_present(
                "simulation_registered_model_version",
                simulation_model.get("registered_model_version"),
            )
            _log_numeric_metrics("simulation", simulation_model.get("metrics"))
        if isinstance(runtime_validation, dict):
            mlflow.log_param("runtime_validation_skipped", bool(runtime_validation.get("skipped", False)))
            _log_param_if_present("runtime_validation_status", runtime_validation.get("status", "executed"))

        mlflow.log_dict(serializable_summary, "pipeline_summary.json")
        return {
            "experiment_name": FULL_PIPELINE_EXPERIMENT_NAME,
            "run_id": run.info.run_id,
            "tracking_uri": resolved_tracking_uri,
        }


def parse_args() -> argparse.Namespace:
    """Construit l'interface en ligne de commande du pipeline complet."""
    parser = argparse.ArgumentParser(
        description="Run the project training pipeline from preparation to runtime validation.",
    )
    parser.add_argument(
        "--skip-preparation",
        action="store_true",
        help="Reuse the existing preparation outputs instead of re-executing preparation.ipynb.",
    )
    parser.add_argument(
        "--skip-runtime-validation",
        action="store_true",
        help="Skip the final smoke test against the runtime service.",
    )
    parser.add_argument(
        "--reuse-simulation-artifact",
        action="store_true",
        help="Reuse the existing P23 artifact instead of forcing a retrain.",
    )
    parser.add_argument(
        "--simulation-sample-size",
        type=int,
        default=200_000,
        help="Maximum number of rows sampled when retraining the simulation model.",
    )
    parser.add_argument(
        "--notebook-timeout-seconds",
        type=int,
        default=7200,
        help="Maximum execution time allowed for each notebook stage.",
    )
    parser.add_argument(
        "--kernel-name",
        default="python3",
        help="Jupyter kernel used to execute notebook-backed stages.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_MLFLOW_TRACKING_URI,
        help="Tracking URI MLflow partage entre entrainement et promotion.",
    )
    parser.add_argument(
        "--historical-version",
        default=None,
        help="Version MLflow historique a promouvoir. Par defaut, le pipeline prend la derniere version.",
    )
    parser.add_argument(
        "--simulation-version",
        default=None,
        help="Version MLflow simulation a promouvoir. Par defaut, le pipeline prend la derniere version.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the pipeline summary as JSON.",
    )
    return parser.parse_args()


def run_full_pipeline(
    *,
    skip_preparation: bool = False,
    skip_runtime_validation: bool = False,
    reuse_simulation_artifact: bool = False,
    simulation_sample_size: int = 200_000,
    notebook_timeout_seconds: int = 7200,
    kernel_name: str = "python3",
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    historical_version: str | None = None,
    simulation_version: str | None = None,
) -> dict[str, object]:
    """Execute les principales etapes de regeneration des artefacts.

    Args:
        skip_preparation: Saute `preparation.ipynb` si les sorties existent deja.
        skip_runtime_validation: Saute le smoke test final.
        reuse_simulation_artifact: Reutilise le modele local existant au lieu de le reentrainer.
        simulation_sample_size: Taille d'echantillon pour le modele local.
        notebook_timeout_seconds: Timeout applique a chaque notebook execute.
        kernel_name: Kernel Jupyter a utiliser.
        tracking_uri: Tracking URI MLflow utilise pour l'entrainement et la promotion.
        historical_version: Version historique a promouvoir, ou derniere version si absent.
        simulation_version: Version simulation a promouvoir, ou derniere version si absent.

    Returns:
        dict[str, object]: Resume des etapes executees et des artefacts verifies.
    """
    tracking_uri = normalize_tracking_uri(tracking_uri)
    results: dict[str, object] = {
        "mlflow": {
            "tracking_uri": tracking_uri,
            "pipeline_experiment": FULL_PIPELINE_EXPERIMENT_NAME,
        }
    }

    if not skip_preparation:
        results["preparation"] = run_preparation(
            timeout_seconds=notebook_timeout_seconds,
            kernel_name=kernel_name,
        )
    else:
        results["preparation"] = {"skipped": True}

    results["historical_model"] = train_historical_model(
        tracking_uri=tracking_uri,
        cv_splits=4,
    )

    results["simulation_model"] = train_simulation_model(
        force_retrain=not reuse_simulation_artifact,
        save_artifact=True,
        sample_size=simulation_sample_size,
        tracking_uri=tracking_uri,
    )
    results["registered_model_promotion"] = promote_registered_models(
        tracking_uri=tracking_uri,
        historical_version=historical_version,
        simulation_version=simulation_version,
        allow_latest_version=True,
    )

    if not skip_runtime_validation:
        results["runtime_validation"] = validate_runtime()
    else:
        results["runtime_validation"] = {"skipped": True}

    results["pipeline_run"] = log_pipeline_summary_to_mlflow(
        results,
        tracking_uri=tracking_uri,
        skip_preparation=skip_preparation,
        skip_runtime_validation=skip_runtime_validation,
        reuse_simulation_artifact=reuse_simulation_artifact,
        simulation_sample_size=simulation_sample_size,
    )

    return results


def main() -> None:
    """Execute le pipeline complet depuis la CLI."""
    args = parse_args()
    summary = run_full_pipeline(
        skip_preparation=args.skip_preparation,
        skip_runtime_validation=args.skip_runtime_validation,
        reuse_simulation_artifact=args.reuse_simulation_artifact,
        simulation_sample_size=args.simulation_sample_size,
        notebook_timeout_seconds=args.notebook_timeout_seconds,
        kernel_name=args.kernel_name,
        tracking_uri=args.tracking_uri,
        historical_version=args.historical_version,
        simulation_version=args.simulation_version,
    )
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return

    print("[pipeline] Completed successfully")


if __name__ == "__main__":
    main()
