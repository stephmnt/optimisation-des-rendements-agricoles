from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'artifacts' / 'mlflow.db').resolve()}"
DEFAULT_MODEL_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "models" / "best_pipeline.joblib"
DEFAULT_METADATA_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "models" / "best_pipeline_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exporte un registered model MLflow vers artifacts/models/best_pipeline.joblib. "
            "Sans --registered-model, le script n'accepte qu'un seul registered model dans le registre."
        )
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="Tracking URI MLflow. Par defaut: base SQLite locale du projet.",
    )
    parser.add_argument(
        "--registered-model",
        default=None,
        help="Nom du registered model MLflow a exporter.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version du registered model a exporter. Par defaut: derniere version disponible.",
    )
    parser.add_argument(
        "--output-model-path",
        default=str(DEFAULT_MODEL_OUTPUT_PATH),
        help="Chemin de sortie du pipeline joblib exporte.",
    )
    parser.add_argument(
        "--output-metadata-path",
        default=str(DEFAULT_METADATA_OUTPUT_PATH),
        help="Chemin de sortie des metadonnees JSON.",
    )
    return parser.parse_args()


def project_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def normalize_registered_model_names(models: list[Any]) -> list[str]:
    return sorted(str(model.name) for model in models)


def resolve_registered_model_name(available_names: list[str], requested_name: str | None = None) -> str:
    if requested_name:
        if requested_name not in available_names:
            available = ", ".join(available_names) if available_names else "aucun"
            raise ValueError(
                f"Registered model introuvable: {requested_name}. "
                f"Modeles disponibles: {available}."
            )
        return requested_name

    if not available_names:
        raise ValueError(
            "Aucun registered model MLflow trouvé. "
            "Sélectionnez d'abord un registered model avec --registered-model."
        )

    if len(available_names) > 1:
        available = ", ".join(available_names)
        raise ValueError(
            "Plusieurs registered models MLflow trouvés. "
            f"Sélectionnez explicitement un registered model avec --registered-model. "
            f"Modeles disponibles: {available}."
        )

    return available_names[0]


def _version_sort_key(version: str) -> tuple[int, str]:
    value = str(version)
    return (int(value), value) if value.isdigit() else (-1, value)


def resolve_model_version(versions: list[Any], requested_version: str | None = None) -> Any:
    if requested_version is not None:
        for version in versions:
            if str(version.version) == str(requested_version):
                return version
        available = ", ".join(str(version.version) for version in versions) if versions else "aucune"
        raise ValueError(
            f"Version introuvable: {requested_version}. Versions disponibles: {available}."
        )

    if not versions:
        raise ValueError("Aucune version disponible pour le registered model sélectionné.")

    return max(versions, key=lambda version: _version_sort_key(str(version.version)))


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def build_export_metadata(
    *,
    existing_metadata: dict[str, Any],
    registered_model_name: str,
    model_version: Any,
    tracking_uri: str,
    model_output_path: Path,
    source_run: Any | None,
) -> dict[str, Any]:
    metadata = dict(existing_metadata)

    metadata.update(
        {
            "artifact_role": "mlflow_registered_model_export",
            "registered_model_name": registered_model_name,
            "registered_model_version": str(model_version.version),
            "registered_model_stage": str(getattr(model_version, "current_stage", "None") or "None"),
            "registered_model_source": str(getattr(model_version, "source", "")),
            "registered_model_run_id": getattr(model_version, "run_id", None),
            "model_uri": f"models:/{registered_model_name}/{model_version.version}",
            "tracking_uri": tracking_uri,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_path": project_relative_path(model_output_path),
            "used_by_final_api": False,
            "consumer": "single_model_export_pipeline",
        }
    )

    if source_run is not None:
        metadata["source_run_name"] = str(source_run.info.run_name)
        metadata["source_experiment_id"] = str(source_run.info.experiment_id)
        metadata["source_run_metrics"] = {
            key: float(value) for key, value in source_run.data.metrics.items()
        }
        metadata["source_run_params"] = dict(source_run.data.params)

    return metadata


def export_registered_model(
    *,
    tracking_uri: str,
    registered_model_name: str,
    model_version: Any,
    model_output_path: Path,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{registered_model_name}/{model_version.version}"
    estimator = mlflow.sklearn.load_model(model_uri)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, model_output_path)


def main() -> None:
    args = parse_args()
    tracking_uri = str(args.tracking_uri)
    model_output_path = Path(args.output_model_path).resolve()
    metadata_output_path = Path(args.output_metadata_path).resolve()

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    registered_models = list(client.search_registered_models())
    available_names = normalize_registered_model_names(registered_models)
    registered_model_name = resolve_registered_model_name(
        available_names,
        requested_name=args.registered_model,
    )

    model_versions = list(client.search_model_versions(f"name = '{registered_model_name}'"))
    selected_version = resolve_model_version(model_versions, requested_version=args.version)
    source_run = client.get_run(selected_version.run_id) if getattr(selected_version, "run_id", None) else None

    export_registered_model(
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
        model_version=selected_version,
        model_output_path=model_output_path,
    )

    existing_metadata = read_json_if_exists(metadata_output_path)
    export_metadata = build_export_metadata(
        existing_metadata=existing_metadata,
        registered_model_name=registered_model_name,
        model_version=selected_version,
        tracking_uri=tracking_uri,
        model_output_path=model_output_path,
        source_run=source_run,
    )
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.write_text(
        json.dumps(json_ready(export_metadata), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"Registered model exporté : {registered_model_name}")
    print(f"Version exportée : {selected_version.version}")
    print(f"Run source : {getattr(selected_version, 'run_id', None)}")
    print(f"Pipeline joblib : {model_output_path}")
    print(f"Metadonnées : {metadata_output_path}")


if __name__ == "__main__":
    main()
