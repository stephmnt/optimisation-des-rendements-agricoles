"""Promouvoit les deux registered models runtime depuis MLflow vers le disque."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from scripts.runtime_model_specs import (
    DEFAULT_MLFLOW_TRACKING_URI,
    DEFAULT_MODELS_DIR,
    HISTORICAL_RUNTIME_MODEL_SPEC,
    RuntimeModelSpec,
    SIMULATION_RUNTIME_MODEL_SPEC,
)


REQUIRED_RUNTIME_METADATA_FIELDS = {
    "runtime_model_role",
    "registered_model_name",
    "registered_model_version",
    "registered_model_run_id",
    "model_uri",
    "tracking_uri",
    "exported_at_utc",
    "artifact_path",
    "metadata_path",
}


def parse_args() -> argparse.Namespace:
    """Construit l'interface CLI du script de promotion runtime."""
    parser = argparse.ArgumentParser(
        description=(
            "Promote the two MLflow registered models used by the FastAPI runtime "
            "and export them to artifacts/models/."
        )
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_MLFLOW_TRACKING_URI,
        help="Tracking URI MLflow. Par defaut: base SQLite locale du projet.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Dossier cible pour les artefacts runtime exportes.",
    )
    parser.add_argument(
        "--historical-registered-model",
        default=None,
        help="Nom du registered model historique a exporter.",
    )
    parser.add_argument(
        "--historical-version",
        default=None,
        help="Version MLflow du modele historique a exporter.",
    )
    parser.add_argument(
        "--simulation-registered-model",
        default=None,
        help="Nom du registered model local/simulation a exporter.",
    )
    parser.add_argument(
        "--simulation-version",
        default=None,
        help="Version MLflow du modele local/simulation a exporter.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime le resume de promotion au format JSON.",
    )
    return parser.parse_args()


def project_relative_path(path: Path) -> str:
    """Retourne un chemin relatif au projet si possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return str(resolved)


def json_ready(value: Any) -> Any:
    """Convertit recursivement les types Python en valeurs serialisables JSON."""
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


def normalize_registered_model_names(models: list[Any]) -> list[str]:
    """Extrait et trie les noms de registered models MLflow."""
    return sorted(str(model.name) for model in models)


def with_models_dir(spec: RuntimeModelSpec, models_dir: Path) -> RuntimeModelSpec:
    """Construit une specification identique avec un dossier cible surcharge."""
    return replace(
        spec,
        output_model_path=models_dir / spec.output_model_path.name,
        output_metadata_path=models_dir / spec.output_metadata_path.name,
    )


def resolve_registered_model_name_for_role(
    *,
    role_spec: RuntimeModelSpec,
    available_names: list[str],
    requested_name: str | None = None,
) -> str:
    """Selectionne le registered model a promouvoir pour un role donne."""
    if requested_name is not None:
        if requested_name not in available_names:
            available = ", ".join(available_names) if available_names else "none"
            raise ValueError(
                f"Requested registered model {requested_name!r} for role "
                f"{role_spec.role!r} was not found. Available registered models: {available}."
            )
        return requested_name

    matching_names = [name for name in available_names if name == role_spec.registered_model_name]
    if not matching_names:
        raise ValueError(
            f"No MLflow registered model found for role {role_spec.role!r}. "
            f"Expected one of: {role_spec.registered_model_name}."
        )
    if len(matching_names) > 1:
        raise ValueError(
            f"Multiple candidate registered models found for role {role_spec.role!r}. "
            f"Please pass --{role_spec.role}-registered-model."
        )
    return matching_names[0]


def _version_sort_key(version: Any) -> tuple[int, str]:
    """Produit une cle de tri robuste pour les versions MLflow."""
    raw_value = str(getattr(version, "version", version))
    return (int(raw_value), raw_value) if raw_value.isdigit() else (-1, raw_value)


def resolve_model_version_for_role(
    versions: list[Any],
    *,
    role_spec: RuntimeModelSpec,
    registered_model_name: str,
    requested_version: str | None = None,
) -> Any:
    """Selectionne strictement la version a exporter pour un role runtime."""
    if requested_version is not None:
        for version in versions:
            if str(version.version) == str(requested_version):
                return version
        available = ", ".join(str(version.version) for version in versions) if versions else "none"
        raise ValueError(
            f"Requested version {requested_version!r} for role {role_spec.role!r} and "
            f"registered model {registered_model_name!r} does not exist. "
            f"Available versions: {available}."
        )

    if not versions:
        raise ValueError(
            f"Registered model exists but no version could be resolved for role "
            f"{role_spec.role!r}."
        )

    if len(versions) > 1:
        available = ", ".join(str(version.version) for version in sorted(versions, key=_version_sort_key))
        raise ValueError(
            f"Multiple versions are available for role {role_spec.role!r} and "
            f"registered model {registered_model_name!r}. "
            f"Please pass --{role_spec.role}-version. Available versions: {available}."
        )

    return versions[0]


def read_json_if_exists(path: Path) -> dict[str, Any]:
    """Charge un JSON local si present, sinon retourne un dictionnaire vide."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def export_registered_model(
    *,
    tracking_uri: str,
    registered_model_name: str,
    model_version: Any,
    model_output_path: Path,
) -> None:
    """Charge un modele depuis MLflow et l'exporte en `joblib` local."""
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{registered_model_name}/{model_version.version}"
    estimator = mlflow.sklearn.load_model(model_uri)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, model_output_path)


def validate_exported_artifact(model_output_path: Path) -> None:
    """Verifie que l'artefact joblib exporte existe et est rechargeable."""
    if not model_output_path.exists():
        raise RuntimeError(f"Exported artifact is missing or cannot be loaded: {model_output_path}")
    try:
        joblib.load(model_output_path)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise RuntimeError(
            f"Exported artifact is missing or cannot be loaded: {model_output_path}"
        ) from exc


def build_export_metadata(
    *,
    existing_metadata: dict[str, Any],
    role_spec: RuntimeModelSpec,
    registered_model_name: str,
    model_version: Any,
    tracking_uri: str,
    model_output_path: Path,
    metadata_output_path: Path,
    source_run: Any | None,
) -> dict[str, Any]:
    """Construit les metadonnees de tracabilite de l'export runtime."""
    metadata = dict(existing_metadata)
    metadata.update(
        {
            "runtime_model_role": role_spec.role,
            "registered_model_name": registered_model_name,
            "registered_model_version": str(model_version.version),
            "registered_model_stage": str(getattr(model_version, "current_stage", "None") or "None"),
            "registered_model_source": str(getattr(model_version, "source", "")),
            "registered_model_run_id": getattr(model_version, "run_id", None),
            "model_uri": f"models:/{registered_model_name}/{model_version.version}",
            "tracking_uri": tracking_uri,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_path": project_relative_path(model_output_path),
            "metadata_path": project_relative_path(metadata_output_path),
            "output_path": project_relative_path(model_output_path),
            "output_metadata_path": project_relative_path(metadata_output_path),
            "role": role_spec.role,
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


def validate_runtime_metadata(metadata: dict[str, Any], *, role_spec: RuntimeModelSpec) -> None:
    """Verifie que les metadonnees exportees sont coherentes pour le runtime."""
    missing_fields = sorted(
        field_name for field_name in REQUIRED_RUNTIME_METADATA_FIELDS if not metadata.get(field_name)
    )
    if missing_fields:
        raise RuntimeError(
            f"Metadata validation failed for role {role_spec.role!r}. "
            f"Missing fields: {', '.join(missing_fields)}."
        )

    if metadata.get("runtime_model_role") != role_spec.role:
        raise RuntimeError(
            f"Metadata validation failed for role {role_spec.role!r}. "
            f"Unexpected runtime_model_role={metadata.get('runtime_model_role')!r}."
        )


def promote_single_registered_model(
    *,
    client: MlflowClient,
    tracking_uri: str,
    role_spec: RuntimeModelSpec,
    available_names: list[str],
    requested_name: str | None = None,
    requested_version: str | None = None,
) -> dict[str, Any]:
    """Promouvoit un registered model runtime unique depuis MLflow."""
    registered_model_name = resolve_registered_model_name_for_role(
        role_spec=role_spec,
        available_names=available_names,
        requested_name=requested_name,
    )
    versions = list(client.search_model_versions(f"name = '{registered_model_name}'"))
    selected_version = resolve_model_version_for_role(
        versions,
        role_spec=role_spec,
        registered_model_name=registered_model_name,
        requested_version=requested_version,
    )
    source_run = client.get_run(selected_version.run_id) if getattr(selected_version, "run_id", None) else None

    try:
        export_registered_model(
            tracking_uri=tracking_uri,
            registered_model_name=registered_model_name,
            model_version=selected_version,
            model_output_path=role_spec.output_model_path,
        )
    except Exception as exc:  # pragma: no cover - defensive branch
        raise RuntimeError(
            f"Export failed for role {role_spec.role!r} and model {registered_model_name!r}."
        ) from exc

    validate_exported_artifact(role_spec.output_model_path)
    existing_metadata = read_json_if_exists(role_spec.output_metadata_path)
    export_metadata = build_export_metadata(
        existing_metadata=existing_metadata,
        role_spec=role_spec,
        registered_model_name=registered_model_name,
        model_version=selected_version,
        tracking_uri=tracking_uri,
        model_output_path=role_spec.output_model_path,
        metadata_output_path=role_spec.output_metadata_path,
        source_run=source_run,
    )
    role_spec.output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    role_spec.output_metadata_path.write_text(
        json.dumps(json_ready(export_metadata), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    validate_runtime_metadata(export_metadata, role_spec=role_spec)

    return {
        "role": role_spec.role,
        "registered_model_name": registered_model_name,
        "registered_model_version": str(selected_version.version),
        "registered_model_run_id": getattr(selected_version, "run_id", None),
        "model_uri": f"models:/{registered_model_name}/{selected_version.version}",
        "artifact_path": project_relative_path(role_spec.output_model_path),
        "metadata_path": project_relative_path(role_spec.output_metadata_path),
    }


def promote_registered_models(
    *,
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    historical_registered_model: str | None = None,
    historical_version: str | None = None,
    simulation_registered_model: str | None = None,
    simulation_version: str | None = None,
) -> dict[str, Any]:
    """Promouvoit les deux registered models runtime depuis MLflow."""
    resolved_models_dir = Path(models_dir).resolve()
    historical_spec = with_models_dir(HISTORICAL_RUNTIME_MODEL_SPEC, resolved_models_dir)
    simulation_spec = with_models_dir(SIMULATION_RUNTIME_MODEL_SPEC, resolved_models_dir)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    registered_models = list(client.search_registered_models())
    available_names = normalize_registered_model_names(registered_models)

    historical_summary = promote_single_registered_model(
        client=client,
        tracking_uri=tracking_uri,
        role_spec=historical_spec,
        available_names=available_names,
        requested_name=historical_registered_model,
        requested_version=historical_version,
    )
    simulation_summary = promote_single_registered_model(
        client=client,
        tracking_uri=tracking_uri,
        role_spec=simulation_spec,
        available_names=available_names,
        requested_name=simulation_registered_model,
        requested_version=simulation_version,
    )
    return {
        "tracking_uri": tracking_uri,
        "models_dir": project_relative_path(resolved_models_dir),
        "historical": historical_summary,
        "simulation": simulation_summary,
    }


def main() -> None:
    """Execute la promotion runtime depuis la CLI."""
    args = parse_args()
    summary = promote_registered_models(
        tracking_uri=str(args.tracking_uri),
        models_dir=args.models_dir,
        historical_registered_model=args.historical_registered_model,
        historical_version=args.historical_version,
        simulation_registered_model=args.simulation_registered_model,
        simulation_version=args.simulation_version,
    )
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return

    for role_name in ("historical", "simulation"):
        role_summary = summary[role_name]
        print(
            "[promotion] "
            f"role={role_summary['role']} "
            f"registered_model={role_summary['registered_model_name']} "
            f"version={role_summary['registered_model_version']} "
            f"artifact={role_summary['artifact_path']}"
        )


if __name__ == "__main__":
    main()
