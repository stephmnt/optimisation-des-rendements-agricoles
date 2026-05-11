"""Helpers MLflow pour journaliser des modeles et des predictions evaluees."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient


SKLEARN_PICKLE_WARNING_PREFIX = (
    "Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution"
)


class _SuppressSklearnPickleWarning(logging.Filter):
    """Filtre le warning MLflow repete sur la serialisation pickle/cloudpickle."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Retourne `False` uniquement pour le warning verbeux attendu."""
        return SKLEARN_PICKLE_WARNING_PREFIX not in record.getMessage()


def configure_mlflow_sklearn_logging() -> None:
    """Rend les logs MLflow sklearn lisibles pendant les entrainements longs."""
    logger = logging.getLogger("mlflow.sklearn")
    if not any(isinstance(item, _SuppressSklearnPickleWarning) for item in logger.filters):
        logger.addFilter(_SuppressSklearnPickleWarning())


configure_mlflow_sklearn_logging()


def sanitize_logged_model_name(raw_name: str) -> str:
    """Construit un nom de modele MLflow stable a partir d'un identifiant brut.

    Args:
        raw_name: Nom de run ou de modele d'origine.

    Returns:
        str: Nom nettoye et compatible avec MLflow.
    """
    candidate = str(raw_name).strip()
    if "::" in candidate:
        candidate = candidate.rsplit("::", 1)[-1]
    if "__" in candidate:
        candidate = candidate.rsplit("__", 1)[-1]

    cleaned = []
    for char in candidate:
        if char.isalnum() or char in {"_", "-", "."}:
            cleaned.append(char)
        else:
            cleaned.append("_")

    normalized = "".join(cleaned).strip("._-")
    return normalized or "model"


def log_named_sklearn_model(estimator: Any, *, model_name: str) -> str:
    """Journalise un estimateur scikit-learn sous un nom MLflow stable.

    Args:
        estimator: Estimateur scikit-learn a enregistrer.
        model_name: Nom descriptif du modele dans le run courant.

    Returns:
        str: Nom effectivement utilise dans MLflow.
    """
    logged_model_name = sanitize_logged_model_name(model_name)
    mlflow.sklearn.log_model(estimator, name=logged_model_name)
    return logged_model_name


def _registered_model_version_sort_key(version: Any) -> tuple[int, str]:
    """Produit une cle de tri robuste pour les versions du registry MLflow."""
    raw_version = str(getattr(version, "version", version))
    return (int(raw_version), raw_version) if raw_version.isdigit() else (-1, raw_version)


def resolve_registered_model_version_for_run(
    *,
    registered_model_name: str,
    run_id: str,
    tracking_uri: str | None = None,
) -> Any:
    """Recupere la version du registry associee a un run MLflow donne.

    Args:
        registered_model_name: Nom du registered model a inspecter.
        run_id: Identifiant du run source.
        tracking_uri: Tracking URI MLflow optionnel.

    Returns:
        Any: Objet version retourne par le client MLflow.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = [
        version
        for version in client.search_model_versions(f"name = '{registered_model_name}'")
        if str(getattr(version, "run_id", "")) == str(run_id)
    ]
    if not versions:
        raise RuntimeError(
            "Registered model version could not be resolved for "
            f"model={registered_model_name!r} and run_id={run_id!r}."
        )
    return max(versions, key=_registered_model_version_sort_key)


def log_and_register_sklearn_model(
    estimator: Any,
    *,
    artifact_name: str,
    registered_model_name: str,
    model_metadata: dict[str, Any] | None = None,
    await_registration_for: int = 300,
) -> dict[str, str]:
    """Journalise un estimateur et l'enregistre comme registered model MLflow.

    Args:
        estimator: Estimateur scikit-learn a enregistrer.
        artifact_name: Nom de l'artefact de run.
        registered_model_name: Nom du registered model cible.
        model_metadata: Metadonnees MLflow optionnelles.
        await_registration_for: Duree d'attente maximale de l'enregistrement.

    Returns:
        dict[str, str]: Contexte de registry resolu apres l'enregistrement.
    """
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("An active MLflow run is required before registering a model.")

    logged_model_name = sanitize_logged_model_name(artifact_name)
    model_info = mlflow.sklearn.log_model(
        estimator,
        name=logged_model_name,
        registered_model_name=registered_model_name,
        metadata=model_metadata,
        await_registration_for=await_registration_for,
    )
    resolved_version = resolve_registered_model_version_for_run(
        registered_model_name=registered_model_name,
        run_id=active_run.info.run_id,
        tracking_uri=mlflow.get_tracking_uri(),
    )
    return {
        "logged_model_name": logged_model_name,
        "registered_model_name": registered_model_name,
        "registered_model_version": str(resolved_version.version),
        "model_uri": f"models:/{registered_model_name}/{resolved_version.version}",
        "run_id": active_run.info.run_id,
        "logged_model_uri": str(getattr(model_info, "model_uri", "")),
    }


class EvaluationPredictionLookupModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc model exposing precomputed evaluation predictions by key lookup.

    This is mainly intended for experiment-tracking consistency when a run does not
    produce a single global fitted estimator, as in local time-series evaluations.
    """

    def load_context(self, context) -> None:
        """Charge les artefacts CSV et JSON exposes au modele pyfunc."""
        predictions_path = Path(context.artifacts["predictions"])
        self.predictions_df = pd.read_csv(predictions_path)

        spec_path = context.artifacts.get("specification")
        if spec_path:
            self.specification = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        else:
            self.specification = {}

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
        """Restitue les predictions pre-calculees correspondant aux cles demandees.

        Args:
            context: Contexte pyfunc MLflow, inutilise ici.
            model_input: Table contenant au minimum `area`, `crop` et `year`.

        Returns:
            pd.DataFrame: Entree enrichie avec les predictions journalisees.
        """
        del context
        input_df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)

        required_columns = ["area", "crop", "year"]
        missing_columns = [column for column in required_columns if column not in input_df.columns]
        if missing_columns:
            raise ValueError(
                "EvaluationPredictionLookupModel requires columns "
                f"{required_columns}, missing {missing_columns}."
            )

        lookup_columns = required_columns.copy()
        predictions_df = self.predictions_df.copy()

        if "split" in input_df.columns and "split" in predictions_df.columns:
            lookup_columns.append("split")
        elif "split" in predictions_df.columns:
            predictions_df = predictions_df.sort_values(lookup_columns + ["split"]).drop_duplicates(
                subset=lookup_columns,
                keep="last",
            )

        predictions_df["year"] = predictions_df["year"].astype(input_df["year"].dtype, copy=False)
        merged = input_df.merge(
            predictions_df,
            on=lookup_columns,
            how="left",
            suffixes=("", "_logged"),
        )
        return merged


def log_evaluation_lookup_model(
    *,
    model_name: str,
    predictions_path: Path | str,
    specification_path: Path | str,
    model_metadata: dict[str, Any] | None = None,
) -> str:
    """Journalise un pyfunc MLflow adosse a des predictions pre-calculees.

    Args:
        model_name: Nom du modele a creer dans MLflow.
        predictions_path: CSV des predictions evaluees.
        specification_path: JSON de specification associe au modele.
        model_metadata: Metadonnees complementaires a propager.

    Returns:
        str: Nom effectivement utilise dans MLflow.
    """
    logged_model_name = sanitize_logged_model_name(model_name)
    predictions_path = Path(predictions_path).resolve()
    specification_path = Path(specification_path).resolve()

    input_example_df = pd.read_csv(predictions_path, nrows=1)
    input_example_columns = [column for column in ["area", "crop", "year", "split"] if column in input_example_df.columns]
    input_example = input_example_df[input_example_columns] if input_example_columns else None

    metadata = {
        "model_kind": "evaluation_prediction_lookup",
        "source_predictions_file": predictions_path.name,
        "source_specification_file": specification_path.name,
    }
    if model_metadata:
        metadata.update(model_metadata)

    mlflow.pyfunc.log_model(
        name=logged_model_name,
        python_model=EvaluationPredictionLookupModel(),
        artifacts={
            "predictions": str(predictions_path),
            "specification": str(specification_path),
        },
        input_example=input_example,
        metadata=metadata,
    )
    return logged_model_name
