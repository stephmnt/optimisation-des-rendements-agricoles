from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd


def sanitize_logged_model_name(raw_name: str) -> str:
    """Return a stable MLflow logged-model name derived from a run/model name."""
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
    """Log a scikit-learn model under a meaningful MLflow logged-model name."""
    logged_model_name = sanitize_logged_model_name(model_name)
    mlflow.sklearn.log_model(estimator, name=logged_model_name)
    return logged_model_name


class EvaluationPredictionLookupModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc model exposing precomputed evaluation predictions by key lookup.

    This is mainly intended for experiment-tracking consistency when a run does not
    produce a single global fitted estimator, as in local time-series evaluations.
    """

    def load_context(self, context) -> None:
        predictions_path = Path(context.artifacts["predictions"])
        self.predictions_df = pd.read_csv(predictions_path)

        spec_path = context.artifacts.get("specification")
        if spec_path:
            self.specification = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        else:
            self.specification = {}

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
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
    """Log a lightweight pyfunc model backed by precomputed evaluation predictions."""
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
