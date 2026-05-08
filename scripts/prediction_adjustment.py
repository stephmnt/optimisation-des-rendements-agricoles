"""Assemble la logique metier finale de prediction et de recommandation.

Le module combine un modele historique `P1` et un modele local `P2/P3` pour
produire un rendement ajuste, une explication interpretable et un classement de
cultures candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from scripts.simulation_dataset import load_normalized_simulation_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_WIDE_DATASET_PATH = PROJECT_ROOT / "artifacts/experiments/experience_1/dataset_consolide_historique_colonnes.csv"
HISTORICAL_MODEL_PATH = PROJECT_ROOT / "artifacts/models/p1_historical_pipeline.joblib"
HISTORICAL_METADATA_PATH = PROJECT_ROOT / "artifacts/models/p1_historical_metadata.json"
SIMULATION_DATASET_PATH = PROJECT_ROOT / "data/simulation/crop_yield.csv"
SIMULATION_MODEL_PATH = PROJECT_ROOT / "artifacts/models/p23_simulation_pipeline.joblib"
SIMULATION_METADATA_PATH = PROJECT_ROOT / "artifacts/models/p23_simulation_metadata.json"

SEED = 42
SIMULATION_SAMPLE_SIZE = 200_000
SIMULATION_FEATURE_COLUMNS = [
    "region",
    "soil_type",
    "rainfall_mm",
    "temperature_celsius",
    "fertilizer_used",
    "irrigation_used",
    "weather_condition",
    "days_to_harvest",
]


@dataclass
class LoadedModel:
    """Couple simple contenant un pipeline charge et ses metadonnees."""

    pipeline: Pipeline
    metadata: dict[str, Any]


def _resolve_path(path: str | Path) -> Path:
    """Resout un chemin absolu ou relatif par rapport au depot."""
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return PROJECT_ROOT / raw_path


def _ensure_parent_dir(path: Path) -> None:
    """Cree le dossier parent d'un artefact si necessaire."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_ready(value: Any) -> Any:
    """Convertit les types numpy et pandas en types JSON-compatibles."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _safe_float(value: Any) -> float:
    """Convertit de maniere defensive un scalaire potentiel en `float`."""
    return float(np.asarray(value).reshape(-1)[0])


def _value_for_display(value: Any) -> Any:
    """Normalise une valeur pour l'affichage ou la serialisation."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def make_dense_onehot_encoder() -> OneHotEncoder:
    """Construit un `OneHotEncoder` dense compatible avec plusieurs versions sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    """Construit le preprocesseur commun aux modeles tabulaires.

    Args:
        feature_frame: Table de caracteristiques de reference.

    Returns:
        ColumnTransformer: Pipeline de pretraitement numerique et categoriel.
    """
    numeric_features = feature_frame.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [col for col in feature_frame.columns if col not in numeric_features]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", make_dense_onehot_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def compute_regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcule les metriques de regression standard du projet."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true_array, y_pred_array)))
    mae = float(mean_absolute_error(y_true_array, y_pred_array))
    r2 = float(r2_score(y_true_array, y_pred_array)) if len(y_true_array) >= 2 else np.nan
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def normalize_label(value: Any) -> str:
    """Normalise une etiquette textuelle issue des datasets ou de l'API."""
    return str(value).strip()


def load_historical_wide_dataset(dataset_path: str | Path = HISTORICAL_WIDE_DATASET_PATH) -> pd.DataFrame:
    """Charge le dataset historique consolide utilise par la brique P1."""
    path = _resolve_path(dataset_path)
    historical_df = pd.read_csv(path)
    historical_df["area"] = historical_df["area"].map(normalize_label)
    historical_df["crop"] = historical_df["crop"].map(normalize_label)
    return historical_df


def load_historical_model(
    model_path: str | Path = HISTORICAL_MODEL_PATH,
    metadata_path: str | Path = HISTORICAL_METADATA_PATH,
) -> LoadedModel:
    """Charge le pipeline historique et ses metadonnees."""
    resolved_model_path = _resolve_path(model_path)
    resolved_metadata_path = _resolve_path(metadata_path)

    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Historical model artifact not found: {resolved_model_path}")
    if not resolved_metadata_path.exists():
        raise FileNotFoundError(f"Historical metadata artifact not found: {resolved_metadata_path}")

    pipeline = joblib.load(resolved_model_path)
    metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
    return LoadedModel(pipeline=pipeline, metadata=metadata)


def load_and_prepare_simulation_dataset(
    simulation_path: str | Path = SIMULATION_DATASET_PATH,
) -> pd.DataFrame:
    """Charge et normalise le dataset de simulation locale."""
    return load_normalized_simulation_dataset(
        _resolve_path(simulation_path),
        boolean_dtype="bool",
    )


def _fit_simulation_pipeline(
    simulation_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    sample_size: int = SIMULATION_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Entraine le modele lineaire local utilise pour P2 et P3."""
    selected_features = feature_columns or SIMULATION_FEATURE_COLUMNS
    sampled_df = simulation_df.sample(n=min(sample_size, len(simulation_df)), random_state=SEED).copy()

    X_all = sampled_df[selected_features].copy()
    y_all = sampled_df["yield_tons_per_hectare"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=SEED,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)

    train_metrics = compute_regression_metrics(y_train, pipeline.predict(X_train))
    test_metrics = compute_regression_metrics(y_test, pipeline.predict(X_test))

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_all)),
            ("regressor", LinearRegression()),
        ]
    )
    final_pipeline.fit(X_all, y_all)

    metadata = {
        "model_name": "linear_regression",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_source": str(_resolve_path(SIMULATION_DATASET_PATH).relative_to(PROJECT_ROOT)),
        "feature_columns": selected_features,
        "sample_size": int(len(sampled_df)),
        "metrics": {
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"],
        },
        "strategy": "2_models_3_predictions_combined",
        "role": "local_adjustment_model_for_P2_and_P3",
    }

    return {
        "pipeline": final_pipeline,
        "metadata": metadata,
        "sampled_df": sampled_df,
    }


def load_or_train_simulation_model(
    *,
    force_retrain: bool = False,
    save_artifact: bool = True,
    simulation_path: str | Path = SIMULATION_DATASET_PATH,
    model_path: str | Path = SIMULATION_MODEL_PATH,
    metadata_path: str | Path = SIMULATION_METADATA_PATH,
    sample_size: int = SIMULATION_SAMPLE_SIZE,
) -> tuple[LoadedModel, pd.DataFrame]:
    """Charge ou regenere le modele local de simulation.

    Args:
        force_retrain: Force le reentrainement meme si les artefacts existent.
        save_artifact: Ecrit les artefacts sur disque si `True`.
        simulation_path: Source tabulaire du modele local.
        model_path: Chemin cible du pipeline serialize.
        metadata_path: Chemin cible des metadonnees JSON.
        sample_size: Taille maximale de l'echantillon d'entrainement.

    Returns:
        tuple[LoadedModel, pd.DataFrame]: Modele local et dataset normalise.
    """
    resolved_model_path = _resolve_path(model_path)
    resolved_metadata_path = _resolve_path(metadata_path)
    simulation_df = load_and_prepare_simulation_dataset(simulation_path)

    if not force_retrain and resolved_model_path.exists() and resolved_metadata_path.exists():
        loaded = LoadedModel(
            pipeline=joblib.load(resolved_model_path),
            metadata=json.loads(resolved_metadata_path.read_text(encoding="utf-8")),
        )
        return loaded, simulation_df

    trained = _fit_simulation_pipeline(simulation_df, sample_size=sample_size)
    loaded = LoadedModel(pipeline=trained["pipeline"], metadata=trained["metadata"])

    if save_artifact:
        _ensure_parent_dir(resolved_model_path)
        joblib.dump(loaded.pipeline, resolved_model_path)
        resolved_metadata_path.write_text(
            json.dumps(_json_ready(loaded.metadata), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    return loaded, simulation_df


def infer_target_year_from_metadata_or_dataset(
    historical_metadata: dict[str, Any],
    historical_df: pd.DataFrame,
) -> int:
    """Determine l'annee cible du modele historique."""
    target_year = historical_metadata.get("target_year")
    if target_year is not None:
        return int(target_year)

    target_columns = [col for col in historical_df.columns if col.startswith("target_yield_t_ha_")]
    available_years = sorted(
        int(col.rsplit("_", 1)[1])
        for col in target_columns
        if historical_df[col].notna().any()
    )
    if not available_years:
        raise ValueError("No usable historical target year found.")
    return int(max(available_years))


def latest_available_from_row(row: pd.Series, prefix: str, years: list[int]) -> tuple[float, int | None]:
    """Recupere la derniere valeur non nulle disponible pour une serie annuelle."""
    for year in sorted(years, reverse=True):
        value = row.get(f"{prefix}_{year}", np.nan)
        if pd.notna(value):
            return float(value), year
    return np.nan, None


def build_historical_reference_frame(
    historical_df: pd.DataFrame,
    *,
    target_year: int,
) -> pd.DataFrame:
    """Construit les reperes climatiques utilises comme reference locale."""
    feature_years = [year for year in range(target_year) if year >= 0]
    rainfall_years = [
        year for year in feature_years if f"average_rain_fall_mm_per_year_{year}" in historical_df.columns
    ]
    temperature_years = [year for year in feature_years if f"avg_temp_{year}" in historical_df.columns]

    reference_df = historical_df[["area", "crop"]].copy()
    reference_df[["reference_rainfall_mm", "reference_rainfall_year"]] = historical_df.apply(
        lambda row: pd.Series(
            latest_available_from_row(row, "average_rain_fall_mm_per_year", rainfall_years)
        ),
        axis=1,
    )
    reference_df[["reference_temperature_celsius", "reference_temperature_year"]] = historical_df.apply(
        lambda row: pd.Series(latest_available_from_row(row, "avg_temp", temperature_years)),
        axis=1,
    )

    crop_fallback_df = reference_df.groupby("crop").agg(
        crop_reference_rainfall_mm=("reference_rainfall_mm", "median"),
        crop_reference_temperature_celsius=("reference_temperature_celsius", "median"),
    ).reset_index()

    return reference_df.merge(crop_fallback_df, on="crop", how="left")


def build_simulation_global_reference(simulation_df: pd.DataFrame) -> dict[str, Any]:
    """Construit un profil global median/modal pour le modele de simulation."""
    return {
        "region": simulation_df["region"].mode().iloc[0],
        "soil_type": simulation_df["soil_type"].mode().iloc[0],
        "rainfall_mm": float(simulation_df["rainfall_mm"].median()),
        "temperature_celsius": float(simulation_df["temperature_celsius"].median()),
        "fertilizer_used": bool(simulation_df["fertilizer_used"].mode().iloc[0]),
        "irrigation_used": bool(simulation_df["irrigation_used"].mode().iloc[0]),
        "weather_condition": simulation_df["weather_condition"].mode().iloc[0],
        "days_to_harvest": float(simulation_df["days_to_harvest"].median()),
    }


def build_reference_profile_from_row(
    row: pd.Series,
    *,
    simulation_global_reference: dict[str, Any],
    selected_simulation_features: list[str],
    overrides: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Construit le profil de reference local pour un couple pays/culture.

    Returns:
        tuple[pd.DataFrame, dict[str, str]]: Profil pret pour l'inference et
        informations de provenance des references pluie/temperature.
    """
    rainfall_source = (
        "row_latest_history"
        if pd.notna(row["reference_rainfall_mm"])
        else "crop_median"
        if pd.notna(row["crop_reference_rainfall_mm"])
        else "simulation_global_default"
    )
    temperature_source = (
        "row_latest_history"
        if pd.notna(row["reference_temperature_celsius"])
        else "crop_median"
        if pd.notna(row["crop_reference_temperature_celsius"])
        else "simulation_global_default"
    )

    profile = dict(simulation_global_reference)
    profile["rainfall_mm"] = (
        float(row["reference_rainfall_mm"])
        if pd.notna(row["reference_rainfall_mm"])
        else float(row["crop_reference_rainfall_mm"])
        if pd.notna(row["crop_reference_rainfall_mm"])
        else float(simulation_global_reference["rainfall_mm"])
    )
    profile["temperature_celsius"] = (
        float(row["reference_temperature_celsius"])
        if pd.notna(row["reference_temperature_celsius"])
        else float(row["crop_reference_temperature_celsius"])
        if pd.notna(row["crop_reference_temperature_celsius"])
        else float(simulation_global_reference["temperature_celsius"])
    )

    if overrides:
        profile.update(overrides)

    profile_df = pd.DataFrame([profile])[selected_simulation_features]
    return profile_df, {
        "rainfall_reference_source": rainfall_source,
        "temperature_reference_source": temperature_source,
    }


class AdjustedYieldService:
    """Service metier principal expose a l'API et a l'interface Streamlit."""

    def __init__(
        self,
        *,
        historical_dataset_path: str | Path = HISTORICAL_WIDE_DATASET_PATH,
        historical_model_path: str | Path = HISTORICAL_MODEL_PATH,
        historical_metadata_path: str | Path = HISTORICAL_METADATA_PATH,
        simulation_dataset_path: str | Path = SIMULATION_DATASET_PATH,
        simulation_model_path: str | Path = SIMULATION_MODEL_PATH,
        simulation_metadata_path: str | Path = SIMULATION_METADATA_PATH,
        force_retrain_simulation: bool = False,
    ) -> None:
        """Initialise les modeles, datasets et catalogues utiles au runtime."""
        self.context = _load_prediction_context(
            historical_dataset_path=historical_dataset_path,
            historical_model_path=historical_model_path,
            historical_metadata_path=historical_metadata_path,
            simulation_dataset_path=simulation_dataset_path,
            simulation_model_path=simulation_model_path,
            simulation_metadata_path=simulation_metadata_path,
            force_retrain_simulation=force_retrain_simulation,
        )

        self.historical_model = self.context["historical_model"]
        self.historical_metadata = self.context["historical_metadata"]
        self.historical_df = self.context["historical_df"]
        self.simulation_model = self.context["simulation_model"]
        self.simulation_metadata = self.context["simulation_metadata"]
        self.simulation_df = self.context["simulation_df"]
        self.simulation_global_reference = self.context["simulation_global_reference"]
        self.strategy_df = self.context["strategy_df"]
        self.target_year = int(self.context["target_year"])
        self.selected_simulation_features = list(self.simulation_metadata["feature_columns"])
        self.available_areas = sorted(self.strategy_df["area"].dropna().unique().tolist())
        self.available_crops = sorted(self.strategy_df["crop"].dropna().unique().tolist())
        self.crops_by_area = {
            area: sorted(area_df["crop"].dropna().unique().tolist())
            for area, area_df in self.strategy_df.groupby("area")
        }
        self.simulation_options = {
            "regions": sorted(self.simulation_df["region"].dropna().unique().tolist()),
            "soil_types": sorted(self.simulation_df["soil_type"].dropna().unique().tolist()),
            "weather_conditions": sorted(self.simulation_df["weather_condition"].dropna().unique().tolist()),
        }
        self._historical_shap_state: dict[str, Any] | None = None

    def _sanitize_overrides(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Nettoie les surcharges de conditions avant utilisation."""
        if not overrides:
            return {}

        cleaned: dict[str, Any] = {}
        for key, value in overrides.items():
            if value is None:
                continue
            if key in {"region", "soil_type", "weather_condition"}:
                cleaned[key] = normalize_label(value)
                continue
            if key in {"fertilizer_used", "irrigation_used"}:
                cleaned[key] = bool(value)
                continue
            cleaned[key] = float(value) if key in {"rainfall_mm", "temperature_celsius", "days_to_harvest"} else value
        return cleaned

    def _get_row(self, area: str, crop: str) -> pd.Series:
        """Recupere la ligne historique unique correspondant au couple demande."""
        return _get_area_crop_row(self.strategy_df, area=area, crop=crop)

    def _predict_p1(self, row: pd.Series) -> float:
        """Calcule la prediction historique P1 a partir d'une ligne consolidee."""
        return _predict_p1_from_row(row, self.historical_model, self.historical_metadata)

    def _map_transformed_feature_to_raw_feature(
        self,
        transformed_feature_name: str,
        raw_feature_names: list[str],
    ) -> str:
        """Ramene un nom de feature transformee vers sa variable brute d'origine."""
        candidates = [transformed_feature_name]
        if "__" in transformed_feature_name:
            parts = transformed_feature_name.split("__")
            candidates.extend("__".join(parts[index:]) for index in range(1, len(parts)))

        for candidate in candidates:
            for raw_feature in sorted(raw_feature_names, key=len, reverse=True):
                if candidate == raw_feature or candidate.startswith(f"{raw_feature}_"):
                    return raw_feature
        return transformed_feature_name

    def _aggregate_transformed_contributions(
        self,
        *,
        transformed_feature_names: list[str],
        contribution_values: np.ndarray,
        raw_feature_names: list[str],
    ) -> dict[str, float]:
        """Agrege les contributions encodees par modalite au niveau variable brute."""
        aggregated: dict[str, float] = {}
        for transformed_feature_name, contribution_value in zip(transformed_feature_names, contribution_values):
            raw_feature_name = self._map_transformed_feature_to_raw_feature(
                transformed_feature_name,
                raw_feature_names,
            )
            aggregated[raw_feature_name] = aggregated.get(raw_feature_name, 0.0) + float(contribution_value)
        return aggregated

    def _ensure_historical_shap_state(self) -> dict[str, Any]:
        """Initialise a la demande l'etat SHAP du modele historique."""
        if self._historical_shap_state is not None:
            return self._historical_shap_state

        try:
            import shap  # type: ignore
        except ModuleNotFoundError:
            self._historical_shap_state = {
                "available": False,
                "status": "missing_dependency",
                "message": "Le package shap n'est pas installe dans l'environnement courant.",
            }
            return self._historical_shap_state

        preprocessor = self.historical_model.named_steps["preprocessor"]
        regressor = self.historical_model.named_steps["regressor"]
        raw_feature_names = list(self.historical_metadata["feature_columns"])
        background_df = self.historical_df[raw_feature_names].sample(
            n=min(200, len(self.historical_df)),
            random_state=SEED,
        )
        background_matrix = preprocessor.transform(background_df)
        transformed_feature_names = list(preprocessor.get_feature_names_out())

        try:
            explainer = shap.Explainer(
                regressor,
                background_matrix,
                feature_names=transformed_feature_names,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._historical_shap_state = {
                "available": False,
                "status": "explainer_initialization_failed",
                "message": f"Impossible d'initialiser SHAP sur le modele historique : {exc}",
            }
            return self._historical_shap_state

        self._historical_shap_state = {
            "available": True,
            "status": "ok",
            "message": None,
            "explainer": explainer,
            "preprocessor": preprocessor,
            "transformed_feature_names": transformed_feature_names,
            "raw_feature_names": raw_feature_names,
        }
        return self._historical_shap_state

    def _explain_historical_prediction(
        self,
        *,
        row: pd.Series,
        p1_prediction: float,
        top_n: int = 10,
    ) -> dict[str, Any]:
        """Produit l'explication SHAP agregee de la prediction P1."""
        shap_state = self._ensure_historical_shap_state()
        if not shap_state["available"]:
            return {
                "available": False,
                "status": shap_state["status"],
                "message": shap_state["message"],
                "model_prediction": p1_prediction,
                "base_value": None,
                "prediction_from_shap": None,
                "top_contributions": [],
            }

        raw_feature_names = shap_state["raw_feature_names"]
        feature_frame = pd.DataFrame([row[raw_feature_names].to_dict()])[raw_feature_names]
        transformed_row = shap_state["preprocessor"].transform(feature_frame)
        shap_values = shap_state["explainer"](transformed_row)
        contribution_vector = np.asarray(shap_values.values)[0]
        aggregated_contributions = self._aggregate_transformed_contributions(
            transformed_feature_names=shap_state["transformed_feature_names"],
            contribution_values=contribution_vector,
            raw_feature_names=raw_feature_names,
        )

        top_contributions = [
            {
                "feature": raw_feature_name,
                "raw_value": _value_for_display(row.get(raw_feature_name)),
                "contribution": float(contribution),
                "abs_contribution": abs(float(contribution)),
            }
            for raw_feature_name, contribution in sorted(
                aggregated_contributions.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:top_n]
        ]

        base_value = _safe_float(shap_values.base_values)
        prediction_from_shap = float(base_value + contribution_vector.sum())

        return {
            "available": True,
            "status": "ok",
            "message": None,
            "model_prediction": p1_prediction,
            "base_value": base_value,
            "prediction_from_shap": prediction_from_shap,
            "top_contributions": top_contributions,
        }

    def _explain_local_adjustment(
        self,
        *,
        reference_profile: pd.DataFrame,
        user_profile: pd.DataFrame,
        p2_prediction: float,
        p3_prediction: float,
        top_n: int = 10,
    ) -> dict[str, Any]:
        """Decompose lineairement l'ajustement local applique entre P2 et P3."""
        preprocessor = self.simulation_model.named_steps["preprocessor"]
        regressor = self.simulation_model.named_steps["regressor"]
        transformed_feature_names = list(preprocessor.get_feature_names_out())
        reference_vector = np.asarray(preprocessor.transform(reference_profile))[0]
        user_vector = np.asarray(preprocessor.transform(user_profile))[0]
        delta_vector = user_vector - reference_vector
        coefficient_vector = np.asarray(regressor.coef_).reshape(-1)
        contribution_vector = delta_vector * coefficient_vector
        aggregated_contributions = self._aggregate_transformed_contributions(
            transformed_feature_names=transformed_feature_names,
            contribution_values=contribution_vector,
            raw_feature_names=self.selected_simulation_features,
        )

        reference_row = reference_profile.iloc[0].to_dict()
        user_row = user_profile.iloc[0].to_dict()
        top_contributions = [
            {
                "feature": raw_feature_name,
                "reference_value": _value_for_display(reference_row.get(raw_feature_name)),
                "user_value": _value_for_display(user_row.get(raw_feature_name)),
                "contribution_delta": float(contribution),
                "abs_contribution_delta": abs(float(contribution)),
            }
            for raw_feature_name, contribution in sorted(
                aggregated_contributions.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:top_n]
        ]

        return {
            "method": "exact_linear_delta_decomposition",
            "reference_prediction": p2_prediction,
            "user_prediction": p3_prediction,
            "total_adjustment": float(p3_prediction - p2_prediction),
            "top_contributions": top_contributions,
        }

    def get_reference_profile(
        self,
        area: str,
        crop: str,
        *,
        reference_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retourne le profil de reference local pour un pays et une culture."""
        row = self._get_row(area, crop)
        normalized_reference_overrides = self._sanitize_overrides(reference_overrides)
        reference_profile, reference_sources = build_reference_profile_from_row(
            row,
            simulation_global_reference=self.simulation_global_reference,
            selected_simulation_features=self.selected_simulation_features,
            overrides=normalized_reference_overrides,
        )
        return {
            "country": normalize_label(area),
            "crop": normalize_label(crop),
            "reference_profile": reference_profile.iloc[0].to_dict(),
            **reference_sources,
        }

    def get_baseline(
        self,
        area: str,
        crop: str,
        *,
        reference_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retourne la prediction historique de base et son profil de reference."""
        row = self._get_row(area, crop)
        reference_payload = self.get_reference_profile(
            area,
            crop,
            reference_overrides=reference_overrides,
        )
        p1 = self._predict_p1(row)
        return {
            "country": normalize_label(area),
            "crop": normalize_label(crop),
            "target_year": self.target_year,
            "p1_historical_prediction": p1,
            "reference_profile": reference_payload["reference_profile"],
            "rainfall_reference_source": reference_payload["rainfall_reference_source"],
            "temperature_reference_source": reference_payload["temperature_reference_source"],
        }

    def predict_adjusted_yield(
        self,
        area: str,
        crop: str,
        user_conditions: dict[str, Any],
        *,
        reference_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Calcule le rendement final ajuste pour une culture donnee.

        Args:
            area: Pays ou zone retenue.
            crop: Culture cible.
            user_conditions: Conditions locales saisies par l'utilisateur.
            reference_overrides: Surcharges appliquees au profil de reference.

        Returns:
            dict[str, Any]: Detail complet des composantes P1, P2, P3 et des
            explications associees.
        """
        row = self._get_row(area, crop)
        normalized_reference_overrides = self._sanitize_overrides(reference_overrides)
        normalized_user_conditions = self._sanitize_overrides(user_conditions)
        merged_user_overrides = {
            **normalized_reference_overrides,
            **normalized_user_conditions,
        }

        reference_profile, reference_sources = build_reference_profile_from_row(
            row,
            simulation_global_reference=self.simulation_global_reference,
            selected_simulation_features=self.selected_simulation_features,
            overrides=normalized_reference_overrides,
        )
        user_profile, _ = build_reference_profile_from_row(
            row,
            simulation_global_reference=self.simulation_global_reference,
            selected_simulation_features=self.selected_simulation_features,
            overrides=merged_user_overrides,
        )

        p1 = self._predict_p1(row)
        p2 = float(self.simulation_model.predict(reference_profile)[0])
        p3 = float(self.simulation_model.predict(user_profile)[0])
        local_adjustment = float(p3 - p2)
        final_prediction = float(max(p1 + local_adjustment, 0.0))
        gap_vs_historical_pct = float(local_adjustment / p1 * 100.0) if p1 != 0 else 0.0
        explanation = {
            "historical_shap": self._explain_historical_prediction(
                row=row,
                p1_prediction=p1,
            ),
            "local_adjustment": self._explain_local_adjustment(
                reference_profile=reference_profile,
                user_profile=user_profile,
                p2_prediction=p2,
                p3_prediction=p3,
            ),
        }

        return {
            "country": normalize_label(area),
            "crop": normalize_label(crop),
            "p1_historical_prediction": p1,
            "p2_reference_simulation": p2,
            "p3_user_simulation": p3,
            "local_adjustment": local_adjustment,
            "gap_vs_historical_pct": gap_vs_historical_pct,
            "final_prediction": final_prediction,
            "reference_profile": reference_profile.iloc[0].to_dict(),
            "user_profile": user_profile.iloc[0].to_dict(),
            "explanation": explanation,
            **reference_sources,
        }

    def recommend_crops(
        self,
        area: str,
        user_conditions: dict[str, Any],
        candidate_crops: list[str] | None = None,
        *,
        reference_overrides: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Classe les cultures candidates pour un pays et des conditions locales."""
        normalized_area = normalize_label(area)
        area_rows = self.strategy_df.loc[self.strategy_df["area"] == normalized_area].copy()
        if area_rows.empty:
            raise ValueError(f"No historical rows found for area={normalized_area!r}.")

        if candidate_crops:
            normalized_candidates = {normalize_label(crop) for crop in candidate_crops if normalize_label(crop)}
            area_rows = area_rows.loc[area_rows["crop"].isin(normalized_candidates)].copy()
            if area_rows.empty:
                raise ValueError(f"No matching crop found for area={normalized_area!r} and provided candidates.")

        normalized_reference_overrides = self._sanitize_overrides(reference_overrides)
        normalized_user_conditions = self._sanitize_overrides(user_conditions)
        merged_user_overrides = {
            **normalized_reference_overrides,
            **normalized_user_conditions,
        }
        recommendation_rows = []
        for _, row in area_rows.sort_values("crop").iterrows():
            reference_profile, reference_sources = build_reference_profile_from_row(
                row,
                simulation_global_reference=self.simulation_global_reference,
                selected_simulation_features=self.selected_simulation_features,
                overrides=normalized_reference_overrides,
            )
            user_profile, _ = build_reference_profile_from_row(
                row,
                simulation_global_reference=self.simulation_global_reference,
                selected_simulation_features=self.selected_simulation_features,
                overrides=merged_user_overrides,
            )

            p1 = self._predict_p1(row)
            p2 = float(self.simulation_model.predict(reference_profile)[0])
            p3 = float(self.simulation_model.predict(user_profile)[0])
            local_adjustment = float(p3 - p2)
            final_prediction = float(max(p1 + local_adjustment, 0.0))
            gap_vs_historical_pct = float(local_adjustment / p1 * 100.0) if p1 != 0 else 0.0

            recommendation_rows.append(
                {
                    "country": normalized_area,
                    "crop": row["crop"],
                    "p1_historical_prediction": p1,
                    "p2_reference_simulation": p2,
                    "p3_user_simulation": p3,
                    "local_adjustment": local_adjustment,
                    "gap_vs_historical_pct": gap_vs_historical_pct,
                    "final_prediction": final_prediction,
                    "rainfall_reference_source": reference_sources["rainfall_reference_source"],
                    "temperature_reference_source": reference_sources["temperature_reference_source"],
                }
            )

        recommendation_df = (
            pd.DataFrame(recommendation_rows)
            .sort_values(["final_prediction", "p1_historical_prediction"], ascending=[False, False])
            .reset_index(drop=True)
        )
        recommendation_df["recommendation_rank"] = np.arange(1, len(recommendation_df) + 1)

        ordered_columns = [
            "country",
            "crop",
            "p1_historical_prediction",
            "p2_reference_simulation",
            "p3_user_simulation",
            "local_adjustment",
            "gap_vs_historical_pct",
            "final_prediction",
            "recommendation_rank",
            "rainfall_reference_source",
            "temperature_reference_source",
        ]
        return recommendation_df[ordered_columns]


def _load_prediction_context(
    *,
    historical_dataset_path: str | Path = HISTORICAL_WIDE_DATASET_PATH,
    historical_model_path: str | Path = HISTORICAL_MODEL_PATH,
    historical_metadata_path: str | Path = HISTORICAL_METADATA_PATH,
    simulation_dataset_path: str | Path = SIMULATION_DATASET_PATH,
    simulation_model_path: str | Path = SIMULATION_MODEL_PATH,
    simulation_metadata_path: str | Path = SIMULATION_METADATA_PATH,
    force_retrain_simulation: bool = False,
) -> dict[str, Any]:
    """Charge l'ensemble des briques necessaires au runtime final."""
    historical_loaded = load_historical_model(
        model_path=historical_model_path,
        metadata_path=historical_metadata_path,
    )
    historical_df = load_historical_wide_dataset(historical_dataset_path)
    simulation_loaded, simulation_df = load_or_train_simulation_model(
        force_retrain=force_retrain_simulation,
        simulation_path=simulation_dataset_path,
        model_path=simulation_model_path,
        metadata_path=simulation_metadata_path,
    )

    target_year = infer_target_year_from_metadata_or_dataset(historical_loaded.metadata, historical_df)
    reference_df = build_historical_reference_frame(historical_df, target_year=target_year)
    strategy_df = historical_df.merge(reference_df, on=["area", "crop"], how="left")

    return {
        "historical_model": historical_loaded.pipeline,
        "historical_metadata": historical_loaded.metadata,
        "historical_df": historical_df,
        "simulation_model": simulation_loaded.pipeline,
        "simulation_metadata": simulation_loaded.metadata,
        "simulation_df": simulation_df,
        "simulation_global_reference": build_simulation_global_reference(simulation_df),
        "strategy_df": strategy_df,
        "target_year": target_year,
    }


def _predict_p1_from_row(
    row: pd.Series,
    historical_model: Pipeline,
    historical_metadata: dict[str, Any],
) -> float:
    """Projette une ligne historique consolidee dans le modele P1."""
    feature_columns = historical_metadata["feature_columns"]
    feature_frame = pd.DataFrame([row[feature_columns].to_dict()])[feature_columns]
    return float(historical_model.predict(feature_frame)[0])


def _get_area_crop_row(strategy_df: pd.DataFrame, area: str, crop: str) -> pd.Series:
    """Retourne la ligne unique correspondant a un couple pays/culture."""
    normalized_area = normalize_label(area)
    normalized_crop = normalize_label(crop)
    filtered = strategy_df.loc[
        (strategy_df["area"] == normalized_area) & (strategy_df["crop"] == normalized_crop)
    ].copy()

    if filtered.empty:
        raise ValueError(f"No historical row found for area={normalized_area!r}, crop={normalized_crop!r}.")
    if len(filtered) > 1:
        raise ValueError(f"Multiple historical rows found for area={normalized_area!r}, crop={normalized_crop!r}.")

    return filtered.iloc[0]


def predict_adjusted_yield(
    *,
    area: str,
    crop: str,
    user_conditions: dict[str, Any],
    historical_dataset_path: str | Path = HISTORICAL_WIDE_DATASET_PATH,
    historical_model_path: str | Path = HISTORICAL_MODEL_PATH,
    historical_metadata_path: str | Path = HISTORICAL_METADATA_PATH,
    simulation_dataset_path: str | Path = SIMULATION_DATASET_PATH,
    simulation_model_path: str | Path = SIMULATION_MODEL_PATH,
    simulation_metadata_path: str | Path = SIMULATION_METADATA_PATH,
    force_retrain_simulation: bool = False,
) -> dict[str, Any]:
    """Helper procedural pour calculer un rendement ajuste sans gerer le service."""
    service = AdjustedYieldService(
        historical_dataset_path=historical_dataset_path,
        historical_model_path=historical_model_path,
        historical_metadata_path=historical_metadata_path,
        simulation_dataset_path=simulation_dataset_path,
        simulation_model_path=simulation_model_path,
        simulation_metadata_path=simulation_metadata_path,
        force_retrain_simulation=force_retrain_simulation,
    )
    return service.predict_adjusted_yield(area=area, crop=crop, user_conditions=user_conditions)


def recommend_crops(
    *,
    area: str,
    user_conditions: dict[str, Any],
    historical_dataset_path: str | Path = HISTORICAL_WIDE_DATASET_PATH,
    historical_model_path: str | Path = HISTORICAL_MODEL_PATH,
    historical_metadata_path: str | Path = HISTORICAL_METADATA_PATH,
    simulation_dataset_path: str | Path = SIMULATION_DATASET_PATH,
    simulation_model_path: str | Path = SIMULATION_MODEL_PATH,
    simulation_metadata_path: str | Path = SIMULATION_METADATA_PATH,
    force_retrain_simulation: bool = False,
) -> pd.DataFrame:
    """Helper procedural pour classer les cultures candidates."""
    service = AdjustedYieldService(
        historical_dataset_path=historical_dataset_path,
        historical_model_path=historical_model_path,
        historical_metadata_path=historical_metadata_path,
        simulation_dataset_path=simulation_dataset_path,
        simulation_model_path=simulation_model_path,
        simulation_metadata_path=simulation_metadata_path,
        force_retrain_simulation=force_retrain_simulation,
    )
    return service.recommend_crops(area=area, user_conditions=user_conditions)
