"""Script Python natif pour reproduire `notebooks/experience_1.ipynb`.

Ce module reconstruit le dataset historique consolide, evalue plusieurs
candidats de regression avec suivi MLflow, puis exporte l'artefact final P1
utilise par l'API et l'interface Streamlit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRFRegressor, XGBRegressor

    XGBOOST_AVAILABLE = True
except ModuleNotFoundError:
    XGBRFRegressor = None
    XGBRegressor = None
    XGBOOST_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlflow_logging import log_named_sklearn_model
from scripts.project_config import DEFAULT_CONFIG_PATH, load_preparation_config


SEED = 42
CV_N_SPLITS = 4
MLFLOW_EXPERIMENT_NAME = "experience_1"
SEARCH_SPACE_DEFINITION = {
    "search_method": "parameter_grid",
    "scope": "all_candidate_families",
    "families": {
        "random_forest_focus": {
            "estimator_kind": "random_forest",
            "model_family": "random_forest",
            "blocks": [
                {
                    "model_name": "random_forest",
                    "tuning_stage": "baseline_grid_point",
                    "regularization_profile": "baseline_grid_point",
                    "grid": {
                        "n_estimators": [300],
                        "max_depth": [12],
                        "min_samples_leaf": [2],
                        "min_samples_split": [2],
                        "max_features": [1.0],
                    },
                },
                {
                    "model_name": "random_forest_regularized",
                    "tuning_stage": "regularized_grid_point",
                    "regularization_profile": "regularized_grid_point",
                    "grid": {
                        "n_estimators": [250],
                        "max_depth": [8],
                        "min_samples_leaf": [4],
                        "min_samples_split": [8],
                        "max_features": [0.6],
                    },
                },
                {
                    "model_name_pattern": "random_forest_search_{index:02d}",
                    "tuning_stage": "systematic_grid_search",
                    "regularization_profile": "parameter_grid_search",
                    "grid": {
                        "n_estimators": [300, 350, 420],
                        "max_depth": [8],
                        "min_samples_leaf": [4],
                        "min_samples_split": [10],
                        "max_features": [0.45],
                    },
                },
            ],
        },
        "xgboost_focus": {
            "estimator_kind": "xgboost",
            "model_family": "xgboost",
            "blocks": [
                {
                    "model_name": "xgboost",
                    "tuning_stage": "baseline_grid_point",
                    "regularization_profile": "baseline_grid_point",
                    "grid": {
                        "n_estimators": [300],
                        "max_depth": [6],
                        "learning_rate": [0.05],
                        "subsample": [0.8],
                        "colsample_bytree": [0.8],
                        "reg_lambda": [1.0],
                        "min_child_weight": [1],
                        "reg_alpha": [0.0],
                        "gamma": [0.0],
                    },
                },
                {
                    "model_name": "xgboost_regularized",
                    "tuning_stage": "regularized_grid_point",
                    "regularization_profile": "regularized_grid_point",
                    "grid": {
                        "n_estimators": [220],
                        "max_depth": [4],
                        "learning_rate": [0.04],
                        "subsample": [0.75],
                        "colsample_bytree": [0.75],
                        "reg_lambda": [3.0],
                        "min_child_weight": [6],
                        "reg_alpha": [0.3],
                        "gamma": [0.1],
                    },
                },
            ],
        },
        "xgboost_random_forest_focus": {
            "estimator_kind": "xgboost_random_forest",
            "model_family": "xgboost_random_forest",
            "blocks": [
                {
                    "model_name": "xgboost_random_forest",
                    "tuning_stage": "baseline_grid_point",
                    "regularization_profile": "baseline_grid_point",
                    "grid": {
                        "n_estimators": [400],
                        "max_depth": [8],
                        "learning_rate": [1.0],
                        "subsample": [0.8],
                        "colsample_bynode": [0.8],
                        "reg_lambda": [1.0],
                        "min_child_weight": [1],
                        "reg_alpha": [0.0],
                    },
                },
                {
                    "model_name": "xgboost_random_forest_regularized",
                    "tuning_stage": "regularized_grid_point",
                    "regularization_profile": "regularized_grid_point",
                    "grid": {
                        "n_estimators": [260],
                        "max_depth": [5],
                        "learning_rate": [1.0],
                        "subsample": [0.7],
                        "colsample_bynode": [0.7],
                        "reg_lambda": [3.0],
                        "min_child_weight": [6],
                        "reg_alpha": [0.2],
                    },
                },
                {
                    "model_name_pattern": "xgboost_random_forest_search_{index:02d}",
                    "tuning_stage": "systematic_grid_search",
                    "regularization_profile": "parameter_grid_search",
                    "grid": {
                        "n_estimators": [220, 280, 300, 320],
                        "max_depth": [5],
                        "learning_rate": [1.0],
                        "subsample": [0.72],
                        "colsample_bynode": [0.68],
                        "reg_lambda": [5.0],
                        "min_child_weight": [7],
                        "reg_alpha": [0.15],
                    },
                },
            ],
        },
    },
}


@dataclass
class ExperiencePaths:
    """Regroupe les chemins produits et consommes par l'experience 1."""

    artifacts_dir: Path
    experience_dir: Path
    cv_dir: Path
    dataset_path: Path
    source_overview_path: Path
    source_quality_path: Path
    summary_path: Path
    missing_summary_path: Path
    model_results_path: Path
    family_results_path: Path
    search_space_path: Path
    mlflow_db_path: Path
    mlflow_artifacts_dir: Path
    mlflow_experiment_artifact_dir: Path
    p1_model_path: Path
    p1_metadata_path: Path


@dataclass
class PreparedSources:
    """Contient les tables nettoyees et le cadrage temporel de l'experience."""

    yield_clean: pd.DataFrame
    rainfall_clean: pd.DataFrame
    pesticides_clean: pd.DataFrame
    temp_clean: pd.DataFrame
    target_year: int
    years: list[int]
    feature_years: list[int]
    selected_yield_years: list[int]


@dataclass
class ModelingContext:
    """Contient les objets necessaires a l'entrainement et a l'evaluation."""

    experience_dataset: pd.DataFrame
    model_df: pd.DataFrame
    target_col: str
    feature_cols: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    train_empty_numeric_features: list[str]
    selected_yield_years: list[int]
    target_year: int
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_all: pd.Series
    groups_train: pd.Series
    onehot_modalities: int
    encoded_feature_count: int


def parse_args() -> argparse.Namespace:
    """Construit l'interface CLI du script.

    Returns:
        argparse.Namespace: Arguments resolves.
    """
    parser = argparse.ArgumentParser(
        description="Run experience_1 as a native Python script and export the P1 artifact.",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Optional path to the project preparation YAML configuration.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI. Defaults to the project SQLite database.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=CV_N_SPLITS,
        help="Number of grouped cross-validation folds on the training split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed used for dataset split and estimators.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final execution summary as JSON.",
    )
    return parser.parse_args()


def build_experience_paths(
    *,
    artifacts_dir: Path,
    tracking_uri: str | None = None,
) -> ExperiencePaths:
    """Construit les chemins utilises par l'experience 1.

    Args:
        artifacts_dir: Dossier `artifacts/` du projet.
        tracking_uri: Tracking URI MLflow optionnel.

    Returns:
        ExperiencePaths: Ensemble des chemins resolus.
    """
    experience_dir = artifacts_dir / "experiments" / MLFLOW_EXPERIMENT_NAME
    cv_dir = experience_dir / "cv"
    models_dir = artifacts_dir / "models"

    experience_dir.mkdir(parents=True, exist_ok=True)
    cv_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    resolved_tracking_uri = tracking_uri or f"sqlite:///{(artifacts_dir / 'mlflow.db').resolve()}"
    mlflow_db_path = Path(resolved_tracking_uri.removeprefix("sqlite:///")).resolve()
    mlflow_artifacts_dir = artifacts_dir / "mlruns"
    mlflow_experiment_artifact_dir = mlflow_artifacts_dir / MLFLOW_EXPERIMENT_NAME
    mlflow_experiment_artifact_dir.mkdir(parents=True, exist_ok=True)

    return ExperiencePaths(
        artifacts_dir=artifacts_dir,
        experience_dir=experience_dir,
        cv_dir=cv_dir,
        dataset_path=experience_dir / "dataset_consolide_historique_colonnes.csv",
        source_overview_path=experience_dir / "source_overview.csv",
        source_quality_path=experience_dir / "source_quality.csv",
        summary_path=experience_dir / "experience_1_summary.csv",
        missing_summary_path=experience_dir / "experience_1_missing_summary.csv",
        model_results_path=experience_dir / "model_results.csv",
        family_results_path=experience_dir / "family_best_results.csv",
        search_space_path=experience_dir / "systematic_search_space.json",
        mlflow_db_path=mlflow_db_path,
        mlflow_artifacts_dir=mlflow_artifacts_dir,
        mlflow_experiment_artifact_dir=mlflow_experiment_artifact_dir,
        p1_model_path=models_dir / "p1_historical_pipeline.joblib",
        p1_metadata_path=models_dir / "p1_historical_metadata.json",
    )


def make_dense_onehot_encoder() -> OneHotEncoder:
    """Construit un OneHotEncoder dense compatible avec plusieurs versions sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def pivot_history(df: pd.DataFrame, index_cols: list[str], value_col: str, years: list[int]) -> pd.DataFrame:
    """Pivote une serie annuelle en format large.

    Args:
        df: Table source en format long.
        index_cols: Colonnes identifiant l'observation.
        value_col: Colonne de valeurs a pivoter.
        years: Annees a conserver et ordonner.

    Returns:
        pd.DataFrame: Table large avec une colonne par annee.
    """
    wide = df.pivot_table(index=index_cols, columns="year", values=value_col, aggfunc="first")
    wide = wide.reindex(columns=years)
    wide.columns = [f"{value_col}_{int(year)}" for year in wide.columns]
    return wide.reset_index()


def compute_regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcule les metriques de regression suivies dans MLflow."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "r2": float(r2_score(y_true_array, y_pred_array)) if len(y_true_array) >= 2 else np.nan,
    }


def build_preprocessor(
    *,
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Construit le preprocesseur tabulaire utilise par tous les candidats."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_dense_onehot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0.0,
    )


def build_model_pipeline(
    estimator: Any,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Assemble le pipeline complet pour un estimateur candidat."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                ),
            ),
            ("regressor", clone(estimator)),
        ]
    )


def register_candidate(
    candidate_registry: dict[str, dict[str, object]],
    model_name: str,
    estimator: Any,
    *,
    model_family: str,
    search_family: str,
    search_method: str,
    search_block_name: str,
    parameter_grid_index: int,
    parameter_grid_size: int,
    regularization_profile: str,
    tuning_stage: str,
) -> None:
    """Enregistre un candidat a evaluer dans le registre local."""
    candidate_registry[model_name] = {
        "estimator": estimator,
        "model_family": model_family,
        "search_family": search_family,
        "search_method": search_method,
        "search_block_name": search_block_name,
        "parameter_grid_index": parameter_grid_index,
        "parameter_grid_size": parameter_grid_size,
        "regularization_profile": regularization_profile,
        "tuning_stage": tuning_stage,
    }


def ensure_xgboost_available() -> None:
    """Valide la disponibilite de `xgboost` avant d'instancier ses modeles.

    Raises:
        ModuleNotFoundError: Si `xgboost` n'est pas installe dans l'environnement
            courant.
    """
    if not XGBOOST_AVAILABLE:
        raise ModuleNotFoundError(
            "xgboost is required to run scripts/experience_1.py. "
            "Install the project dependencies from requirements.txt before training P1."
        )


def make_estimator(estimator_kind: str, *, seed: int, params: dict[str, Any]) -> Any:
    """Construit un estimateur sklearn/xgboost a partir d'un type logique.

    Args:
        estimator_kind: Type d'estimateur declare dans la definition de grille.
        seed: Graine aleatoire globale.
        params: Hyperparametres de l'estimateur.

    Returns:
        Any: Estimateur pret a etre clone dans le pipeline.
    """
    if estimator_kind == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    if estimator_kind == "xgboost":
        ensure_xgboost_available()
        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    if estimator_kind == "xgboost_random_forest":
        ensure_xgboost_available()
        return XGBRFRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    raise ValueError(f"Unknown estimator kind: {estimator_kind}")


def _model_name_from_block(block: dict[str, Any], *, combo_index: int, combo_count: int) -> str:
    """Construit le nom stable d'un candidat issu d'un bloc de grille."""
    explicit_name = block.get("model_name")
    if explicit_name is not None:
        return str(explicit_name)
    pattern = block.get("model_name_pattern")
    if pattern is None:
        raise ValueError("Each search block must define either model_name or model_name_pattern.")
    return str(pattern).format(index=combo_index, total=combo_count)


def expand_search_blocks() -> list[dict[str, Any]]:
    """Deplie les blocs de grilles en liste plate de candidats systematiques.

    Returns:
        list[dict[str, Any]]: Candidats prets a etre instancies.
    """
    expanded_specs: list[dict[str, Any]] = []
    search_method = str(SEARCH_SPACE_DEFINITION["search_method"])
    families = dict(SEARCH_SPACE_DEFINITION["families"])

    for search_family, family_definition in families.items():
        estimator_kind = str(family_definition["estimator_kind"])
        model_family = str(family_definition["model_family"])
        blocks = list(family_definition["blocks"])

        for block in blocks:
            parameter_grid = list(ParameterGrid(dict(block["grid"])))
            combo_count = len(parameter_grid)
            if combo_count == 0:
                raise ValueError(f"Empty parameter grid for search family {search_family!r}.")

            for combo_index, params in enumerate(parameter_grid, start=1):
                expanded_specs.append(
                    {
                        "model_name": _model_name_from_block(
                            block,
                            combo_index=combo_index,
                            combo_count=combo_count,
                        ),
                        "params": dict(params),
                        "estimator_kind": estimator_kind,
                        "model_family": model_family,
                        "search_family": str(search_family),
                        "search_method": search_method,
                        "search_block_name": str(
                            block.get("model_name")
                            or block.get("model_name_pattern")
                            or search_family
                        ),
                        "parameter_grid_index": combo_index,
                        "parameter_grid_size": combo_count,
                        "tuning_stage": str(block["tuning_stage"]),
                        "regularization_profile": str(block["regularization_profile"]),
                    }
                )

    return expanded_specs


def build_candidate_models(seed: int) -> dict[str, dict[str, object]]:
    """Construit le portefeuille de modeles candidats de l'experience.

    Args:
        seed: Graine passee aux estimateurs.

    Returns:
        dict[str, dict[str, object]]: Specification complete des candidats.
    """
    candidate_models: dict[str, dict[str, object]] = {}

    for spec in expand_search_blocks():
        register_candidate(
            candidate_models,
            str(spec["model_name"]),
            make_estimator(
                str(spec["estimator_kind"]),
                seed=seed,
                params=dict(spec["params"]),
            ),
            model_family=str(spec["model_family"]),
            search_family=str(spec["search_family"]),
            search_method=str(spec["search_method"]),
            search_block_name=str(spec["search_block_name"]),
            parameter_grid_index=int(spec["parameter_grid_index"]),
            parameter_grid_size=int(spec["parameter_grid_size"]),
            regularization_profile=str(spec["regularization_profile"]),
            tuning_stage=str(spec["tuning_stage"]),
        )

    return candidate_models


def prepare_sources(
    *,
    config: dict[str, object],
    paths: ExperiencePaths,
) -> PreparedSources:
    """Charge et nettoie les tables historiques de l'experience 1."""
    min_year = int(config["MIN_YEAR"])
    current_year = date.today().year
    years = list(range(min_year, current_year + 1))

    yield_source = pd.read_csv(config["YIELD_PATH"])
    rainfall_source = pd.read_csv(config["RAINFALL_PATH"], na_values=[".."])
    pesticides_source = pd.read_csv(config["PESTICIDES_PATH"])
    temp_source = pd.read_csv(config["TEMP_PATH"])

    source_overview = pd.DataFrame(
        [
            {
                "fichier": "yield.csv",
                "lignes": yield_source.shape[0],
                "colonnes": yield_source.shape[1],
                "nan_detectes": int(yield_source.isna().sum().sum()),
            },
            {
                "fichier": "rainfall.csv",
                "lignes": rainfall_source.shape[0],
                "colonnes": rainfall_source.shape[1],
                "nan_detectes": int(rainfall_source.isna().sum().sum()),
            },
            {
                "fichier": "pesticides.csv",
                "lignes": pesticides_source.shape[0],
                "colonnes": pesticides_source.shape[1],
                "nan_detectes": int(pesticides_source.isna().sum().sum()),
            },
            {
                "fichier": "temp.csv",
                "lignes": temp_source.shape[0],
                "colonnes": temp_source.shape[1],
                "nan_detectes": int(temp_source.isna().sum().sum()),
            },
        ]
    )
    source_overview.to_csv(paths.source_overview_path, index=False)

    yield_clean = (
        yield_source.loc[:, ["Area", "Item", "Year", "Value"]]
        .rename(columns={"Area": "area", "Item": "crop", "Year": "year", "Value": "target_yield_t_ha"})
        .assign(
            area=lambda df: df["area"].astype("string").str.strip(),
            crop=lambda df: df["crop"].astype("string").str.strip(),
            year=lambda df: pd.to_numeric(df["year"], errors="coerce").astype("Int64"),
            target_yield_t_ha=lambda df: pd.to_numeric(df["target_yield_t_ha"], errors="coerce") / 10000,
        )
        .dropna(subset=["area", "crop", "year"])
    )
    yield_clean = yield_clean.loc[yield_clean["year"].between(min_year, current_year, inclusive="both")].copy()
    yield_clean["year"] = yield_clean["year"].astype(int)

    target_year = int(yield_clean["year"].max())
    feature_years = [year for year in years if year < target_year]
    selected_yield_years = feature_years[-3:]

    rainfall_clean = (
        rainfall_source.loc[:, [" Area", "Year", "average_rain_fall_mm_per_year"]]
        .rename(columns={" Area": "area", "Year": "year"})
        .assign(
            area=lambda df: df["area"].astype("string").str.strip(),
            year=lambda df: pd.to_numeric(df["year"], errors="coerce").astype("Int64"),
            average_rain_fall_mm_per_year=lambda df: pd.to_numeric(
                df["average_rain_fall_mm_per_year"],
                errors="coerce",
            ),
        )
        .dropna(subset=["area", "year"])
    )
    rainfall_clean = rainfall_clean.loc[
        rainfall_clean["year"].between(min_year, current_year, inclusive="both")
    ].copy()
    rainfall_clean["year"] = rainfall_clean["year"].astype(int)
    rainfall_clean = rainfall_clean.drop_duplicates(subset=["area", "year"], keep="first")

    pesticides_clean = (
        pesticides_source.loc[:, ["Area", "Year", "Value"]]
        .rename(columns={"Area": "area", "Year": "year", "Value": "pesticides_tonnes"})
        .assign(
            area=lambda df: df["area"].astype("string").str.strip(),
            year=lambda df: pd.to_numeric(df["year"], errors="coerce").astype("Int64"),
            pesticides_tonnes=lambda df: pd.to_numeric(df["pesticides_tonnes"], errors="coerce"),
        )
        .dropna(subset=["area", "year"])
    )
    pesticides_clean = pesticides_clean.loc[
        pesticides_clean["year"].between(min_year, current_year, inclusive="both")
    ].copy()
    pesticides_clean["year"] = pesticides_clean["year"].astype(int)
    pesticides_clean = pesticides_clean.drop_duplicates(subset=["area", "year"], keep="first")

    temp_clean = (
        temp_source.loc[:, ["year", "country", "avg_temp"]]
        .rename(columns={"country": "area"})
        .assign(
            area=lambda df: df["area"].astype("string").str.strip(),
            year=lambda df: pd.to_numeric(df["year"], errors="coerce").astype("Int64"),
            avg_temp=lambda df: pd.to_numeric(df["avg_temp"], errors="coerce"),
        )
        .dropna(subset=["area", "year"])
    )
    temp_clean = temp_clean.loc[temp_clean["year"].between(min_year, current_year, inclusive="both")].copy()
    temp_clean["year"] = temp_clean["year"].astype(int)
    temp_clean = temp_clean.groupby(["area", "year"], as_index=False)["avg_temp"].mean()

    source_quality = pd.DataFrame(
        [
            {
                "table": "yield_clean",
                "cle": "area + crop + year",
                "doublons_sur_cle": int(yield_clean.duplicated(subset=["area", "crop", "year"]).sum()),
                "nan_totaux": int(yield_clean.isna().sum().sum()),
            },
            {
                "table": "rainfall_clean",
                "cle": "area + year",
                "doublons_sur_cle": int(rainfall_clean.duplicated(subset=["area", "year"]).sum()),
                "nan_totaux": int(rainfall_clean.isna().sum().sum()),
            },
            {
                "table": "pesticides_clean",
                "cle": "area + year",
                "doublons_sur_cle": int(pesticides_clean.duplicated(subset=["area", "year"]).sum()),
                "nan_totaux": int(pesticides_clean.isna().sum().sum()),
            },
            {
                "table": "temp_clean",
                "cle": "area + year",
                "doublons_sur_cle": int(temp_clean.duplicated(subset=["area", "year"]).sum()),
                "nan_totaux": int(temp_clean.isna().sum().sum()),
            },
        ]
    )
    source_quality.to_csv(paths.source_quality_path, index=False)

    return PreparedSources(
        yield_clean=yield_clean,
        rainfall_clean=rainfall_clean,
        pesticides_clean=pesticides_clean,
        temp_clean=temp_clean,
        target_year=target_year,
        years=years,
        feature_years=feature_years,
        selected_yield_years=selected_yield_years,
    )


def build_experience_dataset(
    prepared: PreparedSources,
    *,
    paths: ExperiencePaths,
) -> pd.DataFrame:
    """Construit et sauvegarde le dataset large de l'experience 1."""
    base_keys = (
        prepared.yield_clean[["area", "crop"]]
        .drop_duplicates()
        .sort_values(["area", "crop"])
        .reset_index(drop=True)
    )

    yield_history_wide = base_keys.merge(
        pivot_history(prepared.yield_clean, ["area", "crop"], "target_yield_t_ha", prepared.years),
        on=["area", "crop"],
        how="left",
        validate="1:1",
    )
    rainfall_history_wide = pivot_history(
        prepared.rainfall_clean,
        ["area"],
        "average_rain_fall_mm_per_year",
        prepared.years,
    )
    pesticides_history_wide = pivot_history(
        prepared.pesticides_clean,
        ["area"],
        "pesticides_tonnes",
        prepared.years,
    )
    temp_history_wide = pivot_history(prepared.temp_clean, ["area"], "avg_temp", prepared.years)

    experience_dataset = (
        yield_history_wide
        .merge(rainfall_history_wide, on="area", how="left", validate="m:1")
        .merge(pesticides_history_wide, on="area", how="left", validate="m:1")
        .merge(temp_history_wide, on="area", how="left", validate="m:1")
        .sort_values(["area", "crop"])
        .reset_index(drop=True)
    )

    missing_summary = (
        experience_dataset.isna()
        .sum()
        .rename("nb_nan")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    missing_summary["part_nan_pct"] = (missing_summary["nb_nan"] / len(experience_dataset) * 100).round(2)
    missing_summary.to_csv(paths.missing_summary_path, index=False)

    experience_summary = pd.DataFrame(
        {
            "indicateur": [
                "nb_lignes",
                "nb_colonnes",
                "annee_cible_modele",
                "part_nan_globale_pct",
                "colonnes_cible_historiques",
                "colonnes_pluie_historiques",
                "colonnes_pesticides_historiques",
                "colonnes_temperature_historiques",
            ],
            "valeur": [
                int(experience_dataset.shape[0]),
                int(experience_dataset.shape[1]),
                prepared.target_year,
                round(experience_dataset.isna().mean().mean() * 100, 2),
                len([col for col in experience_dataset.columns if col.startswith("target_yield_t_ha_")]),
                len([col for col in experience_dataset.columns if col.startswith("average_rain_fall_mm_per_year_")]),
                len([col for col in experience_dataset.columns if col.startswith("pesticides_tonnes_")]),
                len([col for col in experience_dataset.columns if col.startswith("avg_temp_")]),
            ],
        }
    )
    experience_summary.to_csv(paths.summary_path, index=False)
    experience_dataset.to_csv(paths.dataset_path, index=False)

    return experience_dataset


def build_modeling_context(
    experience_dataset: pd.DataFrame,
    *,
    target_year: int,
    feature_years: list[int],
    selected_yield_years: list[int],
    seed: int,
) -> ModelingContext:
    """Prepare la matrice d'entrainement et la separation train/test."""
    target_col = f"target_yield_t_ha_{target_year}"

    feature_cols = ["crop"]
    feature_cols += [f"target_yield_t_ha_{year}" for year in selected_yield_years]
    feature_cols += [f"average_rain_fall_mm_per_year_{year}" for year in feature_years]
    feature_cols += [f"pesticides_tonnes_{year}" for year in feature_years]
    feature_cols += [f"avg_temp_{year}" for year in feature_years]
    feature_cols = [col for col in feature_cols if col in experience_dataset.columns]

    base_columns = ["area"] + [col for col in feature_cols if col != "area"] + [target_col]
    base_columns = list(dict.fromkeys(base_columns))
    model_df = experience_dataset[base_columns].copy()
    model_df = model_df.dropna(subset=[target_col]).reset_index(drop=True)

    categorical_features = ["crop"]
    numeric_features = [col for col in feature_cols if col not in categorical_features]

    X = model_df[feature_cols].copy()
    y = model_df[target_col].copy()
    groups = model_df["area"].copy()

    group_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(group_split.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)

    train_empty_numeric_features = [col for col in numeric_features if not X_train[col].notna().any()]
    numeric_features = [col for col in numeric_features if col not in train_empty_numeric_features]
    feature_cols = categorical_features + numeric_features
    X_train = X_train[feature_cols].copy()
    X_test = X_test[feature_cols].copy()

    probe_preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    X_train_prepared = probe_preprocessor.fit_transform(X_train)
    onehot_encoder = probe_preprocessor.named_transformers_["cat"].named_steps["onehot"]
    onehot_modalities = int(sum(len(categories) for categories in onehot_encoder.categories_))
    encoded_feature_count = int(X_train_prepared.shape[1])

    return ModelingContext(
        experience_dataset=experience_dataset,
        model_df=model_df,
        target_col=target_col,
        feature_cols=feature_cols,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        train_empty_numeric_features=train_empty_numeric_features,
        selected_yield_years=selected_yield_years,
        target_year=target_year,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_all=y,
        groups_train=groups_train,
        onehot_modalities=onehot_modalities,
        encoded_feature_count=encoded_feature_count,
    )


def run_group_cross_validation(
    estimator: Any,
    *,
    context: ModelingContext,
    cv_n_splits: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Execute la cross-validation groupee sur le split train.

    Args:
        estimator: Estimateur a evaluer.
        context: Contexte de modelisation.
        cv_n_splits: Nombre cible de folds.

    Returns:
        tuple[pd.DataFrame, dict[str, float]]: Resultats detaille par fold et
        statistiques resumees.
    """
    unique_group_count = int(context.groups_train.nunique())
    effective_splits = min(cv_n_splits, unique_group_count)
    if effective_splits < 2:
        raise ValueError("Grouped cross-validation requires at least 2 distinct groups.")

    group_kfold = GroupKFold(n_splits=effective_splits)
    fold_rows: list[dict[str, float | int]] = []

    for fold_id, (cv_train_idx, cv_val_idx) in enumerate(
        group_kfold.split(context.X_train, context.y_train, groups=context.groups_train),
        start=1,
    ):
        X_cv_train = context.X_train.iloc[cv_train_idx].reset_index(drop=True)
        X_cv_val = context.X_train.iloc[cv_val_idx].reset_index(drop=True)
        y_cv_train = context.y_train.iloc[cv_train_idx].reset_index(drop=True)
        y_cv_val = context.y_train.iloc[cv_val_idx].reset_index(drop=True)
        groups_cv_train = context.groups_train.iloc[cv_train_idx].reset_index(drop=True)
        groups_cv_val = context.groups_train.iloc[cv_val_idx].reset_index(drop=True)

        pipeline = build_model_pipeline(
            estimator,
            numeric_features=context.numeric_features,
            categorical_features=context.categorical_features,
        )
        pipeline.fit(X_cv_train, y_cv_train)

        train_pred = pipeline.predict(X_cv_train)
        val_pred = pipeline.predict(X_cv_val)

        train_metrics = compute_regression_metrics(y_cv_train, train_pred)
        val_metrics = compute_regression_metrics(y_cv_val, val_pred)
        fold_rows.append(
            {
                "fold": fold_id,
                "train_rows": int(len(X_cv_train)),
                "val_rows": int(len(X_cv_val)),
                "train_areas": int(groups_cv_train.nunique()),
                "val_areas": int(groups_cv_val.nunique()),
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "overfit_gap_rmse": float(val_metrics["rmse"] - train_metrics["rmse"]),
                "overfit_ratio_rmse": (
                    float(val_metrics["rmse"] / train_metrics["rmse"])
                    if train_metrics["rmse"] > 0
                    else np.nan
                ),
            }
        )

    cv_fold_df = pd.DataFrame(fold_rows)
    cv_summary = {
        "cv_n_splits": int(effective_splits),
        "cv_train_mae_mean": float(cv_fold_df["train_mae"].mean()),
        "cv_train_rmse_mean": float(cv_fold_df["train_rmse"].mean()),
        "cv_train_r2_mean": float(cv_fold_df["train_r2"].mean()),
        "cv_val_mae_mean": float(cv_fold_df["val_mae"].mean()),
        "cv_val_mae_std": float(cv_fold_df["val_mae"].std(ddof=0)),
        "cv_val_rmse_mean": float(cv_fold_df["val_rmse"].mean()),
        "cv_val_rmse_std": float(cv_fold_df["val_rmse"].std(ddof=0)),
        "cv_val_r2_mean": float(cv_fold_df["val_r2"].mean()),
        "cv_val_r2_std": float(cv_fold_df["val_r2"].std(ddof=0)),
        "cv_overfit_gap_rmse_mean": float(cv_fold_df["overfit_gap_rmse"].mean()),
        "cv_overfit_ratio_rmse_mean": float(cv_fold_df["overfit_ratio_rmse"].mean()),
    }
    return cv_fold_df, cv_summary


def ensure_mlflow_experiment(paths: ExperiencePaths, *, tracking_uri: str) -> None:
    """Prepare le backend MLflow et garantit l'emplacement des artefacts.

    Args:
        paths: Chemins de l'experience.
        tracking_uri: Tracking URI MLflow.
    """
    mlflow.set_tracking_uri(tracking_uri)
    while mlflow.active_run() is not None:
        mlflow.end_run()

    tracking_db_path = paths.mlflow_db_path
    experiment_artifact_uri = paths.mlflow_experiment_artifact_dir.resolve().as_uri()
    tracking_db_path.parent.mkdir(parents=True, exist_ok=True)
    paths.mlflow_artifacts_dir.mkdir(parents=True, exist_ok=True)

    if tracking_db_path.exists():
        connection = sqlite3.connect(tracking_db_path)
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiments'")
        if cursor.fetchone() is not None:
            cursor.execute(
                "SELECT experiment_id, name, artifact_location FROM experiments WHERE name = ?",
                (MLFLOW_EXPERIMENT_NAME,),
            )
            existing_row = cursor.fetchone()
            if existing_row is not None:
                experiment_id, _, current_artifact_location = existing_row
                current_artifact_dir = Path(str(current_artifact_location).removeprefix("file://")).resolve()
                target_artifact_dir = paths.mlflow_experiment_artifact_dir.resolve()
                if current_artifact_dir.exists() and current_artifact_dir != target_artifact_dir:
                    for child in current_artifact_dir.iterdir():
                        destination = target_artifact_dir / child.name
                        if not destination.exists():
                            shutil.move(str(child), str(destination))
                    if current_artifact_dir.exists() and current_artifact_dir.is_dir() and not any(current_artifact_dir.iterdir()):
                        current_artifact_dir.rmdir()
                cursor.execute(
                    "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
                    (experiment_artifact_uri, experiment_id),
                )
                cursor.execute(
                    """
                    UPDATE runs
                    SET artifact_uri = REPLACE(artifact_uri, ?, ?)
                    WHERE experiment_id = ? AND artifact_uri LIKE ?
                    """,
                    (
                        str(current_artifact_dir),
                        str(target_artifact_dir),
                        experiment_id,
                        f"{current_artifact_dir}%",
                    ),
                )
                connection.commit()
        connection.close()

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(MLFLOW_EXPERIMENT_NAME, artifact_location=experiment_artifact_uri)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def evaluate_candidates(
    candidate_models: dict[str, dict[str, object]],
    *,
    context: ModelingContext,
    paths: ExperiencePaths,
    cv_n_splits: int,
    tracking_uri: str,
) -> pd.DataFrame:
    """Evalue les candidats, journalise dans MLflow et sauvegarde les resultats."""
    ensure_mlflow_experiment(paths, tracking_uri=tracking_uri)
    paths.search_space_path.write_text(json.dumps(SEARCH_SPACE_DEFINITION, indent=2), encoding="utf-8")

    results: list[dict[str, Any]] = []
    for model_name, model_spec in candidate_models.items():
        estimator = model_spec["estimator"]
        estimator_params = estimator.get_params()
        cv_fold_df, cv_summary = run_group_cross_validation(
            estimator,
            context=context,
            cv_n_splits=cv_n_splits,
        )

        pipeline = build_model_pipeline(
            estimator,
            numeric_features=context.numeric_features,
            categorical_features=context.categorical_features,
        )
        with mlflow.start_run(run_name=f"{MLFLOW_EXPERIMENT_NAME}__{model_name}") as run:
            pipeline.fit(context.X_train, context.y_train)

            train_pred = pipeline.predict(context.X_train)
            test_pred = pipeline.predict(context.X_test)

            train_metrics = compute_regression_metrics(context.y_train, train_pred)
            test_metrics = compute_regression_metrics(context.y_test, test_pred)
            overfit_gap_rmse = float(test_metrics["rmse"] - train_metrics["rmse"])
            overfit_ratio_rmse = (
                float(test_metrics["rmse"] / train_metrics["rmse"])
                if train_metrics["rmse"] > 0
                else np.nan
            )

            cv_fold_path = paths.cv_dir / f"{model_name}_cv_folds.csv"
            cv_summary_path = paths.cv_dir / f"{model_name}_cv_summary.json"
            model_params_path = paths.cv_dir / f"{model_name}_params.json"
            cv_fold_df.to_csv(cv_fold_path, index=False)
            cv_summary_path.write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")
            model_params_path.write_text(json.dumps(estimator_params, default=str, indent=2), encoding="utf-8")

            mlflow.log_param("experience_name", MLFLOW_EXPERIMENT_NAME)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_family", model_spec["model_family"])
            mlflow.log_param("search_family", model_spec["search_family"])
            mlflow.log_param("search_method", model_spec["search_method"])
            mlflow.log_param("search_block_name", model_spec["search_block_name"])
            mlflow.log_param("parameter_grid_index", model_spec["parameter_grid_index"])
            mlflow.log_param("parameter_grid_size", model_spec["parameter_grid_size"])
            mlflow.log_param("tuning_stage", model_spec["tuning_stage"])
            mlflow.log_param("target_col", context.target_col)
            mlflow.log_param("target_year", context.target_year)
            mlflow.log_param("split_strategy", "GroupShuffleSplit(area)")
            mlflow.log_param("cross_validation_strategy", "GroupKFold(area)_on_train_only")
            mlflow.log_param("area_used_as_feature", False)
            mlflow.log_param("selected_feature_count", len(context.feature_cols))
            mlflow.log_param("numeric_feature_count", len(context.numeric_features))
            mlflow.log_param("categorical_feature_count", len(context.categorical_features))
            mlflow.log_param(
                "dropped_train_empty_numeric_feature_count",
                len(context.train_empty_numeric_features),
            )
            mlflow.log_param("selected_yield_year_count", len(context.selected_yield_years))
            mlflow.log_param("selected_yield_year_start", int(context.selected_yield_years[0]))
            mlflow.log_param("selected_yield_year_end", int(context.selected_yield_years[-1]))
            mlflow.log_param("encoded_feature_count", context.encoded_feature_count)
            mlflow.log_param("onehot_modalities", context.onehot_modalities)
            mlflow.log_param("cv_n_splits", cv_summary["cv_n_splits"])
            mlflow.log_param("regularization_profile", model_spec["regularization_profile"])

            mlflow.log_metric("train_mae", train_metrics["mae"])
            mlflow.log_metric("train_rmse", train_metrics["rmse"])
            mlflow.log_metric("train_r2", train_metrics["r2"])
            mlflow.log_metric("test_mae", test_metrics["mae"])
            mlflow.log_metric("test_rmse", test_metrics["rmse"])
            mlflow.log_metric("test_r2", test_metrics["r2"])
            mlflow.log_metric("overfit_gap_rmse", overfit_gap_rmse)
            mlflow.log_metric("overfit_ratio_rmse", overfit_ratio_rmse)
            mlflow.log_metric("cv_train_mae_mean", cv_summary["cv_train_mae_mean"])
            mlflow.log_metric("cv_train_rmse_mean", cv_summary["cv_train_rmse_mean"])
            mlflow.log_metric("cv_train_r2_mean", cv_summary["cv_train_r2_mean"])
            mlflow.log_metric("cv_val_mae_mean", cv_summary["cv_val_mae_mean"])
            mlflow.log_metric("cv_val_mae_std", cv_summary["cv_val_mae_std"])
            mlflow.log_metric("cv_val_rmse_mean", cv_summary["cv_val_rmse_mean"])
            mlflow.log_metric("cv_val_rmse_std", cv_summary["cv_val_rmse_std"])
            mlflow.log_metric("cv_val_r2_mean", cv_summary["cv_val_r2_mean"])
            mlflow.log_metric("cv_val_r2_std", cv_summary["cv_val_r2_std"])
            mlflow.log_metric("cv_overfit_gap_rmse_mean", cv_summary["cv_overfit_gap_rmse_mean"])
            mlflow.log_metric("cv_overfit_ratio_rmse_mean", cv_summary["cv_overfit_ratio_rmse_mean"])

            mlflow.log_artifact(str(paths.summary_path))
            mlflow.log_artifact(str(paths.missing_summary_path))
            mlflow.log_artifact(str(paths.dataset_path))
            mlflow.log_artifact(str(paths.search_space_path))
            mlflow.log_artifact(str(cv_fold_path))
            mlflow.log_artifact(str(cv_summary_path))
            mlflow.log_artifact(str(model_params_path))
            log_named_sklearn_model(pipeline, model_name=model_name)

            results.append(
                {
                    "model": model_name,
                    "model_family": model_spec["model_family"],
                    "search_family": model_spec["search_family"],
                    "search_method": model_spec["search_method"],
                    "search_block_name": model_spec["search_block_name"],
                    "parameter_grid_index": model_spec["parameter_grid_index"],
                    "parameter_grid_size": model_spec["parameter_grid_size"],
                    "tuning_stage": model_spec["tuning_stage"],
                    "regularization_profile": model_spec["regularization_profile"],
                    "train_mae": train_metrics["mae"],
                    "train_rmse": train_metrics["rmse"],
                    "train_r2": train_metrics["r2"],
                    "cv_val_mae_mean": cv_summary["cv_val_mae_mean"],
                    "cv_val_mae_std": cv_summary["cv_val_mae_std"],
                    "cv_val_rmse_mean": cv_summary["cv_val_rmse_mean"],
                    "cv_val_rmse_std": cv_summary["cv_val_rmse_std"],
                    "cv_val_r2_mean": cv_summary["cv_val_r2_mean"],
                    "cv_overfit_gap_rmse_mean": cv_summary["cv_overfit_gap_rmse_mean"],
                    "cv_overfit_ratio_rmse_mean": cv_summary["cv_overfit_ratio_rmse_mean"],
                    "test_mae": test_metrics["mae"],
                    "test_rmse": test_metrics["rmse"],
                    "test_r2": test_metrics["r2"],
                    "overfit_gap_rmse": overfit_gap_rmse,
                    "overfit_ratio_rmse": overfit_ratio_rmse,
                    "run_id": run.info.run_id,
                }
            )

    results_df = pd.DataFrame(results).sort_values(["test_rmse", "cv_val_rmse_mean"]).reset_index(drop=True)
    results_df["global_rank"] = np.arange(1, len(results_df) + 1)
    results_df["family_rank"] = (
        results_df.groupby("search_family")["cv_val_rmse_mean"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    results_df.to_csv(paths.model_results_path, index=False)

    family_best_df = (
        results_df.sort_values(["search_family", "cv_val_rmse_mean", "test_rmse"])
        .groupby("search_family", as_index=False)
        .first()
    )
    family_best_df.to_csv(paths.family_results_path, index=False)

    with mlflow.start_run(run_name=f"{MLFLOW_EXPERIMENT_NAME}__summary"):
        mlflow.log_param("experience_name", MLFLOW_EXPERIMENT_NAME)
        mlflow.log_param("models_tested", ",".join(candidate_models.keys()))
        mlflow.log_param("selected_feature_strategy", "no_area_plus_recent_3_yield_years")
        mlflow.log_param("cross_validation_strategy", "GroupKFold(area)_on_train_only")
        mlflow.log_param("search_method", SEARCH_SPACE_DEFINITION["search_method"])
        mlflow.log_param("search_scope", SEARCH_SPACE_DEFINITION["scope"])
        mlflow.log_metric("best_test_rmse", float(results_df.loc[0, "test_rmse"]))
        mlflow.log_metric("best_test_r2", float(results_df.loc[0, "test_r2"]))
        mlflow.log_metric("best_cv_val_rmse_mean", float(results_df.loc[0, "cv_val_rmse_mean"]))
        mlflow.log_artifact(str(paths.model_results_path))
        mlflow.log_artifact(str(paths.family_results_path))
        mlflow.log_artifact(str(paths.search_space_path))

    return results_df


def export_p1_artifact(
    candidate_models: dict[str, dict[str, object]],
    *,
    results_df: pd.DataFrame,
    context: ModelingContext,
    paths: ExperiencePaths,
) -> dict[str, Any]:
    """Re-entraine le meilleur modele sur tout le dataset et exporte P1."""
    best_model_name = str(results_df.loc[0, "model"])
    best_model_spec = candidate_models[best_model_name]

    p1_training_X = context.model_df[context.feature_cols].copy()
    p1_training_y = context.y_all.copy()

    p1_pipeline = build_model_pipeline(
        best_model_spec["estimator"],
        numeric_features=context.numeric_features,
        categorical_features=context.categorical_features,
    )
    p1_pipeline.fit(p1_training_X, p1_training_y)

    p1_metadata = {
        "artifact_role": "P1_historical_prediction_model",
        "training_notebook": "notebooks/experience_1.ipynb",
        "training_script": "scripts/experience_1.py",
        "training_entrypoint": "scripts/experience_1.py",
        "model_name": best_model_name,
        "model_family": best_model_spec["model_family"],
        "search_family": best_model_spec["search_family"],
        "search_method": best_model_spec["search_method"],
        "search_block_name": best_model_spec["search_block_name"],
        "parameter_grid_index": int(best_model_spec["parameter_grid_index"]),
        "parameter_grid_size": int(best_model_spec["parameter_grid_size"]),
        "tuning_stage": best_model_spec["tuning_stage"],
        "regularization_profile": best_model_spec["regularization_profile"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_source": str(paths.dataset_path),
        "target_year": int(context.target_year),
        "target_column": context.target_col,
        "feature_columns": context.feature_cols,
        "selected_yield_years": context.selected_yield_years,
        "area_role": "group_only_not_feature",
        "split_strategy": "GroupShuffleSplit(area, test_size=0.2, random_state=42)",
        "metrics": {
            "test_rmse": float(results_df.loc[0, "test_rmse"]),
            "test_mae": float(results_df.loc[0, "test_mae"]),
            "test_r2": float(results_df.loc[0, "test_r2"]),
            "cv_val_rmse_mean": float(results_df.loc[0, "cv_val_rmse_mean"]),
            "cv_val_mae_mean": float(results_df.loc[0, "cv_val_mae_mean"]),
            "cv_val_r2_mean": float(results_df.loc[0, "cv_val_r2_mean"]),
        },
        "mlflow_run_id": str(results_df.loc[0, "run_id"]) if "run_id" in results_df.columns else None,
    }

    joblib.dump(p1_pipeline, paths.p1_model_path)
    paths.p1_metadata_path.write_text(
        json.dumps(p1_metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return p1_metadata


def run_experience_1(
    *,
    config_path: str | Path | None = None,
    tracking_uri: str | None = None,
    cv_n_splits: int = CV_N_SPLITS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Execute l'experience 1 complete en Python natif.

    Args:
        config_path: Chemin optionnel vers le YAML de configuration du projet.
        tracking_uri: Tracking URI MLflow optionnel.
        cv_n_splits: Nombre de folds pour la cross-validation groupee.
        seed: Graine aleatoire globale.

    Returns:
        dict[str, Any]: Resume des artefacts et du meilleur modele retenu.
    """
    resolved_config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    config = load_preparation_config(resolved_config_path, ensure_dirs=True)
    resolved_tracking_uri = tracking_uri or f"sqlite:///{(Path(config['ARTIFACTS_DIR']) / 'mlflow.db').resolve()}"
    paths = build_experience_paths(
        artifacts_dir=Path(config["ARTIFACTS_DIR"]),
        tracking_uri=resolved_tracking_uri,
    )

    print(f"[experience_1] Configuration loaded from: {resolved_config_path.resolve()}")
    print(f"[experience_1] Tracking MLflow: {resolved_tracking_uri}")
    print(f"[experience_1] Experiment directory: {paths.experience_dir.resolve()}")

    prepared_sources = prepare_sources(config=config, paths=paths)
    print(
        "[experience_1] Target year: "
        f"{prepared_sources.target_year} | Selected yield years: {prepared_sources.selected_yield_years}"
    )

    experience_dataset = build_experience_dataset(prepared_sources, paths=paths)
    print(f"[experience_1] Historical wide dataset saved: {paths.dataset_path.resolve()}")

    modeling_context = build_modeling_context(
        experience_dataset,
        target_year=prepared_sources.target_year,
        feature_years=prepared_sources.feature_years,
        selected_yield_years=prepared_sources.selected_yield_years,
        seed=seed,
    )

    candidate_models = build_candidate_models(seed)
    results_df = evaluate_candidates(
        candidate_models,
        context=modeling_context,
        paths=paths,
        cv_n_splits=cv_n_splits,
        tracking_uri=resolved_tracking_uri,
    )
    p1_metadata = export_p1_artifact(
        candidate_models,
        results_df=results_df,
        context=modeling_context,
        paths=paths,
    )

    print(f"[experience_1] Model results saved: {paths.model_results_path.resolve()}")
    print(f"[experience_1] Family best results saved: {paths.family_results_path.resolve()}")
    print(f"[experience_1] P1 pipeline saved: {paths.p1_model_path.resolve()}")
    print(f"[experience_1] P1 metadata saved: {paths.p1_metadata_path.resolve()}")

    return {
        "dataset_path": str(paths.dataset_path),
        "model_results_path": str(paths.model_results_path),
        "family_results_path": str(paths.family_results_path),
        "p1_model_path": str(paths.p1_model_path),
        "p1_metadata_path": str(paths.p1_metadata_path),
        "best_model_name": str(results_df.loc[0, "model"]),
        "best_test_rmse": float(results_df.loc[0, "test_rmse"]),
        "best_test_r2": float(results_df.loc[0, "test_r2"]),
        "tracked_models": list(results_df["model"]),
        "p1_metadata": p1_metadata,
    }


def main() -> None:
    """Execute le script depuis la ligne de commande."""
    args = parse_args()
    summary = run_experience_1(
        config_path=args.config_path,
        tracking_uri=args.tracking_uri,
        cv_n_splits=args.cv_splits,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
