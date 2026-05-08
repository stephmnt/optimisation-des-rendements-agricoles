"""Orchestre la chaine locale du projet, de la preparation a la validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline_utils import ensure_paths_exist, execute_notebook, relative_to_project
from scripts.promote_registered_model import promote_registered_models
from scripts.run_preparation import run_preparation
from scripts.runtime_model_specs import DEFAULT_MLFLOW_TRACKING_URI
from scripts.train_historical_model import train_historical_model
from scripts.train_simulation_model import train_simulation_model
from scripts.validate_runtime import validate_runtime


EXPERIENCE_2_NOTEBOOK_PATH = Path("notebooks/experience_2.ipynb")
EXPERIENCE_2_OUTPUTS = [
    Path("artifacts/experiments/experience_2/dataset_series_temporelles.csv"),
    Path("artifacts/experiments/experience_2/model_results.csv"),
    Path("artifacts/experiments/experience_2/experience_2_summary.csv"),
]
EXPERIENCE_3_NOTEBOOK_PATH = Path("notebooks/experience_3.ipynb")


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
        "--run-experience-2",
        action="store_true",
        help="Optionally execute the abandoned complementary temporal notebook.",
    )
    parser.add_argument(
        "--skip-runtime-validation",
        action="store_true",
        help="Skip the final smoke test against the runtime service.",
    )
    parser.add_argument(
        "--run-experience-3",
        action="store_true",
        help="Also execute notebooks/experience_3.ipynb after the artifacts are rebuilt.",
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
        "--json",
        action="store_true",
        help="Print the pipeline summary as JSON.",
    )
    return parser.parse_args()


def run_full_pipeline(
    *,
    skip_preparation: bool = False,
    run_experience_2: bool = False,
    skip_runtime_validation: bool = False,
    run_experience_3: bool = False,
    reuse_simulation_artifact: bool = False,
    simulation_sample_size: int = 200_000,
    notebook_timeout_seconds: int = 7200,
    kernel_name: str = "python3",
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
) -> dict[str, object]:
    """Execute les principales etapes de regeneration des artefacts.

    Args:
        skip_preparation: Saute `preparation.ipynb` si les sorties existent deja.
        run_experience_2: Execute explicitement le notebook temporel abandonne.
        skip_runtime_validation: Saute le smoke test final.
        run_experience_3: Execute aussi le notebook de verification de stack.
        reuse_simulation_artifact: Reutilise le modele local existant au lieu de le reentrainer.
        simulation_sample_size: Taille d'echantillon pour le modele local.
        notebook_timeout_seconds: Timeout applique a chaque notebook execute.
        kernel_name: Kernel Jupyter a utiliser.
        tracking_uri: Tracking URI MLflow utilise pour l'entrainement et la promotion.

    Returns:
        dict[str, object]: Resume des etapes executees et des artefacts verifies.
    """
    results: dict[str, object] = {}

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

    if run_experience_2:
        print(f"[experience_2] Executing {relative_to_project(EXPERIENCE_2_NOTEBOOK_PATH)}")
        execute_notebook(
            EXPERIENCE_2_NOTEBOOK_PATH,
            timeout_seconds=notebook_timeout_seconds,
            kernel_name=kernel_name,
        )
        resolved_outputs = ensure_paths_exist(EXPERIENCE_2_OUTPUTS, label="experience_2 outputs")
        print("[experience_2] Outputs validated")
        results["experience_2"] = {
            "notebook": relative_to_project(EXPERIENCE_2_NOTEBOOK_PATH),
            "outputs": [relative_to_project(path) for path in resolved_outputs],
        }

    results["simulation_model"] = train_simulation_model(
        force_retrain=not reuse_simulation_artifact,
        save_artifact=True,
        sample_size=simulation_sample_size,
        tracking_uri=tracking_uri,
    )
    results["registered_model_promotion"] = promote_registered_models(
        tracking_uri=tracking_uri,
    )

    if run_experience_3:
        print(f"[experience_3] Executing {relative_to_project(EXPERIENCE_3_NOTEBOOK_PATH)}")
        execute_notebook(
            EXPERIENCE_3_NOTEBOOK_PATH,
            timeout_seconds=notebook_timeout_seconds,
            kernel_name=kernel_name,
        )
        results["experience_3"] = {
            "notebook": relative_to_project(EXPERIENCE_3_NOTEBOOK_PATH),
        }

    if not skip_runtime_validation:
        results["runtime_validation"] = validate_runtime()
    else:
        results["runtime_validation"] = {"skipped": True}

    return results


def main() -> None:
    """Execute le pipeline complet depuis la CLI."""
    args = parse_args()
    summary = run_full_pipeline(
        skip_preparation=args.skip_preparation,
        run_experience_2=args.run_experience_2,
        skip_runtime_validation=args.skip_runtime_validation,
        run_experience_3=args.run_experience_3,
        reuse_simulation_artifact=args.reuse_simulation_artifact,
        simulation_sample_size=args.simulation_sample_size,
        notebook_timeout_seconds=args.notebook_timeout_seconds,
        kernel_name=args.kernel_name,
        tracking_uri=args.tracking_uri,
    )
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return

    print("[pipeline] Completed successfully")


if __name__ == "__main__":
    main()
