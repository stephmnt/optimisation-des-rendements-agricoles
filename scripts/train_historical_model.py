"""Point d'entree CLI pour regenerer les artefacts historiques P1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experience_1 import run_experience_1
from scripts.pipeline_utils import ensure_paths_exist, relative_to_project


EXPERIENCE_1_SCRIPT_PATH = Path("scripts/experience_1.py")
HISTORICAL_OUTPUTS = [
    Path("artifacts/experiments/experience_1/dataset_consolide_historique_colonnes.csv"),
    Path("artifacts/experiments/experience_1/model_results.csv"),
    Path("artifacts/models/p1_historical_pipeline.joblib"),
    Path("artifacts/models/p1_historical_metadata.json"),
]
HISTORICAL_METADATA_PATH = Path("artifacts/models/p1_historical_metadata.json")


def parse_args() -> argparse.Namespace:
    """Construit l'interface en ligne de commande du script."""
    parser = argparse.ArgumentParser(
        description="Execute experience_1 headlessly and validate the historical model artifacts.",
    )
    parser.add_argument("--tracking-uri", default=None, help="Optional MLflow tracking URI override.")
    parser.add_argument("--cv-splits", type=int, default=4, help="Number of grouped CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for the experiment.")
    return parser.parse_args()


def train_historical_model(
    *,
    tracking_uri: str | None = None,
    cv_splits: int = 4,
    seed: int = 42,
) -> dict[str, object]:
    """Execute `scripts/experience_1.py` et valide les artefacts historiques.

    Args:
        tracking_uri: Tracking URI MLflow optionnel.
        cv_splits: Nombre de folds pour la CV groupee.
        seed: Graine aleatoire globale.

    Returns:
        dict[str, object]: Resume du modele historique et de ses artefacts.
    """
    print(f"[historical] Executing {relative_to_project(EXPERIENCE_1_SCRIPT_PATH)}")
    run_experience_1(
        tracking_uri=tracking_uri,
        cv_n_splits=cv_splits,
        seed=seed,
    )
    resolved_outputs = ensure_paths_exist(HISTORICAL_OUTPUTS, label="historical model outputs")
    metadata = json.loads((PROJECT_ROOT / HISTORICAL_METADATA_PATH).read_text(encoding="utf-8"))
    metrics = metadata.get("metrics", {})
    print(
        "[historical] Outputs validated "
        f"(model={metadata.get('model_name')}, test_rmse={metrics.get('test_rmse')}, test_r2={metrics.get('test_r2')})"
    )
    return {
        "script": relative_to_project(EXPERIENCE_1_SCRIPT_PATH),
        "training_notebook_reference": metadata.get("training_notebook"),
        "outputs": [relative_to_project(path) for path in resolved_outputs],
        "model_name": metadata.get("model_name"),
        "target_year": metadata.get("target_year"),
        "metrics": metrics,
    }


def main() -> None:
    """Execute le script historique depuis la CLI."""
    args = parse_args()
    train_historical_model(
        tracking_uri=args.tracking_uri,
        cv_splits=args.cv_splits,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
