"""Point d'entree CLI pour la brique de simulation locale P2/P3."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline_utils import ensure_paths_exist, relative_to_project
from scripts.prediction_adjustment import (
    SIMULATION_METADATA_PATH,
    SIMULATION_MODEL_PATH,
    load_or_train_simulation_model,
)


SIMULATION_OUTPUTS = [
    SIMULATION_MODEL_PATH,
    SIMULATION_METADATA_PATH,
]


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
    return parser.parse_args()


def train_simulation_model(
    *,
    force_retrain: bool = False,
    save_artifact: bool = True,
    sample_size: int = 200_000,
) -> dict[str, object]:
    """Charge ou reentraine le modele local de simulation.

    Args:
        force_retrain: Force le reentrainement meme si les artefacts existent.
        save_artifact: Ecrit les artefacts sur disque si `True`.
        sample_size: Nombre maximal de lignes echantillonnees pour l'entrainement.

    Returns:
        dict[str, object]: Resume du dataset utilise, des metriques et des sorties.
    """
    loaded_model, simulation_df = load_or_train_simulation_model(
        force_retrain=force_retrain,
        save_artifact=save_artifact,
        sample_size=sample_size,
    )

    output_paths: list[str] = []
    if save_artifact:
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
    )


if __name__ == "__main__":
    main()
