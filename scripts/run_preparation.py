"""Point d'entree CLI pour executer `notebooks/preparation.ipynb` en headless."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline_utils import ensure_paths_exist, execute_notebook, relative_to_project


PREPARATION_NOTEBOOK_PATH = Path("notebooks/preparation.ipynb")
PREPARATION_OUTPUTS = [
    Path("data/dataset_consolide.csv"),
    Path("artifacts/pca/pca_summary.csv"),
    Path("artifacts/pca/pca_explained_variance.png"),
]


def parse_args() -> argparse.Namespace:
    """Construit l'interface en ligne de commande du script."""
    parser = argparse.ArgumentParser(
        description="Execute the preparation notebook headlessly and validate its outputs.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
        help="Maximum execution time allowed for the notebook.",
    )
    parser.add_argument(
        "--kernel-name",
        default="python3",
        help="Jupyter kernel used to execute the notebook.",
    )
    return parser.parse_args()


def run_preparation(*, timeout_seconds: int = 3600, kernel_name: str = "python3") -> dict[str, object]:
    """Execute le notebook de preparation et valide ses sorties principales.

    Args:
        timeout_seconds: Temps maximal laisse au notebook.
        kernel_name: Kernel Jupyter a utiliser.

    Returns:
        dict[str, object]: Resume des artefacts verifies.
    """
    print(f"[prepare] Executing {relative_to_project(PREPARATION_NOTEBOOK_PATH)}")
    execute_notebook(
        PREPARATION_NOTEBOOK_PATH,
        timeout_seconds=timeout_seconds,
        kernel_name=kernel_name,
    )
    resolved_outputs = ensure_paths_exist(PREPARATION_OUTPUTS, label="preparation outputs")
    print("[prepare] Outputs validated")
    return {
        "notebook": relative_to_project(PREPARATION_NOTEBOOK_PATH),
        "outputs": [relative_to_project(path) for path in resolved_outputs],
    }


def main() -> None:
    """Execute le script de preparation depuis la CLI."""
    args = parse_args()
    run_preparation(
        timeout_seconds=args.timeout_seconds,
        kernel_name=args.kernel_name,
    )


if __name__ == "__main__":
    main()
