"""Briques communes pour les scripts CLI du pipeline du projet."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    """Resout un chemin absolu ou relatif par rapport a la racine du depot."""
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return PROJECT_ROOT / raw_path


def relative_to_project(path: str | Path) -> str:
    """Rend un chemin plus lisible en le relativisant a la racine du projet."""
    resolved_path = resolve_project_path(path)
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved_path)


def ensure_paths_exist(paths: Iterable[str | Path], *, label: str) -> list[Path]:
    """Valide l'existence d'une liste de chemins attendus.

    Args:
        paths: Chemins a verifier.
        label: Libelle utilise dans le message d'erreur.

    Returns:
        list[Path]: Chemins resolus et verifies.
    """
    resolved_paths = [resolve_project_path(path) for path in paths]
    missing_paths = [path for path in resolved_paths if not path.exists()]
    if missing_paths:
        formatted_paths = ", ".join(relative_to_project(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing {label}: {formatted_paths}")
    return resolved_paths


def execute_notebook(
    notebook_path: str | Path,
    *,
    timeout_seconds: int = 3600,
    kernel_name: str = "python3",
    working_directory: str | Path | None = None,
) -> Path:
    """Execute un notebook Jupyter en mode headless.

    Args:
        notebook_path: Notebook a executer.
        timeout_seconds: Timeout global applique par `nbconvert`.
        kernel_name: Kernel Jupyter a utiliser.
        working_directory: Repertoire de travail pour l'execution.

    Returns:
        Path: Chemin resolu du notebook execute.
    """
    import nbformat
    from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

    resolved_notebook_path = resolve_project_path(notebook_path)
    if not resolved_notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {relative_to_project(resolved_notebook_path)}")

    execution_directory = (
        resolve_project_path(working_directory)
        if working_directory is not None
        else PROJECT_ROOT
    )

    with resolved_notebook_path.open("r", encoding="utf-8") as notebook_handle:
        notebook = nbformat.read(notebook_handle, as_version=4)

    executor = ExecutePreprocessor(timeout=timeout_seconds, kernel_name=kernel_name)
    try:
        executor.preprocess(
            notebook,
            {"metadata": {"path": str(execution_directory)}},
        )
    except CellExecutionError as exc:
        raise RuntimeError(
            f"Notebook execution failed: {relative_to_project(resolved_notebook_path)}"
        ) from exc

    return resolved_notebook_path
