"""Construit et valide le payload de deploiement Hugging Face Space.

Le but est d'eviter la duplication entre les jobs `build` et `deploy` du
workflow GitHub Actions, et d'aligner cette logique avec la validation runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEPLOYMENT_REQUIRED_ARTIFACTS = [
    Path("artifacts/models/p1_historical_pipeline.joblib"),
    Path("artifacts/models/p1_historical_metadata.json"),
    Path("artifacts/models/p23_simulation_pipeline.joblib"),
    Path("artifacts/models/p23_simulation_metadata.json"),
    Path("artifacts/experiments/experience_1/dataset_consolide_historique_colonnes.csv"),
]

PAYLOAD_DIRECTORIES = [
    Path("config"),
    Path("scripts"),
    Path("streamlit/src"),
    Path("streamlit/icones"),
    Path("data/simulation"),
    Path("artifacts/models"),
    Path("artifacts/experiments/experience_1"),
]

PAYLOAD_DIRECTORY_SPECS = [
    (Path("scripts"), Path("scripts")),
    (Path("streamlit/src"), Path("streamlit/src")),
    (Path("streamlit/icones"), Path("streamlit/icones")),
]

PAYLOAD_FILE_SPECS = [
    (Path("Dockerfile"), Path("Dockerfile")),
    (Path("config/nginx.conf"), Path("config/nginx.conf")),
    (Path("streamlit/requirements.txt"), Path("streamlit/requirements.txt")),
    (Path("data/dataset_consolide.csv"), Path("data/dataset_consolide.csv")),
    (Path("data/simulation/crop_yield.csv"), Path("data/simulation/crop_yield.csv")),
    (Path("main.py"), Path("main.py")),
    (Path("artifacts/models/p1_historical_pipeline.joblib"), Path("artifacts/models/p1_historical_pipeline.joblib")),
    (Path("artifacts/models/p1_historical_metadata.json"), Path("artifacts/models/p1_historical_metadata.json")),
    (Path("artifacts/models/p23_simulation_pipeline.joblib"), Path("artifacts/models/p23_simulation_pipeline.joblib")),
    (Path("artifacts/models/p23_simulation_metadata.json"), Path("artifacts/models/p23_simulation_metadata.json")),
    (
        Path("artifacts/experiments/experience_1/dataset_consolide_historique_colonnes.csv"),
        Path("artifacts/experiments/experience_1/dataset_consolide_historique_colonnes.csv"),
    ),
]

OPTIONAL_PAYLOAD_FILE_SPECS = [
    (Path("agriculture.png"), Path("agriculture.png")),
]

SPACE_README_CONTENT = """---
title: Rendement Agricole
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8501
tags:
- streamlit
- agriculture
pinned: false
short_description: Démo Streamlit + FastAPI de rendement agricole
license: mit
---

# Rendement Agricole

Ce Space Docker expose une interface Streamlit connectée à une API FastAPI interne dans le même conteneur.

- UI Streamlit : port public `8501`
- API FastAPI : port interne `127.0.0.1:8000`
- logique servie : API finale `main.py` basee sur 2 modeles et 3 predictions combinees
"""


def _resolve_under(root: str | Path, relative_path: str | Path) -> Path:
    """Resout un chemin relatif a une racine de travail."""
    base_root = Path(root)
    raw_path = Path(relative_path)
    if raw_path.is_absolute():
        return raw_path
    return base_root / raw_path


def validate_deployment_artifacts(
    *,
    source_root: str | Path = PROJECT_ROOT,
) -> list[Path]:
    """Valide la presence des artefacts deployables indispensables.

    Args:
        source_root: Racine du depot a valider.

    Returns:
        list[Path]: Liste des artefacts resolus et verifies.
    """
    resolved_root = Path(source_root)
    resolved_paths = [_resolve_under(resolved_root, path) for path in DEPLOYMENT_REQUIRED_ARTIFACTS]
    missing_paths = [path for path in resolved_paths if not path.exists()]
    if missing_paths:
        formatted = ", ".join(str(path.relative_to(resolved_root)) for path in missing_paths)
        raise FileNotFoundError(
            "Missing deployment artifact in repository checkout: "
            f"{formatted}. Commit this file or regenerate it before rerunning the workflow."
        )
    return resolved_paths


def build_space_payload(
    *,
    source_root: str | Path = PROJECT_ROOT,
    output_dir: str | Path = ".hf_space_build",
) -> Path:
    """Construit le payload Docker envoye sur Hugging Face Space.

    Args:
        source_root: Racine du depot source.
        output_dir: Dossier de sortie du payload.

    Returns:
        Path: Repertoire final du payload.
    """
    resolved_root = Path(source_root)
    resolved_output_dir = _resolve_under(resolved_root, output_dir)

    if resolved_output_dir.exists():
        shutil.rmtree(resolved_output_dir)

    for directory in PAYLOAD_DIRECTORIES:
        (resolved_output_dir / directory).mkdir(parents=True, exist_ok=True)

    for source_dir, target_dir in PAYLOAD_DIRECTORY_SPECS:
        shutil.copytree(
            _resolve_under(resolved_root, source_dir),
            resolved_output_dir / target_dir,
            dirs_exist_ok=True,
        )

    for source_file, target_file in PAYLOAD_FILE_SPECS:
        destination = resolved_output_dir / target_file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_resolve_under(resolved_root, source_file), destination)

    for source_file, target_file in OPTIONAL_PAYLOAD_FILE_SPECS:
        resolved_source_file = _resolve_under(resolved_root, source_file)
        if resolved_source_file.exists():
            destination = resolved_output_dir / target_file
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved_source_file, destination)

    (resolved_output_dir / "README.md").write_text(SPACE_README_CONTENT, encoding="utf-8")
    return resolved_output_dir


def parse_args() -> argparse.Namespace:
    """Construit l'interface CLI du script."""
    parser = argparse.ArgumentParser(
        description="Validate deployment artifacts and build the Hugging Face Space payload.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate deployable artifacts.")
    validate_parser.add_argument(
        "--source-root",
        default=str(PROJECT_ROOT),
        help="Repository root to validate.",
    )

    build_parser = subparsers.add_parser("build", help="Build the Hugging Face payload.")
    build_parser.add_argument(
        "--source-root",
        default=str(PROJECT_ROOT),
        help="Repository root used as payload source.",
    )
    build_parser.add_argument(
        "--output-dir",
        default=".hf_space_build",
        help="Output directory used for the generated payload.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute la validation ou la construction du payload depuis la CLI."""
    args = parse_args()
    if args.command == "validate":
        validate_deployment_artifacts(source_root=args.source_root)
        print("[deploy] Deployment artifacts validated")
        return

    validate_deployment_artifacts(source_root=args.source_root)
    payload_dir = build_space_payload(
        source_root=args.source_root,
        output_dir=args.output_dir,
    )
    print(f"[deploy] Space payload built at {payload_dir}")


if __name__ == "__main__":
    main()
