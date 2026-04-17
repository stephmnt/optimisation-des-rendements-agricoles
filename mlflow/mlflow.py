from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BACKEND_URI = f"sqlite:///{(PROJECT_ROOT / 'artifacts' / 'mlflow.db').resolve()}"
DEFAULT_ARTIFACTS_DIR = (PROJECT_ROOT / "artifacts" / "mlruns").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lance l'interface graphique MLflow pour ce projet."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Hôte d'écoute de l'interface MLflow.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port d'écoute de l'interface MLflow.",
    )
    parser.add_argument(
        "--backend-store-uri",
        default=DEFAULT_BACKEND_URI,
        help="URI du backend store MLflow. Par défaut : base SQLite locale du projet.",
    )
    parser.add_argument(
        "--default-artifact-root",
        default=DEFAULT_ARTIFACTS_DIR.as_uri(),
        help="Racine des artefacts MLflow. Par défaut : artifacts/mlruns.",
    )
    return parser.parse_args()


def ensure_backend_target(raw_value: str) -> str:
    if raw_value.startswith("sqlite:///"):
        db_path = Path(raw_value.removeprefix("sqlite:///"))
        if not db_path.is_absolute():
            db_path = (PROJECT_ROOT / db_path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"

    backend_dir = Path(raw_value)
    if not backend_dir.is_absolute():
        backend_dir = (PROJECT_ROOT / backend_dir).resolve()
    backend_dir.mkdir(parents=True, exist_ok=True)
    return str(backend_dir)


def ensure_artifact_root(raw_value: str) -> tuple[str, Path]:
    if raw_value.startswith("file://"):
        artifact_root = Path(raw_value.removeprefix("file://"))
    else:
        artifact_root = Path(raw_value)
        if not artifact_root.is_absolute():
            artifact_root = (PROJECT_ROOT / artifact_root).resolve()

    artifact_root.mkdir(parents=True, exist_ok=True)
    return artifact_root.as_uri(), artifact_root


def artifact_location_to_path(raw_value: str) -> Path:
    if raw_value.startswith("file://"):
        return Path(raw_value.removeprefix("file://")).resolve()
    return Path(raw_value).resolve()


def migrate_sqlite_artifact_locations(backend_uri: str, artifact_root: Path) -> None:
    if not backend_uri.startswith("sqlite:///"):
        return

    db_path = Path(backend_uri.removeprefix("sqlite:///")).resolve()
    if not db_path.exists():
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT experiment_id, name, artifact_location FROM experiments ORDER BY experiment_id")
    experiments = cur.fetchall()

    for experiment_id, experiment_name, artifact_location in experiments:
        target_dir = (artifact_root / experiment_name).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_location = str(target_dir)

        current_dir = artifact_location_to_path(artifact_location)
        if current_dir.exists() and current_dir != target_dir:
            for child in current_dir.iterdir():
                destination = target_dir / child.name
                if not destination.exists():
                    shutil.move(str(child), str(destination))
            if current_dir.exists() and current_dir.is_dir() and not any(current_dir.iterdir()):
                current_dir.rmdir()

        if artifact_location != target_location:
            cur.execute(
                "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
                (target_location, experiment_id),
            )
            current_prefix = str(current_dir)
            cur.execute(
                """
                UPDATE runs
                SET artifact_uri = REPLACE(artifact_uri, ?, ?)
                WHERE experiment_id = ? AND artifact_uri LIKE ?
                """,
                (current_prefix, target_location, experiment_id, f"{current_prefix}%"),
            )

    conn.commit()
    conn.close()


def main() -> None:
    args = parse_args()
    backend_uri = ensure_backend_target(args.backend_store_uri)
    default_artifact_root_uri, artifact_root_path = ensure_artifact_root(args.default_artifact_root)
    migrate_sqlite_artifact_locations(backend_uri, artifact_root_path)
    ui_url = f"http://{args.host}:{args.port}"

    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--backend-store-uri",
        backend_uri,
        "--default-artifact-root",
        default_artifact_root_uri,
    ]

    print(f"Projet : {PROJECT_ROOT}")
    print(f"Backend store : {backend_uri}")
    print(f"Racine artefacts : {artifact_root_path}")
    print(f"Interface MLflow : {ui_url}")
    print("Commande :", " ".join(command))

    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
