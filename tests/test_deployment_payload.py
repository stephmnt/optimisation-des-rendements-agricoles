"""Tests du script de construction du payload Hugging Face."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.deployment_payload import build_space_payload, validate_deployment_artifacts


def _write_text(root: Path, relative_path: str, content: str = "x") -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_validate_deployment_artifacts_detects_missing_files(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    _write_text(repo_root, "artifacts/models/p1_historical_pipeline.joblib")
    _write_text(repo_root, "artifacts/models/p1_historical_metadata.json")
    _write_text(repo_root, "artifacts/models/p23_simulation_pipeline.joblib")
    _write_text(repo_root, "artifacts/models/p23_simulation_metadata.json")

    with pytest.raises(FileNotFoundError, match="dataset_consolide_historique_colonnes.csv"):
        validate_deployment_artifacts(source_root=repo_root)


def test_build_space_payload_copies_expected_runtime_files(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    _write_text(repo_root, "Dockerfile")
    _write_text(repo_root, "config/nginx.conf")
    _write_text(repo_root, "streamlit/requirements.txt")
    _write_text(repo_root, "data/dataset_consolide.csv")
    _write_text(repo_root, "data/simulation/crop_yield.csv")
    _write_text(repo_root, "main.py")
    _write_text(repo_root, "artifacts/models/p1_historical_pipeline.joblib")
    _write_text(repo_root, "artifacts/models/p1_historical_metadata.json")
    _write_text(repo_root, "artifacts/models/p23_simulation_pipeline.joblib")
    _write_text(repo_root, "artifacts/models/p23_simulation_metadata.json")
    _write_text(
        repo_root,
        "artifacts/experiments/experience_1/dataset_consolide_historique_colonnes.csv",
    )
    _write_text(repo_root, "scripts/example.py")
    _write_text(repo_root, "streamlit/src/streamlit_app.py")
    _write_text(repo_root, "streamlit/icones/icon.txt")

    payload_dir = build_space_payload(source_root=repo_root, output_dir=".payload")

    assert payload_dir == repo_root / ".payload"
    assert (payload_dir / "Dockerfile").exists()
    assert (payload_dir / "config/nginx.conf").exists()
    assert (payload_dir / "scripts/example.py").exists()
    assert (payload_dir / "streamlit/src/streamlit_app.py").exists()
    assert (payload_dir / "streamlit/icones/icon.txt").exists()
    assert (payload_dir / "artifacts/models/p1_historical_pipeline.joblib").exists()
    assert (payload_dir / "artifacts/models/p23_simulation_pipeline.joblib").exists()
    assert (payload_dir / "README.md").exists()
