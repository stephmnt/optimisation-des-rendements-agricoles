from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "project_paths.yaml"


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_preparation_directories(config: dict[str, object]) -> dict[str, object]:
    artifacts_dir = config["ARTIFACTS_DIR"]
    pca_artifacts_dir = config["PCA_ARTIFACTS_DIR"]
    dataset_path = config["DATASET_PATH"]

    if isinstance(artifacts_dir, Path):
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(pca_artifacts_dir, Path):
        pca_artifacts_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(dataset_path, Path):
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

    return config


def load_preparation_config(
    config_path: Path | None = None,
    *,
    ensure_dirs: bool = False,
) -> dict[str, object]:
    path = config_path or DEFAULT_CONFIG_PATH
    raw_config = yaml.safe_load(path.read_text())
    preparation = raw_config["preparation"]

    config: dict[str, object] = {}
    for key, value in preparation["paths"].items():
        config[key] = _resolve_path(value)

    for key, value in preparation["parameters"].items():
        config[key] = value

    if ensure_dirs:
        ensure_preparation_directories(config)

    return config
