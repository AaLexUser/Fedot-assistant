import os
import tempfile
from pathlib import Path

from platformdirs import user_data_dir


def _resolve_path(env_var: str, default: Path) -> Path:
    raw_value = os.getenv(env_var)
    if raw_value:
        return Path(raw_value).expanduser()
    return default


def get_package_dir() -> Path:
    return Path(__file__).resolve().parent


def get_data_dir() -> Path:
    """Base directory for persistent UI data such as cached sample datasets."""
    return _resolve_path("FEDOTLLM_DATA_DIR", Path(user_data_dir("fedotllm")))


def get_ui_data_dir() -> Path:
    default = Path(tempfile.gettempdir()) / "fedotllm" / "user_data"
    if os.getenv("FEDOTLLM_DATA_DIR"):
        default = get_data_dir() / "user_data"
    return _resolve_path("FEDOTLLM_UI_DATA_DIR", default)


def get_ui_sample_dataset_dir() -> Path:
    return _resolve_path(
        "FEDOTLLM_UI_SAMPLE_DATASET_DIR", get_data_dir() / "sample_dataset"
    )


def get_ui_sample_dataset_archive_path() -> Path:
    return _resolve_path(
        "FEDOTLLM_UI_SAMPLE_DATASET_ARCHIVE", get_data_dir() / "sample_data.zip"
    )


def get_artifacts_dir() -> Path:
    return _resolve_path(
        "FEDOTLLM_ARTIFACTS_DIR",
        Path(tempfile.gettempdir()) / "fedotllm" / "artifacts",
    )
