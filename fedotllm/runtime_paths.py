import os
from pathlib import Path


def _resolve_path(env_var: str, default: Path) -> Path:
    raw_value = os.getenv(env_var)
    if raw_value:
        return Path(raw_value).expanduser()
    return default


def get_package_dir() -> Path:
    return Path(__file__).resolve().parent


def get_ui_dir() -> Path:
    return get_package_dir() / "ui"


def get_ui_data_dir() -> Path:
    return _resolve_path("FEDOTLLM_UI_DATA_DIR", get_ui_dir() / "user_data")


def get_ui_sample_dataset_dir() -> Path:
    return _resolve_path(
        "FEDOTLLM_UI_SAMPLE_DATASET_DIR", get_ui_dir() / "sample_dataset"
    )


def get_ui_sample_dataset_archive_path() -> Path:
    return _resolve_path(
        "FEDOTLLM_UI_SAMPLE_DATASET_ARCHIVE", get_ui_dir() / "sample_data.zip"
    )
