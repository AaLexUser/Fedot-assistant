"""Streamlit UI defaults, labels, and session state — not used by the core fedotllm library."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from fedotllm.runtime_paths import (
    get_ui_data_dir,
    get_ui_sample_dataset_archive_path,
    get_ui_sample_dataset_dir,
)

BASE_DATA_DIR = str(get_ui_data_dir())

# Preset configurations (display labels → defaults shown in the UI)
PRESET_DEFAULT_CONFIG = {
    "Лучшее качество": {"time_limit": "4 ч", "feature_generation": False},
    "Высокое качество": {"time_limit": "10 мин", "feature_generation": False},
    "Среднее качество": {"time_limit": "3 мин", "feature_generation": False},
}
DEFAULT_PRESET = "Среднее качество"

DEFAULT_AUTOML_ENGINE = "fedot"

PRESET_MAPPING = {
    "Лучшее качество": "best_quality",
    "Высокое качество": "high_quality",
    "Среднее качество": "medium_quality",
}
PRESET_OPTIONS = ["Лучшее качество", "Высокое качество", "Среднее качество"]

DEFAULT_TIME_LIMIT = "5 мин"

TIME_LIMIT_OPTIONS = ["5 мин", "10 мин", "30 мин", "1 ч", "2 ч", "4 ч"]

# LLM configurations (display name → LiteLLM / router model id)
LLM_MAPPING = {
    "Gemma4-31B": "models/gemma-4-31b-it",
    "Qwen3.5-27B": "qwen/qwen3.5-27b",
    "GPT 5 Nano": "openai/gpt-5-nano",
    "Qwen3.5-35B-A3B": "qwen/qwen3.5-35b-a3b",
    "GPT 5 Mini": "openai/gpt-5-mini",
    "GLM 4.7 Flash": "z-ai/glm-4.7-flash",
    "GLM 5": "z-ai/glm-5",
    "Kimi K2.5": "moonshotai/kimi-k2.5",
}

LLM_OPTIONS = [
    "Gemma4-31B",
    "GLM 4.7 Flash",
    "GPT 5 Nano",
    "GPT 5 Mini",
    "Qwen3.5-27B",
    "Qwen3.5-35B-A3B",
    "GLM 5",
    "Kimi K2.5",
]

# Provider configuration (display name → API base URL)
BASE_URL_MAPPING = {
    "Gemma4-31B": "http://localhost:8787/v1",
    "Qwen3.5-27B": "https://openrouter.ai/api/v1",
    "GPT 5 Nano": "https://openrouter.ai/api/v1",
    "Qwen3.5-35B-A3B": "https://openrouter.ai/api/v1",
    "GPT 5 Mini": "https://openrouter.ai/api/v1",
    "GLM 4.7 Flash": "https://openrouter.ai/api/v1",
    "GLM 5": "https://openrouter.ai/api/v1",
    "Kimi K2.5": "https://openrouter.ai/api/v1",
}

INITIAL_STAGE = {
    "Анализ задачи": [],
    "Генерация признаков": [],
    "Обучение модели": [],
    "Предсказание": [],
}

DEFAULT_SESSION_VALUES = {
    "config_overrides": [],
    "preset": DEFAULT_PRESET,
    "time_limit": DEFAULT_TIME_LIMIT,
    "automl_engine": "fedot",
    "llm": LLM_OPTIONS[0],
    "pid": None,
    "logs": "",
    "process": None,
    "clicked": False,
    "task_running": False,
    "output_file": None,
    "output_filename": None,
    "task_description": "",
    "sample_description": "",
    "return_code": None,
    "task_canceled": False,
    "uploaded_files": {},
    "sample_files": {},
    "selected_dataset": None,
    "sample_dataset_dir": None,
    "description_uploader_key": 0,
    "sample_dataset_selector": None,
    "current_stage": None,
    "feature_generation": False,
    "stage_status": {},
    "show_remaining_time": False,
    "model_path": None,
    "elapsed_time": 0,
    "progress_bar": None,
    "increment": 2,
    "zip_path": None,
    "stage_container": deepcopy(INITIAL_STAGE),
    "start_time": None,
    "remaining_time": 0,
    "start_model_train_time": 0,
}

# Keys are English substrings matched against log lines; values are progress fractions
STATUS_BAR_STAGE = {
    "task loaded": 10,
    "model training starts": 25,
    "model training complete": 80,
    "prediction starts": 90,
}

STAGE_TO_STATUS_BAR = {
    "task loaded": "Начало анализа задачи",
    "model training starts": "Начало обучения модели",
    "model training complete": "Обучение модели завершено",
    "prediction starts": "Начало предсказания",
}

STAGE_COMPLETE_SIGNAL = [
    "Анализ задачи завершен",
    "Генерация признаков завершена",
    "Обучение модели завершено",
    "Предсказание завершено",
]

STAGE_TASK_UNDERSTANDING = "Анализ задачи"
STAGE_FEATURE_GENERATION = "Генерация признаков"
STAGE_MODEL_TRAINING = "Обучение модели"
STAGE_PREDICTION = "Предсказание"

MSG_TASK_UNDERSTANDING = "Task understanding starts"
MSG_FEATURE_GENERATION = "Automatic feature generation starts"
MSG_MODEL_TRAINING = "Model training starts"
MSG_PREDICTION = "Prediction starts"

STAGE_MESSAGES = {
    MSG_TASK_UNDERSTANDING: STAGE_TASK_UNDERSTANDING,
    MSG_FEATURE_GENERATION: STAGE_FEATURE_GENERATION,
    MSG_MODEL_TRAINING: STAGE_MODEL_TRAINING,
    MSG_PREDICTION: STAGE_PREDICTION,
}

DATASET_OPTIONS = ["Пример датасета", "Загрузить свой"]

CAPTIONS = [
    "Запустить на готовом примере",
    "Загрузите train (обязательно), test (обязательно) и output (опционально)",
]

LOGO_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "static" / "page_icon.png"
)
SUCCESS_MESSAGE = """
        🎉 Готово! Если инструмент оказался полезен, будем рады звезде на [GitHub](https://github.com/AaLexUser/Fedot-assistant) ⭐
        """
S3_URL = (
    "https://drive.google.com/uc?export=download&id=1N4GNZ69yTEIUT-XDmXrx35CTVfwecMwh"
)
LOCAL_ZIP_PATH = str(get_ui_sample_dataset_archive_path())
EXTRACT_DIR = str(get_ui_sample_dataset_dir())
IGNORED_MESSAGES = [
    "Не удалось определить файл с примером решения, установлено значение None.",
    "Слишком много запросов, подождите перед повторной попыткой",
]
