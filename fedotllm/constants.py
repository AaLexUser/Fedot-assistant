from copy import deepcopy

from fedotllm.runtime_paths import (
    get_ui_data_dir,
    get_ui_sample_dataset_archive_path,
    get_ui_sample_dataset_dir,
)

# Task Inference
NO_FILE_IDENTIFIED = "NO_FILE_IDENTIFIED"
NO_ID_COLUMN_IDENTIFIED = "NO_ID_COLUMN_IDENTIFIED"
NO_TIMESTAMP_COLUMN_IDENTIFIED = "NO_TIMESTAMP_COLUMN_IDENTIFIED"

# Supported File Types
TEXT_EXTENSIONS = [".txt", ".md", ".json", ".yml", ".yaml", ".xml", ".log"]

CSV_SUFFIXES = [".csv"]
PARQUET_SUFFIXES = [".parquet", ".pq"]
EXCEL_SUFFIXES = [".xlsx", ".xls"]
DATA_EXTENSIONS = CSV_SUFFIXES + PARQUET_SUFFIXES + EXCEL_SUFFIXES

# Data types and files
TRAIN = "train"
TEST = "test"
OUTPUT = "output"
STATIC_FEATURES = "static_features"
DESCRIPTION = "description"

# Task types
TABULAR = "tabular"
MULTIMODAL = "multimodal"
TIME_SERIES = "time_series"
TASK_TYPES = [TABULAR, MULTIMODAL, TIME_SERIES]

# Problem types
BINARY = "binary"
MULTICLASS = "multiclass"
REGRESSION = "regression"
TIME_SERIES = "time_series"
PROBLEM_TYPES = [BINARY, MULTICLASS, REGRESSION, TIME_SERIES]
CLASSIFICATION_PROBLEM_TYPES = [BINARY, MULTICLASS]

# Presets/Configs
CONFIGS = "configs"
MEDIUM_QUALITY = "medium_quality"
HIGH_QUALITY = "high_quality"
BEST_QUALITY = "best_quality"
DEFAULT_QUALITY = BEST_QUALITY
PRESETS = [MEDIUM_QUALITY, HIGH_QUALITY, BEST_QUALITY]

# Metrics
ROC_AUC = "roc_auc"
LOG_LOSS = "log_loss"
ACCURACY = "accuracy"
F1 = "f1"
QUADRARIC_KAPPA = "quadratic_kappa"
BALANCED_ACCURACY = "balanced_accuracy"
ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
MEAN_SQUARED_ERROR = "mean_squared_error"
MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
R2 = "r2"
ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR = "root_mean_squared_logarithmic_error"
SYMMETRIC_MEAN_ABSOLUTE_PERCENTAGE_ERROR = "symmetric_mean_absolute_percentage_error"

CLASSIFICATION_PROBA_EVAL_METRIC = [ROC_AUC, LOG_LOSS, F1]

METRICS_DESCRIPTION = {
    ROC_AUC: "Area under the receiver operating characteristics (ROC) curve",
    LOG_LOSS: "Log loss, also known as logarithmic loss",
    ACCURACY: "Accuracy",
    F1: "F1 score",
    QUADRARIC_KAPPA: "Quadratic kappa, i.e., the Cohen kappa metric",
    BALANCED_ACCURACY: "Balanced accuracy, i.e., the arithmetic mean of sensitivity and specificity",
    ROOT_MEAN_SQUARED_ERROR: "Root mean squared error (RMSE)",
    MEAN_SQUARED_ERROR: "Mean squared error (MSE)",
    MEAN_ABSOLUTE_ERROR: "Mean absolute_error (MAE)",
    R2: "R-squared",
    ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR: "Root mean squared logarithmic error (RMSLE)",
    SYMMETRIC_MEAN_ABSOLUTE_PERCENTAGE_ERROR: "Symmetric mean absolute percentage error (SMAPE)",
}

METRICS_BY_PROBLEM_TYPE = {
    BINARY: [ROC_AUC, LOG_LOSS, ACCURACY, F1, QUADRARIC_KAPPA, BALANCED_ACCURACY],
    MULTICLASS: [ROC_AUC, LOG_LOSS, ACCURACY, F1, QUADRARIC_KAPPA, BALANCED_ACCURACY],
    REGRESSION: [
        ROOT_MEAN_SQUARED_ERROR,
        MEAN_SQUARED_ERROR,
        MEAN_ABSOLUTE_ERROR,
        R2,
        ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR,
    ],
    TIME_SERIES: [
        ROOT_MEAN_SQUARED_ERROR,
        MEAN_SQUARED_ERROR,
        MEAN_ABSOLUTE_ERROR,
        ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR,
        SYMMETRIC_MEAN_ABSOLUTE_PERCENTAGE_ERROR,
    ],
}

PREFERED_METRIC_BY_PROBLEM_TYPE = {
    BINARY: ROC_AUC,
    MULTICLASS: ROC_AUC,
    REGRESSION: ROOT_MEAN_SQUARED_ERROR,
    TIME_SERIES: ROOT_MEAN_SQUARED_ERROR,
}

DEFAULT_FORECAST_HORIZON = 1
NO_FORECAST_HORIZON_IDENTIFIED = "NO_FORECAST_HORIZON_IDENTIFIED"

WHITE_LIST_LLM = [
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "meta.llama3-1-405b-instruct-v1:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "gpt-4o-2024-08-06",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
]

#  The below constants are for UI
BASE_DATA_DIR = str(get_ui_data_dir())


# Preset configurations
PRESET_DEFAULT_CONFIG = {
    "Лучшее качество": {"time_limit": "4 ч", "feature_generation": False},
    "Высокое качество": {"time_limit": "10 мин", "feature_generation": False},
    "Среднее качество": {"time_limit": "3 мин", "feature_generation": False},
}
DEFAULT_PRESET = "Среднее качество"

DEFAULT_AUTOML_ENGINE = "fedot"

AUTOML_ENGINE_OPTIONS = ["fedot", "autogluon"]

PRESET_MAPPING = {
    "Лучшее качество": "best_quality",
    "Высокое качество": "high_quality",
    "Среднее качество": "medium_quality",
}
PRESET_OPTIONS = ["Лучшее качество", "Высокое качество", "Среднее качество"]

# Time limit configurations (in seconds)
TIME_LIMIT_MAPPING = {
    "3 мин": 180,
    "10 мин": 600,
    "30 мин": 1800,
    "1 ч": 3600,
    "2 ч": 7200,
    "4 ч": 14400,
}

DEFAULT_TIME_LIMIT = "3 мин"

TIME_LIMIT_OPTIONS = ["3 мин", "10 мин", "30 мин", "1 ч", "2 ч", "4 ч"]

# LLM configurations
LLM_MAPPING = {
    "Qwen3.5-27B": "qwen/qwen3.5-27b",
    "GPT 5 Nano": "openai/gpt-5-nano",
    "Qwen3.5-35B-A3B": "qwen/qwen3.5-35b-a3b",
    "GPT 5 Mini": "openai/gpt-5-mini",
    "GLM 4.7 Flash": "z-ai/glm-4.7-flash",
    "GLM 5": "z-ai/glm-5",
    "Kimi K2.5": "moonshotai/kimi-k2.5",
}

LLM_OPTIONS = ["Qwen3.5-27B", "Qwen3.5-35B-A3B", "GPT 5 Nano", "GPT 5 Mini", "GLM 4.7 Flash", "GLM 5", "Kimi K2.5"]

# Provider configuration
BASE_URL_MAPPING = {
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
# Initial Session state
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

# Message to display different logging stage
# Keys are English log messages, values are progress percentages
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

# Stage Names
STAGE_TASK_UNDERSTANDING = "Анализ задачи"
STAGE_FEATURE_GENERATION = "Генерация признаков"
STAGE_MODEL_TRAINING = "Обучение модели"
STAGE_PREDICTION = "Предсказание"

# Log Messages
MSG_TASK_UNDERSTANDING = "Task understanding starts"
MSG_FEATURE_GENERATION = "Automatic feature generation starts"
MSG_MODEL_TRAINING = "Model training starts"
MSG_PREDICTION = "Prediction starts"

# Mapping
STAGE_MESSAGES = {
    MSG_TASK_UNDERSTANDING: STAGE_TASK_UNDERSTANDING,
    MSG_FEATURE_GENERATION: STAGE_FEATURE_GENERATION,
    MSG_MODEL_TRAINING: STAGE_MODEL_TRAINING,
    MSG_PREDICTION: STAGE_PREDICTION,
}
# DataSet Options
DATASET_OPTIONS = ["Пример датасета", "Загрузить свой"]

# Captions under DataSet Options
CAPTIONS = [
    "Запустить на готовом примере",
    "Загрузите train (обязательно), test (обязательно) и output (опционально)",
]

LOGO_PATH = str(get_ui_sample_dataset_dir().parent / "static" / "page_icon.png")
SUCCESS_MESSAGE = """
        🎉 Готово! Если инструмент оказался полезен, будем рады звезде на [GitHub](https://github.com/AaLexUser/Fedot-assistant) ⭐
        """
S3_URL = "https://drive.google.com/uc?export=download&id=1N4GNZ69yTEIUT-XDmXrx35CTVfwecMwh"
LOCAL_ZIP_PATH = str(get_ui_sample_dataset_archive_path())
EXTRACT_DIR = str(get_ui_sample_dataset_dir())
IGNORED_MESSAGES = [
    "Не удалось определить файл с примером решения, установлено значение None.",
    "Слишком много запросов, подождите перед повторной попыткой",
]
