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
