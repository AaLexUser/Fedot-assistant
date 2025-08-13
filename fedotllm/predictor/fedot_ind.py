import logging
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import psutil
import torch
from fedot.core.pipelines.pipeline import Pipeline
from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.api_init import ApiManager
from fedot_ind.api.utils.checkers_collections import ApiConfigCheck
from fedot_ind.core.repository.config_repository import (
    DEFAULT_CLF_API_CONFIG,
    DEFAULT_REG_API_CONFIG,
    DEFAULT_TSF_API_CONFIG,
)
from golem.core.dag.graph_utils import graph_structure

from fedotllm.tabular import TabularDataset

from ..constants import (
    ACCURACY,
    BINARY,
    CLASSIFICATION_PROBA_EVAL_METRIC,
    F1,
    LOG_LOSS,
    MEAN_ABSOLUTE_ERROR,
    MEAN_SQUARED_ERROR,
    MULTICLASS,
    R2,
    REGRESSION,
    ROC_AUC,
    ROOT_MEAN_SQUARED_ERROR,
    SYMMETRIC_MEAN_ABSOLUTE_PERCENTAGE_ERROR,
    TIME_SERIES,
)
from ..task import PredictionTask
from ..utils import unpack_omega_config
from .base import Predictor

logger = logging.getLogger(__name__)

METRICS_TO_FEDOT_IND = {
    ROC_AUC: "roc_auc",
    LOG_LOSS: "neg_log_loss",
    ACCURACY: "accuracy",
    F1: "f1",
    ROOT_MEAN_SQUARED_ERROR: "rmse",
    MEAN_SQUARED_ERROR: "mse",
    MEAN_ABSOLUTE_ERROR: "mae",
    R2: "r2",
    SYMMETRIC_MEAN_ABSOLUTE_PERCENTAGE_ERROR: "smape",
}

PROBLEM_TO_FEDOT_IND = {
    BINARY: "classification",
    MULTICLASS: "classification",
    REGRESSION: "regression",
    TIME_SERIES: "ts_forecasting",
}

PROBLEM_TO_API_CONFIG = {
    BINARY: DEFAULT_CLF_API_CONFIG,
    MULTICLASS: DEFAULT_CLF_API_CONFIG,
    REGRESSION: DEFAULT_REG_API_CONFIG,
    TIME_SERIES: DEFAULT_TSF_API_CONFIG,
}


class FedotIndustrialTabularPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: Optional[FedotIndustrial] = None
        self.problem_type: Optional[str] = None
        self.eval_metric: Optional[str] = None

    def fit(
        self, task: PredictionTask, time_limit: Optional[float] = None
    ) -> "FedotIndustrialTabularPredictor":
        self.eval_metric = task.eval_metric
        self.problem_type = task.problem_type

        predictor_init_kwargs = self.get_init_kwargs(task, time_limit)

        logger.info(f"Fitting {self.__class__.__name__}")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")
        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
        }

        input_data = self.prepare_industrial_data(task, is_for_predict=False)
        self.predictor = FedotIndustrial(**predictor_init_kwargs)
        self.predictor.fit(input_data)
        self.predictor.shutdown()

        self.metadata["graph_structure"] = graph_structure(
            self.get_current_pipeline(self.predictor.manager)
        )
        return self

    def predict(self, task: PredictionTask) -> TabularDataset:
        input_data = self.prepare_industrial_data(task, is_for_predict=True)
        if (
            task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC
            and self.problem_type in [BINARY, MULTICLASS]
        ):
            predictions = self.predictor.predict_proba(input_data)
        else:
            predictions = self.predictor.predict(input_data)

        return TabularDataset(
            predictions, columns=[task.label_column], index=task.test_data.index
        )

    def save_artifacts(self, path: str, task: PredictionTask) -> None:
        self.get_current_pipeline(self.predictor.manager).save(path)

    @staticmethod
    def get_current_pipeline(manager: ApiManager) -> Pipeline:
        if manager.condition_check.solver_is_fedot_class(manager.solver):
            return manager.solver.current_pipeline
        return manager.solver

    def prepare_industrial_data(
        self, task: PredictionTask, is_for_predict: Optional[bool] = False
    ) -> tuple[np.ndarray, np.ndarray]:
        data = task.test_data if is_for_predict else task.train_data
        series = data.to_numpy().squeeze()
        return series, series

    @staticmethod
    def configure_dask_cluster():
        logical_cores = psutil.cpu_count()
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0

        if cuda_available:
            # GPU workload: 1 worker per GPU, balance threads
            n_workers = min(gpu_count, 4)  # Cap at 4 workers to avoid over-subscription
            threads_per_worker = max(logical_cores // n_workers, 1)
        else:
            n_workers = max(logical_cores // 2, 1)  # Max parallelism for small machines
            threads_per_worker = 2  # Avoid GIL for CPU-bound tasks

        return {
            "n_workers": n_workers,
            "threads_per_worker": threads_per_worker,
            "memory_limit": "auto",
        }

    def get_init_kwargs(
        self, task: PredictionTask, time_limit: Optional[float] = None
    ) -> dict:
        predictor_init_kwargs = {
            "task": PROBLEM_TO_FEDOT_IND[self.problem_type],
            "problem": PROBLEM_TO_FEDOT_IND[self.problem_type],
            "timeout": time_limit,
            "metric": METRICS_TO_FEDOT_IND[self.eval_metric],
            "quality_loss": METRICS_TO_FEDOT_IND[self.eval_metric],
            "tuning_timeout": time_limit,
            **self.configure_dask_cluster(),
            **unpack_omega_config(self.config.predictor_init_kwargs),
        }

        default_config = PROBLEM_TO_API_CONFIG[self.problem_type]

        predictor_init_kwargs = ApiConfigCheck().update_config_with_kwargs(
            default_config, **predictor_init_kwargs
        )
        return predictor_init_kwargs


class FedotIndustrialTimeSeriesPredictor(FedotIndustrialTabularPredictor):
    def __init__(self, config: Any):
        super().__init__(config)
        self.predictor: Optional[FedotIndustrialTimeSeriesPredictor] = None
        self.historical_data: Optional[np.ndarray] = None

    def get_init_kwargs(
        self, task: PredictionTask, time_limit: Optional[float] = None
    ) -> dict:
        predictor_init_kwargs = {
            "task": "ts_forecasting",
            "problem": "ts_forecasting",
            "timeout": time_limit,
            "metric": METRICS_TO_FEDOT_IND[self.eval_metric],
            "quality_loss": METRICS_TO_FEDOT_IND[self.eval_metric],
            "forecast_length": task.forecast_horizon,
            "tuning_timeout": time_limit,
            **self.configure_dask_cluster(),
            **unpack_omega_config(self.config.predictor_init_kwargs),
        }

        default_config = DEFAULT_TSF_API_CONFIG
        predictor_init_kwargs = ApiConfigCheck().update_config_with_kwargs(
            default_config, **predictor_init_kwargs
        )
        return predictor_init_kwargs

    def predict(self, task: PredictionTask) -> TabularDataset:
        input_data = self.prepare_industrial_data(task, is_for_predict=True)
        predictions = self.predictor.predict(input_data)
        return TabularDataset(
            predictions, columns=[task.label_column], index=task.test_data.index
        )

    def prepare_industrial_data(
        self, task: PredictionTask, is_for_predict: Optional[bool] = False
    ) -> tuple[np.ndarray, np.ndarray]:
        data = task.test_data if is_for_predict else task.train_data

        timestamp_col = task.timestamp_column
        if timestamp_col and timestamp_col in data.columns:
            # Convert to datetime and set as index
            data[timestamp_col] = pd.to_datetime(data[timestamp_col])
            data = data.set_index(timestamp_col)
        series = data.to_numpy().squeeze()

        if is_for_predict:
            return self.historical_data, series

        self.historical_data = series
        return series, series[-task.forecast_horizon :]
