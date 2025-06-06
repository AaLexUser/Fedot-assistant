import numpy as np
from fedot.core.pipelines.pipeline import Pipeline
from fedot_ind.api.utils.api_init import ApiManager
from fedot_ind.api.utils.checkers_collections import ApiConfigCheck
from fedot_ind.core.repository.config_repository import (
    DEFAULT_TSF_API_CONFIG,
    DEFAULT_CLF_API_CONFIG,
    DEFAULT_REG_API_CONFIG
)

from .base import Predictor
from typing import Any, Dict, Optional
from collections import defaultdict
from fedot_ind.api.main import FedotIndustrial
from ..task import PredictionTask
from ..utils import unpack_omega_config
from golem.core.dag.graph_utils import graph_structure
from fedotllm.tabular import TabularDataset
import logging

from ..constants import (
    ROC_AUC,
    LOG_LOSS,
    ACCURACY,
    F1,
    ROOT_MEAN_SQUARED_ERROR,
    MEAN_SQUARED_ERROR,
    MEAN_ABSOLUTE_ERROR,
    R2,
    BINARY,
    MULTICLASS,
    REGRESSION,
    CLASSIFICATION_PROBA_EVAL_METRIC,
    TIME_SERIES,
    SYMMETRIC_MEAN_ABSOLUTE_PERCENTAGE_ERROR,
)

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
    SYMMETRIC_MEAN_ABSOLUTE_PERCENTAGE_ERROR: "smape"
}

PROBLEM_TO_FEDOT_IND = {
    BINARY: "classification",
    MULTICLASS: "classification",
    REGRESSION: "regression",
    TIME_SERIES: "ts_forecasting"
}

PROBLEM_TO_API_CONFIG = {
    BINARY: DEFAULT_CLF_API_CONFIG,
    MULTICLASS: DEFAULT_CLF_API_CONFIG,
    REGRESSION: DEFAULT_REG_API_CONFIG,
    TIME_SERIES: DEFAULT_TSF_API_CONFIG
}


class FedotIndustrialTimeSeriesPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: Optional[FedotIndustrial] = None
        self.problem_type: Optional[str] = None
        self.eval_metric: Optional[str] = None

    def fit(self, task: PredictionTask, time_limit: Optional[float] = None) -> "FedotIndustrialTimeSeriesPredictor":
        self.eval_metric = task.eval_metric
        self.problem_type = task.problem_type

        predictor_init_kwargs = {
            "task": PROBLEM_TO_FEDOT_IND[self.problem_type],
            "problem": PROBLEM_TO_FEDOT_IND[self.problem_type],
            "timeout": time_limit,
            "metric": METRICS_TO_FEDOT_IND[self.eval_metric],
            "quality_loss": METRICS_TO_FEDOT_IND[self.eval_metric],
            "forecast_length": task.forecast_horizon,
            **unpack_omega_config(self.config.predictor_init_kwargs)
        }

        default_config = PROBLEM_TO_API_CONFIG[self.problem_type]
        predictor_init_kwargs = ApiConfigCheck().update_config_with_kwargs(default_config, **predictor_init_kwargs)

        logger.info("Fitting FedotIndustrial TimeseriesPredictor")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")
        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
        }

        input_data = self.prepare_industrial_data(task, is_for_forecast=False)
        self.predictor = FedotIndustrial(**predictor_init_kwargs)
        self.predictor.fit(input_data)
        self.predictor.shutdown()

        self.metadata['graph_structure'] = graph_structure(self.get_current_pipeline(self.predictor.manager))
        return self

    def predict(self, task: PredictionTask) -> TabularDataset:
        input_data = self.prepare_industrial_data(task, is_for_forecast=True)
        if task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC and self.problem_type in [
            BINARY,
            MULTICLASS
        ]:
            return TabularDataset(self.predictor.predict_proba(input_data))
        return TabularDataset(self.predictor.predict(input_data))

    def save_artifacts(self, path: str) -> None:
        self.get_current_pipeline(self.predictor.manager).save(path)

    @staticmethod
    def get_current_pipeline(manager: ApiManager) -> Pipeline:
        if manager.condition_check.solver_is_fedot_class(manager.solver):
            return manager.solver.current_pipeline
        return manager.solver

    @staticmethod
    def prepare_industrial_data(task: PredictionTask,
                                is_for_forecast: Optional[bool] = False) -> tuple[np.ndarray, np.ndarray]:
        data = task.test_data if is_for_forecast else task.train_data
        series = data.to_numpy()
        if is_for_forecast:
            return series, series
        return series, series[-task.forecast_horizon:]
