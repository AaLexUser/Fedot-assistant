from fedot.core.pipelines.pipeline import Pipeline
from fedot_ind.api.utils.api_init import ApiManager
from fedot_ind.core.repository.config_repository import DEFAULT_TSF_API_CONFIG

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
    CLASSIFICATION_PROBA_EVAL_METRIC, TIME_SERIES, SYMMETRIC_MEAN_ABSOLUTE_PERCENTAGE_ERROR
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


class FedotIndustrialTimeSeriesPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: Optional[FedotIndustrial] = None

    def fit(self, task: PredictionTask, time_limit: Optional[float] = None) -> "FedotIndustrialTimeSeriesPredictor":
        eval_metric = task.eval_metric
        self.problem_type = task.problem_type

        predictor_init_kwargs = {
            "problem": PROBLEM_TO_FEDOT_IND[self.problem_type],
            "timeout": time_limit,
            "metric": METRICS_TO_FEDOT_IND[eval_metric],
            **unpack_omega_config(self.config.predictor_init_kwargs)
        }
        predictor_fit_kwargs = self.config.predictor_fit_kwargs

        predictor_init_kwargs = DEFAULT_TSF_API_CONFIG
        # TODO: convert native configs to FI api config

        logger.info("Fitting FedotIndustrial TimeseriesPredictor")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")

        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
        }
        self.predictor = FedotIndustrial(**predictor_init_kwargs)
        # TODO: ensure data is passed as tuple(np.ndarray, np.ndarray)
        self.predictor.fit((task.train_data.values, task.label_column))
        self.predictor.shutdown()

        self.metadata['graph_structure'] = graph_structure(self.get_current_pipeline(self.predictor.manager))
        return self

    def predict(self, task: PredictionTask) -> TabularDataset:
        # TODO: ensure data is passed as tuple(np.ndarray, np.ndarray)
        if task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC and self.problem_type in [
            BINARY,
            MULTICLASS
        ]:
            return self.predictor.predict_proba(
                task.test_data
            )
        else:
            return self.predictor.predict(task.test_data)

    @staticmethod
    def get_current_pipeline(manager: ApiManager) -> Pipeline:
        if manager.condition_check.solver_is_fedot_class(manager.solver):
            return manager.solver.current_pipeline
        return manager.solver
