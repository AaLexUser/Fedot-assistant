from .base import Predictor
from typing import Any, Dict, Optional
from collections import defaultdict
from fedot.api.main import Fedot
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
    CLASSIFICATION_PROBA_EVAL_METRIC
)

logger = logging.getLogger(__name__)

METRICS_TO_FEDOT = {
    ROC_AUC: "roc_auc",
    LOG_LOSS: "neg_log_loss",
    ACCURACY: "accuracy",
    F1: "f1",
    ROOT_MEAN_SQUARED_ERROR: "rmse",
    MEAN_SQUARED_ERROR: "mse",
    MEAN_ABSOLUTE_ERROR: "mae",
    R2: "r2"
}

PROBLEM_TO_FEDOT = {
    BINARY: "classification",
    MULTICLASS: "classification",
    REGRESSION: "regression"
}

class FedotTabularPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: Fedot = None
        self.problem_type: str = None
    
    def fit(self, task: PredictionTask, time_limit: Optional[float] = None) -> "FedotTabularPredictor":
        eval_metric = task.eval_metric
        self.problem_type = task.problem_type
        
        predictor_init_kwargs = {
            "problem": PROBLEM_TO_FEDOT[task.problem_type],
            "timeout": time_limit,
            "metric": METRICS_TO_FEDOT[eval_metric], 
            **unpack_omega_config(self.config.predictor_init_kwargs)  
        }
        
        predictor_fit_kwargs = self.config.predictor_fit_kwargs
        
        logger.info("Fitting Fedot TabularPredictor")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")
        logger.info(f"predictor_fit_kwargs: {predictor_fit_kwargs}")
        
        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
            "predictor_fit_kwargs": predictor_fit_kwargs
        }
        
        self.predictor = Fedot(**predictor_init_kwargs).fit(
            task.train_data, task.label_column, **unpack_omega_config(predictor_fit_kwargs)
        )
        
        self.metadata['graph_structure'] = graph_structure(self.predictor.current_pipeline)
        return self
    
    def predict(self, task: PredictionTask) -> TabularDataset:
        if task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC and self.problem_type in [
            BINARY,
            MULTICLASS
        ]:
            return self.predictor.predict_proba(
                task.test_data
            )
        else:
            return self.predictor.predict(task.test_data)
        