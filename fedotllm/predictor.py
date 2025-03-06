"""Predictors solve tabular prediction tasks"""
from collections import defaultdict
from typing import Any, Dict, Optional

from autogluon.tabular import TabularDataset, TabularPredictor

from .task import TabularPredictionTask


class Predictor:
    def fit(self, task: TabularPredictionTask, time_limit: Optional[float] = None) -> "Predictor":
        return self
    
    def predict(self, task: TabularPredictionTask) -> Any:
        raise NotImplementedError
    
    def fit_predict(self, task: TabularPredictionTask) -> Any:
        return self.fit(task).predict(task)
    
class AutogluonTabularPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.tabular_predictor: TabularPredictor = None