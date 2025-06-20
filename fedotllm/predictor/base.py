from ..task import PredictionTask
from typing import Optional, Any


class Predictor:
    def fit(
        self, task: PredictionTask, time_limit: Optional[float] = None
    ) -> "Predictor":
        return self

    def predict(self, task: PredictionTask) -> Any:
        raise NotImplementedError

    def fit_predict(self, task: PredictionTask) -> Any:
        return self.fit(task).predict(task)

    def save_artifacts(self, path: str) -> None:
        raise NotImplementedError
