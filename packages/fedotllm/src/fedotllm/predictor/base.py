from typing import Any, Optional

from ..task import PredictionTask


class Predictor:
    def fit(
        self, task: PredictionTask, time_limit: Optional[float] = None
    ) -> "Predictor":
        return self

    def predict(self, task: PredictionTask) -> Any:
        raise NotImplementedError

    def fit_predict(self, task: PredictionTask) -> Any:
        return self.fit(task).predict(task)

    def save_artifacts(self, path: str, task: PredictionTask) -> None:
        raise NotImplementedError
