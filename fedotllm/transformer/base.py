from ..task import PredictionTask


class TransformTimeoutError(TimeoutError):
    pass


class BaseTransformer:
    def __init__(self, *args, **kwargs):
        pass

    @property
    def name(self):
        return self.__class__.__name__

    def transform(self, task: PredictionTask, *args, **kwargs) -> PredictionTask:
        return task

    def fit(self, task: PredictionTask, *args, **kwargs) -> "BaseTransformer":
        return self

    def fit_transform(self, task: PredictionTask, *args, **kwargs) -> PredictionTask:
        return self.fit(task).transform(task)

    def __call__(self, task: PredictionTask, *args, **kwargs) -> PredictionTask:
        return self.transform(task)
