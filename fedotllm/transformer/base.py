from ..task import TabularPredictionTask

class TransformTimeoutError(TimeoutError):
    pass

class BaseTransformer:
    def __init__(self, *args, **kwargs):
        pass
    
    @property
    def name(self):
        return self.__class__.__name__
    
    def transform(self, task: TabularPredictionTask, *args, **kwargs) -> TabularPredictionTask:
        return task
    
    def fit(self, task: TabularPredictionTask, *args, **kwargs) -> "BaseTransformer":
        return self
    
    def fit_transform(self, task: TabularPredictionTask, *args, **kwargs) -> TabularPredictionTask:
        return self.fit(task).transform(task)
    
    def __call__(self, task: TabularPredictionTask, *args, **kwargs) -> TabularPredictionTask:
        return self.transform(task)