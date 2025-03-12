from omegaconf import DictConfig
from .llm import AssistantChatOpenAI
from .predictor import (
    FedotTabularPredictor,
    FedotMultiModalPredictor,
    AutogluonTabularPredictor,
    AutogluonMultimodalPredictor,
    AutogluonTimeSeriesPredictor
)
from .task import PredictionTask
from .utils import get_feature_transformers_config
from .task_inference import (
    TaskInference,
    DataFileNameInference,
    DescriptionFileNameInference,
    LabelColumnInference,
    TaskTypeInference,
    ProblemTypeInference,
    TestIDColumnInference,
    TrainIDColumnInference,
    OutputIDColumnInference,
    EvalMetricInference,
    TimestampColumnInference,
    StaticFeaturesFileNameInference,
    ForecastHorizonInference,
)
import threading
from typing import Optional, Any, List
import logging
import sys
from contextlib import contextmanager
import signal
from hydra.utils import instantiate
from .constants import (
    TABULAR,
    MULTIMODAL,
    REGRESSION,
    TIME_SERIES
)

logger = logging.getLogger(__name__)

@contextmanager
def timeout(seconds: int, error_message: Optional[str] = None):
    if sys.platform == "win32":
        # Windows implementation using threading
        timer = threading.Timer(seconds, lambda: (_ for _ in ()).throw(TimeoutError(error_message)))
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
    else:
        # Unix impementation using SIGALRM
        def handle_timeout(signum, frame):
            raise TimeoutError(error_message)
        
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
        

class PredictionAssistant:
    """A TabularPredictionAssistant performs a supervised tabular learning task"""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.llm = AssistantChatOpenAI(config.llm)
        self.feature_transformers_config = get_feature_transformers_config(config)
        
    def handle_exception(self, stage: str, exception: Exception):
        raise Exception(str(exception), stage)
    
    def _run_task_inference_preprocessors(
        self,
        task_inference_preprocessors: List[TaskInference],
        task: PredictionTask
    ):
        for preprocessor_class in task_inference_preprocessors:
            preprocessor = preprocessor_class(llm=self.llm)
            try:
                with timeout(
                    seconds=self.config.task_preprocessors_timeout,
                    error_message=f"Task inference preprocessing time out: {preprocessor_class}",
                ):
                    task = preprocessor.transform(task)
            except Exception as e:
                self.handle_exception(f"Task inference preprocessing: {preprocessor_class}", e)
        

    def inference_task(self, task: PredictionTask) -> PredictionTask:
        logger.info("Task understanding starts...")
        task_inference_preprocessors = [
            DescriptionFileNameInference,
            DataFileNameInference,
            LabelColumnInference,
            TaskTypeInference,
            ProblemTypeInference
        ]
        
        if self.config.detect_and_drop_id_column:
            task_inference_preprocessors += [
                OutputIDColumnInference,
                TrainIDColumnInference,
                TestIDColumnInference
            ]
        
        if self.config.infer_eval_metric:
            task_inference_preprocessors += [EvalMetricInference]
            
        self._run_task_inference_preprocessors(task_inference_preprocessors, task)
        
        # Task type specific
                        
        if task.problem_type == TIME_SERIES:
            timeseries_inference_preprocessors = [
                TimestampColumnInference,
                StaticFeaturesFileNameInference,
                ForecastHorizonInference
            ]        
        
            self._run_task_inference_preprocessors(timeseries_inference_preprocessors, task)
        
        bold_start = "\033[1m"
        bold_end = "\033[0m"
        
        logger.info(f"{bold_start}Total number of prompt tokens:{bold_end} {self.llm.input_}")
        logger.info(f"{bold_start}Total number of completion tokens:{bold_end} {self.llm.output_}")
        logger.info("Task understanding complete!")
        return task
    
    def preprocess_task(self, task: PredictionTask) -> PredictionTask:
        task = self.inference_task(task)
        if task.task_type == TABULAR and self.feature_transformers_config:
            logger.info("Automatic feature generation starts...")
            fe_transformers = [instantiate(ft_config) for ft_config in self.feature_transformers_config]
            for fe_transformer in fe_transformers:
                try:
                    with timeout(
                        seconds=self.config.task_preprocessors_timeout,
                        error_message=f"Task preprocessing timed out: {fe_transformer.name}"
                    ):
                        task = fe_transformer.fit_transform(task)
                except Exception as e:
                    self.handle_exception(f"Task preprocessing: {fe_transformer.name}", e)
            logger.info("Automatic feature generation complete!")
        else:
            logger.info("Automatic feature generation is disabled or not supported")
        return task
        
    def fit_predictor(self, task: PredictionTask, time_limit: float):
        match self.config.automl.enabled:
            case "fedot":
                match task.task_type:
                    case "tabular":
                        self.predictor = FedotTabularPredictor(self.config.automl.fedot)
                    case "multimodal":
                        self.predictor = FedotMultiModalPredictor(self.config.automl.fedot)
                    case _:
                        raise ValueError(f"Fedot doesn't support {task.task_type} tasks")
            case "autogluon":
                match task.task_type:
                    case "tabular":
                        self.predictor = AutogluonTabularPredictor(self.config.automl.autogluon)
                    case "multimodal":
                        self.predictor = AutogluonMultimodalPredictor(self.config.automl.autogluon)
                    case "time_series":
                        self.predictor = AutogluonTimeSeriesPredictor(self.config.automl.autogluon)
                    case _:
                        raise ValueError(f"AutoGluon doesn't support {task.task_type} tasks")      
            case _:
                raise ValueError("Unknown automl framework: {self.config.automl.enabled}")
        try:
            if isinstance(self.predictor, FedotTabularPredictor):
                time_limit = time_limit / 60
            self.predictor.fit(task, time_limit=time_limit)
        except Exception as e:
            self.handle_exception("Predictor Fit", e)
    
    def predict(self, task: PredictionTask) -> Any:
        try:
            return self.predictor.predict(task)
        except Exception as e:
            self.handle_exception("Predictor Predict", e)