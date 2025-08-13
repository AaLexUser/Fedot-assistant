import logging
import signal
import sys
import threading
from contextlib import contextmanager
from typing import Any, List, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig

from .constants import TABULAR, TIME_SERIES
from .llm import AssistantChatOpenAI
from .task import PredictionTask
from .task_inference import (
    DataFileNameInference,
    DescriptionFileNameInference,
    EvalMetricInference,
    ForecastHorizonInference,
    LabelColumnInference,
    OutputIDColumnInference,
    ProblemTypeInference,
    StaticFeaturesFileNameInference,
    TaskInference,
    TaskTypeInference,
    TestIDColumnInference,
    TimestampColumnInference,
    TrainIDColumnInference,
)
from .utils import get_feature_transformers_config

logger = logging.getLogger(__name__)


@contextmanager
def timeout(seconds: int, error_message: Optional[str] = None):
    if sys.platform == "win32":
        # Windows implementation using threading
        timer = threading.Timer(
            seconds, lambda: (_ for _ in ()).throw(TimeoutError(error_message))
        )
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
        self, task_inference_preprocessors: List[TaskInference], task: PredictionTask
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
                self.handle_exception(
                    f"Task inference preprocessing: {preprocessor_class}", e
                )

    def inference_task(self, task: PredictionTask) -> PredictionTask:
        logger.info("Task understanding starts...")
        task_inference_preprocessors = [
            DescriptionFileNameInference,
            DataFileNameInference,
            LabelColumnInference,
            TaskTypeInference,
            ProblemTypeInference,
        ]

        if self.config.detect_and_drop_id_column:
            task_inference_preprocessors += [
                OutputIDColumnInference,
                TrainIDColumnInference,
                TestIDColumnInference,
            ]

        if self.config.infer_eval_metric:
            task_inference_preprocessors += [EvalMetricInference]

        self._run_task_inference_preprocessors(task_inference_preprocessors, task)

        # Task type specific

        if task.problem_type == TIME_SERIES:
            timeseries_inference_preprocessors = [
                TimestampColumnInference,
                StaticFeaturesFileNameInference,
                ForecastHorizonInference,
            ]

            self._run_task_inference_preprocessors(
                timeseries_inference_preprocessors, task
            )

        bold_start = "\033[1m"
        bold_end = "\033[0m"

        logger.info(
            f"{bold_start}Total number of prompt tokens:{bold_end} {self.llm.input_}"
        )
        logger.info(
            f"{bold_start}Total number of completion tokens:{bold_end} {self.llm.output_}"
        )
        logger.info("Task understanding complete!")
        return task

    def preprocess_task(self, task: PredictionTask) -> PredictionTask:
        task = self.inference_task(task)
        if task.task_type == TABULAR and self.feature_transformers_config:
            logger.info("Automatic feature generation starts...")
            fe_transformers = [
                instantiate(ft_config) for ft_config in self.feature_transformers_config
            ]
            for fe_transformer in fe_transformers:
                try:
                    with timeout(
                        seconds=self.config.task_preprocessors_timeout,
                        error_message=f"Task preprocessing timed out: {fe_transformer.name}",
                    ):
                        task = fe_transformer.fit_transform(task)
                except Exception as e:
                    self.handle_exception(
                        f"Task preprocessing: {fe_transformer.name}", e
                    )
            logger.info("Automatic feature generation complete!")
        else:
            logger.info("Automatic feature generation is disabled or not supported")
        return task

    def fit_predictor(self, task: PredictionTask, time_limit: float):
        match self.config.automl.enabled:
            case "fedot":
                match task.task_type:
                    case "tabular":
                        from .predictor.fedot import FedotTabularPredictor

                        self.predictor = FedotTabularPredictor(self.config.automl.fedot)
                    case "multimodal":
                        from .predictor.fedot import FedotMultiModalPredictor

                        self.predictor = FedotMultiModalPredictor(
                            self.config.automl.fedot
                        )
                    case "time_series":
                        from .predictor.fedot import FedotTimeSeriesPredictor

                        self.predictor = FedotTimeSeriesPredictor(
                            self.config.automl.fedot
                        )
                    case _:
                        raise ValueError(
                            f"Fedot doesn't support {task.task_type} tasks"
                        )
            case "fedot_ind":
                match task.task_type:
                    case "tabular":
                        from .predictor.fedot_ind import FedotIndustrialTabularPredictor

                        self.predictor = FedotIndustrialTabularPredictor(
                            self.config.automl.fedot_ind
                        )
                    case "time_series":
                        from .predictor.fedot_ind import (
                            FedotIndustrialTimeSeriesPredictor,
                        )

                        self.predictor = FedotIndustrialTimeSeriesPredictor(
                            self.config.automl.fedot_ind
                        )
                    case _:
                        raise ValueError(
                            f"Fedot.Industrial doesn't support {task.task_type} tasks"
                        )
            case "autogluon":
                match task.task_type:
                    case "tabular":
                        from .predictor.autogluon import AutogluonTabularPredictor

                        self.predictor = AutogluonTabularPredictor(
                            self.config.automl.autogluon
                        )
                    case "multimodal":
                        from .predictor.autogluon import AutogluonMultimodalPredictor

                        self.predictor = AutogluonMultimodalPredictor(
                            self.config.automl.autogluon
                        )
                    case "time_series":
                        from .predictor.autogluon import AutogluonTimeSeriesPredictor

                        self.predictor = AutogluonTimeSeriesPredictor(
                            self.config.automl.autogluon
                        )
                    case _:
                        raise ValueError(
                            f"AutoGluon doesn't support {task.task_type} tasks"
                        )
            case _:
                raise ValueError(
                    "Unknown automl framework: {self.config.automl.enabled}"
                )
        try:
            if self.config.automl.enabled in ["fedot", "fedot_ind"]:
                time_limit = time_limit / 60
            self.predictor.fit(task, time_limit=time_limit)
        except Exception as e:
            self.handle_exception("Predictor Fit", e)

    def predict(self, task: PredictionTask) -> Any:
        try:
            return self.predictor.predict(task)
        except Exception as e:
            self.handle_exception("Predictor Predict", e)
