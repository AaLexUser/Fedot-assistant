from .base import Predictor
from typing import Any, Dict, Optional
from collections import defaultdict
from fedot.api.main import Fedot
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.repository.dataset_types import DataTypesEnum
from ..task import PredictionTask
from ..utils import unpack_omega_config
from golem.core.dag.graph_utils import graph_structure
from fedotllm.tabular import TabularDataset
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
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
    TIME_SERIES,
    CLASSIFICATION_PROBA_EVAL_METRIC,
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
    R2: "r2",
}

PROBLEM_TO_FEDOT = {
    BINARY: "classification",
    MULTICLASS: "classification",
    REGRESSION: "regression",
    TIME_SERIES: "ts_forecasting",
}


# TODO: choose targer_size more reasonably
def _load_images_from_dataframe(df, image_column, target_size=(128, 128)):
    images = []
    for image_path in tqdm(df[image_column], desc="Loading images"):
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = img_array.astype(np.float32) / sum(target_size) - 1  # Normalization

        # Ensure the image has 3 channels (RGB)
        if len(img_array.shape) == 2:  # Grayscale image
            img_array = np.stack((img_array,) * 3, axis=-1)  # Convert to 3 channels
        elif img_array.shape[2] == 4:  # RGBA image
            img_array = img_array[:, :, :3]  # Drop the alpha channel
        images.append(img_array)

    images_array = np.array(images)
    return images_array


def prepare_multi_model_data(
    data: pd.DataFrame,
    task: PredictionTask,
) -> MultiModalData:
    task_problem_type = Task(TaskTypesEnum(PROBLEM_TO_FEDOT[task.problem_type]))
    sources = {}

    table_features = data.copy()
    target = (
        data[task.label_column].to_numpy()
        if task.label_column in data.columns
        else None
    )
    if target is not None:
        table_features = table_features.drop(task.label_column, axis=1)

    if task.images_column is not None:
        logger.info(f"Found images column: {task.images_column}")
        data_img = InputData.from_image(
            images=_load_images_from_dataframe(data, task.images_column),
            labels=target,
            task=task_problem_type,
        )
        table_features = table_features.drop(task.images_column, axis=1)
        sources.update({"data_source_img": data_img})

    if len(task.text_columns) > 0:
        logger.info(f"Found {task.text_columns} text columns.")
        data_text = InputData(
            idx=data[task.text_columns].index.to_numpy(),
            features=data[task.text_columns].to_numpy(),
            target=target,
            task=task_problem_type,
            data_type=DataTypesEnum.text,
            features_names=data[task.text_columns].columns.to_numpy(),
        )
        table_features = table_features.drop(task.text_columns, axis=1)
        sources.update({"data_source_text": data_text})

    if len(table_features.columns) > 0:
        logger.info(f"Found table features: {len(table_features.columns)}.")
        data_table = InputData(
            idx=table_features.index.to_numpy(),
            features=table_features.to_numpy(),
            target=target,
            task=task_problem_type,
            data_type=DataTypesEnum.table,
            features_names=table_features.columns.to_numpy(),
        )

        sources.update({"data_source_table": data_table})

    return MultiModalData(sources)


class FedotTabularPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: Fedot = None
        self.problem_type: str = None

    def fit(
        self, task: PredictionTask, time_limit: Optional[float] = None
    ) -> "FedotTabularPredictor":
        eval_metric = task.eval_metric
        self.problem_type = task.problem_type

        predictor_init_kwargs = {
            "problem": PROBLEM_TO_FEDOT[task.problem_type],
            "timeout": time_limit,
            "metric": METRICS_TO_FEDOT[eval_metric],
            **unpack_omega_config(self.config.predictor_init_kwargs),
        }

        predictor_fit_kwargs = self.config.predictor_fit_kwargs

        logger.info("Fitting Fedot TabularPredictor")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")
        logger.info(f"predictor_fit_kwargs: {predictor_fit_kwargs}")

        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
            "predictor_fit_kwargs": predictor_fit_kwargs,
        }

        self.predictor = Fedot(**predictor_init_kwargs)
        self.predictor.fit(
            task.train_data,
            task.label_column,
            **unpack_omega_config(predictor_fit_kwargs),
        )

        self.metadata["graph_structure"] = graph_structure(
            self.predictor.current_pipeline
        )
        return self

    def predict(self, task: PredictionTask) -> TabularDataset:
        if (
            task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC
            and self.problem_type in [BINARY, MULTICLASS]
        ):
            predictions = self.predictor.predict_proba(task.test_data)
        else:
            predictions = self.predictor.predict(task.test_data)
        return pd.DataFrame(predictions, columns=[task.label_column])

    def save_artifacts(self, path: str):
        self.predictor.current_pipeline.save(path)


class FedotMultiModalPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: Fedot = None
        self.problem_type: str = None

    def fit(
        self, task: PredictionTask, time_limit: Optional[float] = None
    ) -> "FedotMultiModalPredictor":
        eval_metric = task.eval_metric
        self.problem_type = task.problem_type

        predictor_init_kwargs = {
            "problem": PROBLEM_TO_FEDOT[task.problem_type],
            "timeout": time_limit,
            "metric": METRICS_TO_FEDOT[eval_metric],
            **unpack_omega_config(self.config.predictor_init_kwargs),
        }

        train_data = prepare_multi_model_data(task.train_data, task)

        predictor_fit_kwargs = self.config.predictor_fit_kwargs

        logger.info("Fitting Fedot TabularPredictor")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")
        logger.info(f"predictor_fit_kwargs: {predictor_fit_kwargs}")

        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
            "predictor_fit_kwargs": predictor_fit_kwargs,
        }

        self.predictor = Fedot(**predictor_init_kwargs)
        self.predictor.fit(
            train_data, task.label_column, **unpack_omega_config(predictor_fit_kwargs)
        )

        self.metadata["graph_structure"] = graph_structure(
            self.predictor.current_pipeline
        )
        return self

    def predict(self, task: PredictionTask) -> TabularDataset:
        test_data = (
            task.test_data.drop(task.label_column, axis=1)
            if task.label_column in task.test_data
            else task.test_data
        )

        test_data = prepare_multi_model_data(test_data, task)

        if (
            task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC
            and self.problem_type in [BINARY, MULTICLASS]
        ):
            predictions = self.predictor.predict_proba(test_data)
        else:
            predictions = self.predictor.predict(test_data)
        return pd.DataFrame(
            predictions, columns=[task.label_column], index=task.test_data.index
        )

    def save_artifacts(self, path: str):
        self.predictor.current_pipeline.save(path)


class FedotTimeSeriesPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: Optional[Fedot] = None
        self.problem_type: Optional[str] = None
        self.eval_metric: Optional[str] = None
        self.historical_data: Optional[np.ndarray] = None

    def fit(
        self, task: PredictionTask, time_limit: Optional[float] = None
    ) -> "FedotTimeSeriesPredictor":
        self.eval_metric = task.eval_metric
        self.problem_type = task.problem_type

        predictor_init_kwargs = {
            "problem": PROBLEM_TO_FEDOT[self.problem_type],
            "timeout": time_limit,
            "metric": METRICS_TO_FEDOT[self.eval_metric],
            "task_params": TsForecastingParams(forecast_length=task.forecast_horizon),
            **unpack_omega_config(self.config.predictor_init_kwargs),
        }

        logger.info("Fitting Fedot TimeseriesPredictor")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")
        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
        }

        input_data = self.prepare_data(task, is_for_forecast=False)
        self.predictor = Fedot(**predictor_init_kwargs)
        self.predictor.fit(input_data)

        self.metadata["graph_structure"] = graph_structure(
            self.predictor.current_pipeline
        )
        return self

    def predict(self, task: PredictionTask) -> TabularDataset:
        input_data = self.prepare_data(task, is_for_forecast=True)
        if (
            task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC
            and self.problem_type in [BINARY, MULTICLASS]
        ):
            return pd.DataFrame(
                self.predictor.predict_proba(input_data),
                columns=[task.label_column],
                index=task.test_data.index,
            )

        return pd.DataFrame(
            self.predictor.predict(input_data),
            columns=[task.label_column],
            index=task.test_data.index,
        )

    def save_artifacts(self, path: str) -> None:
        self.predictor.current_pipeline.save(path)

    def prepare_data(
        self, task: PredictionTask, is_for_forecast: Optional[bool] = False
    ) -> np.ndarray:
        data = task.test_data if is_for_forecast else task.train_data

        timestamp_col = task.timestamp_column
        if timestamp_col and timestamp_col in data.columns:
            # Convert to datetime and set as index
            data[timestamp_col] = pd.to_datetime(data[timestamp_col])
            data = data.set_index(timestamp_col)
        series = data.to_numpy().squeeze()

        if is_for_forecast:
            return self.historical_data

        self.historical_data = series
        return series
