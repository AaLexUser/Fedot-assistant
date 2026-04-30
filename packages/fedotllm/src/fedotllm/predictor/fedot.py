import logging
import os
from collections import defaultdict
from typing import Any, Dict, Optional, cast

import joblib
import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.composer.metrics import QualityMetric
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedotllm.tabular import TabularDataset
from golem.core.dag.graph_utils import graph_structure
from PIL import Image
from sklearn.metrics import mean_squared_log_error
from tqdm import tqdm

from ..constants import (
    ACCURACY,
    BINARY,
    CLASSIFICATION_PROBA_EVAL_METRIC,
    F1,
    LOG_LOSS,
    MEAN_ABSOLUTE_ERROR,
    MEAN_SQUARED_ERROR,
    MULTICLASS,
    MULTILABEL,
    R2,
    REGRESSION,
    ROC_AUC,
    ROOT_MEAN_SQUARED_ERROR,
    ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR,
    TIME_SERIES,
)
from ..task import PredictionTask
from ..utils import unpack_omega_config
from .base import Predictor
from .targets import (
    per_label_problem_type,
    series_from_binary_proba,
    series_from_predictions,
    split_tabular_task,
    task_is_multilabel,
    task_label_columns,
    time_limit_per_label,
    validate_binary_multilabel_targets,
)

logger = logging.getLogger(__name__)


class FedotRMSLE(QualityMetric):
    default_value = np.finfo(np.float64).max

    @staticmethod
    def metric(reference: InputData, predicted: OutputData) -> float:
        return float(
            np.sqrt(
                mean_squared_log_error(
                    y_true=reference.target,
                    y_pred=predicted.predict,
                )
            )
        )


FEDOT_RMSLE_METRIC = FedotRMSLE.get_value

METRICS_TO_FEDOT = {
    ROC_AUC: "roc_auc",
    LOG_LOSS: "neg_log_loss",
    ACCURACY: "accuracy",
    F1: "f1",
    ROOT_MEAN_SQUARED_ERROR: "rmse",
    ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR: FEDOT_RMSLE_METRIC,
    MEAN_SQUARED_ERROR: "mse",
    MEAN_ABSOLUTE_ERROR: "mae",
    R2: "r2",
}

PROBLEM_TO_FEDOT = {
    BINARY: "classification",
    MULTICLASS: "classification",
    MULTILABEL: "classification",
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
    assert task.problem_type is not None
    task_problem_type = Task(TaskTypesEnum(PROBLEM_TO_FEDOT[task.problem_type]))
    sources = {}

    table_features = data.copy()
    label_column = task.label_column
    target = data[label_column].to_numpy() if label_column is not None and label_column in data.columns else None
    if target is not None:
        assert label_column is not None
        table_features = table_features.drop(label_column, axis=1)

    if task.images_column is not None:
        logger.info(f"Found images column: {task.images_column}")
        data_img = InputData.from_image(
            images=_load_images_from_dataframe(data, task.images_column),
            labels=cast(Any, target),
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
        self.predictors: Dict[str, Fedot] = {}
        self.problem_type: Optional[str] = None

    def fit(self, task: PredictionTask, time_limit: Optional[float] = None) -> "FedotTabularPredictor":
        eval_metric = task.eval_metric
        assert eval_metric is not None
        self.problem_type = task.problem_type
        train_x, train_y, test_x = split_tabular_task(task)
        label_columns = task_label_columns(task)
        if task_is_multilabel(task):
            validate_binary_multilabel_targets(train_y)
        predictor_fit_kwargs = self.config.predictor_fit_kwargs

        logger.info("Fitting Fedot TabularPredictor")
        logger.info(f"predictor_fit_kwargs: {predictor_fit_kwargs}")

        self.metadata |= {
            "predictor_fit_kwargs": predictor_fit_kwargs,
        }
        for label in label_columns:
            predictor_init_kwargs = {
                "problem": PROBLEM_TO_FEDOT[per_label_problem_type(task)],
                "timeout": time_limit_per_label(time_limit, len(label_columns)),
                "metric": METRICS_TO_FEDOT[eval_metric],
                **unpack_omega_config(self.config.predictor_init_kwargs),
            }
            logger.info(f"predictor_init_kwargs[{label}]: {predictor_init_kwargs}")
            predictor = Fedot(**cast(Any, predictor_init_kwargs))
            predictor.fit(
                train_x,
                train_y[label],
                **cast(Any, unpack_omega_config(predictor_fit_kwargs)),
            )
            self.predictors[label] = predictor
            self.metadata["predictor_init_kwargs"][label] = predictor_init_kwargs
            assert predictor.current_pipeline is not None
            self.metadata["graph_structure"][label] = graph_structure(predictor.current_pipeline)
        self.metadata["test_columns"] = test_x.columns.to_list()
        return self

    def predict(self, task: PredictionTask) -> TabularDataset:
        _, _, test_x = split_tabular_task(task)
        predictions = []
        label_problem_type = per_label_problem_type(task)
        label_columns = task_label_columns(task)
        for label in label_columns:
            predictor = self.predictors[label]
            if task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC and label_problem_type in [BINARY, MULTICLASS]:
                predictions.append(
                    series_from_binary_proba(
                        predictor.predict_proba(test_x),
                        label=label,
                        index=task.test_data.index,
                    )
                )
            else:
                predictions.append(
                    series_from_predictions(
                        predictor.predict(test_x),
                        label=label,
                        index=task.test_data.index,
                    )
                )
        return pd.concat(predictions, axis=1)

    def save_artifacts(self, path: str, task: PredictionTask):
        artifacts = {
            "trained_model": self,
            "train_data": task.train_data,
            "test_data": task.test_data,
            "out_data": task.sample_submission_data,
        }
        full_save_path_pkl_file = f"{path}/artifacts.pkl"
        os.makedirs(path, exist_ok=True)

        with open(full_save_path_pkl_file, "wb") as f:
            joblib.dump(artifacts, f)

        for label, predictor in self.predictors.items():
            assert predictor.current_pipeline is not None
            predictor.current_pipeline.save(os.path.join(path, f"fedot_{label}"))


class FedotMultiModalPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: Optional[Fedot] = None
        self.problem_type: Optional[str] = None

    def fit(self, task: PredictionTask, time_limit: Optional[float] = None) -> "FedotMultiModalPredictor":
        eval_metric = task.eval_metric
        assert eval_metric is not None
        assert task.problem_type is not None
        self.problem_type = task.problem_type

        predictor_init_kwargs = {
            "problem": PROBLEM_TO_FEDOT[task.problem_type],
            "timeout": time_limit,
            "metric": METRICS_TO_FEDOT[eval_metric],
            **unpack_omega_config(self.config.predictor_init_kwargs),
        }

        train_only_cols = [col for col in task.columns_in_train_but_not_test if col != task.label_column]
        aligned_train = task.train_data.drop(columns=train_only_cols, errors="ignore")
        train_data = prepare_multi_model_data(aligned_train, task)

        predictor_fit_kwargs = self.config.predictor_fit_kwargs

        logger.info("Fitting Fedot TabularPredictor")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")
        logger.info(f"predictor_fit_kwargs: {predictor_fit_kwargs}")

        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
            "predictor_fit_kwargs": predictor_fit_kwargs,
        }

        self.predictor = Fedot(**cast(Any, predictor_init_kwargs))
        label_column = task.label_column
        assert label_column is not None
        self.predictor.fit(train_data, label_column, **cast(Any, unpack_omega_config(predictor_fit_kwargs)))

        assert self.predictor.current_pipeline is not None
        self.metadata["graph_structure"] = graph_structure(self.predictor.current_pipeline)
        return self

    def predict(self, task: PredictionTask) -> TabularDataset:
        assert self.predictor is not None
        label_column = task.label_column
        assert label_column is not None
        test_data = task.test_data.drop(label_column, axis=1) if label_column in task.test_data else task.test_data

        test_data = prepare_multi_model_data(test_data, task)

        if task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC and self.problem_type in [BINARY, MULTICLASS]:
            predictions = self.predictor.predict_proba(test_data)
        else:
            predictions = self.predictor.predict(test_data)
        return pd.DataFrame(predictions, columns=pd.Index([label_column]), index=task.test_data.index)

    def save_artifacts(self, path: str, task: PredictionTask):
        artifacts = {
            "trained_model": self,
            "train_data": task.train_data,
            "test_data": task.test_data,
            "out_data": task.sample_submission_data,
        }
        full_save_path_pkl_file = f"{path}/artifacts.pkl"
        os.makedirs(path, exist_ok=True)

        with open(full_save_path_pkl_file, "wb") as f:
            joblib.dump(artifacts, f)

        assert self.predictor is not None
        assert self.predictor.current_pipeline is not None
        self.predictor.current_pipeline.save(path)


class FedotTimeSeriesPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: Optional[Fedot] = None
        self.problem_type: Optional[str] = None
        self.eval_metric: Optional[str] = None
        self.historical_data: Optional[np.ndarray] = None

    def fit(self, task: PredictionTask, time_limit: Optional[float] = None) -> "FedotTimeSeriesPredictor":
        self.eval_metric = task.eval_metric
        self.problem_type = task.problem_type
        assert self.problem_type is not None
        assert self.eval_metric is not None

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
        self.predictor = Fedot(**cast(Any, predictor_init_kwargs))
        self.predictor.fit(input_data)

        assert self.predictor.current_pipeline is not None
        self.metadata["graph_structure"] = graph_structure(self.predictor.current_pipeline)
        return self

    def predict(self, task: PredictionTask) -> TabularDataset:
        assert self.predictor is not None
        label_column = task.label_column
        assert label_column is not None
        input_data = self.prepare_data(task, is_for_forecast=True)
        if task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC and self.problem_type in [BINARY, MULTICLASS]:
            predictions = self.predictor.predict_proba(input_data)
        else:
            predictions = self.predictor.predict(input_data)

        pred_len = len(predictions)
        return pd.DataFrame(
            predictions,
            columns=pd.Index([label_column]),
            index=task.test_data.index[:pred_len],
        )

    def save_artifacts(self, path: str, task: PredictionTask) -> None:
        artifacts = {
            "trained_model": self,
            "train_data": task.train_data,
            "test_data": task.test_data,
            "out_data": task.sample_submission_data,
        }
        full_save_path_pkl_file = f"{path}/artifacts.pkl"
        os.makedirs(path, exist_ok=True)

        with open(full_save_path_pkl_file, "wb") as f:
            joblib.dump(artifacts, f)
        assert self.predictor is not None
        assert self.predictor.current_pipeline is not None
        self.predictor.current_pipeline.save(path)

    def prepare_data(self, task: PredictionTask, is_for_forecast: Optional[bool] = False) -> np.ndarray:
        data = task.test_data if is_for_forecast else task.train_data

        timestamp_col = task.timestamp_column
        if timestamp_col and timestamp_col in data.columns:
            # Convert to datetime and set as index
            data[timestamp_col] = pd.to_datetime(data[timestamp_col])
            data = data.set_index(timestamp_col)
        series = data.to_numpy().squeeze()

        if is_for_forecast:
            assert self.historical_data is not None
            return self.historical_data

        self.historical_data = series
        return series
