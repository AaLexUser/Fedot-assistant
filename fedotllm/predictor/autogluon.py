"""Predictors solve tabular prediction tasks"""

import logging
import os
import shutil
from collections import defaultdict
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.core.metrics import make_scorer
from autogluon.multimodal import MultiModalPredictor
from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.metrics import mean_squared_log_error

from fedotllm.tabular import TabularDataset

from ..constants import (
    BINARY,
    CLASSIFICATION_PROBA_EVAL_METRIC,
    MEAN_ABSOLUTE_ERROR,
    MEAN_SQUARED_ERROR,
    MULTICLASS,
    ROOT_MEAN_SQUARED_ERROR,
    ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR,
)
from ..task import PredictionTask
from ..utils import unpack_omega_config
from .base import Predictor

logger = logging.getLogger(__name__)


def rmsle_func(y_true, y_pred, **kwargs):
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred, **kwargs))


root_mean_square_logarithmic_error = make_scorer(
    ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR, rmsle_func, optimum=0, greater_is_better=False
)


class AutogluonTabularPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: TabularPredictor = None

    def save_dataset_details(self, task: PredictionTask) -> None:
        for key, data in (("train", task.train_data), ("test", task.test_data)):
            self.metadata["dataset_summery"][key] = data.describe().to_dict()
            self.metadata["feature_metadata_raw"][key] = FeatureMetadata.from_df(
                data
            ).to_dict()
            self.metadata["feature_missing_values"] = (
                data.isna().sum() / len(data)
            ).to_dict()

    def fit(self, task, time_limit=None):
        """Trains an AutoGluon TabularPredictor with parsed arguments. Saves trained predictor
        to `self.predictor`

        Raises
        ------
        Exception
            TabularPredictor fit failures
        """
        eval_metric = task.eval_metric
        if eval_metric == ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR:
            eval_metric = root_mean_square_logarithmic_error

        predictor_init_kwargs = {
            "learner_kwargs": {"ignored_columns": task.columns_in_train_but_not_test},
            "label": task.label_column,
            "problem_type": task.problem_type,
            "eval_metric": eval_metric,
            **unpack_omega_config(self.config.predictor_init_kwargs),
        }

        predictor_fit_kwargs = self.config.predictor_fit_kwargs.copy()
        predictor_fit_kwargs.pop("time_limit", None)

        logger.info("Fitting AutoGluon TabularPredictor")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")
        logger.info(f"predictor_fit_kwargs: {predictor_fit_kwargs}")

        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
            "predictor_fit_kwargs": predictor_fit_kwargs,
        }

        self.save_dataset_details(task)
        self.predictor = TabularPredictor(**predictor_init_kwargs).fit(
            task.train_data,
            **unpack_omega_config(predictor_fit_kwargs),
            time_limit=time_limit,
        )

        self.metadata["leaderboard"] = self.predictor.leaderboard().to_dict()
        return self

    def predict(self, task: PredictionTask) -> TabularDataset:
        if (
            task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC
            and self.predictor.problem_type in [BINARY, MULTICLASS]
        ):
            return self.predictor.predict_proba(
                task.test_data,
                as_multiclass=(self.predictor.problem_type == MULTICLASS),
            )
        else:
            return self.predictor.predict(task.test_data)

    def save_artifacts(self, path: str, task: PredictionTask):
        artifacts = {
            "trained_model": self,
            "train_data": task.train_data,
            "test_data": task.test_data,
            "out_data": task.sample_submission_data,
        }

        ag_model_dir = self.predictor.path
        full_save_path_pkl_file = f"{path}/artifacts.pkl"
        os.makedirs(path, exist_ok=True)

        with open(full_save_path_pkl_file, "wb") as f:
            joblib.dump(artifacts, f)

        src_dir = os.path.abspath(ag_model_dir)
        dst_dir = os.path.join(os.path.abspath(path), os.path.basename(src_dir.rstrip(os.sep)))

        if src_dir == dst_dir:
            logger.warning(
                "Skipping model directory copy because source and destination are the same: %s",
                src_dir,
            )
        else:
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)


class AutogluonMultimodalPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: MultiModalPredictor = None

    def save_dataset_details(self, task: PredictionTask) -> None:
        for key, data in (("train", task.train_data), ("test", task.test_data)):
            self.metadata["dataset_summery"][key] = data.describe().to_dict()
            self.metadata["feature_metadata_raw"][key] = FeatureMetadata.from_df(
                data
            ).to_dict()
            self.metadata["feature_missing_values"] = (
                data.isna().sum() / len(data)
            ).to_dict()

    def fit(self, task, time_limit=None):
        """Trains an AutoGluon MultiModalPredictor with parsed arguments. Saves trained predictor
        to `self.predictor`

        Raises
        ------
        Exception
            MultiModalPredictor fit failures
        """
        eval_metric = task.eval_metric
        if eval_metric == ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR:
            eval_metric = root_mean_square_logarithmic_error

        predictor_init_kwargs = {
            "label": task.label_column,
            "problem_type": task.problem_type,
            "eval_metric": eval_metric,
            **unpack_omega_config(self.config.predictor_init_kwargs),
        }

        predictor_fit_kwargs = self.config.predictor_fit_kwargs.copy()
        predictor_fit_kwargs.pop("time_limit", None)

        logger.info("Fitting AutoGluon MultiModalPredictor")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")
        logger.info(f"predictor_fit_kwargs: {predictor_fit_kwargs}")

        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
            "predictor_fit_kwargs": predictor_fit_kwargs,
        }

        self.save_dataset_details(task)
        self.predictor = MultiModalPredictor(**predictor_init_kwargs).fit(
            task.train_data,
            **unpack_omega_config(predictor_fit_kwargs),
            time_limit=time_limit,
        )

        return self

    def predict(self, task: PredictionTask) -> TabularDataset:
        if (
            task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC
            and self.predictor.problem_type in [BINARY, MULTICLASS]
        ):
            return self.predictor.predict_proba(
                task.test_data,
                as_multiclass=(self.predictor.problem_type == MULTICLASS),
            )
        else:
            return self.predictor.predict(task.test_data)

    def save_artifacts(self, path: str, task: PredictionTask):
        artifacts = {
            "trained_model": self,
            "train_data": task.train_data,
            "test_data": task.test_data,
            "out_data": task.sample_submission_data,
        }

        ag_model_dir = self.predictor.path
        full_save_path_pkl_file = f"{path}/artifacts.pkl"
        os.makedirs(path, exist_ok=True)

        with open(full_save_path_pkl_file, "wb") as f:
            joblib.dump(artifacts, f)

        src_dir = os.path.abspath(ag_model_dir)
        dst_dir = os.path.join(os.path.abspath(path), os.path.basename(src_dir.rstrip(os.sep)))
        if src_dir == dst_dir:
            logger.warning(
                "Skipping model directory copy because source and destination are the same: %s",
                src_dir,
            )
        else:
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)


METRIC_TO_TIMESERIES = {
    ROOT_MEAN_SQUARED_ERROR: "RMSE",
    MEAN_ABSOLUTE_ERROR: "MAE",
    MEAN_SQUARED_ERROR: "MSE",
    ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR: "RMSLE",
}


class AutogluonTimeSeriesPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: MultiModalPredictor = None

    def fit(self, task, time_limit=None):
        """Trains an AutoGluon TimeSeriesPredictor with parsed arguments.

        Parameters
        ----------
        task : PredictionTask
            Task containing training data and metadata
        time_limit : float, optional
            Time limit for training in seconds

        Returns
        -------
        self : AutogluonTimeSeriesPredictor
            Fitted predictor instance

        Raises
        ------
        ValueError
            If required columns are missing from training data
        """
        eval_metric = task.eval_metric
        if eval_metric == ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR:
            eval_metric = root_mean_square_logarithmic_error
        train_data_prepared, freq_str = self._prepare_time_series_data(task)

        train_data = TimeSeriesDataFrame.from_data_frame(
            train_data_prepared["data"],
            id_column=train_data_prepared["id_column"],
            timestamp_column=task.timestamp_column,
            static_features_df=task.static_features_data,
        )

        predictor_init_kwargs = {
            "target": task.label_column,
            "prediction_length": task.forecast_horizon,
            "eval_metric": METRIC_TO_TIMESERIES[eval_metric],
            "freq": freq_str,
            **unpack_omega_config(self.config.predictor_init_kwargs),
        }
        predictor_fit_kwargs = self.config.predictor_fit_kwargs.copy()
        predictor_fit_kwargs.pop("time_limit", None)

        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
            "predictor_fit_kwargs": predictor_fit_kwargs,
        }

        logger.info("Fitting AutoGluon TimeSeriesPredictor")

        self.predictor = TimeSeriesPredictor(**predictor_init_kwargs).fit(
            train_data,
            **unpack_omega_config(predictor_fit_kwargs),
            time_limit=time_limit,
        )

        return self

    def _prepare_time_series_data(self, task):
        """Prepare time series data for AutoGluon training.

        Handles univariate series by adding item_id column and infers frequency
        from timestamp data for irregular time series.

        Returns
        -------
        dict
            Dictionary with 'data' and 'id_column' keys
        str
            Frequency string for AutoGluon
        """
        if task.train_id_column and task.train_id_column not in task.train_data.columns:
            raise ValueError(
                f"train_id_column '{task.train_id_column}' not found in training data"
            )

        # Handle univariate time series
        if task.train_id_column is None:
            train_data_copy = task.train_data.copy()
            train_data_copy["item_id"] = "series_1"
            id_column_to_use = "item_id"
        else:
            train_data_copy = task.train_data.copy()
            id_column_to_use = task.train_id_column

        # Infer frequency from timestamps
        freq_str = self._infer_frequency(train_data_copy, task.timestamp_column)

        return {"data": train_data_copy, "id_column": id_column_to_use}, freq_str

    def _infer_frequency(self, data, timestamp_column):
        """Infer frequency from timestamp data and convert to AutoGluon format."""
        data[timestamp_column] = pd.to_datetime(data[timestamp_column])
        time_diffs = data[timestamp_column].diff().dropna()

        if len(time_diffs.mode()) == 0:
            return "H"  # Default to hourly

        most_common_freq = time_diffs.mode().iloc[0]

        # Map common frequencies to AutoGluon format
        freq_mapping = {
            pd.Timedelta(hours=1): "H",
            pd.Timedelta(minutes=30): "30min",
            pd.Timedelta(days=1): "D",
        }

        return freq_mapping.get(most_common_freq, "H")

    def predict(self, task: PredictionTask) -> TabularDataset:
        return self.predictor.predict(task.train_data)

    def save_artifacts(self, path: str, task: PredictionTask):
        artifacts = {
            "trained_model": self,
            "train_data": task.train_data,
            "test_data": task.test_data,
            "out_data": task.sample_submission_data,
        }

        ag_model_dir = self.predictor.path
        full_save_path_pkl_file = f"{path}/artifacts.pkl"
        os.makedirs(path, exist_ok=True)

        with open(full_save_path_pkl_file, "wb") as f:
            joblib.dump(artifacts, f)

        src_dir = os.path.abspath(ag_model_dir)
        dst_dir = os.path.join(os.path.abspath(path), os.path.basename(src_dir.rstrip(os.sep)))
        if src_dir == dst_dir:
            logger.warning(
                "Skipping model directory copy because source and destination are the same: %s",
                src_dir,
            )
        else:
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
