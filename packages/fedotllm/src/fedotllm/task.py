"""A task encapsulates the data for a data science task or project. It contains descriptions, data, metadata."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from fedotllm.tabular import TabularDataset, load_pd

from .constants import (
    BINARY,
    DEFAULT_FORECAST_HORIZON,
    DESCRIPTION,
    MULTILABEL,
    MULTIMODAL,
    NO_TIMESTAMP_COLUMN_IDENTIFIED,
    OUTPUT,
    PREFERED_METRIC_BY_PROBLEM_TYPE,
    REGRESSION,
    STATIC_FEATURES,
    TABULAR,
    TEST,
    TIME_SERIES,
    TRAIN,
)

logger = logging.getLogger(__name__)


class PredictionTask:
    """A task contains data and metadata for a tabular machine learning task, including datasets, metadata such as
    problem type, test_id_column, etc.
    """

    def __init__(
        self,
        filepaths: List[Path],
        metadata: Dict[str, Any],
        name: Optional[str] = "",
        description: Optional[str] = "",
        cache_data: bool = True,
    ):
        self.metadata: Dict[str, Any] = {
            "name": name,
            "description": description,
            "data_description_file": None,
            "task_type": None,
            "label_columns": [],
            "problem_type": None,
            "eval_metric": None,  # string, keying Autogluon Tabular metrics
            "test_id_column": None,
            "images_column": None,
            "forecast_horizon": None,
        }

        self.metadata.update(metadata)

        self.filepaths = filepaths
        self.cache_data = cache_data

        self.files_mapping: Dict[str, Union[Path, None]] = {
            DESCRIPTION: None,
            TRAIN: None,
            TEST: None,
            OUTPUT: None,
            STATIC_FEATURES: None,
        }

        # TODO: each data split can have multiple files
        self.dataset_mapping: Dict[str, Union[Path, TabularDataset, None]] = {
            TRAIN: None,
            TEST: None,
            OUTPUT: None,
            STATIC_FEATURES: None,
        }

    def __repr__(self) -> str:
        return f"PredictionTask(name={self.metadata['name']}, description={self.metadata['description'][:100]}, {len(self.dataset_mapping)} datasets)"

    @classmethod
    def from_path(cls, task_root_dir: Path, name: Optional[str] = None) -> "PredictionTask":
        # Get all filenames under task_root_dir
        task_data_filenames = []
        for entry in task_root_dir.iterdir():
            if entry.is_file():
                # Get just the filename (no subdir paths)
                task_data_filenames.append(entry.name)

        return cls(
            filepaths=[task_root_dir / fn for fn in task_data_filenames],
            metadata=dict(
                name=task_root_dir.name,
            ),
        )

    def load_task_data(self, dataset_key: str) -> Optional[pd.DataFrame]:
        """Load the competition file for the task."""
        if dataset_key not in self.dataset_mapping:
            raise ValueError(f"Dataset type {dataset_key} not found for task {self.metadata.get('name', 'Unknown')}")
        dataset = self.dataset_mapping[dataset_key]
        if dataset is None:
            return None
        if isinstance(dataset, pd.DataFrame):
            return load_pd(dataset)
        elif isinstance(dataset, TabularDataset):
            return dataset
        else:
            if isinstance(dataset, Path):
                filepath = dataset
            else:
                filepath = Path(dataset)

            if filepath.suffix == ".json":
                raise TypeError(f"File {filepath.name} has unsupported type: json")
            else:
                return load_pd(filepath)

    def _set_task_files(
        self,
        dataset_name_mapping: Dict[str, Union[str, Path, TabularDataset]],
    ):
        """Set the task files for the task."""
        for k, v in dataset_name_mapping.items():
            if v is None:
                self.dataset_mapping[k] = None
            elif isinstance(v, TabularDataset):
                self.dataset_mapping[k] = v
            elif isinstance(v, Path):
                self.dataset_mapping[k] = load_pd(v) if self.cache_data else v
            elif isinstance(v, str):
                filepath = next(
                    iter([path for path in self.filepaths if path.name == v]),
                    self.filepaths[0].parent / v,
                )
                if not filepath.is_file():
                    raise ValueError(f"File {v} not found in task {self.metadata['name']}")
                else:
                    self.dataset_mapping[k] = load_pd(filepath) if self.cache_data else filepath
            else:
                raise TypeError(f"Unsupported type for dataset_mapping: {type(v)}")

    @property
    def description(self) -> Optional[str]:
        return self.metadata.get("description", "")

    @property
    def task_type(self) -> Optional[str]:
        return self.metadata["task_type"] or self._find_task_type_in_description()

    @task_type.setter
    def task_type(self, task_type: str) -> None:
        self.metadata["task_type"] = task_type

    @property
    def train_data(self) -> TabularDataset:
        data = self.load_task_data(TRAIN)
        if data is None:
            raise ValueError("Train data is not set")
        return data

    @train_data.setter
    def train_data(self, data: Union[str, Path, TabularDataset]) -> None:
        if isinstance(data, (str, Path)):
            self.files_mapping[TRAIN] = Path(data)
        self._set_task_files({TRAIN: data})

    @property
    def test_data(self) -> TabularDataset:
        test_data = self.load_task_data(TEST)
        if test_data is None:
            if self.task_type == TIME_SERIES:
                return self._create_time_series_test_data()
            raise ValueError("Test data is not set")
        return test_data

    @test_data.setter
    def test_data(self, data: Union[str, Path, TabularDataset]) -> None:
        if isinstance(data, (str, Path)):
            self.files_mapping[TEST] = Path(data)
        self._set_task_files({TEST: data})

    @property
    def sample_submission_data(self) -> Optional[TabularDataset]:
        return self.load_task_data(OUTPUT)

    @sample_submission_data.setter
    def sample_submission_data(self, data: Union[str, Path, TabularDataset]) -> None:
        if isinstance(data, (str, Path)):
            self.files_mapping[OUTPUT] = Path(data)
        if self.sample_submission_data is not None:
            raise ValueError("Output data already set for task")
        self._set_task_files({OUTPUT: data})

    @property
    def static_features_data(self) -> Optional[TabularDataset]:
        return self.load_task_data(STATIC_FEATURES)

    @static_features_data.setter
    def static_features_data(self, data: Union[str, Path, TabularDataset]) -> None:
        if isinstance(data, (str, Path)):
            self.files_mapping[STATIC_FEATURES] = Path(data)
        self._set_task_files({STATIC_FEATURES: data})

    @property
    def data_description_file(self) -> Optional[str]:
        return self.metadata["data_description_file"]

    @data_description_file.setter
    def data_description_file(self, data: str) -> None:
        if isinstance(data, (str, Path)):
            self.files_mapping[DESCRIPTION] = Path(data)
        self.metadata["data_description_file"] = data

    @property
    def forecast_horizon(self) -> int:
        horizon = None
        data = self.metadata["forecast_horizon"]
        if isinstance(data, str):
            horizon = _safe_int_conversion(data)
        elif isinstance(data, int):
            horizon = data
        if horizon is None:
            horizon = DEFAULT_FORECAST_HORIZON
        return horizon

    @forecast_horizon.setter
    def forecast_horizon(self, data: int | str):
        self.metadata["forecast_horizon"] = data

    @property
    def output_columns(self) -> Optional[List[str]]:
        sample_submission_data = self.sample_submission_data
        if sample_submission_data is not None:
            return sample_submission_data.columns.to_list()

        label_columns = self.label_columns
        if not label_columns:
            return None

        if self.task_type == TIME_SERIES:
            timestamp_column = self.metadata.get("timestamp_column")
            return [col for col in [timestamp_column, self.label_column] if col]

        return label_columns

    @property
    def label_columns(self) -> List[str]:
        if self.metadata.get("label_columns"):
            return list(self.metadata["label_columns"])
        return self._infer_label_columns_from_sample_submission_data()

    @label_columns.setter
    def label_columns(self, label_columns: Union[str, List[str]]) -> None:
        if isinstance(label_columns, str):
            label_columns = [label_columns]
        self.metadata["label_columns"] = list(label_columns)

    @property
    def label_column(self) -> Optional[str]:
        label_columns = self.label_columns
        return label_columns[0] if label_columns else None

    @label_column.setter
    def label_column(self, label_column: Union[str, List[str]]) -> None:
        self.label_columns = label_column

    @property
    def timestamp_column(self) -> Optional[str]:
        """Return the timestamp column for the task."""
        if "timestamp_column" in self.metadata and self.metadata["timestamp_column"]:
            if self.metadata["timestamp_column"] == NO_TIMESTAMP_COLUMN_IDENTIFIED:
                return None
            return self.metadata["timestamp_column"]
        else:
            # should ideally never be called after TimestampColumnInference has run
            return self._find_timestamp_column_in_train()

    @timestamp_column.setter
    def timestamp_column(self, data: str) -> None:
        self.metadata["timestamp_column"] = data

    @property
    def columns_in_train_but_not_test(self) -> List[str]:
        if self.test_data is None:
            return []
        return list(set(self.train_data.columns) - set(self.test_data.columns))

    @property
    def data_description(self) -> str:
        return self.metadata.get("data_description", self.description)

    @property
    def evaluation_description(self) -> str:
        return self.metadata.get("evaluation_description", self.description)

    @property
    def test_id_column(self) -> Optional[str]:
        return self.metadata.get("test_id_column", None)

    @test_id_column.setter
    def test_id_column(self, test_id_column: str) -> None:
        self.metadata["test_id_column"] = test_id_column

    @property
    def train_id_column(self) -> Optional[str]:
        return self.metadata.get("train_id_column", None)

    @train_id_column.setter
    def train_id_column(self, train_id_column: str) -> None:
        self.metadata["train_id_column"] = train_id_column

    @property
    def output_id_column(self) -> Optional[str]:
        return self.metadata.get(
            "output_id_column",
            self.sample_submission_data.columns[0] if self.sample_submission_data is not None else None,
        )

    @output_id_column.setter
    def output_id_column(self, output_id_column: str) -> None:
        self.metadata["output_id_column"] = output_id_column

    @property
    def problem_type(self) -> Optional[str]:
        return self.metadata["problem_type"] or self._find_problem_type_in_description()

    @problem_type.setter
    def problem_type(self, problem_type: str) -> None:
        self.metadata["problem_type"] = problem_type

    @property
    def eval_metric(self) -> Optional[str]:
        return self.metadata["eval_metric"] or (
            PREFERED_METRIC_BY_PROBLEM_TYPE[self.problem_type] if self.problem_type else None
        )

    @eval_metric.setter
    def eval_metric(self, eval_metric: str) -> None:
        self.metadata["eval_metric"] = eval_metric

    @property
    def images_column(self) -> Optional[str]:
        return self.metadata["images_column"] or (
            self._find_path_column_in_train() if self.train_data is not None else None
        )

    @property
    def text_columns(self) -> List[str]:
        return self._find_text_columns_in_train()

    @property
    def is_multilabel(self) -> bool:
        return len(self.label_columns) > 1

    def _infer_label_columns_from_sample_submission_data(self) -> List[str]:
        sample_submission_data = self.sample_submission_data
        if sample_submission_data is None:
            return []

        relevant_output_cols = sample_submission_data.columns.to_list()
        if self.output_id_column in relevant_output_cols:
            relevant_output_cols = [col for col in relevant_output_cols if col != self.output_id_column]
        if not relevant_output_cols:
            return []

        existing_output_cols = [col for col in relevant_output_cols if col in self.train_data.columns]
        if len(existing_output_cols) == len(relevant_output_cols):
            return existing_output_cols

        if len(existing_output_cols) == 1:
            return existing_output_cols

        output_set = set(col.lower() for col in relevant_output_cols)
        for col in self.train_data.columns:
            unique_values = set(str(val).lower() for val in self.train_data[col].unique() if pd.notna(val))
            if output_set and (output_set == unique_values or output_set.issubset(unique_values)):
                return [col]

        raise ValueError("Unable to infer the label columns. Please specify them manually.")

    def _find_text_columns_in_train(self) -> List[str]:
        if self.train_data is not None:
            return [
                col
                for col in self.train_data.columns
                if _column_contains_text(self.train_data[col])
                and col != self.images_column
                and col != self.train_id_column
            ]
        return []

    def _find_timestamp_column_in_train(self) -> Optional[str]:
        """Find column that contain timestamp"""
        if self.train_data is not None:
            datetime_cols = [col for col in self.train_data.columns if self.train_data[col].dtype.kind == "M"]
            if len(datetime_cols) > 0:
                return datetime_cols[0]
            for column in self.train_data.columns:
                if column in self.label_columns:
                    continue
                if self.train_data[column].dtype.kind in ("i", "u", "f"):
                    continue

                parsed = pd.to_datetime(self.train_data[column], errors="coerce")
                if not parsed.empty and parsed.notna().all():
                    return column
        return None

    def _find_path_column_in_train(self) -> Optional[str]:
        """Find column that contain paths"""
        path_columns = []
        for col in self.train_data.columns:
            try:
                path = Path(str(self.train_data[col][0]))
                if path.exists():
                    path_columns.append(col)
            except Exception:
                continue
        if len(path_columns) > 0:
            return path_columns[0]
        return None

    def _find_task_type_in_description(self) -> Optional[str]:
        desc = (self.description or "").lower()
        # Check in priority order
        if any(kw in desc for kw in ["tabular"]):
            return TABULAR
        if any(kw in desc for kw in ["multimodal", "image"]):
            return MULTIMODAL
        if any(kw in desc for kw in ["timeseries", "time series"]):
            return TIME_SERIES
        return None

    def _find_problem_type_in_description(self) -> Optional[str]:
        description = (self.description or "").lower()
        if "regression" in description:
            return REGRESSION
        elif len(self.label_columns) > 1:
            return MULTILABEL
        elif "classification" in description:
            return BINARY
        else:
            return None

    def _infer_time_series_frequency(self, index: pd.DatetimeIndex) -> pd.offsets.BaseOffset:
        if len(index) < 2:
            raise ValueError("At least two timestamps are required to infer forecast frequency.")

        inferred = pd.infer_freq(index)
        if inferred is not None:
            return to_offset(inferred)

        diffs = index.to_series().diff().dropna()
        if diffs.empty:
            raise ValueError("Could not infer forecast frequency from timestamps.")

        return to_offset(diffs.mode().iloc[0])

    def _create_time_series_test_data(self) -> TabularDataset:
        assert self.timestamp_column is not None
        label_column = self.label_column
        assert label_column is not None
        history_index = pd.DatetimeIndex(self.train_data[self.timestamp_column], name=self.timestamp_column)
        frequency = self._infer_time_series_frequency(history_index)
        forecast_index = pd.date_range(
            start=history_index[-1] + frequency,
            periods=self.forecast_horizon,
            freq=frequency,
            name=self.timestamp_column,
        )
        return pd.DataFrame(index=forecast_index, columns=pd.Index([label_column]), data=np.nan)


def _safe_int_conversion(string_value: str):
    try:
        return int(string_value)
    except ValueError:
        logger.warning(f"Could not covert '{string_value}' to an integer")
        return None


def _is_float_compatible(column: pd.Series) -> bool:
    """
    :param column: pandas series with data
    :return: True if column contains only float or nan values
    """
    nans_number = column.isna().sum()
    converted_column = pd.to_numeric(column, errors="coerce")
    result_nans_number = converted_column.isna().sum()
    failed_objects_number = result_nans_number - nans_number
    non_nan_all_objects_number = len(column) - nans_number
    failed_ratio = failed_objects_number / non_nan_all_objects_number
    return failed_ratio < 0.5


def _column_contains_text(column: pd.Series) -> bool:
    """
    Column contains text if:
    1. it's not float or float compatible
    (e.g. ['1.2', '2.3', '3.4', ...] is float too)
    2. fraction of unique values (except nans) is more than 0.95

    :param column: pandas series with data
    :return: True if column contains text
    """
    FRACTION_OF_UNIQUE_VALUES = 0.95
    if column.dtype == object and not _is_float_compatible(column):
        unique_num = len(column.unique())
        nan_num = pd.isna(column).sum()
        return (
            unique_num / len(column) > FRACTION_OF_UNIQUE_VALUES
            if nan_num == 0
            else (unique_num - 1) / (len(column) - nan_num) > FRACTION_OF_UNIQUE_VALUES
        )
    return False
