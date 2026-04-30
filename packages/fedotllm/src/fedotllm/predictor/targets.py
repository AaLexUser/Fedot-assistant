from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..constants import BINARY, MULTILABEL, TEST
from ..task import PredictionTask


def task_label_columns(task: PredictionTask) -> list[str]:
    if hasattr(task, "label_columns"):
        label_columns = getattr(task, "label_columns")
        if label_columns:
            return list(label_columns)
    if hasattr(task, "label_column") and getattr(task, "label_column"):
        return [getattr(task, "label_column")]
    return []


def task_is_multilabel(task: PredictionTask) -> bool:
    if hasattr(task, "is_multilabel"):
        return bool(getattr(task, "is_multilabel"))
    return len(task_label_columns(task)) > 1


def split_tabular_task(
    task: PredictionTask,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_columns = task_label_columns(task)
    train_data = task.train_data
    if isinstance(task, PredictionTask):
        raw_test_data = task.load_task_data(TEST)
        test_data = raw_test_data if raw_test_data is not None else train_data.iloc[0:0].copy()
    else:
        test_data = getattr(task, "test_data", train_data.iloc[0:0].copy())
    return (
        train_data.drop(columns=label_columns, errors="ignore"),
        train_data[label_columns].copy(),
        test_data.drop(columns=label_columns, errors="ignore"),
    )


def per_label_problem_type(task: PredictionTask) -> str:
    if task.problem_type == MULTILABEL:
        return BINARY
    assert task.problem_type is not None
    return task.problem_type


def validate_binary_multilabel_targets(target_data: pd.DataFrame) -> None:
    for column in target_data.columns:
        values = set(target_data[column].dropna().unique())
        if not values.issubset({0, 1, False, True}):
            raise ValueError(f"Multi-label target column '{column}' must be binary, got values: {sorted(values)}")


def time_limit_per_label(time_limit: float | None, label_count: int) -> float | None:
    if time_limit is None or label_count <= 0:
        return time_limit
    return time_limit / label_count


def series_from_binary_proba(
    predictions: pd.DataFrame | pd.Series | np.ndarray | list,
    *,
    label: str,
    index: Iterable,
) -> pd.Series:
    if isinstance(predictions, pd.DataFrame):
        if label in predictions.columns:
            series = predictions[label]
        else:
            series = predictions.iloc[:, -1]
        return pd.Series(series.to_numpy(), index=index, name=label)

    if isinstance(predictions, pd.Series):
        return pd.Series(predictions.to_numpy(), index=index, name=label)

    array = np.asarray(predictions)
    if array.ndim > 1:
        array = array[:, -1]
    return pd.Series(array, index=index, name=label)


def series_from_predictions(
    predictions: pd.Series | np.ndarray | list,
    *,
    label: str,
    index: Iterable,
) -> pd.Series:
    if isinstance(predictions, pd.Series):
        return pd.Series(predictions.to_numpy(), index=index, name=label)

    array = np.asarray(predictions)
    if array.ndim > 1:
        if array.shape[1] != 1:
            raise ValueError(f"Expected one prediction column for '{label}', got shape {array.shape}")
        array = array[:, 0]
    return pd.Series(array, index=index, name=label)
