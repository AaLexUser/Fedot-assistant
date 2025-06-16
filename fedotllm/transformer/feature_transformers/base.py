from ..base import BaseTransformer, TransformTimeoutError
import pandas as pd
from fedotllm.task import PredictionTask
import logging
import traceback
from typing import Tuple
import copy

logger = logging.getLogger(__name__)


class BaseFeatureTransformer(BaseTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fit_dataframes(
        self, train_X: pd.DataFrame, train_y: pd.Series, **kwargs
    ) -> None:
        raise NotImplementedError

    def fit(self, task: PredictionTask, *args, **kwargs) -> "BaseFeatureTransformer":
        try:
            train_x = task.train_data.drop(
                columns=task.columns_in_train_but_not_test + [task.train_id_column],
                errors="ignore",
            )
            train_y = task.train_data[task.label_column]

            self._fit_dataframes(
                train_X=train_x,
                train_y=train_y,
                target_column_name=task.label_column,
                problem_type=task.problem_type,
                dataset_description=task.metadata["description"],
            )
        except TransformTimeoutError:
            logger.warning(f"FeatureTransformer {self.__class__.__name__} timed out.")
        except Exception:
            logger.warning(
                f"FeatureTransformer {self.__class__.__name__} failed to fit."
            )
            logger.warning(traceback.format_exc())
        finally:
            return self

    def _transform_dataframes(
        self, train_X: pd.DataFrame, test_X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def transform(self, task: PredictionTask, *args, **kwargs) -> PredictionTask:
        try:
            train_x = task.train_data.drop(
                columns=task.columns_in_train_but_not_test + [task.train_id_column],
                errors="ignore",
            )
            label_column_available_in_test_set = False
            if task.label_column in train_x.columns:
                # Label Column also present in test data
                # requires explicit dropping of the column
                # from train set before feature transformation
                train_x = train_x.drop(columns=[task.label_column])
            train_y = task.train_data[task.label_column]

            if task.test_id_column in task.test_data.columns:
                test_x = task.test_data.drop(columns=[task.test_id_column])
            else:
                # Skip if test_id_column is not found
                test_x = task.test_data
            if task.label_column in test_x.columns:
                # Label Column also present in test data
                # requires explicit dropping of the column
                # from test set before feature transformation
                label_column_available_in_test_set = True
                test_x = test_x.drop(columns=[task.label_column])
                test_y = task.test_data[task.label_column]

            train_x, test_x = self._transform_dataframes(train_X=train_x, test_X=test_x)

            # add back label columns
            transformed_train_data = pd.concat(
                [train_x, train_y.rename(task.label_column)], axis=1
            )
            if label_column_available_in_test_set:
                # Add back label column to test set as it was available before
                transformed_test_data = pd.concat(
                    [test_x, test_y.rename(task.label_column)], axis=1
                )
            else:
                transformed_test_data = test_x

            # add back id columns
            if task.train_id_column in task.train_data.columns:
                transformed_train_data = pd.concat(
                    [transformed_train_data, task.train_data[task.train_id_column]],
                    axis=1,
                )
            if task.test_id_column in task.test_data.columns:
                transformed_test_data = pd.concat(
                    [transformed_test_data, task.test_data[task.test_id_column]], axis=1
                )

            task = copy.deepcopy(task)
            task.train_data = transformed_test_data
            task.test_data = transformed_test_data
        except Exception as e:
            logger.warning(
                f"FeatureTransformer {self.__class__.__name__} failed to transform. Error: {str(e)}"
            )
        finally:
            return task
