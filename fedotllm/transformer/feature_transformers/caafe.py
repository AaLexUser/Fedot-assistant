import logging
import warnings
from typing import Mapping, Tuple

import pandas as pd

from fedotllm.constants import BINARY, MULTICLASS

from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)
warnings.filterwarnings(action="ignore")

try:
    from caafe import CAAFEClassifier
    from caafe.run_llm_code import run_llm_code
except ImportError:
    raise ImportError(
        "CAAFE required for feature generation but not installed. Please intall with `pip install caafe@git+https://github.com/AnirudhDagar/CAAFE.git@main`"
    )


class CAAFETransformer(BaseFeatureTransformer):
    identifier = "caafe"

    def __init__(
        self,
        num_iterations: int = 2,
        optimization_metric: str = "roc",
        eval_model: str = "lightgdm",
        **kwargs,
    ) -> None:
        pd.set_option("future.no_silent_downcasting", True)

        self.iterations = num_iterations
        self.optimization_metric = optimization_metric
        self.eval_model = eval_model

        # Initialize the base classifier
        if self.eval_model == "tab_pfn":
            from tabpfn import TabPFNClassifier

            clf_no_feat_eng = TabPFNClassifier(
                device="cpu", N_ensemble_configurations=16
            )
        elif self.eval_model == "lightgdm":
            from lightgbm import LGBMClassifier

            clf_no_feat_eng = LGBMClassifier()
        else:
            raise ValueError(f"Unsupported CAAFE eval model: {self.eval_model}")

        self.caafe_clf = CAAFEClassifier(
            base_classifier=clf_no_feat_eng,
            optimization_metric=self.optimization_metric,
            iterations=self.iterations,
            display_method="print",
            **kwargs,
        )

        self.metadata = {
            "transformer": "CAAFE",
        }

    def _fit_dataframes(
        self,
        train_X: pd.DataFrame,
        train_y: pd.Series,
        *,
        target_column_name: str,
        problem_type: str = "binary",
        dataset_description: str = "",
        **kwargs,
    ) -> None:
        if problem_type not in (BINARY, MULTICLASS):
            logger.info(
                "Feature transformer CAAFE only supports classification problems."
            )
            return

        categorical_target = not pd.api.types.is_numeric_dtype(train_y)
        if categorical_target:
            encoded_y, _ = train_y.factorize()

        self.caafe_clf.fit(
            train_X.to_numpy(),
            encoded_y if categorical_target else train_y.to_numpy(),
            dataset_description,
            train_X.columns,
            target_column_name,
        )

        logger.info("CAAFE generated features:")
        logger.info(self.caafe_clf.code)

    def _transform_dataframes(
        self, train_X: pd.DataFrame, test_X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        transformed_train_X = run_llm_code(self.caafe_clf.code, train_X)
        transformed_test_X = run_llm_code(self.caafe_clf.code, test_X)

        return transformed_train_X, transformed_test_X

    def get_metadata(self) -> Mapping:
        return self.metadata
