import logging
import os
import time
import warnings
from typing import Any, Mapping, Tuple

import pandas as pd
from fedotllm.constants import BINARY, MULTICLASS
from omegaconf import OmegaConf

from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)
warnings.filterwarnings(action="ignore")

try:
    from caafe import CAAFEClassifier
    from caafe.llm_clients import litellm_client as _caafe_litellm_client
    from caafe.prompting import prompt_generator as _caafe_prompt_generator
    from caafe.prompting import utils as _caafe_prompt_utils
    from caafe.run_llm_code import run_llm_code
except ImportError:
    raise ImportError(
        "CAAFE required for feature generation but not installed. Please install with `pip install caafe@git+https://github.com/AaLexUser/CAAFE.git@main`"
    )

_CAAFE_INTEGRATION_PATCHED = False

_CAAFE_EMPTY_RETRY_NUDGE_DEFAULT = (
    "Your last reply had no visible assistant text. You must respond with a single "
    "Python code block wrapped in markdown fences exactly as in the instructions "
    "(```python ... ```). Do not return an empty message."
)


def _caafe_empty_content_retry_settings() -> tuple[int, float]:
    retries = int(os.getenv("CAAFE_LLM_EMPTY_CONTENT_RETRIES", "5"))
    backoff_s = float(os.getenv("CAAFE_LLM_EMPTY_CONTENT_BACKOFF_S", "0.5"))
    return max(1, retries), max(0.0, backoff_s)


def _build_caafe_completion_kwargs(
    completion_params: Mapping[str, Any], request_kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    completion_kwargs = {
        key: (
            OmegaConf.to_container(value, resolve=True)
            if OmegaConf.is_config(value)
            else value
        )
        for key, value in {**completion_params, **request_kwargs}.items()
    }
    completion_kwargs.pop("max_completion_tokens", None)
    if not completion_kwargs.get("extra_body"):
        completion_kwargs.pop("extra_body", None)
    return completion_kwargs


def _apply_caafe_integration_patches() -> None:
    """Retry empty LLM completions (common with some GLM APIs); harden extract_code."""
    global _CAAFE_INTEGRATION_PATCHED
    if _CAAFE_INTEGRATION_PATCHED:
        return

    _orig_extract = _caafe_prompt_utils.extract_code

    def extract_code_safe(response: str) -> str:
        if response is None:
            max_r, _ = _caafe_empty_content_retry_settings()
            raise ValueError(
                f"CAAFE LLM returned empty message content after {max_r} attempts. "
                "Increase CAAFE_LLM_EMPTY_CONTENT_RETRIES or check "
                "CAAFE_LLM_API_KEY / CAAFE_LLM_MODEL / CAAFE_LLM_BASE_URL."
            )
        if isinstance(response, str) and response.strip() == "":
            raise ValueError(
                "CAAFE LLM returned only whitespace after retries. "
                "Check CAAFE_LLM_MODEL and API credentials."
            )
        return _orig_extract(response)

    def query_retry_empty(self, messages: str | list[dict[str, Any]], **kwargs):
        import litellm

        max_retries, backoff_s = _caafe_empty_content_retry_settings()
        nudge = os.getenv(
            "CAAFE_LLM_EMPTY_RETRY_NUDGE", _CAAFE_EMPTY_RETRY_NUDGE_DEFAULT
        )
        msg_list: list[dict[str, Any]] = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else list(messages)
        )
        last: str | None = None
        for attempt in range(1, max_retries + 1):
            response = litellm.completion(
                messages=msg_list,
                **_build_caafe_completion_kwargs(self.completion_params, kwargs),
            )
            last = response.choices[0].message.content
            if last is not None and str(last).strip():
                return last
            if attempt < max_retries:
                delay = backoff_s * attempt
                logger.warning(
                    "CAAFE LLM returned empty message content (attempt %s/%s); "
                    "retrying in %.2fs with a follow-up user message (common with GLM).",
                    attempt,
                    max_retries,
                    delay,
                )
                msg_list = msg_list + [{"role": "user", "content": nudge}]
                time.sleep(delay)
        return last

    _caafe_litellm_client.LiteLLMClient.query = query_retry_empty
    _caafe_prompt_utils.extract_code = extract_code_safe
    _caafe_prompt_generator.extract_code = extract_code_safe
    _CAAFE_INTEGRATION_PATCHED = True


_apply_caafe_integration_patches()


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
