import hashlib
import json
import logging
import os
import pprint
from pathlib import Path
from typing import Any, Dict, List, cast

from dotenv import load_dotenv
from langfuse.decorators import observe
from omegaconf import DictConfig, OmegaConf
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fedotllm.runtime_paths import get_llm_cache_dir
from fedotllm.utils.configs import load_config

load_dotenv()

logger = logging.getLogger(__name__)


class AssistantChatOpenAI:
    def __init__(self, config: DictConfig):
        self.history_ = []
        self.input_ = 0
        self.output_ = 0

        self.model = config.model
        self.base_url = config.get("base_url", None)
        self.temperature = config.get("temperature", 0)
        self.max_tokens = config.get("max_tokens", 512)
        raw_extra_body = config.get("extra_body", None)
        if raw_extra_body is not None and OmegaConf.is_config(raw_extra_body):
            raw_extra_body = OmegaConf.to_container(raw_extra_body, resolve=True)
        self.extra_body = raw_extra_body or None
        self.cache_dir = get_llm_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if "FEDOTLLM_LLM_API_KEY" in os.environ:
            api_key = os.environ["FEDOTLLM_LLM_API_KEY"]
        else:
            raise Exception("OpenAI API env variable FEDOTLLM_LLM_API_KEY not set")

        logger.info(f"FedotLLM is using model {config.model} to assist you with the task.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "extra_body": self.extra_body,
            "cache_dir": str(self.cache_dir),
            "history": self.history_,
            "input": self.input_,
            "output": self.output_,
        }

    def _cache_path(self, messages: List[Dict[str, str]]) -> Path:
        cache_key = json.dumps(
            {
                "messages": messages,
                "model": self.model,
                "base_url": self.base_url,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "extra_body": self.extra_body,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return self.cache_dir / f"{hashlib.sha256(cache_key.encode()).hexdigest()}.txt"

    def _append_history(
        self,
        messages: List[Dict[str, str]],
        output: Any,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        self.history_.append(
            {
                "input": messages,
                "output": pprint.pformat(output),
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            }
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                RuntimeError,
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                InternalServerError,
            )
        ),
        reraise=True,
    )
    @observe()
    def invoke(self, messages: List[Dict[str, str]]):
        cache_path = self._cache_path(messages)
        try:
            cached_content = cache_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            cached_content = None
        except OSError as exc:
            logger.warning("Failed to read LLM cache file %s: %s", cache_path, exc)
            cached_content = None

        if cached_content is not None:
            self._append_history(
                messages=messages,
                output={"cached": True, "content": cached_content},
                prompt_tokens=0,
                completion_tokens=0,
            )
            return cached_content

        request_kwargs: dict[str, Any] = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            # "max_completion_tokens": self.max_tokens,
        }
        if self.extra_body is not None:
            request_kwargs["extra_body"] = self.extra_body

        response = self.client.chat.completions.create(**cast(Any, request_kwargs))

        if getattr(response, "error", None):
            raise RuntimeError(f"LLM request failed: {response.error}")

        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("LLM response did not include any choices")

        prompt_tokens = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        completion_tokens = getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if content is None:
            # Log response details for debugging before retrying
            logger.warning(f"LLM response missing content. Response: {response}")
            raise RuntimeError("LLM response did not include message content")

        self.input_ += prompt_tokens
        self.output_ += completion_tokens

        self._append_history(
            messages=messages,
            output=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        try:
            tmp_cache_path = cache_path.with_suffix(".tmp")
            tmp_cache_path.write_text(content, encoding="utf-8")
            tmp_cache_path.replace(cache_path)
        except OSError as exc:
            logger.warning("Failed to write LLM cache file %s: %s", cache_path, exc)
        return content


if __name__ == "__main__":
    config = load_config()

    assistant = AssistantChatOpenAI(config.llm)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Russia?"},
    ]

    response = assistant.invoke(messages)
    print("Response:", response)
    print("History:", assistant.describe())
