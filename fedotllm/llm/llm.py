import logging
import os
import pprint
from typing import Any, Dict, List, Optional

import openai
import tiktoken
from omegaconf import DictConfig
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def count_tokens(
    text: str, model: Optional[str] = None, encoding_name: Optional[str] = None
) -> int:
    """Counts the number of tokens in a given text using specified model or encoding.

    This function uses tiktoken to encode text and count tokens. It can use model-specific
    encoding or a specified encoding name. If neither is provided, it defaults to 'cl100k_base'.

    Args:
        text (str): The text to count tokens for.
        model (Optional[str]): The name of the model to use for token counting.
            If provided, its encoding will be used.
        encoding_name (Optional[str]): The name of the encoding to use.
            Only used if model is None.

    Returns:
        int: The number of tokens in the text.

    Example:
        >>> count_tokens("Hello, world!", model="gpt-35-turbo")
        4
        >>> count_tokens("Hello, world!", encoding_name="cl100k_base")
        4
    """
    if model is not None:
        encoding_name = tiktoken.encoding_for_model(model).name
    if encoding_name is not None:
        return len(tiktoken.get_encoding(encoding_name).encode(text)) if text else 0
    else:
        return len(tiktoken.get_encoding("cl100k_base").encode(text)) if text else 0


def clip_text(
    text: str,
    model: str = "gpt-4o",
    max_tokens: int = 8000,
    trim_ratio: float = 0.75,
) -> str:
    max_tokens = int(trim_ratio * max_tokens)
    tokens = count_tokens(text, model=model)
    if tokens <= max_tokens:
        return text
    else:
        return text[:max_tokens]


class AssistantChatOpenAI:
    def __init__(self, config: DictConfig):
        self.history_ = []
        self.input_ = 0
        self.output_ = 0

        self.model = config.model
        self.base_url = config.get("base_url", None)
        self.temperature = config.get("temperature", 0)
        self.max_tokens = config.get("max_tokens", 512)
        self.max_input_tokens = config.get("max_input_tokens", 4096)

        if "OPENAI_API_KEY" in os.environ:
            api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise Exception("OpenAI API env variable OPENAI_API_KEY not set")

        logger.info(
            f"FedotLLM is using model {config.model} from OpenAI to assist you with the task."
        )

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "history": self.history_,
            "input": self.input_,
            "output": self.output_,
        }

    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def invoke(self, messages: List[Dict[str, str]]):
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
        )

        self.input_ += response.usage.prompt_tokens
        self.output_ += response.usage.completion_tokens

        self.history_.append(
            {
                "input": messages,
                "output": pprint.pformat(response),
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        )
        return response.choices[0].message.content
