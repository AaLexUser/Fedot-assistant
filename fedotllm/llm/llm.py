import logging
import os
import pprint
from typing import Any, Dict, List

from dotenv import load_dotenv
from langfuse.decorators import observe
from omegaconf import DictConfig
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

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

        if "FEDOTLLM_LLM_API_KEY" in os.environ:
            api_key = os.environ["FEDOTLLM_LLM_API_KEY"]
        else:
            raise Exception("OpenAI API env variable FEDOTLLM_LLM_API_KEY not set")

        logger.info(
            f"FedotLLM is using model {config.model} to assist you with the task."
        )

        self.client = OpenAI(
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
    @observe()
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
