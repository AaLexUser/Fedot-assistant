import os
import logging
import openai
from typing import List, Dict, Any
from omegaconf import DictConfig
import pprint
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class AssistantChatOpenAI:
    def __init__(self, config: DictConfig):
        self.history_ = []
        self.input_ = 0
        self.output_ = 0

        self.model = config.model
        self.base_url = config.base_url
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

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
            "output": self.output_
        }

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
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