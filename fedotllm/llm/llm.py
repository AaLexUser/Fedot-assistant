import os
import logging
import openai
from typing import List, Dict
from omegaconf import DictConfig
import pprint

logger = logging.getLogger(__name__)


class AssistantChatOpenAI:
    def __init__(self, config: DictConfig):
        self.history_ = []
        self.input_ = 0
        self.output_ = 0

        self.model = config.model
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
            base_url=config.base_url,
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