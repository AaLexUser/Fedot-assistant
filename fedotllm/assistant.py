from omegaconf import DictConfig, OmegaConf
from .llm import AssistantChatOpenAI
from .predictor import AutogluonTabularPredictor
from .utils import get_feature_transformers_config


class TabularPredictionAssistant:
    """A TabularPredictionAssistant performs a supervised tabular learning task"""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.llm = AssistantChatOpenAI(config)
        self.predictor = AutogluonTabularPredictor(config.autogluon)
        self.feature_transformers_config = get_feature_transformers_config(config)
        