from typing import List, Optional, Union
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.language_models import LanguageModelLike
from owlai.core.interfaces import ModelOperations
from owlai.core.config import ModelConfig


class ModelManager(ModelOperations):
    """Manages model interactions and token counting"""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._model = None

    @property
    def model(self) -> LanguageModelLike:
        """Lazy initialization of the model"""
        if self._model is None:
            self._model = ChatOpenAI(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
            )
        return self._model

    def get_completion(self, messages: List[BaseMessage]) -> str:
        """Get completion from model"""
        if not messages:
            raise ValueError("Messages list cannot be empty")

        try:
            response = self.model.invoke(messages)
            if isinstance(response, str):
                return response
            return str(response)
        except Exception as e:
            raise Exception(f"Error getting model completion: {str(e)}")

    def count_tokens(self, message: Union[BaseMessage, List[BaseMessage]]) -> int:
        """Count tokens in a message or list of messages"""
        messages = [message] if isinstance(message, BaseMessage) else message

        if not isinstance(messages, list):
            raise ValueError("Expected a message or list of messages")

        total_tokens = 0
        for msg in messages:
            if not isinstance(msg, BaseMessage):
                raise ValueError("Invalid message type")
            # Use the model's token counter if available
            if hasattr(self.model, "get_num_tokens"):
                total_tokens += self.model.get_num_tokens(str(msg.content))
            # Fallback to a simple approximation
            else:
                total_tokens += len(str(msg.content).split())

        return total_tokens
