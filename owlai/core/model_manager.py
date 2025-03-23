from typing import List, Optional, Union
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.language_models import LanguageModelLike
from langchain.chat_models import init_chat_model
from owlai.core.config import ModelConfig
from owlai.core.interfaces import ModelOperations
from owlai.core.logging_setup import get_logger


class ModelManager(ModelOperations):
    """Manages model interactions and token counting"""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._model = None
        self.logger = get_logger("model")
        self.logger.info(
            "Initializing ModelManager",
            extra={
                "provider": model_config.provider,
                "model_name": model_config.model_name,
                "temperature": model_config.temperature,
                "max_tokens": model_config.max_tokens,
            },
        )

    def __repr__(self) -> str:
        """String representation of the model manager"""
        self.logger.debug("Generating string representation of ModelManager")
        return (
            f"ModelManager(provider='{self.model_config.provider}', "
            f"model_name='{self.model_config.model_name}', "
            f"temperature={self.model_config.temperature}, "
            f"max_tokens={self.model_config.max_tokens}, "
            f"context_size={self.model_config.context_size}, "
            f"model_initialized={self._model is not None})"
        )

    def get_model(self) -> LanguageModelLike:
        """Get the underlying model instance"""
        if self._model is None:
            self.logger.debug("Creating new model instance")
            try:
                self._model = init_chat_model(
                    model=self.model_config.model_name,
                    model_provider=self.model_config.provider,
                    temperature=self.model_config.temperature,
                    max_tokens=self.model_config.max_tokens,
                )
                self.logger.info(
                    "Model initialized successfully",
                    extra={
                        "model_name": self.model_config.model_name,
                        "provider": self.model_config.provider,
                    },
                )
            except Exception as e:
                self.logger.error(
                    "Failed to initialize model",
                    extra={
                        "error": str(e),
                        "model_name": self.model_config.model_name,
                        "provider": self.model_config.provider,
                    },
                )
                raise
        return self._model

    def get_completion(self, messages: List[BaseMessage]) -> AIMessage:
        """Get completion from model"""
        if not messages:
            self.logger.warning("Attempted to get completion with empty messages")
            raise ValueError("Messages list cannot be empty")

        try:
            self.logger.debug(
                "Getting model completion",
                extra={
                    "message_count": len(messages),
                    "last_message": str(messages[-1].content) if messages else None,
                },
            )
            response = self.get_model().invoke(messages)
            if isinstance(response, str):
                result = AIMessage(content=response)
            else:
                result = response

            self.logger.info(
                "Model completion successful",
                extra={
                    "message_count": len(messages),
                    "response_length": len(result.content),
                },
            )
            return result
        except Exception as e:
            self.logger.error(
                "Failed to get model completion",
                extra={
                    "error": str(e),
                    "message_count": len(messages),
                },
            )
            raise Exception(f"Error getting model completion: {str(e)}")

    def count_tokens(self, message: Union[BaseMessage, List[BaseMessage]]) -> int:
        """Count tokens in a message or list of messages"""
        messages = [message] if isinstance(message, BaseMessage) else message

        if not isinstance(messages, list):
            self.logger.warning(
                "Invalid message type for token counting",
                extra={"type": type(message).__name__},
            )
            raise ValueError("Expected a message or list of messages")

        try:
            total_tokens = 0
            for msg in messages:
                if not isinstance(msg, BaseMessage):
                    self.logger.warning(
                        "Invalid message type in list",
                        extra={"type": type(msg).__name__},
                    )
                    raise ValueError("Invalid message type")
                # Simple approximation for token counting
                total_tokens += len(str(msg.content).split())

            self.logger.debug(
                "Token count completed",
                extra={
                    "message_count": len(messages),
                    "total_tokens": total_tokens,
                },
            )
            return total_tokens
        except Exception as e:
            self.logger.error(
                "Failed to count tokens",
                extra={
                    "error": str(e),
                    "message_count": len(messages),
                },
            )
            raise
