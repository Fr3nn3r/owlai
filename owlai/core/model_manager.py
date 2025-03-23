from typing import List, Optional, Union, Dict, Any, cast
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langchain_core.language_models import LanguageModelLike, BaseChatModel
from langchain_core.runnables import RunnableSerializable
from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from owlai.core.config import ModelConfig
from owlai.core.interfaces import ModelOperations
from owlai.core.logging_setup import get_logger, debug_print


class ModelManager(ModelOperations):
    """Manages model interactions and token counting

    This class provides a unified interface to different LLM providers through LangChain's
    BaseChatModel abstraction. It handles model initialization, tool binding, and token counting.
    """

    def __init__(self, model_config: ModelConfig):
        """Initialize the ModelManager with the given configuration.

        Args:
            model_config: Configuration for the model including provider, model name,
                         temperature, and max tokens.
        """
        self.model_config = model_config
        self._model: Optional[Any] = None
        self._tools: List[BaseTool] = []  # Store the actual tools
        self.logger = get_logger("model")
        self.logger.info(
            f"Initializing ModelManager: provider={model_config.provider}, model_name={model_config.model_name}, "
            f"temperature={model_config.temperature}, max_tokens={model_config.max_tokens}"
        )

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool with the model

        Args:
            tool: The tool to register with the model.
        """
        self.logger.debug(
            f"Registering tool: {tool.name} with model: {self.model_config.model_name}"
        )

        # Store the tool
        self._tools.append(tool)

        # If model exists, bind tools immediately
        if self._model:
            self.logger.debug(f"Binding {len(self._tools)} tools to existing model")
            # We know this is a BaseChatModel if it's not None
            self._model = self._model.bind_tools(self._tools)
        else:
            # If model doesn't exist yet, it will be bound when initialized
            self.logger.debug(
                f"Tool registered, will be bound when model is initialized"
            )

    def get_model(self) -> Any:
        """Get the chat model.

        Returns:
            A configured and initialized chat model instance.
        """
        if not self._model:
            self.logger.debug("Creating new model instance")
            try:
                # Initialize the model using init_chat_model
                model = init_chat_model(
                    model=self.model_config.model_name,
                    model_provider=self.model_config.provider,
                    temperature=self.model_config.temperature,
                    max_tokens=self.model_config.max_tokens,
                )

                # Bind tools if any are registered
                if self._tools:
                    self.logger.debug(f"Binding {len(self._tools)} tools to model")
                    self._model = model.bind_tools(self._tools)
                else:
                    self._model = model

                self.logger.info(
                    f"Model initialized successfully: model_name={self.model_config.model_name}, "
                    f"provider={self.model_config.provider}, tool_count={len(self._tools)}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize model: error={str(e)}, model_name={self.model_config.model_name}, "
                    f"provider={self.model_config.provider}"
                )
                raise
        return self._model

    def get_completion(self, messages: List[BaseMessage]) -> AIMessage:
        """Get a completion from the chat model.

        Args:
            messages: List of messages to send to the model.

        Returns:
            The model's response as an AIMessage.
        """
        model = self.get_model()
        debug_print(self.logger, "Getting completion", model)
        debug_print(self.logger, "Messages", messages)
        response = model.invoke(messages)
        return cast(AIMessage, response)

    def count_tokens(self, message: Union[BaseMessage, List[BaseMessage]]) -> int:
        """Count tokens in a message or list of messages

        Args:
            message: A single message or list of messages to count tokens for.

        Returns:
            Total token count.
        """
        messages = [message] if isinstance(message, BaseMessage) else message

        if not isinstance(messages, list):
            self.logger.warning(
                f"Invalid message type for token counting: type={type(message).__name__}"
            )
            raise ValueError("Expected a message or list of messages")

        try:
            model = self.get_model()
            total_tokens = 0

            if hasattr(model, "get_num_tokens_from_messages"):
                total_tokens = model.get_num_tokens_from_messages(messages)
            else:
                # Fallback for models without token counting
                for msg in messages:
                    if not isinstance(msg, BaseMessage):
                        self.logger.warning(
                            f"Invalid message type in list: type={type(msg).__name__}"
                        )
                        raise ValueError("Invalid message type")
                    # Simple approximation for token counting
                    total_tokens += len(str(msg.content).split())

            self.logger.debug(
                f"Token count completed: message_count={len(messages)}, total_tokens={total_tokens}"
            )
            return total_tokens
        except Exception as e:
            self.logger.error(
                f"Failed to count tokens: error={str(e)}, message_count={len(messages)}"
            )
            raise
