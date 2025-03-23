from typing import List, Optional
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from owlai.core.interfaces import MessageOperations
from owlai.core.config import ModelConfig
from owlai.core.logging_setup import get_logger


class MessageManager(MessageOperations):
    """Manages message history and token counting"""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._messages: List[BaseMessage] = []
        self.fifo_mode = False
        self.logger = get_logger("messages")

    def append_message(self, message: BaseMessage) -> None:
        """Add a message to the history"""
        if not isinstance(message, BaseMessage):
            self.logger.warning(
                f"Invalid message type: {type(message).__name__}, expected BaseMessage"
            )
            raise ValueError("Message must be a BaseMessage instance")

        # Convert AIMessages with tool role to proper ToolMessages
        if (
            isinstance(message, AIMessage)
            and message.additional_kwargs.get("role") == "tool"
        ):
            # For tool response messages, convert to proper ToolMessage
            tool_call_id = message.additional_kwargs.get("tool_call_id", "")
            name = message.additional_kwargs.get("name", "")

            # Create a proper ToolMessage
            preserved_message = ToolMessage(
                content=message.content,
                tool_call_id=tool_call_id,
                name=name,
            )
            self._messages.append(preserved_message)
            self.logger.debug(
                f"Converted AIMessage to ToolMessage: tool_call_id={tool_call_id}, name={name}"
            )
        else:
            # For non-tool messages, preserve all additional kwargs
            preserved_message = message
            if isinstance(message, AIMessage):
                # For AI messages, ensure we preserve tool calls if present
                if hasattr(message, "tool_calls") and message.tool_calls:
                    preserved_message = AIMessage(
                        content=message.content,
                        additional_kwargs={
                            "tool_calls": [
                                {
                                    "id": tc.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": tc.get("name", ""),
                                        "arguments": str(tc.get("args", {})),
                                    },
                                }
                                for tc in message.tool_calls
                            ],
                        },
                    )
            self._messages.append(preserved_message)
            self.logger.debug(
                f"Added message: type={type(message).__name__}, content_length={len(message.content)}"
            )

        total_tokens = sum(self._count_tokens(msg) for msg in self._messages)

        if total_tokens > self.model_config.context_size:
            self.fifo_mode = True
            while total_tokens > self.model_config.context_size and self._messages:
                removed = self._messages.pop(0)
                total_tokens -= self._count_tokens(removed)

    def get_message_history(self) -> List[BaseMessage]:
        """Get the message history"""
        return self._messages

    def clear_history(self) -> None:
        """Clear the message history"""
        message_count = len(self._messages)
        self._messages = []
        self.fifo_mode = False
        self.logger.debug(f"Cleared {message_count} messages from history")

    def _count_tokens(self, message: BaseMessage) -> int:
        """Count tokens in a message"""
        if hasattr(message, "response_metadata") and message.response_metadata:
            return message.response_metadata.get("token_usage", {}).get(
                "total_tokens", 0
            )
        return 0
