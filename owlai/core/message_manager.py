from typing import List
from langchain_core.messages import BaseMessage, SystemMessage
from owlai.core.interfaces import MessageOperations
from owlai.core.config import ModelConfig


class MessageManager(MessageOperations):
    """Manages message history and context window"""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._messages: List[BaseMessage] = []
        self.fifo_mode = False

    def append_message(self, message: BaseMessage) -> None:
        """Add a message to history, managing context window"""
        if not isinstance(message, BaseMessage):
            raise ValueError("Invalid message type")

        self._messages.append(message)
        total_tokens = sum(self._count_tokens(msg) for msg in self._messages)

        if total_tokens > self.model_config.context_size:
            self.fifo_mode = True
            while total_tokens > self.model_config.context_size and self._messages:
                removed = self._messages.pop(0)
                total_tokens -= self._count_tokens(removed)

    def get_message_history(self) -> List[BaseMessage]:
        """Get current message history"""
        return self._messages.copy()

    def clear_history(self) -> None:
        """Clear message history"""
        self._messages.clear()
        self.fifo_mode = False

    def _count_tokens(self, message: BaseMessage) -> int:
        """Count tokens in a message"""
        if hasattr(message, "response_metadata") and message.response_metadata:
            return message.response_metadata.get("token_usage", {}).get(
                "total_tokens", 0
            )
        return 0
