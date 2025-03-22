from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelLike


class MessageOperations(ABC):
    """Interface for message operations"""

    @abstractmethod
    def append_message(self, message: BaseMessage) -> None:
        """Add a message to history"""
        pass

    @abstractmethod
    def get_message_history(self) -> List[BaseMessage]:
        """Get current message history"""
        pass

    @abstractmethod
    def clear_history(self) -> None:
        """Clear message history"""
        pass


class ToolOperations(ABC):
    """Interface for tool operations"""

    @abstractmethod
    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool"""
        pass

    @abstractmethod
    def get_tool(self, name: str) -> BaseTool:
        """Get a tool by name"""
        pass

    @abstractmethod
    def invoke_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Invoke a tool with given arguments"""
        pass


class ModelOperations(ABC):
    """Interface for model operations"""

    @abstractmethod
    def get_completion(self, messages: List[BaseMessage]) -> str:
        """Get completion from model"""
        pass

    @abstractmethod
    def count_tokens(self, message: Union[BaseMessage, List[BaseMessage]]) -> int:
        """Count tokens in a message or list of messages"""
        pass

    @abstractmethod
    def get_model(self) -> LanguageModelLike:
        """Get the underlying model instance"""
        pass
