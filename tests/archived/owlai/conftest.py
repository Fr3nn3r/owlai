"""
                  ,_,
                 (O,O)
                 (   )
                 -"-"-
Common fixtures for OwlAI pytest testing.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Optional

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
)
from langchain_core.language_models.chat_models import BaseChatModel


class MockChatModel(BaseChatModel):
    """
    A simple mock chat model that can be used for testing.
    This approach creates a complete mock implementation of BaseChatModel.
    """

    def __init__(self, responses=None):
        """
        Initialize the mock chat model with predefined responses.

        Args:
            responses: A list of responses to return or a single response.
                      If a list, responses will be returned in sequence.
        """
        super().__init__()
        # Store responses as a public attribute for test access
        self.responses = []

        if responses is None:
            self.responses = [AIMessage(content="Default mock response")]
        elif isinstance(responses, list):
            self.responses = responses
        else:
            self.responses = [responses]

        self.response_index = 0
        self.invoke_count = 0
        self.invoke_inputs = []

    @property
    def _llm_type(self) -> str:
        """Return the type of language model."""
        return "mock_chat_model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """
        Mock implementation of the _generate method.
        """
        raise NotImplementedError("This mock is for synchronous invoke only")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """
        Mock implementation of the _agenerate method.
        """
        raise NotImplementedError("This mock is for synchronous invoke only")

    def invoke(self, input, config=None):
        """
        Mock implementation of the invoke method.
        """
        self.invoke_count += 1
        self.invoke_inputs.append(input)

        if self.response_index >= len(self.responses):
            # Cycle back to the first response if we've used them all
            self.response_index = 0

        response = self.responses[self.response_index]
        self.response_index += 1

        # Add metadata for token counting
        if isinstance(response, AIMessage) and not hasattr(
            response, "response_metadata"
        ):
            response.response_metadata = {"token_usage": {"total_tokens": 100}}

        return response

    def bind_tools(self, tools):
        """
        Mock implementation of bind_tools that returns self for simplicity.
        """
        return self


@pytest.fixture
def mock_chat_model():
    """
    Fixture that returns a simple mock chat model.
    """
    model = MagicMock()
    # Configure default response
    model.invoke.return_value = AIMessage(content="Default mock response")
    # Add metadata for token counting
    model.invoke.return_value.response_metadata = {"token_usage": {"total_tokens": 50}}
    # Make bind_tools return the model itself
    model.bind_tools.return_value = model
    return model


@pytest.fixture
def init_chat_model_patch(mock_chat_model):
    """
    Fixture that patches the init_chat_model function to return a mock chat model.
    """
    with patch("owlai.core.init_chat_model", return_value=mock_chat_model) as mock:
        yield mock


@pytest.fixture
def mock_tool():
    """
    Fixture that creates a mock tool.
    """
    from langchain_core.tools import BaseTool

    tool = MagicMock(spec=BaseTool)
    tool.name = "test_tool"
    tool.invoke.return_value = "Tool result"
    return tool


@pytest.fixture
def mock_agent_graph():
    """
    Fixture that creates a mock agent graph for langgraph testing.
    """
    graph = MagicMock()
    return graph


@pytest.fixture
def create_react_agent_patch(mock_agent_graph):
    """
    Fixture that patches the create_react_agent function.
    """
    with patch("owlai.core.create_react_agent", return_value=mock_agent_graph) as mock:
        yield mock
