"""Unit tests for core.py module."""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from owlai.core import (
    DefaultAgentInput,
    LLMConfig,
    OwlAgent,
    user_context,
)


class MockToolSchema(BaseModel):
    """Mock schema for testing."""

    param: str


@pytest.fixture
def mock_chat_model():
    """Create a mock chat model."""
    mock = Mock()
    mock.invoke.return_value = AIMessage(
        content="Test response",
        response_metadata={"token_usage": {"total_tokens": 100}},
    )
    mock.bind_tools = Mock(return_value=mock)
    return mock


@pytest.fixture
def mock_tool():
    """Create a mock tool."""
    mock = Mock(spec=BaseTool)
    mock.name = "test_tool"
    mock.invoke.return_value = "Tool result"
    mock.args_schema = MockToolSchema
    return mock


@pytest.fixture
def owl_agent():
    """Create a test OwlAgent instance."""
    config = LLMConfig(
        model_provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=2048,
        context_size=4096,
        tools_names=["test_tool"],
    )
    agent = OwlAgent(
        name="test_agent",
        description="Test agent",
        llm_config=config,
        system_prompt="You are a test agent.",
    )
    return agent


def test_default_agent_input():
    """Test DefaultAgentInput model validation."""
    # Valid input
    input_data = DefaultAgentInput(query="test query")
    assert input_data.query == "test query"

    # Invalid input
    with pytest.raises(ValidationError):
        DefaultAgentInput()


def test_llm_config():
    """Test LLMConfig model validation."""
    # Valid config
    config = LLMConfig(
        model_provider="openai",
        model_name="gpt-3.5-turbo",
    )
    assert config.model_provider == "openai"
    assert config.model_name == "gpt-3.5-turbo"
    assert config.temperature == 0.1  # default value
    assert config.max_tokens == 2048  # default value
    assert config.context_size == 4096  # default value
    assert config.tools_names == []  # default value

    # Invalid config
    with pytest.raises(ValidationError):
        LLMConfig()


def test_owl_agent_initialization():
    """Test OwlAgent initialization."""
    config = LLMConfig(
        model_provider="openai",
        model_name="gpt-3.5-turbo",
    )
    agent = OwlAgent(
        name="test_agent",
        description="Test agent",
        llm_config=config,
        system_prompt="You are a test agent.",
    )
    assert agent.name == "test_agent"
    assert agent.description == "Test agent"
    assert agent.system_prompt == "You are a test agent."
    assert agent.total_tokens == 0
    assert agent.fifo_message_mode is False
    assert agent.callable_tools == []
    assert agent._chat_model_cache is None
    assert agent._tool_dict == {}
    assert agent._message_history == []


def test_chat_model_property(owl_agent, mock_chat_model):
    """Test chat_model property initialization."""
    with patch("owlai.core.init_chat_model") as mock_init:
        mock_init.return_value = mock_chat_model
        # Access the property to trigger initialization
        chat_model = owl_agent.chat_model
        assert chat_model == mock_chat_model
        mock_init.assert_called_once_with(
            model="gpt-3.5-turbo",
            model_provider="openai",
            temperature=0.1,
            max_tokens=2048,
        )


def test_init_callable_tools(owl_agent, mock_tool, mock_chat_model):
    """Test initialization of callable tools."""
    with patch("owlai.core.init_chat_model") as mock_init:
        mock_init.return_value = mock_chat_model
        tools = [mock_tool]
        owl_agent.init_callable_tools(tools)
        assert owl_agent.callable_tools == tools
        assert owl_agent._tool_dict == {"test_tool": mock_tool}
        mock_chat_model.bind_tools.assert_called_once_with(tools)


def test_token_count(owl_agent):
    """Test token counting for different message types."""
    # Test OpenAI token counting
    openai_message = AIMessage(
        content="test",
        response_metadata={"token_usage": {"total_tokens": 100}},
    )
    assert owl_agent._token_count(openai_message) == 100

    # Test Anthropic token counting
    owl_agent.llm_config.model_provider = "anthropic"
    anthropic_message = AIMessage(
        content="test",
        response_metadata={
            "usage": {"input_tokens": 50, "output_tokens": 50},
        },
    )
    assert owl_agent._token_count(anthropic_message) == 100

    # Test unsupported provider
    owl_agent.llm_config.model_provider = "unsupported"
    assert owl_agent._token_count(anthropic_message) == -1

    # Test non-AIMessage
    human_message = HumanMessage(content="test")
    assert owl_agent._token_count(human_message) == 0


def test_append_message(owl_agent):
    """Test message appending and FIFO mode."""
    # Initialize message history with system message
    system_message = SystemMessage(content="You are a test agent.")
    owl_agent._message_history = [system_message]

    # Test initial message
    human_message = HumanMessage(content="test")
    owl_agent.append_message(human_message)
    assert len(owl_agent._message_history) == 2
    assert owl_agent._message_history[1] == human_message

    # Test FIFO mode activation
    owl_agent.total_tokens = owl_agent.llm_config.context_size + 1
    owl_agent.append_message(human_message)
    assert owl_agent.fifo_message_mode is True
    assert len(owl_agent._message_history) == 2  # Should maintain size


def test_process_tool_calls(owl_agent, mock_tool):
    """Test processing of tool calls."""
    # Setup
    owl_agent._tool_dict = {"test_tool": mock_tool}
    owl_agent.llm_config.tools_names = ["test_tool"]
    tool_call = {
        "name": "test_tool",
        "args": {"param": "value"},
        "id": "test_id",
        "type": "function",  # Required by langchain
    }

    # Test successful tool call
    response = AIMessage(
        content="test",
        tool_calls=[tool_call],
    )
    owl_agent._process_tool_calls(response)
    mock_tool.invoke.assert_called_once_with({"param": "value"})

    # Test invalid tool call
    invalid_response = AIMessage(
        content="test",
        tool_calls=[
            {
                "name": "invalid_tool",
                "args": {},
                "id": "invalid_id",
                "type": "function",
            }
        ],
    )
    initial_history_len = len(owl_agent._message_history)
    owl_agent._process_tool_calls(invalid_response)
    # Message history should have one more message (the error message)
    assert len(owl_agent._message_history) == initial_history_len + 1


def test_message_invoke(owl_agent, mock_chat_model):
    """Test message invocation."""
    with patch("owlai.core.init_chat_model") as mock_init:
        mock_init.return_value = mock_chat_model
        response = owl_agent.message_invoke("test message")
        assert response == "Test response"
        assert len(owl_agent._message_history) == 3  # system + user + response
        mock_chat_model.invoke.assert_called_once()


@pytest.mark.asyncio
async def test_async_message_invoke(owl_agent, mock_chat_model):
    """Test async message invocation."""
    with patch("owlai.core.init_chat_model") as mock_init:
        mock_init.return_value = mock_chat_model
        response = await owl_agent._arun("test message")
        assert response == "Test response"


def test_error_handling(owl_agent, mock_chat_model):
    """Test error handling in message_invoke."""
    with patch("owlai.core.init_chat_model") as mock_init:
        mock_init.return_value = mock_chat_model
        mock_chat_model.invoke.side_effect = Exception("Test error")
        response = owl_agent.message_invoke("test message")
        assert response == "Error: Test error"
