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
    LLMConfig,
    OwlAgent,
)


class MockToolSchema(BaseModel):
    """Mock schema for testing."""

    param: str

    model_config = {"extra": "ignore"}


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
def mock_streaming_chat_model():
    """Create a mock chat model specifically for streaming tests."""

    class MockStreamingChatModel:
        def __init__(self):
            self.invoke_called = False
            self.astream_called = False

        def invoke(self, messages):
            self.invoke_called = True
            # Return a response with tool calls to trigger astream
            return AIMessage(
                content="Test response",
                response_metadata={"token_usage": {"total_tokens": 100}},
                tool_calls=[
                    {
                        "name": "test_tool",
                        "args": {"param": "value"},
                        "id": "test_id",
                        "type": "function",
                    }
                ],
            )

        async def astream(self, messages):
            self.astream_called = True
            yield AIMessage(content="Test response")

        def bind_tools(self, tools):
            return self

    return MockStreamingChatModel()


@pytest.fixture
def mock_tool():
    """Create a mock tool."""
    mock = Mock(spec=BaseTool)
    mock.name = "test_tool"
    mock.description = "A test tool for unit testing"
    mock.return_direct = False
    mock.invoke.return_value = "Tool result"
    mock.args_schema = MockToolSchema
    mock.tool_call_schema = MockToolSchema
    return mock


@pytest.fixture
def mock_tool_factory():
    """Create a mock ToolFactory that returns our mock tool."""
    mock_factory = Mock()
    return mock_factory


@pytest.fixture
def owl_agent(mock_tool, mock_chat_model):
    """Create a test OwlAgent instance with mocked dependencies."""
    # Patch the ToolFactory.get_tools to return our mock tool
    with patch("owlai.services.toolbox.ToolFactory.get_tools") as mock_get_tools:
        mock_get_tools.return_value = [mock_tool]

        # Patch the init_chat_model to return our mock chat model
        with patch("owlai.core.init_chat_model") as mock_init:
            mock_init.return_value = mock_chat_model

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
                version="1.0",
            )

            return agent


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
    """Test OwlAgent initialization with automatic tool loading."""
    mock_tool = Mock(spec=BaseTool)
    mock_tool.name = "test_tool"

    # Patch both the ToolFactory and chat model initialization
    with patch("owlai.services.toolbox.ToolFactory.get_tools") as mock_get_tools:
        mock_get_tools.return_value = [mock_tool]

        with patch("owlai.core.init_chat_model") as mock_init:
            mock_chat_model = Mock()
            mock_chat_model.bind_tools.return_value = mock_chat_model
            mock_init.return_value = mock_chat_model

            config = LLMConfig(
                model_provider="openai",
                model_name="gpt-3.5-turbo",
                tools_names=["test_tool"],
            )

            agent = OwlAgent(
                name="test_agent",
                description="Test agent",
                llm_config=config,
                system_prompt="You are a test agent.",
                version="1.0",
            )

            # Verify initialization
            assert agent.name == "test_agent"
            assert agent.description == "Test agent"
            assert agent.system_prompt == "You are a test agent."
            assert agent.total_tokens == 0
            assert agent.fifo_message_mode is False

            # Verify tools were automatically initialized
            assert len(agent.callable_tools) == 1
            assert agent.callable_tools[0] == mock_tool

            # Check tool dictionary contents by key
            assert list(agent._tool_dict.keys()) == ["test_tool"]
            assert agent._tool_dict["test_tool"] == mock_tool

            assert len(agent._message_history) == 1

            # Verify chat model was properly set up
            mock_get_tools.assert_called_once_with(["test_tool"])
            mock_chat_model.bind_tools.assert_called_once()


def test_chat_model_property(owl_agent, mock_chat_model):
    """Test chat_model property already initialized during agent creation."""
    # The chat model should already be initialized in the owl_agent fixture
    assert owl_agent.chat_model == mock_chat_model


def test_init_callable_tools():
    """Test explicitly calling init_callable_tools after initialization."""
    mock_tool = Mock(spec=BaseTool)
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool for unit testing"
    mock_tool.args_schema = MockToolSchema
    mock_tool.tool_call_schema = MockToolSchema

    # Create a new tool to add after initialization
    new_mock_tool = Mock(spec=BaseTool)
    new_mock_tool.name = "new_test_tool"
    new_mock_tool.description = "Another test tool"
    new_mock_tool.args_schema = MockToolSchema
    new_mock_tool.tool_call_schema = MockToolSchema

    with patch("owlai.services.toolbox.ToolFactory.get_tools") as mock_get_tools:
        # Return the initial mock tool for automatic initialization
        mock_get_tools.return_value = [mock_tool]

        with patch("owlai.core.init_chat_model") as mock_init:
            mock_chat_model = Mock()
            mock_chat_model.bind_tools.return_value = mock_chat_model
            mock_init.return_value = mock_chat_model

            config = LLMConfig(
                model_provider="openai",
                model_name="gpt-3.5-turbo",
                tools_names=["test_tool"],
            )

            agent = OwlAgent(
                name="test_agent",
                description="Test agent",
                llm_config=config,
                system_prompt="You are a test agent.",
                version="1.0",
            )

            # Verify initial tools
            assert len(agent.callable_tools) == 1
            assert agent.callable_tools[0] == mock_tool
            assert "test_tool" in agent._tool_dict
            assert agent._tool_dict["test_tool"] == mock_tool

            # Now explicitly call init_callable_tools with a new tool
            agent.init_callable_tools([new_mock_tool])

            # Verify tools list was updated
            assert len(agent.callable_tools) == 1
            assert agent.callable_tools[0] == new_mock_tool

            # Check that both tools are in the tool dictionary
            # Note: The implementation currently adds to the dictionary rather than replacing it
            assert "test_tool" in agent._tool_dict
            assert "new_test_tool" in agent._tool_dict
            assert agent._tool_dict["test_tool"] == mock_tool
            assert agent._tool_dict["new_test_tool"] == new_mock_tool

            # Verify the chat model was bound to the new tools
            mock_chat_model.bind_tools.assert_called_with([new_mock_tool])


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
    # No need to set up tool_dict as it's already set up in the fixture
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
    mock_tool.invoke.assert_called_with({"param": "value"})

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
    # The chat model is already patched in the owl_agent fixture
    response = owl_agent.message_invoke("test message")
    assert response == "Test response"
    # Check that messages were added to history
    assert len(owl_agent._message_history) >= 2  # At least system + user message
    assert mock_chat_model.invoke.called


@pytest.mark.asyncio
async def test_async_message_invoke(mock_streaming_chat_model, mock_tool):
    """Test async message invocation."""
    with patch("owlai.services.toolbox.ToolFactory.get_tools") as mock_get_tools:
        mock_get_tools.return_value = [mock_tool]

        with patch("owlai.core.init_chat_model") as mock_init:
            mock_init.return_value = mock_streaming_chat_model

            config = LLMConfig(
                model_provider="openai",
                model_name="gpt-3.5-turbo",
                tools_names=["test_tool"],
            )

            agent = OwlAgent(
                name="test_agent",
                description="Test agent",
                llm_config=config,
                system_prompt="You are a test agent.",
                version="1.0",
            )

            try:
                # Collect all chunks from the async generator
                chunks = []
                async for chunk in agent.stream_message("test message"):
                    chunks.append(chunk)

                # Assert that astream was called
                assert (
                    mock_streaming_chat_model.astream_called
                ), "astream method was not called"

                # Concatenate all chunks to get the full response
                response = "".join(chunks)
                assert (
                    response == "Test response"
                ), f"Expected 'Test response', got '{response}'"

            except Exception as e:
                pytest.fail(f"Test failed with exception: {str(e)}")


def test_error_handling(owl_agent, mock_chat_model):
    """Test error handling in message_invoke."""
    # Set up error in invoke
    mock_chat_model.invoke.side_effect = Exception("Test error")

    # Call method that should handle the error
    response = owl_agent.message_invoke("test message")

    # Verify error handling
    assert "Error: Test error" in response
