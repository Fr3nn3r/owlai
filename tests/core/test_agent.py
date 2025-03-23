import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from owlai.core.agent import OwlAgent
from owlai.core.config import AgentConfig, ModelConfig


class MockTool(BaseTool):
    """Mock tool for testing"""

    name: str = "mock_tool"
    description: str = "A mock tool for testing"

    def _run(self, tool_input: str = "") -> str:
        return "mock result"


@pytest.fixture
def agent_config():
    model_config = ModelConfig(provider="openai", model_name="gpt-4")
    return AgentConfig(
        name="test_agent",
        description="Test agent",
        system_prompt="You are a test agent",
        llm_config=model_config,
    )


@pytest.fixture
def mock_message_manager():
    manager = Mock()
    manager.get_message_history.return_value = []
    return manager


@pytest.fixture
def mock_tool_manager():
    """Create a properly configured mock tool manager with _tools dictionary"""
    manager = MagicMock()
    # Create a mock dictionary for _tools
    manager._tools = {}
    # Return an empty list of keys when accessing _tools.keys()
    return manager


@pytest.fixture
def mock_model_manager():
    manager = Mock()
    # Return AIMessage instead of string
    manager.get_completion.return_value = AIMessage(content="Test response")
    return manager


@pytest.fixture
def agent(agent_config, mock_message_manager, mock_tool_manager, mock_model_manager):
    agent = OwlAgent(agent_config)
    agent.message_manager = mock_message_manager
    agent.tool_manager = mock_tool_manager
    agent.model_manager = mock_model_manager
    return agent


def test_agent_initialization(agent):
    """Test agent initialization"""
    assert agent.config is not None
    assert agent.message_manager is not None
    assert agent.tool_manager is not None
    assert agent.model_manager is not None


def test_message_invoke(agent):
    """Test message invocation without tool calls"""
    # Create a proper AIMessage with no tool calls
    ai_message = AIMessage(content="Test response")
    agent.model_manager.get_completion.return_value = ai_message

    agent.message_manager.get_message_history.return_value = [
        SystemMessage(content="You are a test agent"),
        HumanMessage(content="Hello"),
    ]

    response = agent.message_invoke("Hello")
    assert response == "Test response"
    agent.message_manager.append_message.assert_called()
    agent.model_manager.get_completion.assert_called_once()


def test_tool_processing(agent):
    """Test tool processing"""
    # Create an AIMessage with tool calls
    ai_message = AIMessage(
        content="I'll use the mock tool",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "mock_tool",
                        "arguments": "{}",
                    },
                }
            ]
        },
    )

    # Set up tool execution to return a specific result
    agent.model_manager.get_completion.side_effect = [
        ai_message,  # First call returns the tool call
        AIMessage(
            content="Final response after tool"
        ),  # Second call after tool execution
    ]

    # Set up tool manager to return our mock tool
    mock_tool = Mock()
    mock_tool.run.return_value = "Mock tool result"
    agent.tool_manager.get_tool.return_value = mock_tool

    agent.message_manager.get_message_history.return_value = [
        SystemMessage(content="You are a test agent"),
        HumanMessage(content="Use the mock tool"),
    ]

    response = agent.message_invoke("Use the mock tool")

    # Verify the tool was called
    agent.tool_manager.get_tool.assert_called_once_with("mock_tool")
    mock_tool.run.assert_called_once()

    # Verify final response
    assert response == "Final response after tool"


def test_multiple_tool_calls(agent):
    """Test handling of messages with multiple tool calls"""
    # Create an AIMessage with multiple tool calls
    ai_message = AIMessage(
        content="I'll use multiple tools",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "tool_1",
                        "arguments": "{}",
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "tool_2",
                        "arguments": "{}",
                    },
                },
            ]
        },
    )

    # We'll just test the first tool call for simplicity
    agent.model_manager.get_completion.side_effect = [
        ai_message,
        AIMessage(content="Final response"),
    ]

    # Set up tool manager to return mock tools
    mock_tool = Mock()
    mock_tool.run.return_value = "Mock tool result"
    agent.tool_manager.get_tool.return_value = mock_tool

    agent.message_manager.get_message_history.return_value = [
        SystemMessage(content="You are a test agent"),
        HumanMessage(content="Use both tools"),
    ]

    response = agent.message_invoke("Use both tools")

    # Since our agent processes the first tool and returns, we expect just one tool call
    agent.tool_manager.get_tool.assert_called_once()
    mock_tool.run.assert_called_once()
    assert response == "Final response"


def test_tool_error_handling(agent):
    """Test tool error handling"""
    # Create an AIMessage with tool calls
    ai_message = AIMessage(
        content="I'll use the tool",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "mock_tool",
                        "arguments": "{}",
                    },
                }
            ]
        },
    )

    agent.model_manager.get_completion.return_value = ai_message

    # Set up tool to raise an exception
    mock_tool = Mock()
    mock_tool.run.side_effect = Exception("Tool execution failed")
    agent.tool_manager.get_tool.return_value = mock_tool

    agent.message_manager.get_message_history.return_value = [
        SystemMessage(content="You are a test agent"),
        HumanMessage(content="Use the tool"),
    ]

    # The agent should raise an exception when the tool fails
    with pytest.raises(Exception) as exc_info:
        agent.message_invoke("Use the tool")

    # Verify the exception message mentions tool execution
    assert "Tool execution failed" in str(exc_info.value)


def test_system_prompt_handling(agent):
    """Test system prompt handling"""
    agent.message_manager.get_message_history.return_value = [
        SystemMessage(content="You are a test agent")
    ]

    messages = agent.message_manager.get_message_history()
    assert len(messages) == 1
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "You are a test agent"


def test_message_history_management(agent):
    """Test message history management"""
    # Just return a simple AIMessage
    agent.model_manager.get_completion.return_value = AIMessage(content="Response")

    agent.message_manager.get_message_history.return_value = [
        SystemMessage(content="You are a test agent"),
        HumanMessage(content="Hello"),
        AIMessage(content="Response"),
    ]

    response = agent.message_invoke("Hello")
    assert response == "Response"

    messages = agent.message_manager.get_message_history()
    assert len(messages) == 3  # System + User + AI
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)


def test_model_error_handling(agent):
    """Test model error handling"""
    agent.model_manager.get_completion.side_effect = Exception("Model error")
    with pytest.raises(Exception) as exc_info:
        agent.message_invoke("Hello")
    assert "Model error" in str(exc_info.value)


def test_empty_message_handling(agent):
    """Test empty message handling"""
    with pytest.raises(ValueError, match="Message cannot be empty"):
        agent.message_invoke("")
    with pytest.raises(ValueError, match="Message cannot be empty"):
        agent.message_invoke("   ")


def test_old_format_tool_call_handling(agent):
    """Test handling of older tool call format"""
    # Create an AIMessage with old format tool calls (no function wrapper)
    old_format_message = AIMessage(
        content="I'll use the mock tool",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_id",
                    "name": "mock_tool",
                    "args": {"param": "value"},
                }
            ]
        },
    )

    agent.model_manager.get_completion.side_effect = [
        old_format_message,
        AIMessage(content="Final response"),
    ]

    # Set up tool manager to return mock tool
    mock_tool = Mock()
    mock_tool.run.return_value = "Mock tool result"
    agent.tool_manager.get_tool.return_value = mock_tool

    response = agent.message_invoke("Use the tool with old format")

    # Verify the tool was called correctly
    agent.tool_manager.get_tool.assert_called_once_with("mock_tool")
    mock_tool.run.assert_called_once()
    assert response == "Final response"


def test_no_tool_calls_handling(agent):
    """Test message with empty tool_calls list"""
    # Create an AIMessage with empty tool calls list
    message = AIMessage(
        content="I don't need to use tools",
        additional_kwargs={"tool_calls": []},  # Empty list
    )

    agent.model_manager.get_completion.return_value = message

    response = agent.message_invoke("No tools needed")

    # Should just return the content without trying to process tools
    assert response == "I don't need to use tools"
    agent.tool_manager.get_tool.assert_not_called()


def test_invalid_tool_call_format_handling(agent):
    """Test handling of invalid tool call format"""
    # Create an AIMessage with invalid tool call format
    invalid_format_message = AIMessage(
        content="This has a malformed tool call",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_id",
                    # Missing name and function
                }
            ]
        },
    )

    agent.model_manager.get_completion.return_value = invalid_format_message

    # Should just return the content without crashing
    response = agent.message_invoke("Invalid tool format")
    assert response == "This has a malformed tool call"
