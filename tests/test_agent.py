import pytest
from unittest.mock import patch, MagicMock
from owlai.core.agent import OwlAgent
from owlai.core.config import AgentConfig, ModelConfig
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage


@pytest.fixture
def mock_tool():
    # Create a mock tool instance
    mock_instance = MagicMock(spec=BaseTool)
    mock_instance.name = "mock_search"
    mock_instance.description = "A mock search tool for testing"
    mock_instance.run = MagicMock(return_value="Mock search result")
    return mock_instance


@pytest.fixture
def agent_config():
    return AgentConfig(
        name="test_agent",
        description="Test agent for unit testing",
        system_prompt="You are a helpful AI assistant that uses tools when appropriate.",
        llm_config=ModelConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.1,
            max_tokens=100,
            context_size=1000,
        ),
    )


@pytest.fixture
def mock_model():
    with patch("owlai.core.model_manager.init_chat_model") as mock:
        mock_instance = MagicMock()
        # Create an AIMessage with tool calls
        message = AIMessage(content="Let me search for that information.")
        message.additional_kwargs = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "mock_search", "arguments": "{}"},
                }
            ]
        }
        mock_instance.invoke.return_value = message
        mock.return_value = mock_instance
        yield mock


def test_agent_uses_tool_when_appropriate(mock_tool, agent_config, mock_model):
    """Test that the agent uses tools when the query requires it"""
    # Create agent and register tool
    agent = OwlAgent(config=agent_config)
    agent.register_tool(mock_tool)

    # Process a query that should trigger the tool
    query = "Search for information about Python programming"
    response = agent.message_invoke(query)

    # Verify the tool was called
    mock_tool.run.assert_called_once_with()

    # Verify the response contains the tool's result
    assert "Mock search result" in response


def test_agent_does_not_use_tool_when_unnecessary(mock_tool, agent_config, mock_model):
    """Test that the agent doesn't use tools for simple queries"""
    # Configure model to not use tools for this query
    mock_model.return_value.invoke.return_value = AIMessage(
        content="Hello! I'm doing well, thank you for asking."
    )

    # Create agent and register tool
    agent = OwlAgent(config=agent_config)
    agent.register_tool(mock_tool)

    # Process a simple query that shouldn't need the tool
    query = "Hello, how are you?"
    response = agent.message_invoke(query)

    # Verify the tool was NOT called
    mock_tool.run.assert_not_called()

    # Verify the response doesn't contain the tool's result
    assert "Mock search result" not in response


def test_agent_handles_tool_errors_gracefully(mock_tool, agent_config, mock_model):
    """Test that the agent handles tool errors gracefully"""
    # Configure the mock tool to raise an exception
    mock_tool.run.side_effect = Exception("Tool error")

    # Create agent and register tool
    agent = OwlAgent(config=agent_config)
    agent.register_tool(mock_tool)

    # Process a query that should trigger the tool
    query = "Search for information about Python programming"
    response = agent.message_invoke(query)

    # Verify the tool was called
    mock_tool.run.assert_called_once_with()

    # Verify the response indicates an error occurred
    assert "error" in response.lower()
