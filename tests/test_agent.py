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
    # Set up the name property
    type(mock_instance).name = property(lambda self: "mock_search")
    # Set up the description property
    type(mock_instance).description = property(
        lambda self: "A mock search tool for testing"
    )
    # Set up the run method
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
        # Create an actual AIMessage instance
        mock_instance.invoke.return_value = AIMessage(
            content="Let me search for that information.",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "mock_search", "arguments": "{}"},
                    }
                ]
            },
        )
        mock.return_value = mock_instance
        mock.return_value.bind_tools = MagicMock(return_value=mock_instance)
        yield mock


def test_agent_uses_tool_when_appropriate(mock_tool, agent_config, mock_model):
    """Test that the agent uses tools when the query requires it"""
    # Create agent and register tool
    agent = OwlAgent(config=agent_config)
    agent.register_tool(mock_tool)

    # Configure mock model response with an actual AIMessage
    mock_response = AIMessage(
        content="Let me search for that information.",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "mock_search", "arguments": "{}"},
                }
            ]
        },
    )
    mock_model.return_value.invoke.return_value = mock_response

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
        content="Hello! I'm doing well, thank you for asking.",
        additional_kwargs={},  # No tool calls
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

    # Configure mock model response with an actual AIMessage
    mock_response = AIMessage(
        content="Let me search for that information.",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "mock_search", "arguments": "{}"},
                }
            ]
        },
    )
    mock_model.return_value.invoke.return_value = mock_response

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


def test_agent_registers_tools_correctly(mock_tool, agent_config, mock_model):
    """Test that tools are registered correctly and available to the model"""
    # Create agent and register tool
    agent = OwlAgent(config=agent_config)
    agent.register_tool(mock_tool)

    # Configure mock model response with an actual AIMessage
    mock_response = AIMessage(
        content="Let me search for that information.",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "mock_search", "arguments": "{}"},
                }
            ]
        },
    )
    mock_model.return_value.invoke.return_value = mock_response

    # Verify tool is registered in ToolManager
    assert mock_tool.name in agent.tool_manager._tools
    assert agent.tool_manager._tools[mock_tool.name] == mock_tool

    # Verify tool details are preserved
    registered_tool = agent.tool_manager._tools[mock_tool.name]
    assert registered_tool.name == mock_tool.name
    assert registered_tool.description == mock_tool.description

    # Process a query that should trigger the tool
    query = "Search for information about Python programming"
    response = agent.message_invoke(query)

    # Get the message history
    history = agent.message_manager.get_message_history()

    # Find the AI message with tool calls (should be the second-to-last message)
    ai_messages = [msg for msg in history if isinstance(msg, AIMessage)]
    assert len(ai_messages) >= 2, "Expected at least 2 AI messages in history"
    ai_message = ai_messages[-2]  # Get the AI message before the tool result

    # Verify the model's response includes tool calls
    assert "tool_calls" in ai_message.additional_kwargs
    assert len(ai_message.additional_kwargs["tool_calls"]) > 0

    # Verify the tool was called with the correct name
    mock_tool.run.assert_called_once_with()

    # Verify the response contains the tool's result
    assert "Mock search result" in response

    # Verify the tool is mentioned in the system prompt
    system_message = agent.message_manager.get_message_history()[0]
    assert mock_tool.name.lower() in system_message.content.lower()
    assert mock_tool.description.lower() in system_message.content.lower()
