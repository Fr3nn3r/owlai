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
        # Create an actual AIMessage instance for the first response
        tool_call_message = AIMessage(
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
        # Create a follow-up response after the tool execution
        final_response = AIMessage(
            content="Based on the search, I found: Mock search result."
        )
        # Set up the invoke method to return these messages in sequence
        mock_instance.invoke.side_effect = [tool_call_message, final_response]

        mock.return_value = mock_instance
        mock.return_value.bind_tools = MagicMock(return_value=mock_instance)
        yield mock


def test_agent_uses_tool_when_appropriate(mock_tool, agent_config, mock_model):
    """Test that the agent uses tools when the query requires it"""
    # Create agent and register tool
    agent = OwlAgent(config=agent_config)
    agent.register_tool(mock_tool)

    # Process a query that should trigger the tool
    query = "Search for information about Python programming"
    response = agent.message_invoke(query)

    # Verify the tool was called with empty string (default behavior)
    mock_tool.run.assert_called_once_with("")

    # Verify the response contains the tool's result
    assert "Mock search result" in response


def test_agent_does_not_use_tool_when_unnecessary(mock_tool, agent_config, mock_model):
    """Test that the agent doesn't use tools for simple queries"""
    # Reset the side_effect and set a direct return value for this test
    mock_model.return_value.invoke.side_effect = None
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
    """Test that the agent properly propagates tool errors"""
    # Configure the mock tool to raise an exception
    mock_tool.run.side_effect = Exception("Tool error")

    # Reset model side_effect
    mock_model.return_value.invoke.side_effect = None

    # Configure mock model response with an actual AIMessage
    mock_model.return_value.invoke.return_value = AIMessage(
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

    # Create agent and register tool
    agent = OwlAgent(config=agent_config)
    agent.register_tool(mock_tool)

    # Process a query that should trigger the tool, but expect an error
    query = "Search for information about Python programming"

    # The agent should propagate the tool error
    with pytest.raises(Exception) as exc_info:
        agent.message_invoke(query)

    # Verify the exception contains the tool error message
    assert "Tool error" in str(exc_info.value)


def test_agent_registers_tools_correctly(mock_tool, agent_config, mock_model):
    """Test that tools are registered correctly and available to the model"""
    # Setup message manager mock with call tracking
    message_manager_mock = MagicMock()
    message_manager_mock.get_message_history.return_value = []

    # Create agent with the custom message manager mock
    agent = OwlAgent(config=agent_config)
    agent.message_manager = message_manager_mock
    agent.register_tool(mock_tool)

    # Verify tool is registered in ToolManager
    assert mock_tool.name in agent.tool_manager._tools
    assert agent.tool_manager._tools[mock_tool.name] == mock_tool

    # Verify tool details are preserved
    registered_tool = agent.tool_manager._tools[mock_tool.name]
    assert registered_tool.name == mock_tool.name
    assert registered_tool.description == mock_tool.description

    # Process a query that triggers the tool
    query = "Search for information about Python programming"
    agent.message_invoke(query)

    # Verify the message_manager.append_message was called
    assert message_manager_mock.append_message.called

    # Verify at least 3 calls (user message, preserved message, tool response)
    assert message_manager_mock.append_message.call_count >= 3

    # Find any calls with AIMessage containing tool_calls
    has_tool_call_message = False
    for call in message_manager_mock.append_message.call_args_list:
        args = call[0]
        if len(args) > 0 and isinstance(args[0], AIMessage):
            additional_kwargs = args[0].additional_kwargs
            if additional_kwargs and "tool_calls" in additional_kwargs:
                has_tool_call_message = True
                break

    # Assert we found a message with tool_calls
    assert (
        has_tool_call_message
    ), "No AIMessage with tool_calls found in append_message calls"
