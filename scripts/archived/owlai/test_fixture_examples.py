"""
                  ,_,
                 (O,O)
                 (   )
                 -"-"-
Examples of using pytest fixtures for stubbing chat models.
"""

import pytest
from unittest.mock import patch

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from owlai.core import OwlAgent, OwlAIAgent
from tests.conftest import MockChatModel


@pytest.fixture
def simple_agent(init_chat_model_patch, mock_chat_model):
    """Fixture that returns a basic pre-configured agent."""
    agent = OwlAgent(
        model_provider="fixture_provider",
        model_name="fixture_model",
        system_prompt="Test from fixture",
    )
    return agent


@pytest.fixture
def agent_with_tool(simple_agent, mock_tool):
    """Fixture that returns an agent with a tool already configured."""
    agent = simple_agent
    agent.tools_names = [mock_tool.name]
    agent._tool_dict = {mock_tool.name: mock_tool}
    return agent


@pytest.fixture
def sequential_response_model():
    """Fixture for a chat model that returns sequential responses."""
    responses = [
        AIMessage(content="First response"),
        AIMessage(content="Second response"),
        AIMessage(content="Third response"),
    ]

    # Add metadata for token count
    for resp in responses:
        resp.response_metadata = {"token_usage": {"total_tokens": 50}}

    return MockChatModel(responses)


def test_using_simple_agent_fixture(simple_agent):
    """Example using the simple_agent fixture."""
    # Configure the response in the mock model that's already injected
    simple_agent.chat_model.invoke.return_value = AIMessage(
        content="Fixture test response"
    )
    simple_agent.chat_model.invoke.return_value.response_metadata = {
        "token_usage": {"total_tokens": 25}
    }

    # Invoke the agent
    response = simple_agent.invoke("Fixture test")

    # Verify the response
    assert response == "Fixture test response"

    # Verify the system prompt was set from the fixture
    assert simple_agent.system_prompt == "Test from fixture"

    # Verify the message history
    assert len(simple_agent._message_history) == 3
    assert isinstance(simple_agent._message_history[0], SystemMessage)
    assert isinstance(simple_agent._message_history[1], HumanMessage)
    assert isinstance(simple_agent._message_history[2], AIMessage)


def test_agent_with_tool_fixture(agent_with_tool, mock_tool):
    """Example using the agent_with_tool fixture with tool invocation."""
    # Configure the response with tool call
    tool_response = AIMessage(content="Using tool")
    tool_response.tool_calls = [
        {"name": mock_tool.name, "arguments": {"test": "value"}, "id": "call1"}
    ]
    tool_response.response_metadata = {"token_usage": {"total_tokens": 50}}

    final_response = AIMessage(content="Tool used successfully")
    final_response.response_metadata = {"token_usage": {"total_tokens": 30}}

    agent_with_tool.chat_model.invoke.side_effect = [
        tool_response,
        final_response,
    ]

    # Invoke the agent
    response = agent_with_tool.invoke("Use the test tool")

    # Verify response
    assert response == "Tool used successfully"

    # Verify tool was called
    mock_tool.invoke.assert_called_once()

    # Verify message history includes tool message
    assert len(agent_with_tool._message_history) == 5  # system, human, ai, tool, ai
    assert isinstance(agent_with_tool._message_history[3], ToolMessage)


@pytest.fixture
def configured_owlai_agent(
    init_chat_model_patch, create_react_agent_patch, mock_agent_graph
):
    """Fixture for an OwlAIAgent with configured graph."""
    # Set up the mock agent graph response
    mock_agent_graph.invoke.return_value = {
        "messages": [
            HumanMessage(content="Test input"),
            AIMessage(content="Test output from fixture"),
        ]
    }

    # Create the agent
    agent = OwlAIAgent(
        model_provider="test",
        model_name="test",
        system_prompt="Test prompt from fixture",
    )

    return agent


def test_owlai_agent_with_fixture(configured_owlai_agent):
    """Test using the configured_owlai_agent fixture."""
    # Invoke the agent
    response = configured_owlai_agent.invoke("Hello from fixture")

    # Verify the response
    assert response == "Test output from fixture"


def test_with_sequential_model(sequential_response_model):
    """Test using the sequential_response_model fixture."""
    # Use the model provided by the fixture
    model = sequential_response_model

    # Patch init_chat_model to return our sequential model
    with patch("owlai.core.init_chat_model", return_value=model):
        # Create agent
        agent = OwlAgent(
            model_provider="test", model_name="test", system_prompt="Sequential test"
        )

        # Make multiple invocations to see different responses
        response1 = agent.invoke("First call")
        response2 = agent.invoke("Second call")
        response3 = agent.invoke("Third call")

        # Verify sequential responses
        assert response1 == "First response"
        assert response2 == "Second response"
        assert response3 == "Third response"

        # Verify invocation count
        assert model.invoke_count == 3
