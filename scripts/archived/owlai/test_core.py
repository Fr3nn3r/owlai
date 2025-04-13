"""
                  ,_,
                 (O,O)
                 (   )
                 -"-"-
Unit tests for the owlai.core module.
"""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from owlai.core import OwlAgent, OwlAIAgent


class TestOwlAgent:
    """
    Test cases for the OwlAgent class in core.py
    """

    @pytest.fixture
    def basic_agent(self, init_chat_model_patch, mock_chat_model):
        """
        Fixture that creates a basic OwlAgent for testing.
        """
        agent = OwlAgent(
            model_provider="test_provider",
            model_name="test_model",
            temperature=0.5,
            max_tokens=1000,
            context_size=2000,
            system_prompt="Test system prompt",
        )
        return agent

    def test_agent_initialization(self, basic_agent, mock_chat_model):
        """
        Test that the agent initializes correctly with the mocked chat model.
        """
        agent = basic_agent
        assert agent.model_provider == "test_provider"
        assert agent.model_name == "test_model"
        assert agent.temperature == 0.5
        assert agent.max_tokens == 1000
        assert agent.context_size == 2000
        assert agent.system_prompt == "Test system prompt"

        # Verify the chat model is our mock
        assert agent.chat_model == mock_chat_model

    def test_chat_model_lazy_loading(self, mock_chat_model):
        """
        Test that the chat_model property lazily loads the model.
        """
        # Create a new agent without accessing _chat_model_cache
        with patch(
            "owlai.core.init_chat_model", return_value=mock_chat_model
        ) as mock_init:
            agent = OwlAgent(
                model_provider="test_provider",
                model_name="test_model",
                system_prompt="Test prompt",
            )

            # Access the chat_model property
            chat_model = agent.chat_model

            # Verify init_chat_model was called with the correct parameters
            mock_init.assert_called_once_with(
                model="test_model",
                model_provider="test_provider",
                temperature=0.1,  # Default value
                max_tokens=2048,  # Default value
            )

            # Verify the chat model is our mock
            assert chat_model == mock_chat_model

    def test_init_callable_tools(self, basic_agent, mock_chat_model, mock_tool):
        """
        Test that tools are properly initialized.
        """
        agent = basic_agent

        # Create a second mock tool
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"

        tools = [mock_tool, mock_tool2]

        # Configure the chat model's bind_tools method
        bound_model = MagicMock()
        mock_chat_model.bind_tools.return_value = bound_model

        # Initialize tools
        result = agent.init_callable_tools(tools)

        # Verify bind_tools was called with the tools
        mock_chat_model.bind_tools.assert_called_once_with(tools)

        # Verify the result is the bound model
        assert result == bound_model

        # Verify the tools were added to the _tool_dict
        assert agent._tool_dict[mock_tool.name] == mock_tool
        assert agent._tool_dict[mock_tool2.name] == mock_tool2

        # Verify the agent's callable_tools was updated
        assert agent.callable_tools == tools

    def test_invoke_basic(self, basic_agent, mock_chat_model):
        """
        Test the basic invoke functionality.
        """
        agent = basic_agent

        # Configure mock response
        mock_response = AIMessage(content="Test response")
        mock_chat_model.invoke.return_value = mock_response

        # Configure mock token count
        mock_response.response_metadata = {"token_usage": {"total_tokens": 50}}

        # Call invoke
        response = agent.invoke("Test message")

        # Verify the response
        assert response == "Test response"

        # Verify the chat model was called correctly
        mock_chat_model.invoke.assert_called_once()

        # Verify messages were added to history
        assert (
            len(agent._message_history) == 3
        )  # System message + human message + ai message
        assert isinstance(agent._message_history[0], SystemMessage)
        assert isinstance(agent._message_history[1], HumanMessage)
        assert agent._message_history[1].content == "Test message"

    def test_invoke_with_tool_calls(self, basic_agent, mock_chat_model, mock_tool):
        """
        Test invoke with tool calls.
        """
        agent = basic_agent

        # Add tool to agent
        agent._tool_dict = {mock_tool.name: mock_tool}
        agent.tools_names = [mock_tool.name]

        # Configure mock response with tool calls
        mock_response1 = AIMessage(
            content="Using tool",
            tool_calls=[
                {
                    "name": mock_tool.name,
                    "args": {"arg1": "value1"},
                    "id": "call_id",
                }
            ],
        )
        mock_response2 = AIMessage(content="Final response")

        # Configure sequential responses
        mock_chat_model.invoke.side_effect = [mock_response1, mock_response2]

        # Mock token count
        mock_response1.response_metadata = {"token_usage": {"total_tokens": 30}}
        mock_response2.response_metadata = {"token_usage": {"total_tokens": 20}}

        # Call invoke
        response = agent.invoke("Test message with tool")

        # Verify the response
        assert response == "Final response"

        # Verify the chat model was called twice
        assert mock_chat_model.invoke.call_count == 2

        # Verify tool was called
        mock_tool.invoke.assert_called_once()

        # Verify messages in history: system, human, ai, tool, ai
        assert len(agent._message_history) == 5
        assert isinstance(agent._message_history[0], SystemMessage)
        assert isinstance(agent._message_history[1], HumanMessage)
        assert isinstance(agent._message_history[2], AIMessage)
        assert isinstance(agent._message_history[3], ToolMessage)
        assert isinstance(agent._message_history[4], AIMessage)
        assert agent._message_history[3].content == "Tool result"


class TestOwlAIAgent:
    """
    Test cases for the OwlAIAgent class in core.py
    """

    @pytest.fixture
    def basic_owlai_agent(
        self, init_chat_model_patch, create_react_agent_patch, mock_agent_graph
    ):
        """
        Fixture that creates a basic OwlAIAgent for testing.
        """
        agent = OwlAIAgent(
            model_provider="test_provider",
            model_name="test_model",
            temperature=0.5,
            max_tokens=1000,
            context_size=2000,
            system_prompt="Test system prompt",
        )
        return agent

    def test_agent_initialization(
        self, basic_owlai_agent, mock_chat_model, mock_agent_graph
    ):
        """
        Test that the agent initializes correctly with the mocked dependencies.
        """
        agent = basic_owlai_agent
        assert agent.model_provider == "test_provider"
        assert agent.model_name == "test_model"
        assert agent.temperature == 0.5
        assert agent.max_tokens == 1000
        assert agent.context_size == 2000
        assert agent.system_prompt == "Test system prompt"

        # Verify the chat model is our mock
        assert agent.chat_model == mock_chat_model

        # Verify the agent graph is our mock
        assert agent.agent_graph == mock_agent_graph

    def test_invoke(self, basic_owlai_agent, mock_agent_graph):
        """
        Test the invoke method.
        """
        agent = basic_owlai_agent

        # Configure mock response
        mock_response = {
            "messages": [
                HumanMessage(content="Test message"),
                AIMessage(content="Test response"),
            ]
        }
        mock_agent_graph.invoke.return_value = mock_response

        # Call invoke
        response = agent.invoke("Test message")

        # Verify the response
        assert response == "Test response"

        # Verify the agent graph was called with the correct parameters
        mock_agent_graph.invoke.assert_called_once()
        args, kwargs = mock_agent_graph.invoke.call_args
        assert list(args[0]["messages"])[0].content == "Test message"
