"""
                  ,_,
                 (O,O)
                 (   )
                 -"-"-
Examples of different ways to stub and mock chat models for testing.
"""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from owlai.core import OwlAgent, OwlAIAgent
from tests.conftest import MockChatModel


class TestStubChatModels:
    """
    Examples of different approaches to stubbing chat models for testing.
    """

    def test_stub_with_mock_class(self, mock_tool):
        """
        Approach 1: Use a complete mock implementation of BaseChatModel.
        """
        # Create predefined responses
        responses = [
            AIMessage(content="First response"),
            AIMessage(
                content="Second response with tool call",
                tool_calls=[
                    {
                        "name": mock_tool.name,
                        "args": {"arg": "value"},
                        "id": "call1",
                    }
                ],
            ),
            AIMessage(content="Final response after tool call"),
        ]

        # Create the mock chat model with predefined responses
        mock_model = MockChatModel(responses)

        # Patch init_chat_model to return our mock model
        with patch("owlai.core.init_chat_model", return_value=mock_model):
            # Create the agent
            agent = OwlAgent(
                model_provider="test", model_name="test", system_prompt="Test prompt"
            )

            # Add the tool
            agent.tools_names = [mock_tool.name]
            agent._tool_dict = {mock_tool.name: mock_tool}

            # Invoke the agent
            response = agent.invoke("Test message")

            # Verify the response is the final message from our sequence
            assert response == "Final response after tool call"

            # Verify that our mock model was invoked the expected number of times
            assert mock_model.invoke_count == 3

            # Verify the tool was called
            mock_tool.invoke.assert_called_once()

    def test_stub_with_magicmock(self, mock_tool):
        """
        Approach 2: Use MagicMock for more flexible control.
        """
        # Create a MagicMock for the chat model
        mock_chat_model = MagicMock()

        # Set up the responses for consecutive calls
        mock_response1 = AIMessage(content="Response with tool call")
        mock_response1.tool_calls = [
            {"name": mock_tool.name, "arguments": {"arg": "value"}, "id": "call1"}
        ]
        mock_response1.response_metadata = {"token_usage": {"total_tokens": 50}}

        mock_response2 = AIMessage(content="Final response")
        mock_response2.response_metadata = {"token_usage": {"total_tokens": 30}}

        # Configure the mock to return different responses on consecutive calls
        mock_chat_model.invoke.side_effect = [mock_response1, mock_response2]

        # Make the bind_tools method return the mock itself
        mock_chat_model.bind_tools.return_value = mock_chat_model

        # Patch init_chat_model
        with patch(
            "owlai.core.init_chat_model", return_value=mock_chat_model
        ) as mock_init:
            # Create the agent
            agent = OwlAgent(
                model_provider="test", model_name="test", system_prompt="Test prompt"
            )

            # Add the tool
            agent.tools_names = [mock_tool.name]
            agent._tool_dict = {mock_tool.name: mock_tool}

            # Invoke the agent
            response = agent.invoke("Test message")

            # Verify the response
            assert response == "Final response"

            # Verify the chat model was called twice
            assert mock_chat_model.invoke.call_count == 2

            # Verify the tool was called
            mock_tool.invoke.assert_called_once()

    def test_stub_owlai_agent(self, mock_chat_model, mock_agent_graph):
        """
        Approach 3: Stub the OwlAIAgent which uses langgraph.
        """
        # Configure patching for init_chat_model and create_react_agent
        with patch(
            "langchain.chat_models.init_chat_model",
            return_value=mock_chat_model,
        ), patch("owlai.core.create_react_agent", return_value=mock_agent_graph):

            # Configure the mock agent graph response
            mock_agent_graph.invoke.return_value = {
                "messages": [
                    HumanMessage(content="Test input"),
                    AIMessage(content="Test output"),
                ]
            }

            # Create the agent
            agent = OwlAgent(
                model_provider="test", model_name="test", system_prompt="Test prompt"
            )

            # Invoke the agent
            response = agent.invoke("Test input")

            # Verify the response
            assert response == "Test output"
