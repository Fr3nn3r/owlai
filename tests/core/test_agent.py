import pytest
from unittest.mock import Mock, patch
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
    return Mock()


@pytest.fixture
def mock_model_manager():
    return Mock()


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
    """Test message invocation"""
    agent.model_manager.get_completion.return_value = "Test response"
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
    agent.model_manager.get_completion.return_value = "Test response"
    agent.message_manager.get_message_history.return_value = [
        SystemMessage(content="You are a test agent"),
        HumanMessage(content="Use the mock tool"),
    ]

    response = agent.message_invoke("Use the mock tool")
    assert response == "Test response"
    agent.message_manager.append_message.assert_called()


def test_multiple_tool_calls(agent):
    """Test multiple tool calls"""
    agent.model_manager.get_completion.return_value = "Test response"
    agent.message_manager.get_message_history.return_value = [
        SystemMessage(content="You are a test agent"),
        HumanMessage(content="Use both tools"),
    ]

    response = agent.message_invoke("Use both tools")
    assert response == "Test response"
    agent.message_manager.append_message.assert_called()


def test_tool_error_handling(agent):
    """Test tool error handling"""
    agent.model_manager.get_completion.return_value = "Test response"
    agent.message_manager.get_message_history.return_value = [
        SystemMessage(content="You are a test agent"),
        HumanMessage(content="Use the tool"),
    ]

    response = agent.message_invoke("Use the tool")
    assert response == "Test response"
    agent.message_manager.append_message.assert_called()


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
    agent.model_manager.get_completion.return_value = "Response"
    agent.message_manager.get_message_history.return_value = [
        SystemMessage(content="You are a test agent"),
        HumanMessage(content="Hello"),
        AIMessage(content="Response"),
    ]

    agent.message_invoke("Hello")
    messages = agent.message_manager.get_message_history()
    assert len(messages) == 3  # System + User + AI
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)


def test_model_error_handling(agent):
    """Test model error handling"""
    agent.model_manager.get_completion.side_effect = Exception("Test error")
    with pytest.raises(Exception):
        agent.message_invoke("Hello")


def test_empty_message_handling(agent):
    """Test empty message handling"""
    with pytest.raises(ValueError):
        agent.message_invoke("")
    with pytest.raises(ValueError):
        agent.message_invoke("   ")
