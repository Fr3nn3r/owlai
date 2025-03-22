import pytest
from unittest.mock import Mock, patch
from langchain_core.tools import BaseTool
from owlai.core.factory import OwlAgentFactory
from owlai.core.config import ModelConfig, AgentConfig


class MockTool(BaseTool):
    """Mock tool for testing"""

    name: str = "mock_tool"
    description: str = "A mock tool for testing"

    def _run(self, **kwargs):
        return "mock result"


@pytest.fixture
def model_config():
    return ModelConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        context_size=2000,
    )


@pytest.fixture
def agent_config(model_config):
    return AgentConfig(
        name="test_agent",
        description="Test agent",
        system_prompt="You are a test agent",
        llm_config=model_config,
    )


def test_factory_creation():
    """Test factory creation"""
    factory = OwlAgentFactory()
    assert isinstance(factory, OwlAgentFactory)


def test_factory_create_agent(agent_config):
    """Test creating an agent"""
    factory = OwlAgentFactory()
    agent = factory.create(config=agent_config)
    assert agent.config == agent_config


def test_factory_create_multiple_agents(agent_config):
    """Test creating multiple agents"""
    factory = OwlAgentFactory()
    agent1 = factory.create(config=agent_config)
    agent2 = factory.create(config=agent_config)

    assert agent1 is not agent2  # Should create new instances
    assert agent1.config == agent2.config  # Should have same config
