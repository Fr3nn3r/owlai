import pytest
from pydantic import ValidationError
from owlai.core.config import ModelConfig, AgentConfig


def test_model_config_valid():
    """Test valid model configuration"""
    config = ModelConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        context_size=2000,
    )
    assert config.provider == "openai"
    assert config.model_name == "gpt-4"
    assert config.temperature == 0.7
    assert config.max_tokens == 1000
    assert config.context_size == 2000


def test_model_config_invalid_provider():
    """Test invalid model provider"""
    with pytest.raises(ValidationError):
        ModelConfig(
            provider="invalid",
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            context_size=2000,
        )


def test_model_config_invalid_temperature():
    """Test invalid temperature values"""
    with pytest.raises(ValidationError):
        ModelConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=2.5,  # > 2.0
            max_tokens=1000,
            context_size=2000,
        )


def test_model_config_invalid_tokens():
    """Test invalid token values"""
    with pytest.raises(ValidationError):
        ModelConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=0,  # <= 0
            context_size=2000,
        )


def test_agent_config_valid():
    """Test valid agent configuration"""
    model_config = ModelConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        context_size=2000,
    )

    config = AgentConfig(
        name="test_agent",
        description="Test agent description",
        system_prompt="You are a test agent.",
        llm_config=model_config,
    )

    assert config.name == "test_agent"
    assert config.description == "Test agent description"
    assert config.system_prompt == "You are a test agent."
    assert config.llm_config == model_config
    assert config.default_queries is None
    assert config.tools_names == []


def test_agent_config_invalid_name():
    """Test invalid agent name"""
    model_config = ModelConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        context_size=2000,
    )

    with pytest.raises(ValidationError):
        AgentConfig(
            name="",  # Empty name
            description="Test agent description",
            system_prompt="You are a test agent.",
            llm_config=model_config,
        )


def test_agent_config_with_default_queries():
    """Test agent configuration with default queries"""
    model_config = ModelConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        context_size=2000,
    )

    config = AgentConfig(
        name="test_agent",
        description="Test agent description",
        system_prompt="You are a test agent.",
        llm_config=model_config,
        default_queries=["query1", "query2"],
    )

    assert config.default_queries == ["query1", "query2"]


def test_agent_config_with_tools():
    """Test agent configuration with tools"""
    model_config = ModelConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        context_size=2000,
    )

    config = AgentConfig(
        name="test_agent",
        description="Test agent description",
        system_prompt="You are a test agent.",
        llm_config=model_config,
        tools_names=["tool1", "tool2"],
    )

    assert config.tools_names == ["tool1", "tool2"]
