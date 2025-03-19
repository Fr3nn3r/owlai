"""Unit tests for the core module."""

import pytest
from owlai.core import OwlAgent


def test_owl_agent_initialization():
    """Test that OwlAgent can be initialized with default values."""
    agent = OwlAgent(system_prompt="You are a helpful assistant.")

    assert agent.model_provider == "openai"
    assert agent.model_name == "gpt-4o-mini"
    assert agent.temperature == 0.1
    assert agent.max_tokens == 2048
    assert agent.system_prompt == "You are a helpful assistant."
    assert agent.total_tokens == 0
    assert agent.fifo_message_mode is False
    assert len(agent.callable_tools) == 0


def test_owl_agent_custom_values():
    """Test that OwlAgent can be initialized with custom values."""
    agent = OwlAgent(
        model_provider="anthropic",
        model_name="claude-3-opus-20240229",
        temperature=0.7,
        max_tokens=4000,
        system_prompt="You are an expert assistant.",
    )

    assert agent.model_provider == "anthropic"
    assert agent.model_name == "claude-3-opus-20240229"
    assert agent.temperature == 0.7
    assert agent.max_tokens == 4000
    assert agent.system_prompt == "You are an expert assistant."
