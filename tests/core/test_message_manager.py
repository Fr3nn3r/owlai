import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from owlai.core.config import ModelConfig
from owlai.core.message_manager import MessageManager


@pytest.fixture
def model_config():
    """Fixture for model configuration"""
    return ModelConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        context_size=2000,
    )


@pytest.fixture
def message_manager(model_config):
    """Fixture for message manager"""
    return MessageManager(model_config)


def test_message_manager_initialization(message_manager):
    """Test message manager initialization"""
    assert message_manager.model_config is not None
    assert len(message_manager.get_message_history()) == 0


def test_append_message(message_manager):
    """Test appending messages"""
    # Test human message
    human_msg = HumanMessage(content="Hello")
    message_manager.append_message(human_msg)
    history = message_manager.get_message_history()
    assert len(history) == 1
    assert history[0] == human_msg

    # Test AI message
    ai_msg = AIMessage(content="Hi there!")
    message_manager.append_message(ai_msg)
    history = message_manager.get_message_history()
    assert len(history) == 2
    assert history[1] == ai_msg


def test_append_message_invalid(message_manager):
    """Test appending invalid message"""
    with pytest.raises(ValueError):
        message_manager.append_message("not a message")


def test_clear_history(message_manager):
    """Test clearing message history"""
    # Add some messages
    message_manager.append_message(HumanMessage(content="Hello"))
    message_manager.append_message(AIMessage(content="Hi"))

    # Clear history
    message_manager.clear_history()

    # Verify history is empty
    assert len(message_manager.get_message_history()) == 0


def test_get_message_history(message_manager):
    """Test getting message history"""
    # Add messages
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi"),
        HumanMessage(content="How are you?"),
        AIMessage(content="I'm good!"),
    ]

    for msg in messages:
        message_manager.append_message(msg)

    # Get history
    history = message_manager.get_message_history()

    # Verify history
    assert len(history) == len(messages)
    for i, msg in enumerate(history):
        assert msg == messages[i]


def test_message_manager_with_system_prompt(model_config):
    """Test message manager with system prompt"""
    system_prompt = "You are a helpful assistant."
    message_manager = MessageManager(model_config)
    message_manager.append_message(SystemMessage(content=system_prompt))

    history = message_manager.get_message_history()
    assert len(history) == 1
    assert isinstance(history[0], SystemMessage)
    assert history[0].content == system_prompt


def test_context_window_management(message_manager):
    """Test FIFO mode activation when context is exceeded"""
    # Add enough messages to exceed context window
    for i in range(10):
        message_manager.append_message(HumanMessage(content=f"Message {i}"))
        message_manager.append_message(AIMessage(content=f"Response {i}"))

    # Add one more message that should trigger FIFO mode
    message_manager.append_message(HumanMessage(content="Final message"))

    history = message_manager.get_message_history()
    assert len(history) > 0  # Should still have messages
    assert history[-1].content == "Final message"  # Last message should be preserved


def test_fifo_mode_behavior(message_manager):
    """Test FIFO mode message management"""
    message_manager.fifo_mode = True
    messages = [
        HumanMessage(content="1"),
        HumanMessage(content="2"),
        HumanMessage(content="3"),
    ]
    for msg in messages:
        message_manager.append_message(msg)
    history = message_manager.get_message_history()
    assert len(history) <= message_manager.model_config.context_size


def test_system_message_handling(message_manager):
    """Test system message handling"""
    system_msg = SystemMessage(content="You are a test agent")
    message_manager.append_message(system_msg)
    history = message_manager.get_message_history()
    assert len(history) == 1
    assert history[0].content == "You are a test agent"


def test_token_counting(message_manager):
    """Test token counting for different message types"""
    # Test with AIMessage
    ai_message = AIMessage(
        content="Test", response_metadata={"token_usage": {"total_tokens": 10}}
    )
    count = message_manager._count_tokens(ai_message)
    assert count == 10

    # Test with non-AIMessage
    human_message = HumanMessage(content="Test")
    count = message_manager._count_tokens(human_message)
    assert count == 0


def test_message_history_ordering(message_manager):
    """Test message history maintains correct order"""
    messages = [
        SystemMessage(content="System"),
        HumanMessage(content="User"),
        AIMessage(content="Assistant"),
    ]
    for msg in messages:
        message_manager.append_message(msg)
    history = message_manager.get_message_history()
    assert [msg.content for msg in history] == ["System", "User", "Assistant"]


def test_error_handling(message_manager):
    """Test error handling in message operations"""
    with pytest.raises(ValueError):
        message_manager.append_message(None)  # Invalid message type
