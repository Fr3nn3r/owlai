import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from owlai.core.model_manager import ModelManager
from owlai.core.config import ModelConfig


class MockTool(BaseTool):
    """Mock tool for testing"""

    name: str = "mock_tool"
    description: str = "A mock tool for testing"

    def _run(self, tool_input: str = "") -> str:
        return f"Mock result for: {tool_input}"


@pytest.fixture
def model_config():
    return ModelConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=1024,
        context_size=4096,
    )


@pytest.fixture
def mock_init_chat_model():
    with patch("owlai.core.model_manager.init_chat_model") as mock:
        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(content="Test response")
        mock_model.get_num_tokens_from_messages.return_value = 100
        mock_model.bind_tools.return_value = mock_model  # Return self for chaining
        mock.return_value = mock_model
        yield mock


def test_model_manager_initialization(model_config):
    """Test basic initialization of ModelManager"""
    manager = ModelManager(model_config)

    assert manager.model_config == model_config
    assert manager._model is None
    assert manager._tools == []


def test_register_tool_before_model_init(model_config, mock_init_chat_model):
    """Test registering a tool before the model is initialized"""
    manager = ModelManager(model_config)
    tool = MockTool()

    # Register the tool (should store it for later binding)
    manager.register_tool(tool)

    assert len(manager._tools) == 1
    assert manager._tools[0] == tool
    assert mock_init_chat_model.call_count == 0  # Model not initialized yet


def test_register_tool_after_model_init(model_config, mock_init_chat_model):
    """Test registering a tool after the model is initialized"""
    manager = ModelManager(model_config)
    tool = MockTool()

    # Get the model first (initializes it)
    model = manager.get_model()

    # Then register the tool (should bind immediately)
    manager.register_tool(tool)

    assert len(manager._tools) == 1
    assert manager._tools[0] == tool
    assert mock_init_chat_model.call_count == 1
    assert model.bind_tools.call_count == 1


def test_model_initialization(model_config, mock_init_chat_model):
    """Test that models are initialized with correct parameters"""
    manager = ModelManager(model_config)

    # Get model to trigger initialization
    model = manager.get_model()

    # Check that init_chat_model was called with correct params
    mock_init_chat_model.assert_called_once_with(
        model=model_config.model_name,
        model_provider=model_config.provider,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
    )


def test_model_caching(model_config, mock_init_chat_model):
    """Test that the model is cached and only initialized once"""
    manager = ModelManager(model_config)

    # Get model multiple times
    model1 = manager.get_model()
    model2 = manager.get_model()
    model3 = manager.get_model()

    # Should only initialize once
    assert mock_init_chat_model.call_count == 1
    assert model1 is model2
    assert model2 is model3


def test_tool_binding(model_config, mock_init_chat_model):
    """Test that tools are correctly bound to the model"""
    manager = ModelManager(model_config)

    # Get model first to initialize it
    model = manager.get_model()

    # Register first tool and check it's bound correctly
    tool1 = MockTool()
    tool1.name = "tool1"
    manager.register_tool(tool1)

    # Verify the first binding call had just tool1
    assert model.bind_tools.call_count == 1
    first_call_args = model.bind_tools.call_args_list[0][0]
    assert len(first_call_args[0]) == 1
    assert first_call_args[0][0] == tool1

    # Register second tool
    tool2 = MockTool()
    tool2.name = "tool2"
    manager.register_tool(tool2)

    # Verify the second binding call had both tools
    assert model.bind_tools.call_count == 2
    second_call_args = model.bind_tools.call_args_list[1][0]
    assert len(second_call_args[0]) == 2
    assert tool1 in second_call_args[0]
    assert tool2 in second_call_args[0]


def test_get_completion(model_config, mock_init_chat_model):
    """Test getting completions from the model"""
    manager = ModelManager(model_config)

    # Create some messages
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="Hello, how are you?"),
    ]

    # Get completion
    response = manager.get_completion(messages)

    # Check model was called with the messages
    model = mock_init_chat_model.return_value
    model.invoke.assert_called_once_with(messages)

    # Check response
    assert isinstance(response, AIMessage)
    assert response.content == "Test response"


def test_count_tokens_with_model_support(model_config, mock_init_chat_model):
    """Test token counting when model supports get_num_tokens_from_messages"""
    manager = ModelManager(model_config)

    # Create some messages
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="Hello, how are you?"),
    ]

    # Count tokens
    token_count = manager.count_tokens(messages)

    # Should use the model's token counting method
    model = mock_init_chat_model.return_value
    model.get_num_tokens_from_messages.assert_called_once_with(messages)

    # Should return the count from the model
    assert token_count == 100


def test_count_tokens_fallback(model_config, mock_init_chat_model):
    """Test token counting fallback when model doesn't support get_num_tokens_from_messages"""
    manager = ModelManager(model_config)

    # Remove token counting method from mock model
    model = mock_init_chat_model.return_value
    del model.get_num_tokens_from_messages

    # Create a message
    message = HumanMessage(content="This is a test message with 8 words")

    # Count tokens
    token_count = manager.count_tokens(message)

    # Should use fallback counting (words in content)
    assert token_count == 8


def test_count_tokens_invalid_input(model_config, mock_init_chat_model):
    """Test token counting with invalid input"""
    manager = ModelManager(model_config)

    # Try to count tokens for non-message, non-list
    with pytest.raises(ValueError, match="Expected a message or list of messages"):
        manager.count_tokens("Not a message")


def test_different_providers(mock_init_chat_model):
    """Test different provider configurations work correctly"""
    # Test OpenAI
    openai_config = ModelConfig(provider="openai", model_name="gpt-4")
    openai_manager = ModelManager(openai_config)
    openai_manager.get_model()

    # Test Anthropic
    anthropic_config = ModelConfig(provider="anthropic", model_name="claude-3-opus")
    anthropic_manager = ModelManager(anthropic_config)
    anthropic_manager.get_model()

    # Test Google
    google_config = ModelConfig(provider="google", model_name="gemini-pro")
    google_manager = ModelManager(google_config)
    google_manager.get_model()

    # Check all calls used correct provider
    calls = mock_init_chat_model.call_args_list
    assert calls[0][1]["model_provider"] == "openai"
    assert calls[1][1]["model_provider"] == "anthropic"
    assert calls[2][1]["model_provider"] == "google"


def test_model_initialization_error(model_config):
    """Test handling of model initialization errors"""
    with patch("owlai.core.model_manager.init_chat_model") as mock:
        # Make init_chat_model raise an exception
        mock.side_effect = Exception("Model initialization failed")

        manager = ModelManager(model_config)

        # Should raise an exception
        with pytest.raises(Exception, match="Model initialization failed"):
            manager.get_model()


def test_tool_execution_error_handling(model_config, mock_init_chat_model):
    """Test handling of errors during tool binding"""
    manager = ModelManager(model_config)

    # Get model
    model = manager.get_model()

    # Make bind_tools raise an exception
    model.bind_tools.side_effect = Exception("Tool binding failed")

    # Register a tool - should propagate the exception from bind_tools
    with pytest.raises(Exception, match="Tool binding failed"):
        manager.register_tool(MockTool())
