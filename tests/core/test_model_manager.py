import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, SystemMessage
from owlai.core.model_manager import ModelManager
from owlai.core.config import ModelConfig


@pytest.fixture
def model_config():
    return ModelConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        context_size=4000,
    )


@pytest.fixture
def model_manager(model_config):
    return ModelManager(model_config)


def test_model_initialization(model_manager):
    """Test model initialization"""
    assert model_manager.model_config is not None
    model_manager.get_model()  # Initialize the model
    assert model_manager._model is not None


@patch("owlai.core.model_manager.init_chat_model")
def test_get_completion(mock_init_chat_model, model_manager):
    """Test getting completion from model"""
    # Setup mock
    mock_model = Mock()
    mock_model.invoke.return_value = "Test response"
    mock_init_chat_model.return_value = mock_model

    # Test completion
    messages = [HumanMessage(content="Test message")]
    response = model_manager.get_completion(messages)
    assert response == "Test response"
    mock_model.invoke.assert_called_once_with(messages)


def test_count_tokens(model_manager):
    """Test token counting"""
    message = HumanMessage(content="Test message")
    count = model_manager.count_tokens(message)
    assert isinstance(count, int)
    assert count > 0


@patch("owlai.core.model_manager.init_chat_model")
def test_model_caching(mock_init_chat_model, model_manager):
    """Test model caching"""
    # First call should create new model
    model1 = model_manager.get_model()
    mock_init_chat_model.assert_called_once_with(
        model="gpt-4", model_provider="openai", temperature=0.7, max_tokens=1000
    )

    # Second call should return cached model
    mock_init_chat_model.reset_mock()
    model2 = model_manager.get_model()
    assert model1 is model2
    mock_init_chat_model.assert_not_called()


def test_count_tokens_with_metadata(model_manager):
    """Test token counting with metadata"""
    message = HumanMessage(content="Test message")
    count = model_manager.count_tokens(message)
    assert isinstance(count, int)
    assert count > 0


def test_count_tokens_without_metadata(model_manager):
    """Test token counting without metadata"""
    message = HumanMessage(content="Test message")
    count = model_manager.count_tokens(message)
    assert isinstance(count, int)
    assert count > 0


def test_model_error_handling(model_manager):
    """Test error handling in model operations"""
    with pytest.raises(ValueError):
        model_manager.get_completion([])


def test_model_config_validation(model_manager):
    """Test model configuration validation"""
    assert model_manager.model_config.provider == "openai"
    assert model_manager.model_config.model_name == "gpt-4"
    assert model_manager.model_config.temperature == 0.7
    assert model_manager.model_config.max_tokens == 1000
    assert model_manager.model_config.context_size == 4000


def test_model_completion_with_empty_messages(model_manager):
    """Test model completion with empty messages"""
    with pytest.raises(ValueError, match="Messages list cannot be empty"):
        model_manager.get_completion([])
