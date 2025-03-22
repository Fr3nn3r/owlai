import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from owlai.core.config import ModelConfig
from owlai.core.model_manager import ModelManager


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
def model_manager(model_config):
    return ModelManager(model_config)


def test_model_initialization(model_manager):
    """Test model initialization"""
    assert model_manager.model_config is not None


@patch("owlai.core.model_manager.ChatOpenAI")
def test_get_completion(mock_chat_openai, model_manager):
    """Test getting completion from model"""
    # Create a mock response that returns a string
    mock_model = Mock()
    mock_model.invoke.return_value = "Test response"
    mock_chat_openai.return_value = mock_model

    model_manager._model = None

    messages = [HumanMessage(content="Hello")]
    response = model_manager.get_completion(messages)
    assert response == "Test response"
    mock_model.invoke.assert_called_once_with(messages)


def test_count_tokens(model_manager):
    """Test token counting"""
    messages = [HumanMessage(content="Hello"), AIMessage(content="Hi there")]
    token_count = model_manager.count_tokens(messages)
    assert isinstance(token_count, int)
    assert token_count > 0


@patch("owlai.core.model_manager.ChatOpenAI")
def test_model_caching(mock_chat_openai, model_manager):
    """Test model caching"""
    mock_model = Mock()
    mock_chat_openai.return_value = mock_model

    model_manager._model = None

    model1 = model_manager.model
    model2 = model_manager.model

    assert model1 is model2  # Should reuse the same model instance
    mock_chat_openai.assert_called_once_with(
        model="gpt-4", temperature=0.7, max_tokens=1000
    )


def test_count_tokens_with_metadata(model_manager):
    """Test token counting with metadata"""
    messages = [
        HumanMessage(content="Hello", additional_kwargs={"metadata": {"key": "value"}}),
        AIMessage(content="Hi there", additional_kwargs={"metadata": {"key": "value"}}),
    ]
    token_count = model_manager.count_tokens(messages)
    assert isinstance(token_count, int)
    assert token_count > 0


def test_count_tokens_without_metadata(model_manager):
    """Test token counting without metadata"""
    messages = [HumanMessage(content="Hello"), AIMessage(content="Hi there")]
    token_count = model_manager.count_tokens(messages)
    assert isinstance(token_count, int)
    assert token_count > 0


def test_model_error_handling(model_manager):
    """Test error handling in model operations"""
    with pytest.raises(ValueError):
        model_manager.get_completion([])


def test_model_config_validation(model_manager):
    """Test model configuration validation"""
    assert model_manager.model_config.model_name == "gpt-4"
    assert model_manager.model_config.temperature == 0.7
    assert model_manager.model_config.max_tokens == 1000


def test_model_completion_with_empty_messages(model_manager):
    """Test model completion with empty messages"""
    with pytest.raises(ValueError):
        model_manager.get_completion([])
