# OwlAI Tests

This directory contains tests for the OwlAI framework.

## Running Tests

To run all tests, use:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/owlai/test_core.py
```

To run a specific test class:

```bash
pytest tests/owlai/test_core.py::TestOwlAgent
```

To run a specific test method:

```bash
pytest tests/owlai/test_core.py::TestOwlAgent::test_invoke_basic
```

## Test Coverage

To run tests with coverage:

```bash
pytest --cov=owlai
```

For a more detailed HTML report:

```bash
pytest --cov=owlai --cov-report=html
```

## Mocking Chat Models

Since OwlAI relies heavily on LLM interactions through langchain's chat models, we provide several approaches to mock these models for testing.

### Approach 1: Using the MockChatModel Class

The `MockChatModel` class in `tests/conftest.py` provides a complete mock implementation of `BaseChatModel`. This is useful when you need to control the sequence of responses and keep track of invocations.

Example:
```python
from tests.conftest import MockChatModel

# Create predefined responses
responses = [
    AIMessage(content="First response"),
    AIMessage(content="Response with tool call", tool_calls=[...]),
    AIMessage(content="Final response")
]

# Create the mock chat model with predefined responses
mock_model = MockChatModel(responses)

# Patch init_chat_model to return our mock model
with patch("owlai.core.init_chat_model", return_value=mock_model):
    # Create and use the agent
    agent = OwlAgent(...)
    response = agent.invoke("Test message")
```

### Approach 2: Using Fixtures and MagicMock

For simpler cases, you can use pytest fixtures with Python's built-in `MagicMock`:

```python
def test_my_function(mock_chat_model):
    # mock_chat_model is provided by a fixture from conftest.py
    
    # Configure the mock to return custom responses
    mock_chat_model.invoke.return_value = AIMessage(content="Custom response")
    
    # Use the agent
    agent = OwlAgent(...)
    response = agent.invoke("Test message")
```

### Approach 3: Testing OwlAIAgent with langgraph

For testing the `OwlAIAgent` class that uses langgraph:

```python
def test_owlai_agent(mock_chat_model, mock_agent_graph):
    # Both fixtures are from conftest.py
    
    # Configure the mock agent graph response
    mock_agent_graph.invoke.return_value = {
        "messages": [HumanMessage(...), AIMessage(...)]
    }
    
    # Create and use the agent
    agent = OwlAIAgent(...)
    response = agent.invoke("Test input")
```

See `tests/owlai/test_stub_chat_models.py` for complete examples of each approach. 