import pytest
from typing import get_type_hints
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from owlai.core.interfaces import MessageOperations, ToolOperations, ModelOperations


def test_message_operations_interface():
    """Test MessageOperations interface"""
    assert hasattr(MessageOperations, "append_message")
    assert hasattr(MessageOperations, "get_message_history")
    assert hasattr(MessageOperations, "clear_history")


def test_tool_operations_interface():
    """Test ToolOperations interface"""
    assert hasattr(ToolOperations, "register_tool")
    assert hasattr(ToolOperations, "get_tool")
    assert hasattr(ToolOperations, "invoke_tool")


def test_model_operations_interface():
    """Test ModelOperations interface"""
    assert hasattr(ModelOperations, "get_completion")
    assert hasattr(ModelOperations, "count_tokens")


def test_interface_method_signatures():
    """Test that interface methods have correct signatures"""
    # MessageOperations
    message_hints = get_type_hints(MessageOperations.append_message)
    assert message_hints["message"] == BaseMessage

    # ToolOperations
    tool_hints = get_type_hints(ToolOperations.register_tool)
    assert tool_hints["tool"] == BaseTool

    # ModelOperations
    model_hints = get_type_hints(ModelOperations.get_completion)
    assert "messages" in model_hints
    assert "return" in model_hints
