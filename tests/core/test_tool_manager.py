import pytest
from unittest.mock import Mock
from langchain_core.tools import BaseTool
from owlai.core.tool_manager import ToolManager


class MockTool(BaseTool):
    """Mock tool for testing"""

    name: str = "mock_tool"
    description: str = "A mock tool for testing"

    def _run(self, tool_input: str = "") -> str:
        return "mock result"


class MockToolWithError(BaseTool):
    name: str = "error_tool"
    description: str = "A tool that raises an error"

    def _run(self, tool_input: str = "") -> str:
        raise ValueError("Test error")


@pytest.fixture
def tool_manager():
    return ToolManager()


def test_register_tool(tool_manager):
    """Test tool registration"""
    tool = MockTool()
    tool_manager.register_tool(tool)
    assert tool_manager.get_tool("mock_tool") == tool


def test_register_duplicate_tool(tool_manager):
    """Test registering duplicate tool"""
    tool1 = MockTool()
    tool2 = MockTool()
    tool_manager.register_tool(tool1)
    with pytest.raises(ValueError):
        tool_manager.register_tool(tool2)


def test_invoke_tool(tool_manager):
    """Test tool invocation"""
    tool = MockTool()
    tool_manager.register_tool(tool)
    result = tool_manager.invoke_tool("mock_tool", {"tool_input": "test"})
    assert result == "mock result"


def test_invoke_nonexistent_tool(tool_manager):
    """Test invoking non-existent tool"""
    with pytest.raises(ValueError):
        tool_manager.invoke_tool("nonexistent_tool", {})


def test_invoke_tool_with_error(tool_manager):
    """Test tool invocation with error"""
    tool = MockToolWithError()
    tool_manager.register_tool(tool)
    with pytest.raises(ValueError):
        tool_manager.invoke_tool("error_tool", {"tool_input": "test"})


def test_get_tool(tool_manager):
    """Test getting tool by name"""
    tool = MockTool()
    tool_manager.register_tool(tool)
    retrieved_tool = tool_manager.get_tool("mock_tool")
    assert retrieved_tool == tool


def test_get_nonexistent_tool(tool_manager):
    """Test getting non-existent tool"""
    with pytest.raises(ValueError):
        tool_manager.get_tool("nonexistent_tool")


def test_tool_registration_validation(tool_manager):
    """Test tool registration validation"""
    with pytest.raises(ValueError):
        tool_manager.register_tool("not a tool")


def test_tool_invocation_validation(tool_manager):
    """Test tool invocation validation"""
    tool = MockTool()
    tool_manager.register_tool(tool)
    with pytest.raises(ValueError):
        tool_manager.invoke_tool("mock_tool", "not a dict")


def test_tool_manager_initialization():
    """Test tool manager initialization"""
    manager = ToolManager()
    assert isinstance(manager, ToolManager)
