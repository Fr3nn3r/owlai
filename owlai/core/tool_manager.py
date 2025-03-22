from typing import Dict, Any
from langchain_core.tools import BaseTool
from owlai.core.interfaces import ToolOperations


class ToolManager(ToolOperations):
    """Manages tool registration and invocation"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool"""
        if not isinstance(tool, BaseTool):
            raise ValueError("Invalid tool type")

        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")

        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        """Get a tool by name"""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found")
        return self._tools[name]

    def invoke_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Invoke a tool with given arguments"""
        if not isinstance(args, dict):
            raise ValueError("Arguments must be a dictionary")

        tool = self.get_tool(name)
        try:
            return tool.run(**args)
        except Exception as e:
            raise ValueError(f"Error invoking tool '{name}': {str(e)}")
