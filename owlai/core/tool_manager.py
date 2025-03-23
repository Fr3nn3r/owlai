from typing import Dict, Any
from langchain_core.tools import BaseTool
from owlai.core.interfaces import ToolOperations
from owlai.core.logging_setup import get_logger


class ToolManager(ToolOperations):
    """Manages tool registration and invocation"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self.logger = get_logger("tool_manager")

    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool"""
        if not isinstance(tool, BaseTool):
            self.logger.error(f"Invalid tool type: {type(tool)}")
            raise ValueError("Invalid tool type")

        if tool.name in self._tools:
            self.logger.error(f"Tool '{tool.name}' is already registered")
            raise ValueError(f"Tool '{tool.name}' is already registered")

        self._tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> BaseTool:
        """Get a tool by name"""
        if name not in self._tools:
            self.logger.error(
                f"Tool '{name}' not found. Available tools: {list(self._tools.keys())}"
            )
            raise ValueError(f"Tool '{name}' not found")
        self.logger.debug(f"Retrieved tool: {name}")
        return self._tools[name]

    def invoke_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Invoke a tool with given arguments"""
        self.logger.info(f"Invoking tool '{name}' with args: {args}")
        if not isinstance(args, dict):
            self.logger.error(f"Invalid arguments type: {type(args)}")
            raise ValueError("Arguments must be a dictionary")

        tool = self.get_tool(name)
        try:
            self.logger.debug(f"Running tool '{name}' with args: {args}")
            result = tool.run(**args)
            self.logger.info(f"Tool '{name}' execution successful: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error invoking tool '{name}': {e}")
            raise ValueError(f"Error invoking tool '{name}': {str(e)}")
