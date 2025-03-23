from typing import Dict, Any
from langchain_core.tools import BaseTool
from owlai.core.interfaces import ToolOperations
from owlai.core.logging_setup import get_logger


class ToolManager(ToolOperations):
    """Manages tool registration and invocation"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._name_map: Dict[str, str] = {}  # Maps normalized names to original names
        self.logger = get_logger("tool_manager")

    def _normalize_name(self, name: str) -> str:
        """Normalize a tool name for case-insensitive matching"""
        return name.lower().strip()

    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool"""
        if not isinstance(tool, BaseTool):
            self.logger.error(f"Invalid tool type: {type(tool)}")
            raise ValueError("Invalid tool type")

        if not tool.name:
            self.logger.error("Tool name cannot be empty")
            raise ValueError("Tool name cannot be empty")

        # Store tool with original name and create normalized name mapping
        original_name = tool.name
        normalized_name = self._normalize_name(original_name)

        if normalized_name in self._name_map:
            self.logger.error(f"Tool '{original_name}' is already registered")
            raise ValueError(f"Tool '{original_name}' is already registered")

        self._tools[original_name] = tool
        self._name_map[normalized_name] = original_name
        self.logger.info(f"Registered tool: {original_name}")

    def get_tool(self, name: str) -> BaseTool:
        """Get a tool by name"""
        if not name:
            self.logger.error("Tool name cannot be empty")
            raise ValueError("Tool name cannot be empty")

        # Try to find the tool using normalized name
        normalized_name = self._normalize_name(name)
        if normalized_name in self._name_map:
            original_name = self._name_map[normalized_name]
            self.logger.debug(f"Retrieved tool: {original_name}")
            return self._tools[original_name]

        # Try to find the tool directly by name
        if name in self._tools:
            self.logger.debug(f"Retrieved tool: {name}")
            return self._tools[name]

        self.logger.error(
            f"Tool '{name}' not found. Available tools: {list(self._tools.keys())}"
        )
        raise ValueError(f"Tool '{name}' not found")

    def invoke_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Invoke a tool with given arguments"""
        if not name:
            self.logger.error("Tool name cannot be empty")
            raise ValueError("Tool name cannot be empty")

        if not isinstance(args, dict):
            self.logger.error(f"Invalid arguments type: {type(args)}")
            raise ValueError("Arguments must be a dictionary")

        try:
            tool = self.get_tool(name)
            self.logger.debug(f"Running tool '{name}' with args: {args}")
            result = tool.run(**args)
            self.logger.info(f"Tool '{name}' execution successful: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error invoking tool '{name}': {e}")
            raise ValueError(f"Error invoking tool '{name}': {str(e)}")
