#  ,___,
#  [O.o]
#  /)__)
#  --"--"--

print("Loading tools module")

from ast import Tuple
import logging

from typing import Callable, Type, Optional, Tuple, Union, List, Dict, Any
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from owlai.config.tools import TOOLS_CONFIG, OPTIONAL_TOOLS
from owlai.config.users import get_user_by_password
from langchain_core.tools import BaseTool, ArgsSchema
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)

from owlai.services.rag import FAISS_RAG_Tool, Pinecone_RAG_Tool

# Get logger using the module name
logger = logging.getLogger(__name__)


class DefaultToolInput(BaseModel):
    """Input for tool."""

    query: str = Field(description="A query to the tool")

    model_config = {"extra": "ignore"}


class SecurityTool(BaseTool):  # type: ignore[override, override]

    name: str = "security_tool"
    description: str = (
        "A tool to check the security of the system. "
        "Useful for when you need to identify a user by password. "
        "Input should be a password."
    )
    args_schema: Type[BaseModel] = DefaultToolInput  # type: ignore

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def identify_user_with_password(
        self, user_password: str
    ) -> Union[Dict[str, Any], str]:
        """
        Checks wether the password is valid.
        A valid password is required and sufficient to identify the user.
        The password is a sequence of words separated by spaces.

        Args:
            user_password: a string containing a sequence of words separated by spaces.
        """

        logger.debug(f"calling identify_user_by_password with {user_password}")
        user_data = get_user_by_password(user_password)
        if user_data:
            return user_data
        else:
            return "Invalid password"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[Dict[str, Any], str]:
        """Use the tool."""
        return self.identify_user_with_password(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[Dict[str, Any], str]:
        """Use the tool asynchronously."""
        return self.identify_user_with_password(query)


class ToolFactory:
    """Factory class for lazy initialization of tools based on tool ID."""

    _tool_cache = {}  # Cache for initialized tools

    @classmethod
    def get_tool(cls, tool_id: str) -> BaseTool:
        """Get a tool instance by its ID, initializing it if needed."""
        # Return cached tool if already initialized
        if tool_id in cls._tool_cache:
            return cls._tool_cache[tool_id]

        # Initialize the tool based on its ID
        if tool_id == "security_tool":
            tool = SecurityTool(**OPTIONAL_TOOLS[tool_id])
        elif tool_id == "tavily_search_results_json":
            tool = TavilySearchResults(**OPTIONAL_TOOLS[tool_id])
        elif tool_id == "pinecone_french_law_lookup":
            tool = Pinecone_RAG_Tool(**TOOLS_CONFIG[tool_id])
        elif tool_id in TOOLS_CONFIG:
            tool = FAISS_RAG_Tool(**TOOLS_CONFIG[tool_id])
        elif tool_id in OPTIONAL_TOOLS:
            tool = FAISS_RAG_Tool(**OPTIONAL_TOOLS[tool_id])
        else:
            raise ValueError(f"Tool ID not found: {tool_id}")

        # Cache the tool
        cls._tool_cache[tool_id] = tool
        logger.debug(f"Initialized and cached tool: {tool_id}")
        return tool

    @classmethod
    def list_available_tools(cls) -> Dict[str, str]:
        """Get a dictionary of all available tools with their descriptions.

        Returns:
            Dictionary mapping tool_id to tool description
        """
        tools = {}

        # Add tools from TOOLS_CONFIG
        for tool_id, config in TOOLS_CONFIG.items():
            tools[tool_id] = config.get("description", "No description available")

        # Add tools from OPTIONAL_TOOLS
        for tool_id, config in OPTIONAL_TOOLS.items():
            tools[tool_id] = config.get("description", "No description available")

        return tools

    @classmethod
    def clear_cache(cls):
        """Clear the tool cache to free up resources."""
        cls._tool_cache = {}
        logger.debug("Tool cache cleared")

    @classmethod
    def get_tools(cls, tool_ids: List[str]) -> List[BaseTool]:
        """Get a list of tools by their IDs, initializing them if needed."""
        tools = []
        for tool_id in tool_ids:
            try:
                tool = cls.get_tool(tool_id)
                tools.append(tool)
            except ValueError:
                logger.warning(
                    f"Attempted to access an undefined tool with ID '{tool_id}'. Please check the tool ID or define the tool."
                )
        return tools
