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
from owlai.config import TOOLBOX_CONFIG, get_user_by_password
from langchain_core.tools import BaseTool, ArgsSchema
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)

from owlai.rag import RAGAgent

# Get logger using the module name
logger = logging.getLogger(__name__)


class SecurityToolInput(BaseModel):
    """Input for the Security tool."""

    query: str = Field(description="a password (sequence of words separated by spaces)")


class SecurityTool(BaseTool):  # type: ignore[override, override]

    name: str = "security_tool"
    description: str = (
        "A tool to check the security of the system. "
        "Useful for when you need to identify a user by password. "
        "Input should be a password."
    )
    args_schema: Type[BaseModel] = SecurityToolInput  # type: ignore

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


class ToolBox:

    _tavily_tool = TavilySearchResults(**TOOLBOX_CONFIG["tavily_search_results_json"])
    _security_tool = SecurityTool(**TOOLBOX_CONFIG["security_tool"])
    _naruto_search = RAGAgent(**TOOLBOX_CONFIG["rag-naruto-v1"])

    def __init__(self):
        self.mapping = {
            "security_tool": self._security_tool,
            "tavily_search_results_json": self._tavily_tool,
            "rag-naruto-v1": self._naruto_search,
        }

    def get_tools(self, keys: list[str]) -> list[Callable]:
        return [self.mapping[key] for key in keys if key in self.mapping]

    def get_tool(self, key: str) -> Callable:
        return self.mapping[key]
