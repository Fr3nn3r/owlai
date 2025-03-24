#  ,___,
#  [O.o]
#  /)__)
#  --"--"--

print("Loading tools module")
from ast import Tuple
import logging

from typing import Callable, Type, Optional, Tuple, Union, List, Dict, Any
from langchain_community.tools.tavily_search.tool import TavilyInput
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from torch import Type
from owlai.db import TOOLS_CONFIG
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)

from langchain_core.tools import BaseTool, ArgsSchema

logger = logging.getLogger("main")

focus_role: str = "qna"


class SecurityToolInput(BaseModel):
    """Input for the Security tool."""

    query: str = Field(description="a password (sequence of words separated by spaces)")


class SecurityTool(BaseTool):

    name: str = "security_tool"
    description: str = (
        "A tool to check the security of the system. "
        "Useful for when you need to identify a user by password. "
        "Input should be a password."
    )
    args_schema: Optional[ArgsSchema] = SecurityToolInput

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
        from .db import get_user_by_password  # Import here to avoid circular imports

        global user_context
        logger.debug(f"calling identify_user_by_password with {user_password}")
        user_data = get_user_by_password(user_password)
        if user_data:
            user_context = f"CONTEXT: {user_data}"
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

    user_context: str = "CONTEXT: "

    _tavily_tool = TavilySearchResults(**TOOLS_CONFIG["tavily_search_results_json"])
    _security_tool = SecurityTool()
    # _owl_system_interpreter = OwlSystemInterpreter(
    #    **TOOLS_CONFIG["owl_system_interpreter"]
    # )
    # _owl_memory_tool = OwlMemoryTool(**TOOLS_CONFIG["owl_memory_tool"])

    def __init__(self):
        self.mapping = {
            "activate_mode": self.activate_mode,
            "security_tool": self._security_tool,
            # "owl_system_interpreter": self._owl_system_interpreter,
            "play_song": self.play_song,
            # "owl_memory_tool": self._owl_memory_tool,
            "tavily_search_results_json": self._tavily_tool,
        }

    def get_tools(self, keys: list[str]) -> list[Callable]:
        return [self.mapping[key] for key in keys if key in self.mapping]

    def get_tool(self, key: str) -> Callable:
        return self.mapping[key]

    def _identify_user_with_password(self, user_password: str) -> str:
        """
        Checks wether the password is valid.
        A valid password is required and sufficient to identify the user.
        The password is a sequence of words separated by spaces.

        Args:
            user_password: a string containing a sequence of words separated by spaces.
        """
        from .db import get_user_by_password  # Import here to avoid circular imports

        global user_context
        logger.debug(f"calling identify_user_by_password with {user_password}")
        user_data = get_user_by_password(user_password)
        if user_data:
            user_context = f"CONTEXT: {user_data}"
            return (
                "User identified: "
                + user_data["first_name"]
                + " "
                + user_data["last_name"]
            )
        else:
            return "Invalid password"

    @tool
    def activate_mode(self, mode: str):
        """
        Activates a different owlai mode.
        Args:
            mode: a string containing the mode to activate.
        """

        global focus_role
        # if mode not in list_roles():
        #    return f"Invalid mode: {mode}"
        focus_role = mode
        message = f"Activated {mode} mode"
        logger.debug(message)
        return message

    @tool
    def play_song(
        self, song_name: str = "Fly Away", artist_name: str = "Lenny Kravitz"
    ):
        """
        Plays a song.
        Args:
            song_name: (required) a string containing the name of the song to play.
            artist_name: (required) a string containing the name of the artist of the song to play.
        """
        from .spotify import play_song_on_spotify

        logger.info(f"Playing song: {song_name} by {artist_name}")
        play_song_on_spotify(song_name, artist_name)

    def get_focus_role(self) -> str:
        self.focus_role = focus_role
        return self.focus_role
