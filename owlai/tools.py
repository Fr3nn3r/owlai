#  ,___,
#  [O.o]
#  /)__)
#  --"--"--

print("Loading tools module")
import logging
import os
import subprocess
from subprocess import CompletedProcess
from typing import Callable
from langchain_core.tools import tool, BaseTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from .core import OwlAgent, sprint
from .db import TOOLS_CONFIG
from .interpreter import LocalPythonInterpreter
from .interpreter import OwlSystemInterpreter
from .rag import LocalRAGTool, OwlMemoryTool
from ragatouille import RAGPretrainedModel

import warnings

warnings.simplefilter("ignore", category=FutureWarning)

logger = logging.getLogger("tools")

focus_role: str = "qna"


class ToolBox:

    toolbox_hook: Callable = None
    toolbox_hook_rag_engine: Callable = None
    user_context: str = "CONTEXT: "

    # _local_python_interpreter = LocalPythonInterpreter(
    #    **TOOLS_CONFIG["python_interpreter"]
    # )
    # _local_rag_tool = LocalRAGTool(**TOOLS_CONFIG["rag_tool"])
    _tavily_tool = TavilySearchResults(**TOOLS_CONFIG["tavily_search_results_json"])
    _owl_system_interpreter = OwlSystemInterpreter(
        **TOOLS_CONFIG["owl_system_interpreter"]
    )
    _owl_memory_tool = OwlMemoryTool(**TOOLS_CONFIG["owl_memory_tool"])

    def __init__(self):
        self.mapping = {
            "activate_mode": self.activate_mode,
            "identify_user_with_password": self.identify_user_with_password,
            "owl_system_interpreter": self._owl_system_interpreter,
            "play_song": self.play_song,
            "owl_memory_tool": self._owl_memory_tool,
            "tavily_search_results_json": self._tavily_tool,
        }

    def get_tools(self, keys: list[str]) -> list[Callable]:
        return [self.mapping[key] for key in keys if key in self.mapping]

    def get_tool(self, key: str) -> Callable:
        return self.mapping[key]

    @tool
    def identify_user_with_password(user_password: str) -> str:
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
    def activate_mode(mode: str):
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
    def run_task(task: str):
        """
        Runs a natural language tasks on the local machine.
        Args:
            script: a natural language string describing the command to run.
        """

        toolbox_hook = _local_python_interpreter._run_system_command
        logger.debug(f"Running system task: {task}")
        command_stdout = toolbox_hook(task)
        if command_stdout.endswith("\n"):
            command_stdout = command_stdout[:-1]
        logger.info(f"Command output: {command_stdout}")
        return command_stdout

    @tool
    def play_song(song_name: str = "Fly Away", artist_name: str = "Lenny Kravitz"):
        """
        Plays a song.
        Args:
            song_name: (required) a string containing the name of the song to play.
            artist_name: (required) a string containing the name of the artist of the song to play.
        """
        from .spotify import play_song_on_spotify

        logger.info(f"Playing song: {song_name} by {artist_name}")
        play_song_on_spotify(song_name, artist_name)

    @tool
    def get_answer_from_knowledge_base(question: str):
        """
        Gets an answer from the knowledge base.
        Args:
            question: a string containing the question to answer.
        """
        toolbox_hook_rag_engine = _local_rag_tool.rag_question
        logger.debug(f"Running RAG question: {question}")
        rag_answer = toolbox_hook_rag_engine(question)
        return rag_answer

    def get_focus_role(self) -> str:
        self.focus_role = focus_role
        return self.focus_role
