#       ,_,
#      (O,O)
#      (   )
#      -"-"-

import subprocess
from subprocess import CompletedProcess
from dotenv import load_dotenv
import os
import yaml
from typing import List, Dict, Any, Optional, Callable
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
import logging.config
import logging
from langchain_core.tools import tool, BaseTool
import sys
import io
import json
from db import (
    USER_DATABASE,
    get_user_by_password,
    CONFIG,
    get_system_prompt_by_role,
    get_default_prompts_by_role,
)
from spotify import play_song_on_spotify
from pydantic.v1 import BaseSettings

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("main_logger")

focus_role = "identification"

user_context = "CONTEXT: "

toolbox_hook: Callable = None


class ToolBox:

    @tool
    def identify_user_with_password(user_password: str) -> str:
        """
        Checks wether the password is valid.
        A valid password is required and sufficient to identify the user.
        The password is a sequence of words separated by spaces.

        Args:
            user_password: a string containing a sequence of words separated by spaces.
        """
        global user_context
        logger.debug(f"calling identify_user_by_password with {user_password}")
        passwords = [user["password"] for user in USER_DATABASE.values()]

        if user_password.lower() in passwords:
            user_data = get_user_by_password(user_password)
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
    def activate(mode: str):
        """
            Activates a different owlai mode.
        Args:
            mode: a string containing the mode to activate. possible values are:
                - identification: activates identification mode
                - system: activates system mode
                - welcome: activates welcome mode
                - command_manager: activates command manager mode
        """
        global focus_role
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
        logger.debug(f"Running system task: {task} {toolbox_hook}")
        command_stdout = toolbox_hook(task)
        if command_stdout.endswith("\n"):
            command_stdout = command_stdout[:-1]
        logger.info(command_stdout)
        return command_stdout

    @tool
    def play_song(song_name: str = "Fly Away", artist_name: str = "Lenny Kravitz"):
        """
            Plays a song.
        Args:
            song_name: (required) a string containing the name of the song to play.
            artist_name: (required) a string containing the name of the artist of the song to play.
        """
        logger.info(f"Playing song: {song_name} by {artist_name}")
        play_song_on_spotify(song_name, artist_name)


toolbox = ToolBox()


class Owl:
    """
    Base class for OwlAI agents.
    
    Attributes:
        role (str): The role of the agent (welcome, system, etc.)
        implementation (str): The model implementation to use
        model_name (str): Name of the model to use
        
    Methods:
        invoke(message: str) -> str: Process a user message
        _process_tool_calls(model_response: AIMessage) -> None: Handle tool calls
    """

    # Global message history shared by all agents
    message_history: List[BaseMessage] = []

    def __init__(
        self,
        role: str = "welcome",
        implementation: str = "openai",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.9,
        max_tokens: int = 2048,
        max_context_tokens: int = 4096,
        tools: List[BaseTool] = [],
        system_prompt: str = "You are a system agent from OwlAI",
    ):
        self.role = role
        self.implementation = implementation
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_context_tokens = max_context_tokens
        self.tools = tools
        self.system_prompt = system_prompt
        self.chat_model = self.__get_chat_model()

        self.fifo_message_mode = False
        self.total_tokens = 0
        if len(tools) > 0:
            self.tools = tools
            self.tools_dict = {tool.name: tool for tool in self.tools}
            self.chat_model = self.chat_model.bind_tools(self.tools)

    def __get_chat_model(self) -> BaseChatModel:
        """Returns a new model instance nased on self parameters, This is the Model Factory."""
        if self.implementation == "openai":
            openai_model = ChatOpenAI(
                name=self.model_name,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )
            return openai_model
        elif self.implementation == "anthropic":
            anthropic_model = ChatAnthropic(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens_to_sample=self.max_tokens,
            )
            return anthropic_model
        elif self.implementation == "ollama":
            meta_model = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                num_predict=self.max_tokens,
            )
            return meta_model
        else:
            raise ValueError(
                f"Unsupported model implementation provider: {self.implementation}"
            )

    def _token_count(self, message: AIMessage):
        metadata = message.response_metadata
        if self.implementation == "openai":
            return metadata["token_usage"]["total_tokens"]
        elif self.implementation == "anthropic":
            anthropic_total_tokens = (
                metadata["usage"]["input_tokens"] + metadata["usage"]["output_tokens"]
            )
            return anthropic_total_tokens
        else:
            logger.warning(
                f"Token count unsupported for model provider: '{self.implementation}'"
            )
            return -1

    def append_message(self, message: BaseMessage):
        if type(message) == AIMessage:
            self.total_tokens = self._token_count(message)
        if (self.total_tokens > self.max_context_tokens) and (
            self.fifo_message_mode == False
        ):
            logger.warning(
                f"Total tokens '{self.total_tokens}' exceeded max context tokens '{self.max_context_tokens}' -> activating FIFO message mode"
            )
            self.fifo_message_mode = True
        if self.fifo_message_mode:
            self.message_history.pop(1)  # Remove the oldest message
            if (
                self.message_history[-1].type == "tool"
            ):  # Remove the tool message if any
                self.message_history.pop(1)
            self.print_message_history()

        self.message_history.append(message)

    def _process_tool_calls(self, model_response: AIMessage) -> None:
        """Process tool calls from the model response and add results to chat history."""
        if not hasattr(model_response, "tool_calls") or not model_response.tool_calls:
            logger.debug("No tool calls in response")
            return

        for tool_call in model_response.tool_calls:
            # Loop over tool calls
            logger.debug(f"Tool call requested: '{tool_call}'")

            # Skip if no tool calls or empty
            if not tool_call or "name" not in tool_call:
                logger.warning(f"Invalid tool call format: '{tool_call}'")
                continue

            # Get tool name and arguments
            tool_name = tool_call["name"].lower()
            tool_args = tool_call.get("arguments", {})

            # Check if tool exists
            if tool_name not in self.tools_dict:
                error_msg = f"Tool '{tool_name}' not found in available tools"
                logger.error(error_msg)
                tool_msg = ToolMessage(
                    content=f"Error: {error_msg}",
                    tool_call_id=tool_call.get("id", "unknown"),
                    name=tool_name,
                )
                self.append_message(tool_msg)
                continue

            # Select the tool
            selected_tool = self.tools_dict[tool_name]

            try:
                # Invoke the tool
                tool_result = selected_tool.invoke(tool_call)

                # Create tool message
                tool_msg = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call.get("id", "unknown"),
                    name=tool_name,
                )

                # Add tool response to history
                self.append_message(tool_msg)

            except Exception as e:
                logger.error(f"Error invoking tool '{tool_name}': '{e}' ({tool_call})")

                # Create error message
                error_content = f"Error executing '{tool_name}': '{str(e)}'"
                tool_msg = ToolMessage(
                    content=error_content,
                    tool_call_id=tool_call.get("id", "unknown"),
                    name=tool_name,
                )
                self.append_message(tool_msg)

    def invoke(self, message: str) -> str:
        # update system prompt with latestcontext
        system_message = SystemMessage(f"{self.system_prompt}\n{user_context}")
        if len(self.message_history) == 0:
            self.message_history.append(system_message)
        else:
            self.message_history[0] = system_message

        self.append_message(HumanMessage(message))  # Add user message to history
        response = self.chat_model.invoke(self.message_history)
        self.append_message(response)  # Add model response to history

        self._process_tool_calls(response)

        if response.tool_calls:  # If tools were called, invoke the model again
            response = self.chat_model.invoke(self.message_history)
            self.append_message(response)  # Add model response to history
            logger.debug(response.content)  # Log the model response

        self._total_tokens = self._token_count(response)
        return response.content  # Return the model response

    def print_message_history(self):
        for index, message in enumerate(self.message_history):
            if index > 0: # skip the system message
                logger.info(
                    f"Message #{index} '{message.type}' '{message.content[:100]  + "..." if len(message.content) > 100 else message.content }'"
                )

    def print_message_metadata(self):
        for index, message in enumerate(self.message_history):
            if message.response_metadata:
                logger.info(
                    f"Message #{index} type: '{message.type}' metadata: '{message.response_metadata}'"
                )

    def reset_message_history(self):
        if len(self.message_history) > 0:
            self.message_history = [self.message_history[0]]
            self.fifo_message_mode = False

    def get_default_prompts(self):
        return CONFIG[self.role]["default_prompts"]

    def run_tests(self):
        if len(CONFIG[self.role]["test_prompts"]) > 0:
            logger.info(f"Running tests for owl role '{self.role}'")
            for test in CONFIG[self.role]["test_prompts"]:
                logger.info(f"USER: {test}")
                self.invoke(test)
        else:
            logger.warning("No test prompts defined for owl role '{self.role}'")

    def print_info(self):
        logger.info(
            f"Role '{self.role}', model provider '{self.implementation}', model name '{self.model_name}', tools {self.tools_dict.keys()}"
        )

    # Add proper resource cleanup:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class LocalPythonInterpreter(Owl):
    """
    Needs to have its own message queue to avoid publishing messages to the global message queue (main context).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lpi_own_message_history = [
            SystemMessage(f"{self.system_prompt}\n{user_context}")
        ]

    def _run_system_command(self, user_query: str) -> str:
        """
        Send a prompt to the model and executes the output as a python script.
        """

        user_message = HumanMessage(content=user_query)
        self.lpi_own_message_history.append(user_message)

        try:

            model_code_message = self.chat_model.invoke(self.lpi_own_message_history)

            # here we could be stream writing to the file
            python_script = model_code_message.content

            self.lpi_own_message_history.append(model_code_message)

            logger.debug(python_script)

        except Exception as e:
            error_message = f"Error generating python script: {str(e)}"
            logger.error(error_message)
            return error_message

        try:
            result = self._execute_as_python(python_script)

        except Exception as e:
            error_message = f"Error executing python script: {str(e)}"
            logger.error(error_message)
            return error_message

        return result.stdout

    ######################################### Script

    _script_count: int = 0

    def _script_label(self):
        return f"SCRIPT {self._script_count:05d}"

    def _next_script(self):

        relative_file_pathname = f"scripts/python/temp_{self._script_count:05d}.py"
        while os.path.exists(relative_file_pathname):
            self._script_count += 1
            return self._next_script()

        return relative_file_pathname

    def _save_to_file(self, script):
        file_relative_pathname = self._next_script()

        with open(file_relative_pathname, "w", encoding="utf-8") as file:
            python_script = f"#{self._script_label()}\n{script}"
            file.write(python_script)

        return file_relative_pathname

    # saves script to a temo file, executes the script handling error, updates the conversation
    # returns a completed process with stdout stderr
    def _execute_as_python(self, code: str) -> CompletedProcess[str]:
        result = None
        try:
            file_relative_pathname = self._save_to_file(code)

            ##################### Execute saved script as python
            result = subprocess.run(
                ["python.exe", file_relative_pathname],
                shell=True,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )

            ######################## Update Conversation
            if len(result.stdout) > 0:
                result_message = f"{self._script_label()} - execution completed - Return code : {result.returncode} - STDOUT: {result.stdout}"
                logger.debug(result_message)
                self.lpi_own_message_history.append(HumanMessage(result_message))

            if len(result.stderr) > 0:
                stderr_message = f"{self._script_label()} - Error/Warning message - Return code : {result.returncode} - STDERR: {result.stderr}"
                logger.warning(stderr_message)
                self.lpi_own_message_history.append(HumanMessage(stderr_message))

        ################ Error handling
        except subprocess.CalledProcessError as cpe:

            error_message = f"{self._script_label()} - execution failed: ProcessError: {str(cpe)} - {cpe.output} - {cpe.stdout} - {cpe.stderr}."
            logger.error(error_message)
            self.lpi_own_message_history.append(HumanMessage(error_message))

        except subprocess.TimeoutExpired as toe:

            error_message = f"{self._script_label()} execution failed: TimeoutExpired: {str(toe)} - {toe.output} - {toe.stdout} - {toe.stderr}."
            logger.error(error_message)
            self.lpi_own_message_history.append(HumanMessage(error_message))

        except Exception as e:

            error_message = (
                f"{self._script_label()} - execution failed: Exception: {str(e)}"
            )
            logger.error(error_message)
            self.lpi_own_message_history.append(HumanMessage(error_message))

        return result


class Edwige:

    def __init__(self):
        self.logger = logging.getLogger("main_logger")
        self.focus_role = "welcome"

    owls: Dict[str, Owl] = {}

    roles = ["system", "welcome", "identification", "command_manager"]

    owls["system"] = Owl(
        role="system",
        implementation="openai",
        model_name="gpt-4o-mini",
        temperature=0.9,
        max_tokens=200,
        max_context_tokens=4096,
        tools=[toolbox.activate, toolbox.run_task, toolbox.play_song],
        system_prompt=get_system_prompt_by_role("system"),
    )

    owls["welcome"] = Owl(
        role="welcome",
        implementation="ollama",
        model_name="llama3.1:8b",
        temperature=0.9,
        max_tokens=1000,
        max_context_tokens=4096,
        tools=[toolbox.activate],
        system_prompt=get_system_prompt_by_role("welcome"),
    )

    owls["identification"] = Owl(
        role="identification",
        implementation="openai",
        model_name="gpt-4o-mini",
        temperature=0.9,
        max_tokens=200,
        max_context_tokens=4096,
        tools=[toolbox.activate, toolbox.identify_user_with_password],
        system_prompt=get_system_prompt_by_role("identification"),
    )

    owls["command_manager"] = LocalPythonInterpreter(
        role="command_manager",
        implementation="anthropic",
        model_name="claude-3-7-sonnet-20250219",
        temperature=0.9,
        max_tokens=2048,
        max_context_tokens=4096,
        tools=[],
        system_prompt=get_system_prompt_by_role("command_manager"),
    )

    # set the toolbox hook to the LocalPythonInterpreter
    global toolbox_hook
    toolbox_hook = owls["command_manager"]._run_system_command

    def get_focus_owl(self):
        logger.debug(f"focus role: {focus_role}")
        return self.owls[focus_role]

    def get_default_prompts(self):
        return get_default_prompts_by_role(self.focus_role)


class OwlConfig(BaseSettings):
    role: str
    implementation: str
    model_name: str
    temperature: float

    def validate_config(self, config: Dict[str, Any]) -> bool:
        required_fields = ["role", "implementation", "model_name"]
        return all(field in config for field in required_fields)
