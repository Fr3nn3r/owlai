import subprocess
from subprocess import CompletedProcess
from dotenv import load_dotenv
import os
import yaml
from typing import List, Dict, Any, Optional
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
import logging.config
import logging  # Loop over test instructions before prompting user
from langchain_core.tools import tool, BaseTool
import sys
import io
import json
from db import USER_DATABASE, get_user_by_password, MODE_RESOURCES
from spotify import play_song_on_spotify
from pydantic import ValidationError

# Load environment variables from .env file
load_dotenv()

with open("logging.yaml", "r") as logger_config:
    config = yaml.safe_load(logger_config)
    logging.config.dictConfig(config)
logger = logging.getLogger("main_logger")

current_mode = "welcome"

user_context = "CONTEXT: "

###### I have to declare tools before instanciating the agents because the tools functions need an instance to be referenced
###### I like to have tools associated with a specific agent, but langchain does not support that. Maybe we'll have a tool manager later.
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
            "User identified: " + user_data["first_name"] + " " + user_data["last_name"]
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
            - old_system: switch to old system mode
    """
    global current_mode
    current_mode = mode
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
    logger.debug(f"Running system task: {task}")
    command_stdout = mode_agent_mapping["python"]._run_system_command(task)
    if command_stdout.endswith("\n"):
        command_stdout = command_stdout[:-1]
    logger.info(command_stdout)
    return command_stdout


@tool
def play_song(song_name: str = "Fly Away", artist_name: Optional[str] = None):
    """
        Plays a song.
    Args:
        song_name: (required) a string containing the name of the song to play.
        artist_name: (optional) a string containing the name of the artist of the song to play.
    """
    logger.info(f"Playing song: {song_name} by {artist_name}")
    play_song_on_spotify(song_name, artist_name)


class BaseAgent:

    # Global message history shared by all agents
    _message_history: List[BaseMessage] = []

    def __init__(
        self,
        system_prompt: str,
        model_provider: str = "openai",
        tools: List[BaseTool] = [],
        max_context_tokens: int = 4096,
        max_output_tokens: int = 2048,
    ):
        self._system_prompt = system_prompt
        self._chat_model = self.__get_chat_model(model_provider)
        self._model_provider = model_provider
        self._max_context_tokens = max_context_tokens
        self._max_output_tokens = max_output_tokens
        self._fifo_message_mode = False
        self._total_tokens = 0
        if len(tools) > 0:
            self._tools = tools
            self._tools_dict = {tool.name: tool for tool in self._tools}
            self._chat_model = self._chat_model.bind_tools(self._tools)

    def __get_chat_model(
        self, model_provider: str = "openai", max_output_tokens: int = 2048
    ) -> BaseChatModel:
        """Returns a new model instance, don't call me."""
        if model_provider == "openai":
            openai_model = ChatOpenAI(
                model_name="gpt-4o-mini", max_tokens=max_output_tokens
            )
            return openai_model
        elif model_provider == "anthropic":
            anthropic_model = ChatAnthropic(
                model="claude-3-sonnet-20240229", max_tokens=max_output_tokens
            )
            return anthropic_model
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")

    def _token_count(self, message: AIMessage):
        metadata = message.response_metadata
        if self._model_provider == "openai":
            return metadata["token_usage"]["total_tokens"]
        elif self._model_provider == "anthropic":
            anthropic_total_tokens = (
                metadata["usage"]["input_tokens"] + metadata["usage"]["output_tokens"]
            )
            return anthropic_total_tokens
        else:
            logger.warning(
                f"Token count unsupported for model provider: {self._model_provider}"
            )
            return -1

    def append_message(self, message: BaseMessage):
        if type(message) == AIMessage:
            self._total_tokens = self._token_count(message)
        if (self._total_tokens > self._max_context_tokens) and (
            self._fifo_message_mode == False
        ):
            logger.warning(
                f"Total tokens {self._total_tokens} exceeded max context tokens {self._max_context_tokens} -> activating FIFO message mode"
            )
            self._fifo_message_mode = True
        if self._fifo_message_mode:
            self._message_history.pop(1)  # Remove the oldest message
            self.print_message_history()

        self._message_history.append(message)

    def _process_tool_calls(self, model_response: AIMessage) -> None:
        """Process tool calls from the model response and add results to chat history."""
        if not hasattr(model_response, "tool_calls") or not model_response.tool_calls:
            logger.debug("No tool calls in response")
            return

        for tool_call in model_response.tool_calls:
            # Loop over tool calls
            logger.debug(f"Tool call requested: {tool_call}")

            # Skip if no tool calls or empty
            if not tool_call or "name" not in tool_call:
                logger.warning(f"Invalid tool call format: {tool_call}")
                continue

            # Get tool name and arguments
            tool_name = tool_call["name"].lower()
            tool_args = tool_call.get("arguments", {})

            # Check if tool exists
            if tool_name not in self._tools_dict:
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
            selected_tool = self._tools_dict[tool_name]

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
                logger.debug(f"Tool '{tool_name}' response: {tool_result}")

            except Exception as e:
                logger.error(f"Error invoking tool {tool_name}: {e} ({tool_call})")

                # Create error message
                error_content = f"Error executing {tool_name}: {str(e)}"
                tool_msg = ToolMessage(
                    content=error_content,
                    tool_call_id=tool_call.get("id", "unknown"),
                    name=tool_name,
                )
                self.append_message(tool_msg)

    def invoke(self, message: str) -> str:

        # update system prompt with latestcontext
        system_message = SystemMessage(f"{self._system_prompt}\n{user_context}")
        if len(self._message_history) == 0:
            self._message_history.append(system_message)
        else:
            self._message_history[0] = system_message

        self.append_message(HumanMessage(message))  # Add user message to history
        model_response = self._chat_model.invoke(
            self._message_history
        )  # Invoke the model
        self.append_message(model_response)  # Add model response to history

        self._process_tool_calls(model_response)

        if model_response.tool_calls:  # If tools were called, invoke the model again
            model_response = self._chat_model.invoke(
                self._message_history
            )  # Invoke the model again
            self.append_message(model_response)  # Add model response to history
            logger.debug(model_response.content)  # Log the model response

        self._total_tokens = self._token_count(model_response)
        return model_response.content  # Return the model response

    def print_message_history(self):
        for index, message in enumerate(self._message_history):
            logger.info(
                f"Message #{index}, {message.type}, {message.content[:100]  + "..." if len(message.content) > 100 else message.content }"
            )

    def print_message_metadata(self):
        for index, message in enumerate(self._message_history):
            if message.response_metadata:
                logger.info(
                    f"Message #{index}, type: {message.type}, metadata: {message.response_metadata}"
                )

    def reset_message_history(self):
        if len(self._message_history) > 0:
            self._message_history = [self._message_history[0]]
            self._fifo_message_mode = False

    def get_default_prompts(self):
        return MODE_RESOURCES[current_mode]["default_prompts"]

    def run_tests(self):
        if(len(MODE_RESOURCES[current_mode]["test_prompts"]) > 0):
            logger.info("Running tests for current mode")
            for test in MODE_RESOURCES[current_mode]["test_prompts"]:
                logger.info(f"USER: {test}")
                self.invoke(test)
        else:
            logger.warning("No test prompts defined for current mode")

class WelcomeAgent(BaseAgent):

    def __init__(self, model_provider: str = "openai"):
        super().__init__(
            system_prompt=MODE_RESOURCES["welcome"]["system_prompt"],
            model_provider=model_provider,
            tools=[activate],
        )


class IdentificationAgent(BaseAgent):
    def __init__(self, model_provider: str = "openai"):
        super().__init__(
            system_prompt=MODE_RESOURCES["identification"]["system_prompt"],
            model_provider=model_provider,
            tools=[identify_user_with_password, activate],
        )


class SystemAgent(BaseAgent):
    def __init__(self, model_provider: str = "openai"):
        super().__init__(
            system_prompt=MODE_RESOURCES["system"]["system_prompt"],
            model_provider=model_provider,
            tools=[activate, run_task, play_song],
        )


class LocalPythonInterpreter(BaseAgent):
    def __init__(self, model_provider: str = "openai"):
        super().__init__(
            system_prompt=MODE_RESOURCES["python"]["system_prompt"],
        )
        self.__own_message_history = [
            SystemMessage(f"{self._system_prompt}\n{user_context}")
        ]

    def _run_system_command(self, user_query: str) -> str:
        """
        Send a prompt to the model and executes the output as a python script.
        """

        user_message = HumanMessage(content=user_query)
        self.__own_message_history.append(user_message)

        try:

            model_code_message = self._chat_model.invoke(self.__own_message_history)

            # here we could be stream writing to the file
            python_script = model_code_message.content

            self.__own_message_history.append(model_code_message)

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
                self.__own_message_history.append(HumanMessage(result_message))

            if len(result.stderr) > 0:
                stderr_message = f"{self._script_label()} - execution failed - Return code : {result.returncode} - STDERR: {result.stderr}"
                logger.warning(stderr_message)
                self.__own_message_history.append(HumanMessage(stderr_message))

        ################ Error handling
        except subprocess.CalledProcessError as cpe:

            error_message = f"{self._script_label()} - execution failed: ProcessError: {str(cpe)} - {cpe.output} - {cpe.stdout} - {cpe.stderr}."
            logger.error(error_message)
            self.__own_message_history.append(HumanMessage(error_message))

        except subprocess.TimeoutExpired as toe:

            error_message = f"{self._script_label()} execution failed: TimeoutExpired: {str(toe)} - {toe.output} - {toe.stdout} - {toe.stderr}."
            logger.error(error_message)
            self.__own_message_history.append(HumanMessage(error_message))

        except Exception as e:

            error_message = (
                f"{self._script_label()} - execution failed: Exception: {str(e)}"
            )
            logger.error(error_message)
            self.__own_message_history.append(HumanMessage(error_message))

        return result


# static instances of the agents
mode_agent_mapping = {
    "identification": IdentificationAgent(),
    "system": SystemAgent(),
    "welcome": WelcomeAgent(),
    "python": LocalPythonInterpreter(),
}


def get_focus_agent():
    return mode_agent_mapping[current_mode]


def get_current_mode():
    return current_mode
