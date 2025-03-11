#       ,_,
#      (O,O)
#      (   )
#      -"-"-

import subprocess
from subprocess import CompletedProcess
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional, Callable
from typing_extensions import TypedDict, deprecated
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.prompt_values import PromptValue
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
import logging.config
import logging
from langchain_core.tools import tool, BaseTool
import re
from owlai.db import (
    USER_DATABASE,
    get_user_by_password,
)
from .spotify import play_song_on_spotify
from pydantic import BaseModel, ValidationError

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from rich.console import Console

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("main_logger")

user_context: str = "CONTEXT: "

focus_role: str = "identification"

class _ToolBox:

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
    def activate_mode(mode: str):
        """
            Activates a different owlai mode.
        Args:
            mode: a string containing the mode to activate. possible values are:
                - identification: activates identification mode
                - system: activates system mode
                - welcome: activates welcome mode
                - command_manager: activates command manager mode
                - qna: activates qna mode
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
        logger.info(f"Playing song: {song_name} by {artist_name}")
        play_song_on_spotify(song_name, artist_name)

    @tool
    def get_answer_from_knowledge_base(question: str):
        """
            Gets an answer from the knowledge base.
        Args:
            question: a string containing the question to answer.
        """
        logger.debug(f"Running RAG question: {question} {toolbox_hook_rag_engine}")
        rag_answer = toolbox_hook_rag_engine(question)
        # logger.debug(rag_answer)
        return rag_answer

        return "This is a test answer"



#toolbox = ToolBox()

class OwlConfig(BaseModel):
    role: str
    implementation: str = "openai"  # default value
    model_name: str
    temperature: float = 0.9
    max_tokens: int = 2048
    max_context_tokens: int = 4096
    tools_names: List[str] = []
    system_prompt: str
    default_prompts: Optional[List[str]] = None
    test_prompts: List[str] = []

    class Config:
        validate_assignment = True

def load_config(role: str, config: dict[str, dict[str, Any]]) -> OwlConfig:
    """Load and validate configuration for a specific owl role."""
    if role not in config:
        raise ValueError(f"Configuration not found for role: {role}")
    
    try:
        # Convert the raw config dict to OwlConfig instance
        config_data = config[role]
        owl_config = OwlConfig(
            role=role,
            implementation=config_data.get("model_provider", "openai"),
            model_name=config_data.get("model_name", "gpt-4o-mini"),
            temperature=config_data.get("temperature", 0.9),
            max_tokens=config_data.get("max_output_tokens", 2048),
            max_context_tokens=config_data.get("max_context_tokens", 4096),
            tools_names=config_data.get("tools_names", []),
            system_prompt=config_data["system_prompt"],
            default_prompts=config_data.get("default_prompts"),
            test_prompts=config_data.get("test_prompts", [])
        )
        return owl_config
    
    except ValidationError as e:
        logger.error(f"Configuration validation failed for role {role}: {e}")
        raise


def list_roles(config) -> List[str]:
    """Returns list of all valid owl roles from CONFIG."""
    return list(config.keys())



class OwlCheek() :

    def __init__(
        self,
        config : OwlConfig,
    ):
        self.config = config
        self.chat_model : BaseChatModel = init_chat_model(
            model=config.model_name,
            model_provider=config.implementation,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

class Owl(OwlCheek):

    def __init__(
        self,
        config : OwlConfig,
        tools: List[BaseTool] = [],
    ):
        super().__init__(config)
        self.max_context_tokens = config.max_context_tokens
        self.message_history: List[BaseMessage] = []


        self.fifo_message_mode = False
        self.total_tokens = 0

        if len(tools) > 0:
            self.callable_tools = tools
            self.tools_dict = {tool.name: tool for tool in self.callable_tools}
            self.chat_model = self.chat_model.bind_tools(self.callable_tools)

        #backward compatibility
        self.system_prompt = config.system_prompt
        self.implementation = config.implementation

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
            # logger.debug(response.content)  # Log the model response

        self._total_tokens = self._token_count(response)
        return response.content  # Return the model response

    def print_message_history(self):
        for index, message in enumerate(self.message_history):
            logger.info(
                f"Message #{index} '{message.type}' '{ (message.content[:100]  + '...' if len(message.content) > 100 else message.content )}'"
            )

    def print_message_metadata(self):
        for index, message in enumerate(self.message_history):
            if message.response_metadata:
                logger.info(
                    f"Message #{index} type: '{message.type}' metadata: '{message.response_metadata}'"
                )

    def print_system_prompt(self):
        if len(self.message_history) > 0:
            logger.info(f"System prompt: '{self.message_history[0].content}'")
        else:
            logger.info(f"System prompt: '{self.system_prompt}'")

    def reset_message_history(self):
        if len(self.message_history) > 0:
            self.message_history = [self.message_history[0]]
            self.fifo_message_mode = False

    def print_info(self):
        sprint(self.config)
    

    # Add proper resource cleanup:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

class _Owl:
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
        #role: str = "welcome",
        implementation: str = "openai",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        max_context_tokens: int = 4096,
        tools: List[BaseTool] = [],
        system_prompt: str = None,
    ):
        #self.role = role
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
            # logger.debug(response.content)  # Log the model response

        self._total_tokens = self._token_count(response)
        return response.content  # Return the model response

    def print_message_history(self):
        for index, message in enumerate(self.message_history):
            logger.info(
                f"Message #{index} '{message.type}' '{ (message.content[:100]  + '...' if len(message.content) > 100 else message.content )}'"
            )

    def print_message_metadata(self):
        for index, message in enumerate(self.message_history):
            if message.response_metadata:
                logger.info(
                    f"Message #{index} type: '{message.type}' metadata: '{message.response_metadata}'"
                )

    def print_system_prompt(self):
        if len(self.message_history) > 0:
            logger.info(f"System prompt: '{self.message_history[0].content}'")
        else:
            logger.info(f"System prompt: '{self.system_prompt}'")

    def reset_message_history(self):
        if len(self.message_history) > 0:
            self.message_history = [self.message_history[0]]
            self.fifo_message_mode = False

    def print_info(self):
        logger.info(
            f"role='{self.role}', model-provider='{self.implementation}', model-name='{self.model_name}', tools {self.tools_dict.keys()}"
        )

    # Add proper resource cleanup:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class _LocalPythonInterpreter(Owl):
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

            # logger.debug(f"Raw script: {python_script}") # a bit too verbose

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

    def remove_python_markup_simple(self, text):
        return text.replace("```", "").replace("```", "")

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


class _LocalRAGTool(Owl):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"
        db_persist_directory = "data/vector"
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

        self.prompt = PromptTemplate.from_template(self.system_prompt)
        self.vector_store = Chroma(
            embedding_function=embeddings, persist_directory=db_persist_directory
        )

    def rag_question(self, question: str) -> str:
        """
        Runs the RAG query against the vector store and returns an answer to the question.

        Args:
            question: a string containing the question to answer.
        """

        retrieved_docs = self.vector_store.similarity_search(query=question, k=4)

        for doc in retrieved_docs:
            logger.debug(f"Document matched: {doc.page_content[:100]} {doc.metadata}")

        docs_content = "\n\n".join(
            doc.page_content.encode("utf-8", errors="ignore").decode("utf-8")
            for doc in retrieved_docs
        )
        # message_with_question_and_context : PromptValue = self.prompt.invoke({"question": question, "context": docs_content})
        message_with_question_and_context = self.prompt.format(
            question=question, context=docs_content
        )
        currated_message_with_question_and_context = (
            message_with_question_and_context.encode("utf-8", errors="ignore").decode(
                "utf-8"
            )
        )
        # logger.debug(f"length [chars]: {len(message_with_question_and_context.to_string())} - {message_with_question_and_context.to_string()}")
        messages = [SystemMessage(currated_message_with_question_and_context)]
        messages = self.chat_model.invoke(messages)

        logger.debug(f"Raw RAG answer: {messages.content}")

        # no history management for now
        return messages.content

    def index_folder(self, folder_path: str):
        """
        Index a folder and its contents (this should be done offline).
        """
        return "Do this offline idiot"


def sprint(*args):
    """A smart print function for JSON-like structures"""
    console = Console()
    for arg in args:
        console.print(arg)  # Normal print with `rich`


class OwlAIAgent:

    def __init__(
        self,
        implementation: str = "openai",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.9,
        max_tokens: int = 2048,
        max_context_tokens: int = 4096,
        tools: List[BaseTool] = [],
        system_prompt: str = None,
    ):
        self.implementation = implementation
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_context_tokens = max_context_tokens
        self.tools = tools
        self.system_prompt = system_prompt
        self.chat_model = init_chat_model(
            model=model_name,
            model_provider=implementation,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.state_memory = MemorySaver()
        self.state_config = {"configurable": {"thread_id": str(id(self))}}

        self.agent_graph = create_react_agent(
            self.chat_model,
            self.tools,
            prompt=SystemMessage(self.system_prompt),
            checkpointer=self.state_memory,
        )

    def invoke(self, message: str) -> str:
        graph_response = self.agent_graph.invoke(
            {"messages": [HumanMessage(message)]}, self.state_config
        )

        if logger.isEnabledFor(logging.DEBUG):
            sprint(graph_response)

        return self._return_last_message_content(graph_response)

    def _return_last_message_content(self, response: Dict[str, Any]) -> str:
        return response["messages"][-1].content

    def print_message_history(self):
        state = self.agent_graph.get_state(self.state_config)
        for index, message in enumerate(state.values["messages"]):
            logger.info(
                f"Message #{index} type: '{message.type}' content: '{ (message.content[:100] + '...' if len(message.content) > 100 else message.content)}'"
            )
            if logger.isEnabledFor(logging.DEBUG):
                sprint(message)

    def print_message_metadata(self):
        state = self.agent_graph.get_state(self.state_config)
        for index, message in enumerate(state.values["messages"]):
            if message.response_metadata:
                logger.info(
                    f"Message #{index} type: '{message.type}' metadata: '{message.response_metadata}'"
                )

    def print_system_prompt(self):
        logger.info(f"System prompt: '{self.system_prompt}'")

    def reset_message_history(self):
        logger.warning("Resetting message history not supported")

    def run_tests(self):
        logger.warning("Running tests not supported (NOT TO BE MANAGED HERE)")

    def print_info(self):
        logger.info(
            f"model-provider='{self.implementation}', model-name='{self.model_name}', tools='{', '.join([t.name for t in self.tools])}'"
        )

    # NOT SURE WHAT THIS IS...
    # Add proper resource cleanup (???):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.warning("Cleanup not supported (???)")

# Remove Edwige class and add this import if needed elsewhere
#from .manager import Edwige


