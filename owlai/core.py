"""
OwlAI Core Module

Note: We are using Pydantic v1 because it's required by langchain-core and other LangChain components.
This is a temporary solution until LangChain fully supports Pydantic v2.
The deprecation warnings are suppressed in pytest configuration.
"""

#  ,_,
# (O,O)
# (   )
# -"-"-
print("Loading core module")

# Guard against duplicate module loading
import sys

from typing import List, Dict, Any, Optional, Union, cast
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
import logging.config
import logging
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime, timezone

from langchain.chat_models import init_chat_model
from owlai.db.memory import Memory

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

import traceback
from owlai.services.system import sprint
from langchain_core.tools import BaseTool, ArgsSchema
from owlai.services.telemetry import RequestLatencyTracker

# Get logger using the module name
logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Configuration for Language Model settings.

    Attributes:
        model_provider (str): Provider of the language model (e.g., 'openai', 'anthropic')
        model_name (str): Name of the specific model to use
        temperature (float): Sampling temperature (default: 0.1)
        max_tokens (int): Maximum tokens per response (default: 2048)
        context_size (int): Maximum context window size (default: 4096)
        tools_names (List[str]): List of available tool names
        tool_choice (str): Tool choice mode for the model (default: None)
    """

    model_provider: str
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 2048
    context_size: int = 4096
    tools_names: List[str] = []  # list of tools this agent can use
    tool_choice: Optional[str] = None  # Tool choice mode for the model


class OwlAgent(BaseModel):
    """Base agent class that implements core functionality for interacting with LLMs.

    This class combines LangChain's BaseTool and Pydantic's BaseModel to provide
    a flexible agent that can be easily configured and used as both a tool and an LLM.
    """

    # JSON defined properties
    name: str = "sad_unamed_owl_agent"
    version: str
    description: str
    # args_schema: Optional[ArgsSchema] = DefaultAgentInput
    llm_config: LLMConfig
    system_prompt: str
    default_queries: Optional[List[str]] = None

    # Runtime updated properties
    total_tokens: int = 0
    fifo_message_mode: bool = False
    callable_tools: List[BaseTool] = []

    # Private attributes
    _chat_model_cache: Any = None
    _tool_dict: Dict[str, BaseTool] = {}
    _message_history: List[BaseMessage] = []
    _memory: Optional[Memory] = None
    _agent_id: Optional[UUID] = None
    _conversation_id: Optional[UUID] = None

    @property
    def chat_model(self) -> BaseChatModel:
        """Lazy initialization of the chat model.

        Returns:
            BaseChatModel: The initialized chat model
        """
        if self._chat_model_cache is None:
            self._chat_model_cache = init_chat_model(
                model=self.llm_config.model_name,
                model_provider=self.llm_config.model_provider,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )
            logger.debug(
                f"Chat model initialized: {self.llm_config.model_name} {self.llm_config.model_provider} {self.llm_config.temperature} {self.llm_config.max_tokens} {self.llm_config.tool_choice}"
            )
        return self._chat_model_cache

    def init_callable_tools(self, tools: List[Any]):
        """Initialize callable tools with the provided tools list.

        Args:
            tools (List[Any]): List of tools to initialize

        Returns:
            BaseChatModel: The chat model with bound tools
        """
        self.callable_tools = tools
        self._chat_model_cache = self.chat_model.bind_tools(tools)
        for tool in tools:
            self._tool_dict[tool.name] = tool
            logger.debug(f"Initialized tool: {tool.name} for agent {self.name}")
        return self._chat_model_cache

    def init_memory(self, memory: Memory) -> None:
        """Initialize the memory system for this agent.

        Args:
            memory (Memory): Memory implementation to use
        """
        self._memory = memory
        # Register agent in memory system
        self._agent_id = memory.get_or_create_agent(self.name, self.version)
        logger.info(
            f"Initialized memory for agent {self.name} with ID {self._agent_id}"
        )

    def _start_new_conversation(self) -> None:
        """Start a new conversation in memory."""
        if self._memory and self._agent_id:
            self._conversation_id = self._memory.create_conversation(
                f"Conversation with {self.name}"
            )
            logger.debug(f"Started new conversation {self._conversation_id}")

    def append_message(self, message: BaseMessage) -> None:
        """Append a message to history and memory.

        Args:
            message (BaseMessage): Message to append
        """
        if type(message) == AIMessage:
            self.total_tokens = self._token_count(message)
            # logger.debug(f"Total tokens: {self.total_tokens} for agent '{id(self)}'")

        # Handle FIFO mode to manage context window
        if (
            not self.fifo_message_mode
            and self.total_tokens > self.llm_config.context_size
        ):
            logger.debug(
                f"Total tokens '{self.total_tokens}' exceeded max context tokens '{self.llm_config.context_size}'; activating FIFO message mode."
            )
            self.fifo_message_mode = True

        if self.fifo_message_mode and self.total_tokens > self.llm_config.context_size:
            # Remove the oldest message
            removed = self._message_history.pop(1)
            self.total_tokens -= self._token_count(removed)
            logger.debug("Oldest message removed from history.")

            # Remove tool message if present as the next message
            while (
                len(self._message_history) > 1
                and self._message_history[1].type == "tool"
            ):
                removed = self._message_history.pop(1)
                self.total_tokens -= self._token_count(removed)
                logger.warning("Tool message removed from history.")

            def print_message_type_history():
                for message in self._message_history:
                    logger.debug(f"Message type: {message.type}")

            # print_message_type_history()
        else:
            self.fifo_message_mode = False

        # Add to message history
        self._message_history.append(message)

        # Log to memory if available
        if self._memory and self._agent_id and self._conversation_id:
            # Start new conversation if needed
            if not self._conversation_id:
                self._start_new_conversation()

            # Log the message
            self._memory.log_message(
                agent_id=self._agent_id,
                conversation_id=self._conversation_id,
                source=message.type,
                content=str(message.content),
                metadata=message.response_metadata,
                tool_calls=getattr(message, "tool_calls", None),
            )

    def _token_count(self, message: Union[AIMessage, BaseMessage]):
        """Count tokens in a message based on the model provider. Should get rid of model_provider dependend code ------------- should be a util function outside owlagent

        Args:
            message (Union[AIMessage, BaseMessage]): Message to count tokens for

        Returns:
            int: Number of tokens in the message, or -1 if unsupported
        """
        if not isinstance(message, AIMessage) or not hasattr(
            message, "response_metadata"
        ):
            logger.warning(
                "Cannot count tokens: message is not an AIMessage or lacks response_metadata"
            )
            return 0

        metadata = message.response_metadata
        if (
            self.llm_config.model_provider == "openai"
            or self.llm_config.model_provider == "mistralai"
        ):
            return metadata.get("token_usage", {}).get("total_tokens", 0)
        elif self.llm_config.model_provider == "anthropic":
            anthropic_total_tokens = (
                metadata["usage"]["input_tokens"] + metadata["usage"]["output_tokens"]
            )
            return anthropic_total_tokens
        else:
            logger.warning(
                f"Token count unsupported for model provider: '{self.llm_config.model_provider}'"
            )
            return -1

    def _process_tool_calls(self, model_response: AIMessage) -> None:
        """Process tool calls from the model response and add results to chat history.

        Args:
            model_response (AIMessage): Response containing tool calls to process
        """
        if not hasattr(model_response, "tool_calls") or not model_response.tool_calls:
            logger.debug("No tool calls in response")
            return

        if len(model_response.tool_calls) > 1:
            logger.warning(
                f"Multiple tool calls in response is experimental: {model_response.tool_calls}"
            )

        for tool_call in model_response.tool_calls:
            # Loop over tool calls
            logger.debug(f"Tool call requested: '{tool_call}'")

            # Skip if no tool calls or empty
            if not tool_call or "name" not in tool_call:
                logger.warning(f"Invalid tool call format: '{tool_call}'")
                continue

            # Get tool name and arguments
            tool_name = tool_call["name"].lower()
            tool_args = tool_call.get("args", {})

            # Check if tool exists
            if tool_name not in self.llm_config.tools_names:
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
            selected_tool = self._tool_dict[tool_name]

            try:
                # Remove 'self' from arguments if present
                if "self" in tool_args:
                    del tool_args["self"]

                logger.debug(f"Invoking tool '{tool_name}' with arguments: {tool_args}")
                # Invoke the tool
                tool_result = selected_tool.invoke(tool_args)

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
                logger.error(f"Stack trace: '{traceback.format_exc()}'")

                # Create error message
                error_content = f"Error executing '{tool_name}': '{str(e)}'"
                tool_msg = ToolMessage(
                    content=error_content,
                    tool_call_id=tool_call.get("id", "unknown"),
                    name=tool_name,
                )
                self.append_message(tool_msg)
                continue

    def message_invoke(self, message: str) -> str:
        """Process a message and return the model's response.

        Args:
            message (str): The input message

        Returns:
            str: The model's response or error message
        """
        try:
            # Start new conversation if needed
            if not self._conversation_id and self._memory:
                self._start_new_conversation()

            # Update system prompt
            system_message = SystemMessage(content=self.system_prompt)
            if len(self._message_history) == 0:
                self._message_history.append(system_message)
            else:
                self._message_history[0] = system_message

            # Add user message
            self.append_message(HumanMessage(content=message))

            # Get model response
            response = self.chat_model.invoke(self._message_history)
            self.append_message(response)

            # Process tool calls if needed
            if isinstance(response, AIMessage):
                self._process_tool_calls(response)

                if hasattr(response, "tool_calls") and response.tool_calls:
                    response = self.chat_model.invoke(self._message_history)
                    self.append_message(response)

                # self._total_tokens = self._token_count(response)

            return str(response.content)

        except Exception as e:
            logger.error(f"Error invoking model '{self.llm_config.model_name}': {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return f"Error: {str(e)}"

    async def stream_message(self, message: str):
        """Stream a response from the agent.

        Args:
            message (str): The input message

        Yields:
            str: Chunks of the model's response
        """
        latency = RequestLatencyTracker(str(id(self)))
        first_token_logged = False
        try:
            logger.info(f"Streaming message for agent {self.name}")
            # Start new conversation if needed
            if not self._conversation_id and self._memory:
                self._start_new_conversation()

            latency.mark("conversation_setup")

            # update system prompt with latestcontext
            system_message = SystemMessage(f"{self.system_prompt}")
            if len(self._message_history) == 0:
                self._message_history.append(system_message)
            else:
                self._message_history[0] = system_message

            self.append_message(HumanMessage(message))
            latency.mark("messages_prepared")

            response = self.chat_model.invoke(self._message_history)
            latency.mark("initial_response_received")

            self.append_message(response)

            # Process tool calls if needed
            if isinstance(response, AIMessage):
                self._process_tool_calls(response)
                latency.mark("tool_calls_processed")

                if hasattr(response, "tool_calls") and response.tool_calls:
                    # Stream the response
                    complete_response = []

                    async for chunk in self.chat_model.astream(self._message_history):
                        if chunk.content:
                            if (
                                not complete_response and not first_token_logged
                            ):  # First token
                                latency.mark("first_token_generated")
                                first_token_logged = True
                            complete_response.append(chunk.content)
                            yield chunk.content

                    final_response = "".join(complete_response)
                    final_message = AIMessage(content=final_response)
                    self.append_message(final_message)
                    latency.mark("streaming_complete")

                else:
                    yield response.content
                    latency.mark("direct_response_sent")
            else:
                raise Exception("Invalid response from model")

            logger.info(f"Streaming message for agent {self.name} completed")
            # Log final latency breakdown
            latencies = latency.get_latency_breakdown()
            # logger.debug(f"Agent processing latencies: {latencies}")

        except Exception as e:
            logger.error(
                f"Error streaming from model '{self.llm_config.model_name}': '{e}'"
            )
            logger.error(f"Stack trace: '{traceback.format_exc()}'")
            yield f"Error: {str(e)}"

    def print_message_history(self):
        """Print the current message history."""
        sprint(self._message_history)

    def print_message_metadata(self):
        """Print metadata for all messages in history."""
        for index, message in enumerate(self._message_history):
            if message.response_metadata:
                logger.info(
                    f"Message #{index} type: '{message.type}' metadata: '{message.response_metadata}'"
                )

    def print_system_prompt(self):
        """Print the current system prompt."""
        if len(self._message_history) > 0:
            logger.info(f"System prompt: '{self._message_history[0].content}'")
        else:
            logger.info(f"System prompt: '{self.system_prompt}'")

    def reset_message_history(self):
        """Reset message history while preserving system message."""
        if len(self._message_history) > 0:
            self._message_history = [self._message_history[0]]
            self.fifo_message_mode = False

    def print_info(self):
        """Print agent information."""
        sprint(self)

    def print_model_info(self):
        """Print model configuration information."""
        logger.debug(
            f"Chat model: {self.llm_config.model_name} {self.llm_config.model_provider} {self.llm_config.temperature} {self.llm_config.max_tokens}"
        )
        sprint(self.chat_model)


class MessageManager:
    """Manage message history and system prompt."""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.message_history = []


class ToolManager:
    """Manage tools."""

    def __init__(self):
        self.tools = []


class ModelManager:
    """Manage model."""

    def __init__(self):
        self.model = None


def main():
    pass


if __name__ == "__main__":
    main()
