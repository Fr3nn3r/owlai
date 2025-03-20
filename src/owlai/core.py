#  ,_,
# (O,O)
# (   )
# -"-"-
print("Loading core module")

from typing import List, Dict, Any, Optional, Union, cast
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.language_models import LanguageModelInput
from langchain_core.tools import BaseTool
from langchain_core.messages.tool import ToolCall
import logging.config
import logging
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

import traceback
from owlai.owlsys import sprint
from langchain_core.tools import BaseTool, ArgsSchema

logger = logging.getLogger("core")

user_context: str = "CONTEXT: "


class DefaultOwlAgentInput(BaseModel):
    """Input schema for DefaultOwlAgent."""

    query: str = Field(description="some natural language input to the agent")


class OwlAgent(BaseTool, BaseModel):

    # JSON defined properties
    name: str = "sad_unamed_owl_agent"
    description: str
    args_schema: Optional[ArgsSchema] = DefaultOwlAgentInput
    system_prompt: str
    model_provider: str
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 2048
    context_size: int = 4096
    tools_names: List[str] = []  # list of tools this agent can use
    default_queries: Optional[List[str]] = None

    # Runtime updated properties
    total_tokens: int = 0
    fifo_message_mode: bool = False
    callable_tools: List[BaseTool] = []

    # Private attribute
    _chat_model_cache: Any = None
    _tool_dict: Dict[str, BaseTool] = {}
    _message_history: List[BaseMessage] = []

    @property
    def chat_model(self) -> BaseChatModel:
        if self._chat_model_cache is None:
            self._chat_model_cache = init_chat_model(
                model=self.model_name,
                model_provider=self.model_provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            logger.debug(
                f"Chat model initialized: {self.model_name} {self.model_provider} {self.temperature} {self.max_tokens}"
            )
        return self._chat_model_cache

    def init_callable_tools(self, tools: List[Any]):
        """Initialize callable tools with the provided tools list."""
        self.callable_tools = tools
        self._chat_model_cache = self.chat_model.bind_tools(tools)
        for tool in tools:
            self._tool_dict[tool.name] = tool
        return self._chat_model_cache

    def _token_count(self, message: Union[AIMessage, BaseMessage]):
        if not isinstance(message, AIMessage) or not hasattr(
            message, "response_metadata"
        ):
            logger.warning(
                "Cannot count tokens: message is not an AIMessage or lacks response_metadata"
            )
            return 0

        metadata = message.response_metadata
        # Should get rid of model_provider dependend code ------------- should be a util function outside owlagent
        if self.model_provider == "openai":
            return metadata["token_usage"]["total_tokens"]
        elif self.model_provider == "anthropic":
            anthropic_total_tokens = (
                metadata["usage"]["input_tokens"] + metadata["usage"]["output_tokens"]
            )
            return anthropic_total_tokens
        else:
            logger.warning(
                f"Token count unsupported for model provider: '{self.model_provider}'"
            )
            return -1

    def append_message(self, message: BaseMessage):
        if type(message) == AIMessage:
            self.total_tokens = self._token_count(message)
        if (self.total_tokens > self.context_size) and (
            self.fifo_message_mode == False
        ):
            logger.warning(
                f"Total tokens '{self.total_tokens}' exceeded max context tokens '{self.context_size}' -> activating FIFO message mode"
            )
            self.fifo_message_mode = True
        if self.fifo_message_mode:
            self._message_history.pop(1)  # Remove the oldest message
            if (
                self._message_history[-1].type == "tool"
            ):  # Remove the tool message if any
                self._message_history.pop(1)

        self._message_history.append(message)

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
            tool_args = tool_call.get("args", {})

            # Check if tool exists
            if tool_name not in self.tools_names:
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

    def invoke(
        self,
        input: Union[str, Dict[str, Any], ToolCall],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        logger.debug(f"Invoking agent with input: {input}")
        return self.message_invoke(str(input))

    def message_invoke(self, message: str) -> str:
        try:
            # update system prompt with latestcontext
            system_message = SystemMessage(f"{self.system_prompt}\n{user_context}")
            if len(self._message_history) == 0:
                self._message_history.append(system_message)
            else:
                self._message_history[0] = system_message

            self.append_message(HumanMessage(message))  # Add user message to history
            response = self.chat_model.invoke(self._message_history)
            self.append_message(response)  # Add model response to history

            # Only process tool calls if response is AIMessage
            if isinstance(response, AIMessage):
                self._process_tool_calls(response)

                if (
                    hasattr(response, "tool_calls") and response.tool_calls
                ):  # If tools were called, invoke the model again
                    response = self.chat_model.invoke(self._message_history)
                    self.append_message(response)  # Add model response to history
                    # logger.debug(response.content)  # Log the model response

                self._total_tokens = self._token_count(response)

            return str(response.content)  # Return the model response as string

        except Exception as e:
            logger.error(f"Error invoking model '{self.model_name}': '{e}'")
            logger.error(f"Stack trace: '{traceback.format_exc()}'")
            return f"Error: {str(e)}"

    def print_message_history(self):
        sprint(self._message_history)

    def print_message_metadata(self):
        for index, message in enumerate(self._message_history):
            if message.response_metadata:
                logger.info(
                    f"Message #{index} type: '{message.type}' metadata: '{message.response_metadata}'"
                )

    def print_system_prompt(self):
        if len(self._message_history) > 0:
            logger.info(f"System prompt: '{self._message_history[0].content}'")
        else:
            logger.info(f"System prompt: '{self.system_prompt}'")

    def reset_message_history(self):
        if len(self._message_history) > 0:
            self._message_history = [self._message_history[0]]
            self.fifo_message_mode = False

    def print_info(self):
        sprint(self)

    def print_model_info(self):
        logger.debug(
            f"Chat model: {self.model_name} {self.model_provider} {self.temperature} {self.max_tokens}"
        )
        sprint(self.chat_model)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        logger.debug(f"Running tool with query: {query}")
        return self.message_invoke(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self.message_invoke(query)


def main():
    pass


if __name__ == "__main__":
    main()
