from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from owlai.core.config import AgentConfig
from owlai.core.message_manager import MessageManager
from owlai.core.tool_manager import ToolManager
from owlai.core.model_manager import ModelManager
from owlai.core.logging_setup import get_logger
from rich.console import Console
from rich.pretty import Pretty
from rich.panel import Panel
from rich.table import Table


class OwlAgent:
    """Main agent class that orchestrates message, tool, and model operations"""

    def __init__(
        self,
        config: AgentConfig,
        message_manager: Optional[MessageManager] = None,
        tool_manager: Optional[ToolManager] = None,
        model_manager: Optional[ModelManager] = None,
    ):
        self.config = config
        self.message_manager = message_manager or MessageManager(config.llm_config)
        self.tool_manager = tool_manager or ToolManager()
        self.model_manager = model_manager or ModelManager(config.llm_config)
        self.logger = get_logger("agent")
        self.console = (
            Console(force_terminal=True) if self.logger.isEnabledFor(10) else None
        )

        # Initialize system message
        self.message_manager.append_message(
            SystemMessage(content=self.config.system_prompt)
        )
        self.logger.info(f"Initialized OwlAgent: {self}")

    def _debug_print(self, title: str, content: any) -> None:
        """Print debug information using rich console if debug is enabled"""
        if self.console and self.logger.isEnabledFor(10):
            self.console.print(Panel(Pretty(content), title=title, border_style="blue"))

    def _debug_table(self, title: str, data: List[dict]) -> None:
        """Print debug information in table format using rich console if debug is enabled"""
        if self.console and self.logger.isEnabledFor(10):
            table = Table(title=title, show_header=True, header_style="bold magenta")
            if data:
                for key in data[0].keys():
                    table.add_column(key)
                for row in data:
                    table.add_row(*[str(v) for v in row.values()])
            self.console.print(table)

    def __repr__(self) -> str:
        """String representation of the agent"""
        return (
            f"OwlAgent(name='{self.config.name}', "
            f"description='{self.config.description}', "
            f"system_prompt_length={len(self.config.system_prompt)}, "
            f"tool_count={len(self.config.tools_names)})"
        )

    def message_invoke(self, message: str) -> str:
        """Process a user message and return the response"""
        if not message or not message.strip():
            self.logger.warning("Empty message received")
            raise ValueError("Message cannot be empty")

        preview = message[:100] + "..." if len(message) > 100 else message
        self.logger.info(f"Processing message: {preview}")

        # Add user message to history
        user_message = HumanMessage(content=message)
        self.message_manager.append_message(user_message)
        self.logger.debug("Added user message to history")

        try:
            # Get model response
            messages = self.message_manager.get_message_history()
            self.logger.debug(f"Getting model completion for {len(messages)} messages")
            ai_message = self.model_manager.get_completion(messages)
            self.message_manager.append_message(ai_message)

            # Add detailed debug logging for AI message structure
            self._debug_print(
                "AI Message Structure",
                {
                    "attributes": dir(ai_message),
                    "additional_kwargs": ai_message.additional_kwargs,
                    "has_tool_calls": hasattr(ai_message, "tool_calls"),
                    "content_length": len(ai_message.content),
                },
            )

            # Process any tool calls
            tool_calls = getattr(
                ai_message, "tool_calls", None
            ) or ai_message.additional_kwargs.get("tool_calls", [])
            self._debug_print(
                "Tool Calls",
                {
                    "from_attribute": getattr(ai_message, "tool_calls", None),
                    "from_kwargs": ai_message.additional_kwargs.get("tool_calls"),
                    "final_value": tool_calls,
                },
            )

            if tool_calls:
                self.logger.info(f"Processing {len(tool_calls)} tool calls")
                for tool_call in tool_calls:
                    tool_name = None
                    tool_args = {}
                    try:
                        # Log the tool call format
                        self._debug_print("Processing Tool Call", tool_call)

                        # Handle both old and new tool call formats
                        if isinstance(tool_call, dict) and "function" in tool_call:
                            tool_name = tool_call["function"]["name"]
                            tool_args = eval(tool_call["function"]["arguments"])
                            self._debug_print(
                                "Extracted Tool Info (New Format)",
                                {"name": tool_name, "args": tool_args},
                            )
                        else:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            self._debug_print(
                                "Extracted Tool Info (Old Format)",
                                {"name": tool_name, "args": tool_args},
                            )

                        if not tool_name:
                            self.logger.error(
                                "Invalid tool call format: missing tool name"
                            )
                            continue

                        self.logger.debug(
                            f"Invoking tool: {tool_name}, args={tool_args}"
                        )
                        tool_result = self.tool_manager.invoke_tool(
                            tool_name, tool_args
                        )
                        # Add tool result to history
                        tool_message = AIMessage(content=f"Tool result: {tool_result}")
                        self.message_manager.append_message(tool_message)
                        self.logger.debug(
                            f"Tool execution successful: {tool_name}, result={tool_result}"
                        )
                        # Return the tool result
                        return str(tool_result)
                    except Exception as e:
                        self.logger.error(
                            f"Tool execution failed: {tool_name or 'unknown'} - {e}"
                        )
                        # Log tool error but continue processing
                        error_message = f"Tool error: {str(e)}"
                        self.message_manager.append_message(
                            AIMessage(content=error_message)
                        )
                        # Return the error message
                        return error_message

            self.logger.info(
                f"Message processing completed: {len(ai_message.content)} chars, "
                f"{len(self.message_manager.get_message_history())} total messages"
            )
            return str(ai_message.content)

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            raise Exception(f"Error processing message: {str(e)}")

    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool with the agent"""
        self.logger.debug(f"Registering tool: {tool}")
        self.tool_manager.register_tool(tool)
        self.logger.info(f"Tool registered: {tool.name}")

    def clear_history(self) -> None:
        """Clear message history and reinitialize system message"""
        message_count = len(self.message_manager.get_message_history())
        self.message_manager.clear_history()
        self.message_manager.append_message(
            SystemMessage(content=self.config.system_prompt)
        )
        self.logger.info(
            f"Cleared message history: {message_count} messages, "
            f"reinitialized system prompt ({len(self.config.system_prompt)} chars)"
        )
