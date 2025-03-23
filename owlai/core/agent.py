from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.messages.tool import ToolCall
from owlai.core.config import AgentConfig
from owlai.core.message_manager import MessageManager
from owlai.core.tool_manager import ToolManager
from owlai.core.model_manager import ModelManager
from owlai.core.logging_setup import get_logger, debug_print, debug_table
from rich.console import Console
import json


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

            # Process any tool calls
            tool_calls = getattr(ai_message, "tool_calls", None)
            if not tool_calls:
                tool_calls = ai_message.additional_kwargs.get("tool_calls", [])

            # Create a new AIMessage with the tool calls preserved
            # Preserve the original AI message with tool calls using langchain's built-in structures
            # This avoids manual dictionary manipulation by using the proper AIMessage constructor
            # with the appropriate structure for tool calls
            preserved_message = AIMessage(
                content=ai_message.content,
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": str(tc.get("args", {})),
                            },
                        }
                        for tc in tool_calls
                    ],
                },
            )
            self.message_manager.append_message(preserved_message)

            if tool_calls:
                self.logger.info(f"Processing {len(tool_calls)} tool calls")
                for tool_call in tool_calls:
                    tool_name = None
                    tool_args = {}
                    args_str = "{}"  # Initialize with empty JSON object string

                    try:
                        # Handle both old and new tool call formats
                        if isinstance(tool_call, dict):
                            if "function" in tool_call:
                                tool_name = tool_call["function"]["name"]
                                # Get arguments string
                                args_str = tool_call["function"].get("arguments", "{}")
                                # Try to parse arguments, fall back to empty dict if invalid
                                try:
                                    if args_str == "{}" or not args_str.strip():
                                        tool_args = {}
                                    else:
                                        # Replace eval with safer json.loads
                                        tool_args = json.loads(args_str)
                                except json.JSONDecodeError:
                                    self.logger.warning(
                                        f"Failed to parse tool arguments: {args_str}"
                                    )
                                    tool_args = {}
                                self.logger.debug(
                                    self.logger,
                                    "Extracted Tool Info (New Format)",
                                    {"name": tool_name, "args": tool_args},
                                )
                            elif "name" in tool_call:
                                tool_name = tool_call["name"]
                                tool_args = tool_call.get("args", {})
                                self.logger.debug(
                                    self.logger,
                                    "Extracted Tool Info (Old Format)",
                                    {"name": tool_name, "args": tool_args},
                                )
                            else:
                                self.logger.error(
                                    f"Invalid tool call format: {tool_call}"
                                )
                                continue
                        else:
                            self.logger.error(
                                f"Tool call is not a dictionary: {tool_call}"
                            )
                            continue

                        if not tool_name:
                            self.logger.error(
                                "Invalid tool call format: missing tool name"
                            )
                            continue

                        try:
                            # Let the tool manager handle case-insensitive matching
                            tool = self.tool_manager.get_tool(tool_name)
                            self.logger.debug(
                                f"Invoking tool: {tool_name}, args={tool_args}"
                            )
                            # Pass the query as tool_input
                            tool_result = tool.run(tool_args.get("query", ""))

                            # Use ToolMessage instead of AIMessage with additional_kwargs
                            tool_message = ToolMessage(
                                content=str(tool_result),
                                name=tool_name,
                                tool_call_id=tool_call.get("id", ""),
                            )
                            self.message_manager.append_message(tool_message)
                            self.logger.debug(
                                f"Tool execution successful: {tool_name}, result={tool_result}"
                            )

                            # Get assistant's response to the tool result
                            messages = self.message_manager.get_message_history()
                            self.logger.debug(
                                f"Getting assistant's response to tool result"
                            )
                            final_response = self.model_manager.get_completion(messages)

                            # Add the assistant's response to the message history
                            if final_response.content:
                                self.message_manager.append_message(
                                    AIMessage(
                                        content=final_response.content,
                                    )
                                )
                            return str(final_response.content or "")

                        except ValueError as e:
                            self.logger.error(f"Tool not found or invalid: {tool_name}")
                            raise e
                        except Exception as e:
                            self.logger.error(
                                f"Tool execution failed: {tool_name} - {e}"
                            )
                            raise e

                    except Exception as e:
                        self.logger.error(f"Error processing tool call: {e}")
                        raise e

            # Return the AI message content if no tool calls
            return str(ai_message.content or "")

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            raise Exception(f"Error processing message: {str(e)}")

    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool with the agent"""
        self.logger.debug(f"Registering tool: {tool}")

        # Register with tool manager for execution
        self.tool_manager.register_tool(tool)

        # Register with model manager which handles tool binding
        self.model_manager.register_tool(tool)

        # Update system prompt with tool information
        system_prompt = self.config.system_prompt
        if self.model_manager._tools:
            system_prompt += "\n\nAvailable tools:\n"
            for t in self.model_manager._tools:
                system_prompt += f"- {t.name}: {t.description}\n"

        # Clear history and reinitialize with updated system prompt
        self.message_manager.clear_history()
        self.message_manager.append_message(SystemMessage(content=system_prompt))

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
