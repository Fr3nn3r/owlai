from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from owlai.core.config import AgentConfig
from owlai.core.message_manager import MessageManager
from owlai.core.tool_manager import ToolManager
from owlai.core.model_manager import ModelManager
from owlai.core.logging_setup import get_logger


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
            response = self.model_manager.get_completion(messages)

            # Create AI message
            ai_message = AIMessage(content=response)
            self.message_manager.append_message(ai_message)
            self.logger.debug(f"Added AI response to history: {len(response)} chars")

            # Process any tool calls
            if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
                self.logger.info(f"Processing {len(ai_message.tool_calls)} tool calls")
                for tool_call in ai_message.tool_calls:
                    try:
                        self.logger.debug(f"Invoking tool: {tool_call['name']}")
                        tool_result = self.tool_manager.invoke_tool(
                            tool_call["name"], tool_call["args"]
                        )
                        # Add tool result to history
                        self.message_manager.append_message(
                            AIMessage(content=f"Tool result: {tool_result}")
                        )
                        self.logger.debug(
                            f"Tool execution successful: {tool_call['name']}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Tool execution failed: {tool_call['name']} - {e}"
                        )
                        # Log tool error but continue processing
                        self.message_manager.append_message(
                            AIMessage(content=f"Tool error: {str(e)}")
                        )

            self.logger.info(
                f"Message processing completed: {len(response)} chars, "
                f"{len(self.message_manager.get_message_history())} total messages"
            )
            return response

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
