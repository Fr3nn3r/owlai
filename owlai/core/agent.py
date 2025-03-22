from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from owlai.core.config import AgentConfig
from owlai.core.message_manager import MessageManager
from owlai.core.tool_manager import ToolManager
from owlai.core.model_manager import ModelManager


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

        # Initialize system message
        self.message_manager.append_message(
            SystemMessage(content=self.config.system_prompt)
        )

    def message_invoke(self, message: str) -> str:
        """Process a user message and return the response"""
        if not message or not message.strip():
            raise ValueError("Message cannot be empty")

        # Add user message to history
        user_message = HumanMessage(content=message)
        self.message_manager.append_message(user_message)

        try:
            # Get model response
            messages = self.message_manager.get_message_history()
            response = self.model_manager.get_completion(messages)

            # Create AI message
            ai_message = AIMessage(content=response)
            self.message_manager.append_message(ai_message)

            # Process any tool calls
            if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
                for tool_call in ai_message.tool_calls:
                    try:
                        tool_result = self.tool_manager.invoke_tool(
                            tool_call["name"], tool_call["args"]
                        )
                        # Add tool result to history
                        self.message_manager.append_message(
                            AIMessage(content=f"Tool result: {tool_result}")
                        )
                    except Exception as e:
                        # Log tool error but continue processing
                        self.message_manager.append_message(
                            AIMessage(content=f"Tool error: {str(e)}")
                        )

            return response

        except Exception as e:
            raise Exception(f"Error processing message: {str(e)}")

    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool with the agent"""
        self.tool_manager.register_tool(tool)

    def clear_history(self) -> None:
        """Clear message history and reinitialize system message"""
        self.message_manager.clear_history()
        self.message_manager.append_message(
            SystemMessage(content=self.config.system_prompt)
        )
