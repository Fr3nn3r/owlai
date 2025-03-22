from typing import List, Optional
from langchain_core.tools import BaseTool
from owlai.core.agent import OwlAgent
from owlai.core.config import AgentConfig, ModelConfig
from owlai.core.logging import LoggingManager
from owlai.core.configuration import ConfigurationManager


class OwlAgentFactory:
    """Factory for creating and configuring OwlAgent instances"""

    def __init__(self, environment: str = "development"):
        self.config_manager = ConfigurationManager(environment)
        self.logging_manager = LoggingManager()

    def create(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> OwlAgent:
        """Create a new OwlAgent instance"""
        try:
            # Load configuration if not provided
            if not config:
                config = self.config_manager.load_config()

            # Create agent
            agent = OwlAgent(config=config)

            # Register tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, BaseTool):
                        raise ValueError(f"Invalid tool type: {type(tool)}")
                    agent.register_tool(tool)

            return agent

        except Exception as e:
            self.logging_manager.error(f"Error creating agent: {str(e)}")
            raise
