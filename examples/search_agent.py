"""
Example of using TavilySearchResults with OwlAI agent.
This example demonstrates how to create an agent capable of performing web searches
using the Tavily API.
"""

import json
import os
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field

from owlai.core.agent import OwlAgent
from owlai.core.config import AgentConfig, ModelConfig
from owlai.core.logging_setup import get_logger, setup_logging

# Load environment variables and setup logging
load_dotenv()
setup_logging()

# Get logger for this example
logger = get_logger("examples.search_agent")


class TavilyConfig(BaseModel):
    """Configuration for Tavily search tool"""

    max_results: int = Field(
        default=3, description="Maximum number of search results to return"
    )


class SearchAgentConfig(BaseModel):
    """Complete configuration for the search agent example"""

    agent: AgentConfig
    model: ModelConfig
    tavily: TavilyConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SearchAgentConfig":
        """Create configuration from dictionary"""
        # First create the model config
        model_config = ModelConfig(**config_dict["model"])

        # Create agent config with the model config as llm_config
        agent_dict = config_dict["agent"].copy()
        agent_dict["llm_config"] = model_config

        return cls(
            agent=AgentConfig(**agent_dict),
            model=model_config,
            tavily=TavilyConfig(**config_dict["tavily"]),
        )


# Configuration
CONFIG = {
    "agent": {
        "name": "search_agent",
        "description": "An agent capable of performing web searches using Tavily",
        "system_prompt": """You are a helpful AI assistant that can search the web for information.
        When asked a question, use the Tavily search tool to find relevant information.
        Always cite your sources and provide clear, concise answers.
        If you're not sure about something, say so rather than making assumptions.""",
        "tools_names": ["tavily_search_results_json"],
    },
    "model": {
        "provider": "openai",
        "model_name": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2048,
        "context_size": 4096,
    },
    "tavily": {"max_results": 3},
}


def create_agent() -> OwlAgent:
    """Create and configure the search agent"""
    logger.info("Creating search agent with configuration", extra={"config": CONFIG})

    # Load and validate configuration
    config = SearchAgentConfig.from_dict(CONFIG)

    # Create Tavily search tool
    tavily_tool = TavilySearchResults(**config.tavily.model_dump())

    # Create and return the agent
    agent = OwlAgent(config.agent)
    agent.tool_manager.register_tool(tavily_tool)

    logger.info(f"Created search agent: {agent}")
    return agent


def main():
    """Main function to demonstrate the search agent"""
    try:
        # Create the agent
        agent = create_agent()

        # Example queries
        queries = [
            "What was the last result of the AC Milan soccer team?",
            "When is the AC Milan next playing?",
        ]

        # Process each query
        for query in queries:
            logger.info("Processing query", extra={"query": query})

            # Get response from agent
            response = agent.message_invoke(query)

            # Print response
            print(f"\nQuery: {query}")
            print(f"Response: {response}\n")

    except Exception as e:
        logger.error("Error in search agent example", extra={"error": str(e)})
        raise


if __name__ == "__main__":
    main()
