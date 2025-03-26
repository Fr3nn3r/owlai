import logging
from typing import List, Dict, Any
from logging import Logger
from pydantic import ValidationError

from owlai.rag import RAGAgent
from owlai.core import OwlAgent
from owlai.db import OWL_AGENTS_CONFIG, RAG_AGENTS_CONFIG
from owlai.tools import ToolBox

logger: Logger = logging.getLogger(__name__)


class AgentManager:
    """OwlAI agent manager"""

    focus_agent: OwlAgent
    _initialized = False

    def __init__(self):
        self.owls: Dict[str, OwlAgent] = {}
        self.names: List[str] = []
        self.toolbox = ToolBox()
        self._lazy_init()

    def _lazy_init(self):
        """Lazy initialization of agents to prevent module reloading"""
        if self._initialized:
            return

        # Initialize RAG agents
        for iagent_config in RAG_AGENTS_CONFIG:
            try:
                # Import RAG components only when needed
                from owlai.rag import create_rag_agent

                agent = create_rag_agent(**iagent_config)
                agent.init_callable_tools(
                    self.toolbox.get_tools(iagent_config["llm_config"]["tools_names"])
                )
                self.owls[agent.name] = agent
                self.names.append(agent.name)
                logging.debug(f"Initialized RAG agent: {agent.name}")
            except ValidationError as e:
                logging.error(f"Validation failed for {iagent_config}: {e}")

        # Initialize Owl agents
        for iagent_config in OWL_AGENTS_CONFIG:
            try:
                # Import Owl components only when needed
                from owlai.core import OwlAgent

                agent = OwlAgent(**iagent_config)
                agent.init_callable_tools(
                    self.toolbox.get_tools(agent.llm_config.tools_names)
                )
                self.owls[agent.name] = agent
                self.names.append(agent.name)
                logging.debug(f"Initialized Owl agent: {agent.name}")
            except ValidationError as e:
                logging.error(f"Validation failed for {iagent_config}: {e}")

        if not self.names:
            raise RuntimeError("No agents were successfully initialized")

        self.focus_agent = self.owls[self.names[0]]
        self._initialized = True
        logging.info(f"AgentManager initialized with {len(self.names)} agents")

    def get_focus_owl(self) -> OwlAgent:
        logger.debug(f"Focus agent: {self.focus_agent.name}")
        return self.focus_agent

    def get_default_queries(self) -> List[str]:
        if self.focus_agent.default_queries is None:
            logger.warning(f"No default queries defined for {self.focus_agent.name}")
            return []

        return self.focus_agent.default_queries

    def run_default_queries(self):
        default_queries = self.get_default_queries()

        if len(default_queries) > 0:
            logger.info(f"Running default queries for {self.focus_agent.name}")
            for test in default_queries:
                logger.info(f"USER: {test}")
                self.focus_agent.invoke(test)
        else:
            logger.warning(f"No default queries defined for {self.focus_agent.name}")

    def get_agents_names(self) -> List[str]:
        return self.names

    def get_agents_info(self) -> List[str]:
        return [f"{agent.name}: {agent.description}" for agent in self.owls.values()]

    def get_agents_default_queries(self) -> List[str]:
        return [
            f"{agent.name}: {', '.join(agent.default_queries) if agent.default_queries else 'No default queries'}"
            for agent in self.owls.values()
        ]

    def invoke_agent(self, agent_name: str, message: str):
        if agent_name not in self.names:
            logger.warning(f"Agent {agent_name} not found")
            return
        self.focus_agent = self.owls[agent_name]
        return self.focus_agent.message_invoke(message)

    def set_focus_agent(self, agent_name: str):
        if agent_name not in self.names:
            logger.warning(f"Agent {agent_name} not found")
            return
        self.focus_agent = self.owls[agent_name]
