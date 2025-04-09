"""
OwlAI agent manager module
"""

import logging
from typing import List, Dict, Any, Optional
from logging import Logger
from pydantic import ValidationError
from sqlalchemy.orm import Session
import asyncio
import time

from owlai.core import OwlAgent
from owlai.services.tools.box import TOOLBOX
from owlai.db.memory import SQLAlchemyMemory
from owlai.services.system import Session

logger: Logger = logging.getLogger(__name__)


class AgentManager:
    """OwlAI agent manager"""

    INACTIVE_TIMEOUT = 30 * 60  # 30 minutes in seconds
    CLEANUP_INTERVAL = 300  # 5 minutes in seconds

    def __init__(
        self, agents_config: Dict[str, Dict[str, Any]], enable_cleanup: bool = True
    ):
        """Initialize AgentManager.

        Args:
            agents_config: Dictionary of agent configurations
            enable_cleanup: Whether to enable cleanup of inactive agents
        """
        self.agents_config = agents_config
        self.active_agents: Dict[str, OwlAgent] = {}
        self.inactive_agents: Dict[str, OwlAgent] = {}
        self.last_used: Dict[str, float] = {}
        self.owls: Dict[str, Optional[OwlAgent]] = {}
        self.names: List[str] = []
        self.db_session = Session()
        self.memory = SQLAlchemyMemory(self.db_session)
        self._initialized = False
        self._focus_agent: Optional[OwlAgent] = None
        self.focus_agent_name = ""
        self.enable_cleanup = enable_cleanup

        # Register agent names but don't initialize them yet
        for agent_key in agents_config.keys():
            agent_config = agents_config[agent_key]
            self.owls[agent_config["name"]] = None
            self.names.append(agent_config["name"])

        if not self.names:
            raise RuntimeError("No agent configurations found")

        # Set focus agent name but don't initialize it yet
        self.focus_agent_name = self.names[0]
        logger.info(f"AgentManager initialized with active/inactive agent management")

    def initialize_agent(self, agent_key: str) -> Optional[OwlAgent]:
        """Initialize a new agent with the given configuration key.

        Args:
            agent_key: The configuration key for the agent

        Returns:
            Initialized OwlAgent or None if initialization fails
        """
        try:
            agent: OwlAgent = OwlAgent(**self.agents_config[agent_key])
            agent.init_callable_tools(
                [TOOLBOX[key] for key in agent.llm_config.tools_names if key in TOOLBOX]
            )
            agent.init_memory(self.memory)
            logger.info(f"Initialized Owl agent: {agent.name}")
            return agent
        except ValidationError as e:
            logger.error(f"Failed to initialize Owl agent {agent_key}: {e}")
            return None

    def _lazy_init(self):
        """Lazy initialization of agents to prevent module reloading"""
        if self._initialized:
            return

        # Initialize Owl agents
        for agent_key in self.agents_config.keys():
            agent = self.initialize_agent(agent_key)
            if agent:
                self.owls[agent.name] = agent
                self.names.append(agent.name)

        if not self.names:
            raise RuntimeError("No agents were successfully initialized")

        self._focus_agent = self.owls[self.names[0]]
        self._initialized = True
        logger.info(f"AgentManager initialized with {len(self.names)} agents")

    def get_focus_owl(self) -> Optional[OwlAgent]:
        """Get the current focus agent, initializing it if necessary."""
        self._focus_agent = self.get_agent(self.focus_agent_name)

        logger.debug(f"Focus agent: {self.focus_agent_name}")
        return self._focus_agent

    def _start_cleanup_task(self):
        """Start the background task for cleaning up inactive agents."""

        async def cleanup_loop():
            while True:
                try:
                    await self._cleanup_inactive_agents()
                    await asyncio.sleep(self.CLEANUP_INTERVAL)
                except Exception as e:
                    logger.error(f"Error in cleanup loop: {e}")
                    await asyncio.sleep(self.CLEANUP_INTERVAL)

        asyncio.create_task(cleanup_loop())
        logger.debug("Started background cleanup task")

    async def _cleanup_inactive_agents(self):
        """Clean up inactive agents that haven't been used for a while."""
        current_time = time.time()
        to_remove = []

        for session_id, agent in self.inactive_agents.items():
            if current_time - self.last_used.get(session_id, 0) > self.INACTIVE_TIMEOUT:
                to_remove.append(session_id)
                logger.debug(f"Marking inactive agent for cleanup: {session_id}")

        for session_id in to_remove:
            del self.inactive_agents[session_id]
            del self.last_used[session_id]
            logger.info(f"Cleaned up inactive agent: {session_id}")

    def get_agent(self, agent_name: str) -> Optional[OwlAgent]:
        """Get an agent by name, initializing it if necessary.

        Args:
            agent_name: Name of the agent to get

        Returns:
            The requested agent or None if not found/initialization fails
        """
        if agent_name not in self.owls:
            logger.error(f"No agent configuration found for {agent_name}")
            return None

        # Initialize agent if not already initialized
        if self.owls[agent_name] is None:
            for key, config in self.agents_config.items():
                if config["name"] == agent_name:
                    self.owls[agent_name] = self.initialize_agent(key)
                    break

        return self.owls[agent_name]

    def mark_inactive(self, session_id: str):
        """Mark an agent as inactive and move it to the inactive pool."""
        if session_id in self.active_agents:
            logger.debug(f"Moving agent to inactive pool: {session_id}")
            agent = self.active_agents.pop(session_id)
            self.inactive_agents[session_id] = agent
            self.last_used[session_id] = time.time()

    def get_active_count(self) -> int:
        """Get the number of currently active agents."""
        return len(self.active_agents)

    def get_inactive_count(self) -> int:
        """Get the number of currently inactive agents."""
        return len(self.inactive_agents)

    def get_agent_status(self, session_id: str) -> str:
        """Get the status of an agent (active/inactive/not found)."""
        if session_id in self.active_agents:
            return "active"
        if session_id in self.inactive_agents:
            return "inactive"
        return "not found"

    def get_default_queries(self) -> List[str]:
        if self._focus_agent.default_queries is None:
            logger.warning(f"No default queries defined for {self._focus_agent.name}")
            return []

        return self._focus_agent.default_queries

    def run_default_queries(self):
        default_queries = self.get_default_queries()

        if len(default_queries) > 0:
            logger.info(f"Running default queries for {self._focus_agent.name}")
            for test in default_queries:
                logger.info(f"USER: {test}")
                self._focus_agent.message_invoke(test)
        else:
            logger.warning(f"No default queries defined for {self._focus_agent.name}")

    def get_agents_names(self) -> List[str]:
        return self.names

    def get_agents_keys(self) -> List[str]:
        return [key for key in self.agents_config.keys()]

    def get_agents_info(self) -> List[str]:
        return [
            f"{agent.name}: {agent.description}"
            for agent in self.owls.values()
            if agent
        ]

    def get_agents_default_queries(self) -> List[str]:
        return [
            f"{agent.name}: {', '.join(agent.default_queries) if agent.default_queries else 'No default queries'}"
            for agent in self.owls.values()
            if agent
        ]

    def invoke_agent(self, agent_name: str, message: str):
        if agent_name not in self.names:
            logger.warning(f"Agent {agent_name} not found")
            return
        self._focus_agent = self.owls[agent_name]
        return self._focus_agent.message_invoke(message)

    def set_focus_agent(self, agent_name: str):
        if agent_name not in self.names:
            logger.warning(f"Agent {agent_name} not found")
            return
        self.focus_agent_name = agent_name

    def cleanup_inactive_agents(self):
        """Clean up inactive agents to free memory if enabled."""
        if not self.enable_cleanup:
            return

        focus_name = self.focus_agent_name
        for name in self.names:
            if name != focus_name and name in self.owls:
                self.owls[name] = None
                logger.debug(f"Cleaned up inactive agent: {name}")
