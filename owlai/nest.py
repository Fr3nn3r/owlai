import logging
from typing import List, Dict, Any, Optional
from logging import Logger
from pydantic import ValidationError
from sqlalchemy.orm import Session
import asyncio
import time
from datetime import datetime, timedelta

from owlai.core import OwlAgent
from owlai.tools import ToolBox
from owlai.memory import SQLAlchemyMemory
from owlai.owlsys import Session

logger: Logger = logging.getLogger(__name__)


class AgentManager:
    """OwlAI agent manager"""

    INACTIVE_TIMEOUT = 30 * 60  # 30 minutes in seconds
    CLEANUP_INTERVAL = 300  # 5 minutes in seconds

    focus_agent: OwlAgent
    _initialized = False

    def __init__(
        self, agents_config: Dict[str, Dict[str, Any]], enable_cleanup: bool = False
    ):
        """Initialize the AgentManager.

        Args:
            agents_config: Dictionary containing agent configurations
            enable_cleanup: Whether to enable automatic cleanup of inactive agents
        """
        self.agents_config = agents_config
        self.active_agents: Dict[str, OwlAgent] = {}
        self.inactive_agents: Dict[str, OwlAgent] = {}
        self.last_used: Dict[str, float] = {}
        self.owls: Dict[str, OwlAgent] = {}
        self.names: List[str] = []
        self.toolbox = ToolBox()
        self.db_session = Session()
        self.memory = SQLAlchemyMemory(self.db_session)
        self._lazy_init()
        self.CLEANUP_INTERVAL = 300  # 5 minutes
        self.INACTIVE_THRESHOLD = 1800  # 30 minutes

        if enable_cleanup:
            self._start_cleanup_task()
        logger.info("AgentManager initialized with active/inactive agent management")

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
                self.toolbox.get_tools(agent.llm_config.tools_names)
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

        self.focus_agent = self.owls[self.names[0]]
        self._initialized = True
        logger.info(f"AgentManager initialized with {len(self.names)} agents")

    def get_focus_owl(self) -> OwlAgent:
        logger.debug(f"Focus agent: {self.focus_agent.name}")
        return self.focus_agent

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

    def get_agent(self, session_id: str, agent_key: str) -> OwlAgent:
        """Get an agent by session ID. If the agent doesn't exist, create it."""
        current_time = time.time()

        # Check active agents first
        if session_id in self.active_agents:
            self.last_used[session_id] = current_time
            return self.active_agents[session_id]

        # Check inactive agents
        if session_id in self.inactive_agents:
            logger.debug(f"Reactivating inactive agent: {session_id}")
            agent = self.inactive_agents.pop(session_id)
            self.active_agents[session_id] = agent
            self.last_used[session_id] = current_time
            agent.reset_message_history()
            return agent

        # Create new agent if not found
        logger.debug(f"Creating new agent for session: {session_id}")
        agent = self.initialize_agent(agent_key)
        if not agent:
            raise RuntimeError(f"Failed to initialize agent {agent_key}")

        self.active_agents[session_id] = agent
        self.last_used[session_id] = current_time
        return agent

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

    def get_agents_keys(self) -> List[str]:
        return [key for key in self.agents_config.keys()]

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
