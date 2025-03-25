#  ,___,
# ( o,o )
# { `" `}
#  " - "

import logging
import logging.config
import yaml
from typing import List, Dict, Any
from logging import Logger
from pydantic import ValidationError
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

from owlai.rag import RAGAgent
from owlai.core import OwlAgent
from owlai.db import OWL_AGENTS_CONFIG, RAG_AGENTS_CONFIG
from owlai.tools import ToolBox

logger: Logger = logging.getLogger(__name__)


class AgentManager:
    """OwlAI agent manager"""

    focus_agent: OwlAgent

    def __init__(self):
        self.owls: Dict[str, OwlAgent] = {}
        self.names: List[str] = []
        self.toolbox = ToolBox()

        # Initialize RAG agents
        for iagent_config in RAG_AGENTS_CONFIG:
            try:
                agent = RAGAgent(**iagent_config)
                agent.init_callable_tools(
                    self.toolbox.get_tools(agent.llm_config.tools_names)
                )
                self.owls[agent.name] = agent
                self.names.append(agent.name)
                logging.debug(f"Initialized RAG agent: {agent.name}")
            except ValidationError as e:
                logging.error(f"Validation failed for {iagent_config}: {e}")

        # Initialize Owl agents
        for iagent_config in OWL_AGENTS_CONFIG:
            try:
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


if __name__ == "__main__":
    # this prevents reloading when multi processing is used
    import time

    start_time = time.time()
    print(f"Application loading please wait...")

    def load_logger_config():
        with open("logging.yaml", "r") as logger_config:
            config = yaml.safe_load(logger_config)
            logging.config.dictConfig(config)

    load_logger_config()
    logger = logging.getLogger("main")
    logger.info(f"Application started in {time.time() - start_time} seconds")

    speak = False
    last_agent = None
    history = None

    edwige = AgentManager()
    while True:
        focus_agent = edwige.get_focus_owl()
        if last_agent is None:
            last_agent = focus_agent
            default_queries = edwige.get_default_queries()
            history = InMemoryHistory(
                list(reversed(default_queries + edwige.get_agents_names()))
            )
        elif last_agent != focus_agent:
            default_queries = edwige.get_default_queries()
            history = InMemoryHistory(
                list(reversed(default_queries + edwige.get_agents_names()))
            )
            last_agent = focus_agent

        help_message = """quit     - Quit the program
exit     - Exit the program
print    - Print the conversation history
prints   - Print the active system prompt
reset    - Reset the conversation (new chat)
speak    - Toggle speech output
focus    - Info about the active agent
model    - Info about the active agent's model
reload   - Reloads owlai package source code
test     - Runs test instructions (active mode)
metadata - Print the conversation metadata
log      - reloads the logger config
list     - list all agents
[agent]  - set the focus agent"""

        user_message = prompt(
            "Enter your message ('exit' or 'help'): ", history=history
        )

        if len(user_message) == 0:
            continue

        if user_message.lower() in ["exit", "quit"]:
            break

        if user_message.lower() in ["help"]:
            logger.info(help_message)
            continue
        if user_message.lower() in ["speak"]:
            speak = not speak
            logger.info(f"Speaking is now {'on' if speak else 'off'}")
            continue

        if user_message.lower() == "print":
            focus_agent.print_message_history()
            continue

        if user_message.lower() == "prints":
            focus_agent.print_system_prompt()
            continue

        if user_message.lower() == "reset":
            focus_agent.reset_message_history()
            continue

        if user_message.lower() == "focus":
            logger.info(f"Focus agent: {focus_agent.name}")
            continue

        if user_message.lower() == "model":
            logger.info(f"Model: {focus_agent.llm_config.model_name}")
            continue

        if user_message.lower() == "reload":
            import importlib
            import owlai

            importlib.reload(owlai)
            continue

        if user_message.lower() == "test":
            focus_agent.invoke("test")
            continue

        if user_message.lower() == "metadata":
            focus_agent.print_message_metadata()
            continue

        if user_message.lower() == "log":
            load_logger_config()
            continue

        if user_message.lower() == "list":
            logger.info(f"Available agents: {edwige.get_agents_names()}")
            continue

        if user_message in edwige.get_agents_names():
            edwige.set_focus_agent(user_message)
            continue

        print(focus_agent.message_invoke(user_message))
