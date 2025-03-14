#  ,___,
# ( o,o )
# { `" `}
#  " - "

import time

start_time = time.time()
print(f"Application loading please wait...")

import logging
import logging.config
import yaml

from .ttsengine import hoot
from .db import CONFIG
import importlib
from dotenv import load_dotenv

from typing import Dict, Any
from .db import CONFIG
from .core import (
    OwlAgent,
    OwlAIAgent,
)

from pydantic import ValidationError

from .db import get_default_prompts_by_role
from .tools import get_focus_role
from .tools import get_tools
import owlai
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

#load_dotenv()

logger = logging.getLogger("main")

class AgentManager:
    """OwlAI agent manager"""

    def __init__(self):

        self.owls: Dict[str, Any] = {}

        for irole in list(CONFIG.keys()):
            try :
                agent = OwlAgent(**CONFIG[irole])
                agent.init_callable_tools(get_tools(agent.tools_names))
                self.owls[irole] = agent
            except ValidationError as e:
                logger.error(f"Configuration validation failed for role {irole}: {e}")

    def get_focus_owl(self):

        logger.debug(f"Active mode: {get_focus_role()}")
        return self.owls[get_focus_role()]

    def get_default_prompts(self):

        return get_default_prompts_by_role(get_focus_role())

    def run_tests(self):

        if len(CONFIG[get_focus_role()]["test_prompts"]) > 0:
            logger.info(f"Running tests for mode '{get_focus_role()}'")
            for test in CONFIG[get_focus_role()]["test_prompts"]:
                logger.info(f"USER: {test}")
                self.owls[get_focus_role()].invoke(test)
        else:
            logger.warning(f"No test prompts defined for owl role '{get_focus_role()}'")


def load_logger_config():
    with open("logging.yaml", "r") as logger_config:
        config = yaml.safe_load(logger_config)
        logging.config.dictConfig(config)


def main():
    try:
        load_logger_config()
        #logger = logging.getLogger("main_logger")
        logger.info(f"Application started in {time.time() - start_time} seconds")
        speak = False

        edwige = AgentManager()
        while True:

            focus_agent = edwige.get_focus_owl()
            default_prompts = edwige.get_default_prompts()
            history = InMemoryHistory(reversed(default_prompts + ["exit"]))

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
log      - reloads the logger config"""

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

            if user_message.lower() == "metadata":
                focus_agent.print_message_metadata()
                continue

            if user_message.lower() == "reload":
                importlib.reload(owlai.db)
                importlib.reload(owlai.tools)
                importlib.reload(owlai.core)
                #importlib.reload(owlai.spotify)
                #importlib.reload(owlai.ttsengine)
                edwige = AgentManager()
                logger.info("Reloaded owlai package")
                continue

            if user_message.lower() == "focus":
                focus_agent.print_info()
                continue

            if user_message.lower() == "model":
                focus_agent.print_model_info()
                continue

            if user_message.lower() == "reset":
                focus_agent.reset_message_history()
                logger.info("Conversation reset")
                continue

            if user_message.lower() == "reset":
                focus_agent.reset_message_history()
                logger.info("Conversation reset")
                continue

            if user_message.lower() == "test":
                edwige.run_tests()
                continue

            if user_message.lower() == "log":
                load_logger_config()
                logger.info("Logger reloaded")
                continue

            try:

                logger.info(f"USER: {user_message}")
                response = focus_agent.invoke(user_message)
                logger.info(f"AI: {response}")
                if speak:
                    hoot(response)

            except Exception as e:
                logger.critical(f"Fatal Error: {e}")
                raise

    except KeyboardInterrupt:
        logger.info("Excution interrupted. Shutting down...")


if __name__ == "__main__":
    main()
