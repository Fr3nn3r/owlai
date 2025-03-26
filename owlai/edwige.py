#  ,___,
# ( o,o )
# { `" `}
#  " - "

import logging
import logging.config
import yaml
import os
from typing import List, Dict, Any
from logging import Logger
from pydantic import ValidationError
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

from owlai.agent_manager import AgentManager
from owlai.owlsys import load_logger_config

logger: Logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # this prevents reloading when multi processing is used
    import time

    start_time = time.time()
    print(f"Application loading please wait...")

    # Load logging config using the shared function
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
