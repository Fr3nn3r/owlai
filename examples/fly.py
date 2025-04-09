#  ,___,
# ( o,o )
# { `" `}
#  " - "
import time

start_time = time.time()
print(f"Application loading please wait...")


import logging
import logging.config

from logging import Logger
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

from owlai.nest import AgentManager
from owlai.services.system import setup_logging
from owlai.config.agents import OWL_AGENTS_CONFIG
from owlai.services.system import sprint

logger: Logger = logging.getLogger(__name__)

if __name__ == "__main__":

    logger.info(f"Application started in {time.time() - start_time} seconds")

    speak = False
    last_agent = None
    history = None

    edwige = AgentManager(agents_config=OWL_AGENTS_CONFIG, enable_cleanup=False)
    while True:

        try:

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

            user_message = prompt("Enter your message ('q' or 'h'): ", history=history)

            if len(user_message) == 0:
                continue

            if user_message.lower() in ["exit", "quit", "q"]:
                break

            if user_message.lower() in ["help", "h"]:
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
                logger.info(
                    f"Focus agent: {focus_agent.name} Model: {focus_agent.llm_config.model_name}"
                )
                continue

            if user_message.lower() == "model":

                sprint(focus_agent.chat_model)
                continue

            if user_message.lower() == "reload":
                import importlib
                import owlai

                importlib.reload(owlai)
                continue

            if user_message.lower() == "test":
                focus_agent.message_invoke("test")
                continue

            if user_message.lower() == "metadata":
                focus_agent.print_message_metadata()
                continue

            if user_message.lower() == "log":
                setup_logging()
                continue

            if user_message.lower() == "list":
                logger.info(f"Available agents: {edwige.get_agents_names()}")
                continue

            if user_message in edwige.get_agents_names():
                edwige.set_focus_agent(user_message)
                continue

            print(focus_agent.message_invoke(user_message))

        except KeyboardInterrupt as kbi:
            logger.info(f"Execution interrupted by Ctrl-C - Goodbye!")
            break
