import time

start_time = time.time()
print(f"Application loading please wait...")

import logging
import logging.config

import logging
import yaml

import cProfile
import pstats


from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

from owlai import Edwige, Owl
from ttsengine import hoot  # takes 2.7 seconds to start

import owlai


import importlib

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Move magic strings to constants:
DEFAULT_ROLE = "welcome"
DEFAULT_ENV = "Athena"
MAX_RETRIES = 5

def load_logger_config():
    with open("logging.yaml", "r") as logger_config:
        config = yaml.safe_load(logger_config)
        logging.config.dictConfig(config)


def main():
    try:
        load_logger_config()
        logger = logging.getLogger("main_logger")
        logger.info(f"Application started in {time.time() - start_time} seconds")
        speak = False

        edwige = Edwige()
        while True:

            focus_agent = edwige.get_focus_owl()
            history = InMemoryHistory(
                reversed(focus_agent.get_default_prompts() + ["exit"])
            )

            help_message = """quit     - Quit the program
exit     - Exit the program
print    - Print the conversation history
prints   - Print the active system prompt
reset    - Reset the conversation (new chat)
speak    - Toggle speech output
mode     - Print the active mode
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
                logger.info(f"AI: speaking is now {'on' if speak else 'off'}")
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
                importlib.reload(owlai)
                logger.info("Reloaded owlai package")
                edwige = Edwige()
                logger.info("Reloaded new Edwige instance")
                continue

            if user_message.lower() == "mode":
                focus_agent.print_info()
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
                focus_agent.run_tests()
                continue

            if user_message.lower() == "log":
                load_logger_config()
                logger.info("Logger reloaded")
                continue

            try:

                logger.info(
                    f"USER: {user_message}"
                )  # This will print in white to terminal
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
