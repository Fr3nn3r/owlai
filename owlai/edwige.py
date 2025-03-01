import time

start_time = time.time()

import logging
import logging.config

import logging
import yaml

import cProfile
import pstats

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory

from owlai import get_focus_agent, get_current_mode
from ttsengine import hoot #takes 2.7 seconds to start

import importlib
import owlai

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Load logger config from YAML file
with open("logging.yaml", "r") as logger_config:
    config = yaml.safe_load(logger_config)
    logging.config.dictConfig(config)
    
logger = logging.getLogger("main_logger")

def main():
    try:
        logger.info(f"Application started in {time.time() - start_time} seconds")
        speak = False
        while True:
            # get static instance from owlai
            focus_agent = get_focus_agent()
            history = InMemoryHistory(focus_agent.get_default_prompts() + ["exit", "reload"]  )

            help_message = """quit     - Quit the program
exit     - Exit the program
print    - Print the conversation history
reset    - Reset the conversation (new chat)
speak    - Toggle speech output
mode     - Print the current mode
reload   - Reloads owlai package source code
test     - Runs test instructions (current mode)
metadata - Print the conversation metadata"""


            user_message = prompt("Enter your message ('exit' or 'help'): ", history=history)

            if len(user_message) == 0: continue

            if user_message.lower() in ["exit", "quit"]: break

            if user_message.lower() in ["help"] : 
                logger.info(help_message)
                continue
            if user_message.lower() in ["speak"]:
                speak = not speak
                logger.info(f"EDWIGE: speaking is now {'on' if speak else 'off'}")  
                continue

            if user_message.lower() == "print":
                focus_agent.print_message_history()
                continue

            if user_message.lower() == "metadata":
                focus_agent.print_message_metadata()
                continue

            if user_message.lower() == "reload":
                importlib.reload(owlai)
                continue

            if user_message.lower() == "mode":
                logger.info(f"EDWIGE: current mode is {get_current_mode()}")
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

            try:

                logger.info(f"\033[97mUSER: {user_message}\033[0m")  # This will print in white to terminal
                response = focus_agent.invoke(user_message)
                logger.info(f"EDWIGE: {response}")
                if speak : hoot(response)

            except Exception as e:
                logger.critical(f"Fatal Error: {e}")
                raise

    except KeyboardInterrupt:
        logger.info("Excution interrupted. Shutting down...")

if __name__ == "__main__":
    main()

