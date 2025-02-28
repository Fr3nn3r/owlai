import logging
import logging.config
import logging
import yaml

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory

from owlai import get_focus_agent, get_current_mode
from localtts import hoot_local

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

# Move these to a config file or constants section at the top
TEST_INSTRUCTIONS = [
    "list the current directory.",
    "remove the temp folder if any exist in the current directory.",
    "create a temp folder in the current directory if it does not exist.",
    "you must always save files in the temp folder", # Added to system prompt for anthropic
    "open an explorer in the temp folder",
    "get some information about the network and put it into a .txt file",
    "give me some information about the hardware and put it into a .txt file in the temp folder",
    "open the last .txt file",
    "open the bbc homepage",
    "display an owl in ascii art",
    "display an owl in ascii art and put it into a .txt file",
    "switch off the screen for 1 second and then back on",
    "it did not work can you try again?",
    "set the brightness of the screen to 50/100",
    "create an excel file with detailed demographic data for the 50 biggest countries by population",
    "open this file with Excel",
    "save this file as an excel file",
    "list the values in the PATH environement variable",
    "list the values of the PATH environement variable in a txt file one per line",
    "open the last txt file",
    "Report all of the USB devices installed into a file",
    "print the file you saved with USB devices in the terminal",
    "set the brightness of the screen back to 100",
    "kill the notepad process",
    "display information about my network connection",
    "minimizes all windows",
    "run the keyboard combination Ctlr + Win + -> ",
]

def hoot(text : str):
    #logger.warning(f"NOT IMPLEMENTED HOOT!!!: {text}")
    hoot_local(text)

def main():
    try:
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

            # Loop over test instructions before prompting user
            test_instructions_length = len(TEST_INSTRUCTIONS)
            test_instructions_index = 40 # switched off for now

            if test_instructions_index < test_instructions_length:
                user_message = TEST_INSTRUCTIONS[test_instructions_index]
                test_instructions_index += 1

            # Prompt user for input in CLI
            else:
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
