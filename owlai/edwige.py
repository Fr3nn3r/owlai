#  ,___,
# ( o,o )
# { `" `}
#  " - "

import time

start_time = time.time()
print(f"Application loading please wait...")

import logging
import logging.config

import logging
import yaml

from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

from .ttsengine import hoot
from .db import CONFIG

import importlib

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

import logging
from typing import Dict, Any
from .db import CONFIG
from .core import (
    Owl,
    OwlAIAgent,
    list_roles,
    load_config,
)
from .tools import (
    toolbox,
    toolbox_hook,
    toolbox_hook_rag_engine,
    LocalPythonInterpreter,
    LocalRAGTool,
)

from .db import get_default_prompts_by_role
from .tools import focus_role
from .db import CONFIG
from .tools import focus_role
from .tools import focus_role
from .tools import get_tools
import owlai

logger = logging.getLogger("main_logger")


class AgentManager:
    """OwlAI agent manager"""

    def __init__(self):
        self.logger = logging.getLogger("main_logger")
        self.owls: Dict[str, Any] = {}
        # self._initialize_owls()

        for irole in list_roles(CONFIG):
            config = load_config(irole, CONFIG)
            self.owls[irole] = Owl(config,get_tools(config.tools_names))

    def _initialize_owls(self):
        """Initialize all owl agents with validated configurations."""
        try:
            # System owl
            system_config = load_owl_config("system")
            self.owls["system"] = Owl(
                role=system_config.role,
                implementation=system_config.implementation,
                model_name=system_config.model_name,
                temperature=system_config.temperature,
                max_tokens=system_config.max_tokens,
                max_context_tokens=system_config.max_context_tokens,
                tools=[toolbox.activate_mode, toolbox.run_task, toolbox.play_song],
                system_prompt=system_config.system_prompt,
            )

            # Welcome owl
            welcome_config = load_owl_config("welcome")
            self.owls["welcome"] = Owl(
                role=welcome_config.role,
                implementation=welcome_config.implementation,
                model_name=welcome_config.model_name,
                temperature=welcome_config.temperature,
                max_tokens=welcome_config.max_tokens,
                max_context_tokens=welcome_config.max_context_tokens,
                tools=[toolbox.activate_mode],
                system_prompt=welcome_config.system_prompt,
            )

            # Identification owl
            identification_config = load_owl_config("identification")
            self.owls["identification"] = OwlAIAgent(
                implementation=identification_config.implementation,
                model_name=identification_config.model_name,
                temperature=identification_config.temperature,
                max_tokens=identification_config.max_tokens,
                max_context_tokens=identification_config.max_context_tokens,
                tools=[toolbox.activate_mode, toolbox.identify_user_with_password],
                system_prompt=identification_config.system_prompt,
            )

            # QnA owl
            qna_config = load_owl_config("qna")
            self.owls["qna"] = Owl(
                role=qna_config.role,
                implementation=qna_config.implementation,
                model_name=qna_config.model_name,
                temperature=qna_config.temperature,
                max_tokens=qna_config.max_tokens,
                max_context_tokens=qna_config.max_context_tokens,
                tools=[toolbox.get_answer_from_knowledge_base, toolbox.activate_mode],
                system_prompt=qna_config.system_prompt,
            )

            # Command manager owl
            cmd_config = load_owl_config("command_manager")
            self.owls["command_manager"] = LocalPythonInterpreter(
                role=cmd_config.role,
                implementation=cmd_config.implementation,
                model_name=cmd_config.model_name,
                temperature=cmd_config.temperature,
                max_tokens=cmd_config.max_tokens,
                max_context_tokens=cmd_config.max_context_tokens,
                tools=[],
                system_prompt=cmd_config.system_prompt,
            )

            # RAG tool owl
            rag_config = load_owl_config("rag_tool")
            self.owls["rag_tool"] = LocalRAGTool(
                role=rag_config.role,
                implementation=rag_config.implementation,
                model_name=rag_config.model_name,
                temperature=rag_config.temperature,
                max_tokens=rag_config.max_tokens,
                max_context_tokens=rag_config.max_context_tokens,
                tools=[],
                system_prompt=rag_config.system_prompt,
            )

            # Set the toolbox hooks
            global toolbox_hook
            toolbox_hook = self.owls["command_manager"]._run_system_command
            global toolbox_hook_rag_engine
            toolbox_hook_rag_engine = self.owls["rag_tool"].rag_question

        except Exception as e:
            self.logger.error(f"Failed to initialize owls: {e}")
            raise

    def get_focus_owl(self):

        logger.debug(f"Active mode: {focus_role}")
        return self.owls[focus_role]

    def get_default_prompts(self):

        return get_default_prompts_by_role(focus_role)

    def run_tests(self):

        if len(CONFIG[focus_role]["test_prompts"]) > 0:
            logger.info(f"Running tests for mode '{focus_role}'")
            for test in CONFIG[focus_role]["test_prompts"]:
                logger.info(f"USER: {test}")
                self.owls[focus_role].invoke(test)
        else:
            logger.warning(f"No test prompts defined for owl role '{focus_role}'")


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
                importlib.reload(owlai.core)
                importlib.reload(owlai.db)
                importlib.reload(owlai.spotify)
                importlib.reload(owlai.ttsengine)
                logger.info("Reloaded owlai package")
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
