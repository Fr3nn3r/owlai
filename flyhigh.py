import asyncio
import logging
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.live import Live
from rich.table import Table
from rich import box
from owlai.nest import AgentManager
from owlai.config import OWL_AGENTS_CONFIG
from owlai.owlsys import setup_logging, sprint
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
import os
import sys
import importlib
from typing import List, Optional

logger = logging.getLogger(__name__)


class RichConsole:
    def __init__(self):
        # Create separate consoles for UI and logs
        self.log_console = Console(stderr=True)  # For logs
        self.ui_console = Console()  # For UI

        # Initialize agent manager
        self.agent_manager = AgentManager(
            agents_config=OWL_AGENTS_CONFIG, enable_cleanup=False
        )
        self.focus_agent = self.agent_manager.get_focus_owl()

        # Message history
        self.messages: List[tuple[str, str]] = []
        self.speak = False

        # Command history
        self.last_agent = None
        self.history = None
        self._init_history()

        # Clear screen for UI console
        self.ui_console.clear()

    def _init_history(self):
        """Initialize command history."""
        if self.last_agent is None or self.last_agent != self.focus_agent:
            self.last_agent = self.focus_agent
            default_queries = self.agent_manager.get_default_queries()
            self.history = InMemoryHistory(
                list(reversed(default_queries + self.agent_manager.get_agents_names()))
            )

    def create_chat_panel(self) -> Panel:
        """Create the main chat display panel."""
        table = Table(box=box.ROUNDED, expand=True, show_header=False)
        table.add_column("Messages", ratio=1)

        for sender, msg in self.messages[-15:]:
            if sender == "System":
                table.add_row(Text(f"[bold blue]{sender}:[/] {msg}"))
            elif sender == "You":
                table.add_row(Text(f"[bold green]{sender}:[/] {msg}"))
            elif sender == "Error":
                table.add_row(Text(f"[bold red]{sender}:[/] {msg}"))
            else:
                table.add_row(Text(f"[bold purple]{sender}:[/] {msg}"))

        return Panel(
            table,
            title="OwlAI Console",
            subtitle=f"Active Agent: {self.focus_agent.name}",
            border_style="bright_blue",
        )

    def display_help(self):
        help_text = """quit     - Quit the program
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

        self.add_message("System", help_text)

    def add_message(self, sender: str, message: str):
        """Add a message to the history."""
        self.messages.append((sender, message))

    def handle_command(self, command: str) -> bool:
        """Handle special commands. Return True if command was handled."""
        cmd = command.lower()

        if cmd in ["exit", "quit", "q"]:
            return False
        elif cmd in ["help", "h"]:
            self.display_help()
        elif cmd == "print":
            self.focus_agent.print_message_history()
        elif cmd == "prints":
            self.focus_agent.print_system_prompt()
        elif cmd == "reset":
            self.focus_agent.reset_message_history()
            self.add_message("System", "Conversation reset")
        elif cmd == "speak":
            self.speak = not self.speak
            self.add_message(
                "System", f"Speaking is now {'on' if self.speak else 'off'}"
            )
        elif cmd == "focus":
            self.add_message(
                "System",
                f"Focus agent: {self.focus_agent.name} Model: {self.focus_agent.llm_config.model_name}",
            )
        elif cmd == "model":
            sprint(self.focus_agent.chat_model)
        elif cmd == "reload":
            import owlai

            importlib.reload(owlai)
            self.add_message("System", "Reloaded owlai package")
        elif cmd == "test":
            self.focus_agent.invoke("test")
        elif cmd == "metadata":
            self.focus_agent.print_message_metadata()
        elif cmd == "log":
            setup_logging()
            self.add_message("System", "Reloaded logging configuration")
        elif cmd == "list":
            agents = self.agent_manager.get_agents_names()
            self.add_message("System", f"Available agents: {', '.join(agents)}")
        elif cmd in self.agent_manager.get_agents_names():
            self.agent_manager.set_focus_agent(cmd)
            self.focus_agent = self.agent_manager.get_focus_owl()
            self._init_history()
            self.add_message("System", f"Switched to agent: {cmd}")
        else:
            return True
        return True

    def run(self):
        """Main run loop."""
        # Show initial help
        self.display_help()

        # Create a new console screen
        with Live(
            self.create_chat_panel(),
            console=self.ui_console,
            refresh_per_second=4,
            screen=True,  # This creates a new alternate screen
        ) as live:
            while True:
                try:
                    # Update the display
                    live.update(self.create_chat_panel())

                    # Get user input with history
                    message = prompt(
                        "Enter your message ('q' or 'h'): ", history=self.history
                    )

                    if not message.strip():
                        continue

                    # Handle commands (without requiring /)
                    if not self.handle_command(message):
                        break

                    # Regular message
                    self.add_message("You", message)

                    try:
                        response = self.focus_agent.message_invoke(message)
                        self.add_message("Assistant", response)
                    except Exception as e:
                        self.add_message("Error", str(e))

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.add_message("Error", f"An error occurred: {str(e)}")


def main():
    # Setup logging
    setup_logging()

    # Create and run console
    console = RichConsole()
    console.run()


if __name__ == "__main__":
    main()
