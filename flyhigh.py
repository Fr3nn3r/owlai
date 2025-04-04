import asyncio
import logging
from textual.app import App
from textual.widgets import Static, Input
from textual.containers import Container, ScrollableContainer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from owlai.nest import AgentManager
from owlai.config import OWL_AGENTS_CONFIG
from owlai.owlsys import setup_logging, sprint
from prompt_toolkit.history import InMemoryHistory
import os
import sys
import json
import subprocess
from pathlib import Path
import importlib
from typing import List, Optional

logger = logging.getLogger(__name__)


def create_terminal_profile():
    """Create a Windows Terminal profile for OwlAI."""
    settings_path = (
        Path.home()
        / "AppData"
        / "Local"
        / "Packages"
        / "Microsoft.WindowsTerminal_8wekyb3d8bbwe"
        / "LocalState"
        / "settings.json"
    )

    if not settings_path.exists():
        logger.warning(
            "Windows Terminal settings not found. Please install Windows Terminal."
        )
        return None

    try:
        with open(settings_path, "r") as f:
            settings = json.load(f)

        # Create our custom profile
        owl_profile = {
            "name": "OwlAI Console",
            "commandline": f"python {os.path.abspath(__file__)} --child",
            "startingDirectory": os.getcwd(),
            "icon": "🦉",
            "background": "#0f111a",
            "backgroundImage": None,  # You can add a background image path here
            "backgroundImageOpacity": 0.3,
            "useAcrylic": True,
            "acrylicOpacity": 0.8,
            "fontFace": "CaskaydiaCove Nerd Font",  # Install this font or use another
            "fontSize": 12,
            "colorScheme": "Tokyo Night",
            "cursorShape": "filledBox",
            "suppressApplicationTitle": True,
            "padding": "8",
            "snapOnInput": True,
            "scrollbarState": "hidden",
        }

        # Add Tokyo Night color scheme if not exists
        tokyo_night = {
            "name": "Tokyo Night",
            "background": "#1a1b26",
            "foreground": "#c0caf5",
            "black": "#15161e",
            "red": "#f7768e",
            "green": "#9ece6a",
            "yellow": "#e0af68",
            "blue": "#7aa2f7",
            "purple": "#bb9af7",
            "cyan": "#7dcfff",
            "white": "#a9b1d6",
            "brightBlack": "#414868",
            "brightRed": "#f7768e",
            "brightGreen": "#9ece6a",
            "brightYellow": "#e0af68",
            "brightBlue": "#7aa2f7",
            "brightPurple": "#bb9af7",
            "brightCyan": "#7dcfff",
            "brightWhite": "#c0caf5",
        }

        # Add our profile if it doesn't exist
        profiles = settings.get("profiles", {}).get("list", [])
        if not any(p.get("name") == "OwlAI Console" for p in profiles):
            profiles.append(owl_profile)
            settings["profiles"]["list"] = profiles

        # Add color scheme if it doesn't exist
        schemes = settings.get("schemes", [])
        if not any(s.get("name") == "Tokyo Night" for s in schemes):
            schemes.append(tokyo_night)
            settings["schemes"] = schemes

        # Save settings
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=4)

        return "OwlAI Console"

    except Exception as e:
        logger.error(f"Failed to create Windows Terminal profile: {e}")
        return None


class ChatView(Static):
    def __init__(self):
        super().__init__("")
        self.messages: List[tuple[str, str]] = []

    def add_message(self, sender: str, message: str):
        self.messages.append((sender, message))
        self.refresh_messages()

    def refresh_messages(self):
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

        # Get the parent OwlApp instance
        owl_app = self.app
        if isinstance(owl_app, OwlApp):
            agent_name = owl_app.focus_agent.name
        else:
            agent_name = "Unknown"

        self.update(
            Panel(
                table,
                title="OwlAI Console",
                subtitle=f"Active Agent: {agent_name}",
                border_style="bright_blue",
            )
        )


class OwlApp(App):
    CSS = """
    Screen {
        background: #0f111a;
    }

    ChatView {
        height: auto;
        border: solid #304878;
        padding: 1;
    }
    
    ScrollableContainer {
        height: 1fr;
        border: solid #304878;
        background: #1a1b26;
    }
    
    Input {
        dock: bottom;
        margin: 1;
        padding: 1;
        background: #24283b;
        color: #c0caf5;
        border: solid #304878;
    }

    Input:focus {
        border: solid #7aa2f7;
    }
    """

    def __init__(self):
        super().__init__()
        self.agent_manager = AgentManager(
            agents_config=OWL_AGENTS_CONFIG, enable_cleanup=False
        )
        self.focus_agent = self.agent_manager.get_focus_owl()
        self.speak = False
        self.history = None
        self.last_agent = None
        self._init_history()

    def _init_history(self):
        """Initialize command history."""
        if self.last_agent is None or self.last_agent != self.focus_agent:
            self.last_agent = self.focus_agent
            default_queries = self.agent_manager.get_default_queries()
            self.history = InMemoryHistory(
                list(reversed(default_queries + self.agent_manager.get_agents_names()))
            )

    def compose(self):
        self.chat_view = ChatView()
        yield Container(
            ScrollableContainer(
                self.chat_view,
            ),
            Input(placeholder="Enter your message ('q' or 'h' for help)"),
        )
        self.display_help()

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

        self.chat_view.add_message("System", help_text)

    def handle_command(self, command: str) -> bool:
        """Handle special commands. Return True if command was handled."""
        cmd = command.lower()

        if cmd in ["exit", "quit", "q"]:
            self.exit()
            return False
        elif cmd in ["help", "h"]:
            self.display_help()
        elif cmd == "print":
            self.focus_agent.print_message_history()
        elif cmd == "prints":
            self.focus_agent.print_system_prompt()
        elif cmd == "reset":
            self.focus_agent.reset_message_history()
            self.chat_view.add_message("System", "Conversation reset")
        elif cmd == "speak":
            self.speak = not self.speak
            self.chat_view.add_message(
                "System", f"Speaking is now {'on' if self.speak else 'off'}"
            )
        elif cmd == "focus":
            self.chat_view.add_message(
                "System",
                f"Focus agent: {self.focus_agent.name} Model: {self.focus_agent.llm_config.model_name}",
            )
        elif cmd == "model":
            sprint(self.focus_agent.chat_model)
        elif cmd == "reload":
            import owlai

            importlib.reload(owlai)
            self.chat_view.add_message("System", "Reloaded owlai package")
        elif cmd == "test":
            self.focus_agent.invoke("test")
        elif cmd == "metadata":
            self.focus_agent.print_message_metadata()
        elif cmd == "log":
            setup_logging()
            self.chat_view.add_message("System", "Reloaded logging configuration")
        elif cmd == "list":
            agents = self.agent_manager.get_agents_names()
            self.chat_view.add_message(
                "System", f"Available agents: {', '.join(agents)}"
            )
        elif cmd in self.agent_manager.get_agents_names():
            self.agent_manager.set_focus_agent(cmd)
            self.focus_agent = self.agent_manager.get_focus_owl()
            self._init_history()
            self.chat_view.add_message("System", f"Switched to agent: {cmd}")
            self.chat_view.refresh_messages()  # Update title with new agent
        else:
            return True
        return True

    async def on_input_submitted(self, message: Input.Submitted):
        """Handle submitted messages."""
        if not message.value.strip():
            return

        # Handle commands
        if not self.handle_command(message.value):
            return

        # Regular message
        self.chat_view.add_message("You", message.value)

        try:
            response = self.focus_agent.message_invoke(message.value)
            self.chat_view.add_message("Assistant", response)
        except Exception as e:
            self.chat_view.add_message("Error", str(e))

        # Clear input
        message.input.value = ""


def main():
    # Setup logging
    setup_logging()

    # If this is the parent process
    if "--child" not in sys.argv:
        # Create Windows Terminal profile
        profile_name = create_terminal_profile()

        if profile_name and os.name == "nt":
            # Launch Windows Terminal with our profile
            try:
                subprocess.Popen(
                    [
                        "wt.exe",
                        "new-tab",
                        "--profile",
                        profile_name,
                        "--title",
                        "OwlAI Console",
                    ]
                )
                sys.exit(0)
            except Exception as e:
                logger.error(f"Failed to launch Windows Terminal: {e}")
                # Fall through to regular launch if Windows Terminal fails

    # Create and run app (either as child process or if Windows Terminal launch failed)
    app = OwlApp()
    app.run()


if __name__ == "__main__":
    main()
