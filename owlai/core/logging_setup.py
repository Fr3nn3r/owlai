import os
import yaml
import logging.config
from typing import Optional, List, Any
import logging
from rich.console import Console
from rich.pretty import Pretty
from rich.panel import Panel
from rich.table import Table


def setup_logging(config_path: Optional[str] = None) -> None:
    """Setup logging configuration from YAML file"""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "logging.yaml"
        )

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Load and validate logging configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Logging configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(f"owlai.{name}")


def debug_print(logger: logging.Logger, title: str, content: Any) -> None:
    """Print debug information using rich console if debug is enabled"""
    console = Console(force_terminal=True) if logger.isEnabledFor(10) else None
    if console and logger.isEnabledFor(10):
        console.print(Panel(Pretty(content), title=title, border_style="blue"))


def debug_table(logger: logging.Logger, title: str, data: List[dict]) -> None:
    """Print debug information in table format using rich console if debug is enabled"""
    console = Console(force_terminal=True) if logger.isEnabledFor(10) else None
    if console and logger.isEnabledFor(10):
        table = Table(title=title, show_header=True, header_style="bold magenta")
        if data:
            for key in data[0].keys():
                table.add_column(key)
            for row in data:
                table.add_row(*[str(v) for v in row.values()])
        console.print(table)
