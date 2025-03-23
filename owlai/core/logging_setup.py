import os
import yaml
import logging.config
from typing import Optional
import logging


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
