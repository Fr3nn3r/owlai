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
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    logger = logging.getLogger(f"owlai.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
