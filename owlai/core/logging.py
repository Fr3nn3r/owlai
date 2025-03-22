import logging
from typing import Optional


class LoggingManager:
    """Manages logging configuration and operations"""

    def __init__(self, level: int = logging.INFO):
        self.logger = logging.getLogger("owlai")
        self.logger.setLevel(level)

        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message"""
        self.logger.critical(message)

    def set_level(self, level: int) -> None:
        """Set logging level"""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
