import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime


class ________LoggingManager:
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

    def _format_message(
        self, message: str, extra: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format message with optional structured data"""
        if extra:
            # Convert datetime objects to ISO format
            formatted_extra = {}
            for key, value in extra.items():
                if isinstance(value, datetime):
                    formatted_extra[key] = value.isoformat()
                else:
                    formatted_extra[key] = value

            # Add structured data as JSON
            return f"{message} | {json.dumps(formatted_extra)}"
        return message

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message with optional structured data"""
        self.logger.debug(self._format_message(message, extra))

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message with optional structured data"""
        self.logger.info(self._format_message(message, extra))

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message with optional structured data"""
        self.logger.warning(self._format_message(message, extra))

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log error message with optional structured data"""
        self.logger.error(self._format_message(message, extra))

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message with optional structured data"""
        self.logger.critical(self._format_message(message, extra))

    def set_level(self, level: int) -> None:
        """Set logging level"""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def log_operation(
        self, operation: str, status: str, extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an operation with its status and optional details"""
        if extra is None:
            extra = {}
        extra["operation"] = operation
        extra["status"] = status
        extra["timestamp"] = datetime.utcnow()

        if status == "success":
            self.info(f"Operation completed: {operation}", extra)
        elif status == "error":
            self.error(f"Operation failed: {operation}", extra)
        else:
            self.warning(f"Operation {status}: {operation}", extra)
