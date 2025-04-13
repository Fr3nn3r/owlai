"""
OwlAI - A Python library for managing AI agents
"""

from owlai.core import LLMConfig, OwlAgent
from owlai.nest import AgentManager
from importlib.metadata import version

__version__ = version("owlai")

__all__ = [
    "LLMConfig",
    "OwlAgent",
    "AgentManager",
]
