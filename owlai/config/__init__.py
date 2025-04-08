"""
Configuration package for OwlAI.

This package contains various configuration components:
- prompts: Prompt templates for LLM interactions
- agents: Agent configurations
- users: User database and related settings
- tools: Tool configurations
"""

from owlai.owlsys import device, env, is_prod, is_dev, is_test

# Import and expose configuration components
from owlai.config.prompts import PROMPT_CONFIG
from owlai.config.agents import OWL_AGENTS_CONFIG, OWL_AGENTS_OPTIONAL_RAG_TOOLS
from owlai.config.users import USER_DATABASE, get_user_by_password
from owlai.config.tools import TOOLS_CONFIG, FRENCH_LAW_QUESTIONS, FR_LAW_PARSER_CONFIG

# Multi-process configuration
enable_multi_process = device == "cuda"

__all__ = [
    "PROMPT_CONFIG",
    "OWL_AGENTS_CONFIG",
    "OWL_AGENTS_OPTIONAL_RAG_TOOLS",
    "USER_DATABASE",
    "get_user_by_password",
    "TOOLS_CONFIG",
    "FRENCH_LAW_QUESTIONS",
    "FR_LAW_PARSER_CONFIG",
    "enable_multi_process",
    "device",
    "env",
    "is_prod",
    "is_dev",
    "is_test",
]
