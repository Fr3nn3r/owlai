"""
OwlAI - A framework for building sentient AI agents
"""

from owlai.core import OwlAgent
from owlai.rag import RAGAgent
from owlai.tools import ToolBox
from owlai.agent_manager import AgentManager
from owlai.db import Base, Agent, Conversation, Message, Feedback, Context

__version__ = "0.1.0"

__all__ = [
    "OwlAgent",
    "RAGAgent",
    "ToolBox",
    "AgentManager",
    "Base",
    "Agent",
    "Conversation",
    "Message",
    "Feedback",
    "Context",
]
