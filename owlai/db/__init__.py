"""
Database module for OwlAI.

This module provides database functionality including:
- SQLAlchemy models for storing conversations and agent interactions
- Memory interface for managing conversation history
- Vector store management for efficient semantic search
"""

from owlai.db.dbmodels import (
    Base,
    Agent,
    Conversation,
    Message,
    Feedback,
    Context,
    VectorStore,
)
from owlai.db.memory import (
    Memory,
    SQLAlchemyMemory,
    MessageDict,
    ConversationDict,
)
from owlai.db.vector_store_manager import (
    save_vector_store,
    load_vector_store,
    import_vector_stores,
)

__all__ = [
    # Database Models
    "Base",
    "Agent",
    "Conversation",
    "Message",
    "Feedback",
    "Context",
    "VectorStore",
    # Memory Interface
    "Memory",
    "SQLAlchemyMemory",
    "MessageDict",
    "ConversationDict",
    # Vector Store Management
    "save_vector_store",
    "load_vector_store",
    "import_vector_stores",
]
