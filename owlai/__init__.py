"""
OwlAI - A framework for building sentient AI agents
"""

from .document_parser import FrenchLawParser
from .vector_store import VectorStore
from .rag_agent import RAGAgent

__version__ = "0.1.0"
__all__ = ["FrenchLawParser", "VectorStore", "RAGAgent"]
