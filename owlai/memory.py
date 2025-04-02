"""
Memory interface for OwlAI agents to store and retrieve conversations and interactions.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID


class Memory(ABC):
    """
    Memory interface for storing and retrieving agent conversations and interactions.
    Implementations can use different storage backends (SQL, Vector DB, etc.).
    """

    @abstractmethod
    def create_agent(self, name: str, version: str) -> UUID:
        """
        Register a new agent in memory.

        Args:
            name: Name of the agent
            version: Version of the agent

        Returns:
            UUID of the created agent
        """
        pass

    @abstractmethod
    def get_or_create_agent(self, name: str, version: str) -> UUID:
        """
        Get an existing agent or create if it doesn't exist.

        Args:
            name: Name of the agent
            version: Version of the agent

        Returns:
            UUID of the existing or newly created agent
        """
        pass

    @abstractmethod
    def create_conversation(self, title: Optional[str] = None) -> UUID:
        """
        Start a new conversation.

        Args:
            title: Optional title for the conversation

        Returns:
            UUID of the created conversation
        """
        pass

    @abstractmethod
    def log_message(
        self,
        agent_id: UUID,
        conversation_id: UUID,
        source: str,  # 'human', 'agent', 'tool'
        content: str,
        metadata: Optional[Dict[Any, Any]] = None,
    ) -> UUID:
        """
        Log a message in a conversation.

        Args:
            agent_id: ID of the agent involved
            conversation_id: ID of the conversation
            source: Source of the message ('human', 'agent', 'tool')
            content: Content of the message
            metadata: Optional metadata about the message

        Returns:
            UUID of the created message
        """
        pass

    @abstractmethod
    def log_feedback(
        self,
        message_id: UUID,
        score: int,
        comments: Optional[str] = None,
        user_id: Optional[UUID] = None,
    ) -> None:
        """
        Log feedback for a message.

        Args:
            message_id: ID of the message being rated
            score: Numerical score/rating
            comments: Optional feedback comments
            user_id: Optional ID of the user giving feedback
        """
        pass

    @abstractmethod
    def get_conversation_history(
        self,
        conversation_id: UUID,
        limit: Optional[int] = None,
        before: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history.

        Args:
            conversation_id: ID of the conversation
            limit: Optional max number of messages to return
            before: Optional timestamp to get messages before

        Returns:
            List of messages with their metadata
        """
        pass

    @abstractmethod
    def get_agent_conversations(
        self, agent_id: UUID, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all conversations involving an agent.

        Args:
            agent_id: ID of the agent
            limit: Optional max number of conversations to return
            offset: Optional offset for pagination

        Returns:
            List of conversations with their metadata
        """
        pass

    @abstractmethod
    def add_context(self, message_id: UUID, context_message_id: UUID) -> None:
        """
        Link a message to another message that provided context.

        Args:
            message_id: ID of the message using the context
            context_message_id: ID of the message providing context
        """
        pass

    @abstractmethod
    def get_message_context(self, message_id: UUID) -> List[Dict[str, Any]]:
        """
        Get all context messages used for a given message.

        Args:
            message_id: ID of the message

        Returns:
            List of context messages with their metadata
        """
        pass

    @abstractmethod
    def search_conversations(
        self, query: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search through conversation history.

        Args:
            query: Search query string
            limit: Optional max number of results

        Returns:
            List of matching messages with their metadata
        """
        pass
