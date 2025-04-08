"""
Memory interface for OwlAI agents to store and retrieve conversations and interactions in database
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, TypedDict, cast
from uuid import UUID
from sqlalchemy import select, desc
from sqlalchemy.orm import Session
from owlai.db.dbmodels import Agent, Conversation, Message, Feedback, Context
from sqlalchemy import Column, String, DateTime, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID as SQLUUID
from sqlalchemy.orm import relationship
import json


class MessageDict(TypedDict):
    """Dictionary representation of a message."""

    id: UUID
    agent_id: UUID
    source: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]]
    feedback: List[dict]


class ConversationDict(TypedDict):
    id: UUID
    title: Optional[str]
    created_at: datetime
    message_count: int


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
        source: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[Dict[str, Any]] = None,
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
    ) -> List[MessageDict]:
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
    ) -> List[ConversationDict]:
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
    def get_message_context(self, message_id: UUID) -> List[MessageDict]:
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
    ) -> List[MessageDict]:
        """
        Search through conversation history.

        Args:
            query: Search query string
            limit: Optional max number of results

        Returns:
            List of matching messages with their metadata
        """
        pass

    @abstractmethod
    def get_last_message_id(self, conversation_id: UUID) -> Optional[UUID]:
        """
        Get the ID of the last message in a conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            UUID of the last message or None if no messages exist
        """
        pass

    @abstractmethod
    def get_preceding_tool_message(self, message_id: UUID) -> Optional[MessageDict]:
        """Get the tool message that precedes a given message in the same conversation.

        Args:
            message_id: ID of the message to find the preceding tool message for

        Returns:
            The preceding tool message if found, None otherwise
        """
        pass


class SQLAlchemyMemory(Memory):
    """
    SQLAlchemy implementation of the Memory interface.
    Uses the existing database models to store conversation history.
    """

    def __init__(self, session: Session):
        """
        Initialize with a SQLAlchemy session.

        Args:
            session: SQLAlchemy session for database operations
        """
        self.session = session

    def create_agent(self, name: str, version: str) -> UUID:
        agent = Agent(name=name, version=version)
        self.session.add(agent)
        self.session.commit()
        return agent.id  # type: ignore

    def get_or_create_agent(self, name: str, version: str) -> UUID:
        # Try to find existing agent
        stmt = select(Agent).where(Agent.name == name, Agent.version == version)
        agent = self.session.execute(stmt).scalar_one_or_none()

        if agent is None:
            # Create new agent if not found
            return self.create_agent(name, version)

        return agent.id  # type: ignore

    def create_conversation(self, title: Optional[str] = None) -> UUID:
        conversation = Conversation(title=title, created_at=datetime.now(timezone.utc))
        self.session.add(conversation)
        self.session.commit()
        return conversation.id  # type: ignore

    def log_message(
        self,
        agent_id: UUID,
        conversation_id: UUID,
        source: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """Log a message to memory.

        Args:
            agent_id (UUID): ID of the agent
            conversation_id (UUID): ID of the conversation
            source (str): Source of the message (e.g., 'user', 'assistant', 'system', 'tool')
            content (str): Content of the message
            metadata (Optional[Dict[str, Any]]): Additional metadata about the message
            tool_calls (Optional[Dict[str, Any]]): Additional tool calls about the message
        Returns:
            UUID: ID of the created message
        """
        message = Message(
            agent_id=agent_id,
            conversation_id=conversation_id,
            source=source,
            content=content,
            message_metadata=json.dumps(metadata) if metadata else None,
            tool_calls=json.dumps(tool_calls) if tool_calls else None,
            timestamp=datetime.now(timezone.utc),
        )
        self.session.add(message)
        self.session.commit()
        return message.id  # type: ignore

    def log_feedback(
        self,
        message_id: UUID,
        score: int,
        comments: Optional[str] = None,
        user_id: Optional[UUID] = None,
    ) -> None:
        feedback = Feedback(
            message_id=message_id,
            score=score,
            comments=comments,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc),
        )
        self.session.add(feedback)
        self.session.commit()

    def get_conversation_history(
        self,
        conversation_id: UUID,
        limit: Optional[int] = None,
        before: Optional[datetime] = None,
    ) -> List[MessageDict]:
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(desc(Message.timestamp))
        )

        if before:
            query = query.where(Message.timestamp < before)

        if limit:
            query = query.limit(limit)

        messages = self.session.execute(query).scalars().all()

        return [
            MessageDict(
                id=cast(UUID, msg.id),
                agent_id=cast(UUID, msg.agent_id),
                source=str(msg.source),
                content=str(msg.content),
                timestamp=datetime.fromtimestamp(
                    msg.timestamp.timestamp(), timezone.utc
                ),
                metadata=(
                    json.loads(str(msg.message_metadata))
                    if msg.message_metadata
                    else None
                ),
                feedback=[
                    {
                        "score": fb.score,
                        "comments": fb.comments,
                        "user_id": fb.user_id,
                        "timestamp": datetime.fromtimestamp(
                            fb.timestamp.timestamp(), timezone.utc
                        ),
                    }
                    for fb in msg.feedback
                ],
            )
            for msg in messages
        ]

    def get_agent_conversations(
        self,
        agent_id: UUID,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[ConversationDict]:
        query = (
            select(Message.conversation_id)
            .where(Message.agent_id == agent_id)
            .group_by(Message.conversation_id)
            .order_by(desc(Message.timestamp))
        )

        if limit:
            query = query.limit(limit)
        if offset:
            query = query.offset(offset)

        result = self.session.execute(query)
        conversation_ids = result.scalars().all()

        conversations: List[ConversationDict] = []
        for conv_id in conversation_ids:
            conv = self.session.get(Conversation, conv_id)
            if conv:
                conversations.append(
                    ConversationDict(
                        id=cast(UUID, conv.id),
                        title=str(conv.title) if conv.title else None,
                        created_at=datetime.fromtimestamp(
                            conv.created_at.timestamp(), timezone.utc
                        ),
                        message_count=len(conv.messages),
                    )
                )

        return conversations

    def add_context(self, message_id: UUID, context_message_id: UUID) -> None:
        context = Context(message_id=message_id, context_id=context_message_id)
        self.session.add(context)
        self.session.commit()

    def get_message_context(self, message_id: UUID) -> List[MessageDict]:
        query = (
            select(Message)
            .join(Context, Context.context_id == Message.id)
            .where(Context.message_id == message_id)
        )

        context_messages = self.session.execute(query).scalars().all()

        return [
            MessageDict(
                id=cast(UUID, msg.id),
                agent_id=cast(UUID, msg.agent_id),
                source=str(msg.source),
                content=str(msg.content),
                timestamp=datetime.fromtimestamp(
                    msg.timestamp.timestamp(), timezone.utc
                ),
                metadata=(
                    json.loads(str(msg.message_metadata))
                    if msg.message_metadata
                    else None
                ),
                feedback=[],  # Context messages don't need feedback
            )
            for msg in context_messages
        ]

    def search_conversations(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> List[MessageDict]:
        search_query = f"%{query}%"
        stmt = (
            select(Message)
            .where(Message.content.ilike(search_query))
            .order_by(desc(Message.timestamp))
        )

        if limit:
            stmt = stmt.limit(limit)

        messages = self.session.execute(stmt).scalars().all()

        return [
            MessageDict(
                id=cast(UUID, msg.id),
                agent_id=cast(UUID, msg.agent_id),
                source=str(msg.source),
                content=str(msg.content),
                timestamp=datetime.fromtimestamp(
                    msg.timestamp.timestamp(), timezone.utc
                ),
                metadata=(
                    json.loads(str(msg.message_metadata))
                    if msg.message_metadata
                    else None
                ),
                feedback=[],  # Search results don't need feedback
            )
            for msg in messages
        ]

    def get_last_message_id(self, conversation_id: UUID) -> Optional[UUID]:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(desc(Message.timestamp))
            .limit(1)
        )
        message = self.session.execute(stmt).scalar_one_or_none()
        return cast(UUID, message.id) if message else None

    def get_preceding_tool_message(self, message_id: UUID) -> Optional[MessageDict]:
        # First get the target message to find its conversation
        target_msg = self.session.execute(
            select(Message).where(Message.id == message_id)
        ).scalar_one_or_none()

        if not target_msg:
            return None

        # Now get the preceding tool message in the same conversation
        query = (
            select(Message)
            .where(Message.conversation_id == target_msg.conversation_id)
            .where(Message.source == "tool")
            .where(Message.timestamp < target_msg.timestamp)
            .order_by(desc(Message.timestamp))
            .limit(1)
        )
        message = self.session.execute(query).scalar_one_or_none()
        if message:
            return MessageDict(
                id=cast(UUID, message.id),
                agent_id=cast(UUID, message.agent_id),
                source=str(message.source),
                content=str(message.content),
                timestamp=datetime.fromtimestamp(
                    message.timestamp.timestamp(), timezone.utc
                ),
                metadata=(
                    json.loads(str(message.message_metadata))
                    if message.message_metadata
                    else None
                ),
                feedback=[],
            )
        return None
