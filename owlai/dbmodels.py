import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, ForeignKey, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Agent(Base):
    __tablename__ = "agents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)

    messages = relationship("Message", back_populates="agent")


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"))
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    source = Column(String, nullable=False)  # e.g., 'human', 'agent', 'tool'
    content = Column(Text, nullable=False)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)

    agent = relationship("Agent", back_populates="messages")
    conversation = relationship("Conversation")
    feedback = relationship("Feedback", back_populates="message")
    context_links = relationship(
        "Context", foreign_keys="[Context.message_id]", back_populates="message"
    )


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"))
    user_id = Column(UUID(as_uuid=True), nullable=True)
    score = Column(Integer, nullable=False)
    comments = Column(Text, nullable=True)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)

    message = relationship("Message", back_populates="feedback")


class Context(Base):
    __tablename__ = "context"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"), nullable=False)
    context_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"), nullable=False)

    message = relationship(
        "Message", foreign_keys=[message_id], back_populates="context_links"
    )
