"""
Tests for the Memory interface implementations.
"""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import UUID
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from owlai.db.memory import SQLAlchemyMemory
from owlai.db.dbmodels import Base, Agent, Conversation, Message, Feedback, Context


@pytest.fixture
def engine():
    """Create a SQLite in-memory database for testing."""
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def session(engine):
    """Create a new database session for testing."""
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture
def memory(session):
    """Create a SQLAlchemyMemory instance for testing."""
    return SQLAlchemyMemory(session)


class TestSQLAlchemyMemory:
    def test_create_agent(self, memory):
        """Test creating a new agent."""
        agent_id = memory.create_agent("TestBot", "1.0.0")
        assert isinstance(agent_id, UUID)

    def test_get_or_create_agent(self, memory):
        """Test get_or_create_agent creates and retrieves agents correctly."""
        # First creation
        agent_id1 = memory.get_or_create_agent("TestBot", "1.0.0")
        assert isinstance(agent_id1, UUID)

        # Should return same ID for same name/version
        agent_id2 = memory.get_or_create_agent("TestBot", "1.0.0")
        assert agent_id1 == agent_id2

        # Should create new agent for different version
        agent_id3 = memory.get_or_create_agent("TestBot", "2.0.0")
        assert agent_id1 != agent_id3

    def test_create_conversation(self, memory):
        """Test creating a new conversation."""
        # Test with title
        conv_id1 = memory.create_conversation("Test Chat")
        assert isinstance(conv_id1, UUID)

        # Test without title
        conv_id2 = memory.create_conversation()
        assert isinstance(conv_id2, UUID)
        assert conv_id1 != conv_id2

    def test_log_message(self, memory):
        """Test logging messages in a conversation."""
        agent_id = memory.create_agent("TestBot", "1.0.0")
        conv_id = memory.create_conversation("Test Chat")

        # Test logging a message
        msg_id = memory.log_message(
            agent_id=agent_id,
            conversation_id=conv_id,
            source="human",
            content="Hello!",
        )
        assert isinstance(msg_id, UUID)

        # Verify message in conversation history
        history = memory.get_conversation_history(conv_id)
        assert len(history) == 1
        assert history[0]["content"] == "Hello!"
        assert history[0]["source"] == "human"

    def test_log_feedback(self, memory):
        """Test logging feedback for messages."""
        agent_id = memory.create_agent("TestBot", "1.0.0")
        conv_id = memory.create_conversation()
        msg_id = memory.log_message(
            agent_id=agent_id,
            conversation_id=conv_id,
            source="agent",
            content="Hello human!",
        )

        # Log feedback
        memory.log_feedback(
            message_id=msg_id,
            score=5,
            comments="Great response!",
        )

        # Verify feedback in conversation history
        history = memory.get_conversation_history(conv_id)
        assert len(history) == 1
        assert len(history[0]["feedback"]) == 1
        assert history[0]["feedback"][0]["score"] == 5
        assert history[0]["feedback"][0]["comments"] == "Great response!"

    def test_get_conversation_history(self, memory):
        """Test retrieving conversation history with various filters."""
        agent_id = memory.create_agent("TestBot", "1.0.0")
        conv_id = memory.create_conversation("Test Chat")

        # Create multiple messages
        now = datetime.now(timezone.utc)
        for i in range(5):
            memory.log_message(
                agent_id=agent_id,
                conversation_id=conv_id,
                source="human" if i % 2 == 0 else "agent",
                content=f"Message {i}",
            )

        # Test basic retrieval
        history = memory.get_conversation_history(conv_id)
        assert len(history) == 5

        # Test limit
        limited = memory.get_conversation_history(conv_id, limit=3)
        assert len(limited) == 3

        # Test before timestamp
        before = memory.get_conversation_history(
            conv_id, before=datetime.now(timezone.utc) + timedelta(seconds=1)
        )
        assert len(before) == 5

    def test_get_agent_conversations(self, memory):
        """Test retrieving all conversations for an agent."""
        agent_id = memory.create_agent("TestBot", "1.0.0")

        # Create multiple conversations
        for i in range(3):
            conv_id = memory.create_conversation(f"Chat {i}")
            memory.log_message(
                agent_id=agent_id,
                conversation_id=conv_id,
                source="human",
                content=f"Hello {i}",
            )

        # Test retrieval
        conversations = memory.get_agent_conversations(agent_id)
        assert len(conversations) == 3

        # Test pagination
        paginated = memory.get_agent_conversations(agent_id, limit=2)
        assert len(paginated) == 2

    def test_context_linking(self, memory):
        """Test linking messages as context."""
        agent_id = memory.create_agent("TestBot", "1.0.0")
        conv_id = memory.create_conversation()

        # Create messages
        context_id = memory.log_message(
            agent_id=agent_id,
            conversation_id=conv_id,
            source="human",
            content="What's the weather?",
        )
        response_id = memory.log_message(
            agent_id=agent_id,
            conversation_id=conv_id,
            source="agent",
            content="It's sunny!",
        )

        # Link context
        memory.add_context(response_id, context_id)

        # Verify context
        context = memory.get_message_context(response_id)
        assert len(context) == 1
        assert context[0]["content"] == "What's the weather?"

    def test_search_conversations(self, memory):
        """Test searching through conversations."""
        agent_id = memory.create_agent("TestBot", "1.0.0")
        conv_id = memory.create_conversation()

        # Create some messages
        memory.log_message(
            agent_id=agent_id,
            conversation_id=conv_id,
            source="human",
            content="Hello, how are you?",
        )
        memory.log_message(
            agent_id=agent_id,
            conversation_id=conv_id,
            source="agent",
            content="I'm doing great, thanks!",
        )
        memory.log_message(
            agent_id=agent_id,
            conversation_id=conv_id,
            source="human",
            content="What's the weather like?",
        )

        # Test search
        results = memory.search_conversations("weather")
        assert len(results) == 1
        assert "weather" in results[0]["content"].lower()

        # Test with limit
        all_results = memory.search_conversations("doing", limit=1)
        assert len(all_results) == 1
