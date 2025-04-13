from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from owlai.dbmodels import Base, Agent, Conversation, Message, Feedback, Context
import uuid
from datetime import datetime

# Replace with your actual DB URI
# DATABASE_URL = "postgresql://postgres:dev@localhost:5432/owlai_dev"
DATABASE_URL = "postgresql+psycopg2://owluser:owlsrock@localhost:5432/owlai_db"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Drop and recreate all tables (dev-only)
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

# Create agent
agent = Agent(name="Owly", version="v1.0.0")
session.add(agent)
session.commit()

# Create conversation
conversation = Conversation(title="Test Chat with Owly")
session.add(conversation)
session.commit()

# Create messages
msg1 = Message(
    agent_id=agent.id,
    conversation_id=conversation.id,
    source="human",
    content="What's the weather like on the Moon?",
)

msg2 = Message(
    agent_id=agent.id,
    conversation_id=conversation.id,
    source="agent",
    content="On the Moon, it's about -173°C at night and 127°C during the day!",
)

session.add_all([msg1, msg2])
session.commit()

# Link context: msg2 used msg1
context_link = Context(message_id=msg2.id, context_id=msg1.id)
session.add(context_link)

# Add feedback to agent response
feedback = Feedback(
    message_id=msg2.id, user_id=None, score=5, comments="Love the lunar vibes, Owly!"
)
session.add(feedback)

session.commit()
print("✅ Dummy data inserted!")
