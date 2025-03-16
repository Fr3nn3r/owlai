from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from typing import cast
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.checkpoint.memory import MemorySaver  # an in-memory checkpointer
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from owlai.db import get_system_prompt_by_role
from owlai.core import ToolBox

from rich.console import Console

# Initialize a chat model
model = init_chat_model(model="gpt-4o-mini", 
                    model_provider="openai", 
                    temperature=0.1,
                    max_tokens=200)

toolbox = ToolBox()

tools = [toolbox.activate_mode, toolbox.identify_user_with_password]



system_message = get_system_prompt_by_role("identification")
# This could also be a SystemMessage object
# system_message = SystemMessage(content="You are a helpful assistant. Respond only in Spanish.")

history = MemorySaver()
graph = create_react_agent(
    model, tools, prompt=system_message, checkpointer=history 
)

config = {"configurable": {"thread_id": "thread-12345"}}
print(
    graph.invoke(
        {
            "messages": [
                ("user", "Hi, I'm polly! Who are you?")
            ]
        },
        config,
    )["messages"][-1].content
)
print("---")
print(
    graph.invoke(
        {"messages": [("user", "My password is red unicorn")]}, config
    )["messages"][-1].content
)
print("---")
print(
    graph.invoke(
        {"messages": [("user", "How many attemps do I have left?")]}, config
    )["messages"][-1].content
)

def sprint(*args):
    """
    A smart print function for JSON-like structures
    
    """
    console = Console()
    for arg in args:
        console.print(arg)  # Normal print with `rich`

sprint(graph.get_state(config))
