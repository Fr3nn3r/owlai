print("Application starting please wait...")
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
import json
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display

load_dotenv()


import json
import yaml

import yaml
from rich.console import Console
from rich.text import Text
from typing import Any

console = Console()


def to_serializable(obj: Any):
    """Recursively converts objects to dictionaries for YAML serialization."""
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(item) for item in obj]
    elif hasattr(obj, "model_dump"):  # Handles Pydantic models (like LangChain)
        return obj.model_dump()
    elif hasattr(obj, "__dict__"):  # Generic Python objects
        return to_serializable(obj.__dict__)
    return obj  # Return as is for primitive types


def colorize_yaml(yaml_text: str):
    """Adds colors to YAML keys and values."""
    lines = yaml_text.split("\n")
    colored_text = Text()
    is_multiline = False  # Track whether we're inside a multi-line value

    for line in lines:
        if ": |" in line:  # Detect multi-line block (`|`)
            key = line.split(": |")[0]
            colored_text.append(key + ": |", style="bold cyan")  # Keep `|` cyan
            colored_text.append("\n")  # Newline after the block literal
            is_multiline = True  # Enable multi-line mode
        elif line.startswith("  ") and is_multiline:  # Multi-line content (indented)
            colored_text.append(
                line + "\n", style="bright_magenta"
            )  # Keep all lines magenta
        else:
            is_multiline = False  # Reset multi-line mode
            if ": " in line:  # Normal key-value pairs
                key, value = line.split(": ", 1)
                colored_text.append(key + ": ", style="bold cyan")  # Cyan keys
                colored_text.append(
                    value + "\n", style="bright_magenta"
                )  # Magenta values
            else:
                colored_text.append(line + "\n", style="dim")  # Keep YAML structure dim

    return colored_text


def sprint2222(*args):
    """
    A smart print function that detects JSON-like structures (dict, list, objects)
    and prints them in YAML format with colors. Otherwise, it behaves like a normal print.
    """
    for arg in args:
        if isinstance(arg, (dict, list)) or hasattr(
            arg, "__dict__"
        ):  # Detect structured data
            print("-------------------------------- RAW")
            console.print(arg)
            print("-------------------------------- CLEANED")
            cleaned_data = to_serializable(arg)  # Convert objects to dictionaries
            console.print(cleaned_data)
            print("-------------------------------- YAMLED")
            yaml_output = yaml.dump(
                cleaned_data, default_flow_style=False, sort_keys=False
            )
            console.print(colorize_yaml(yaml_output))  # Print with colors
        else:
            console.print(arg)  # Normal print with `rich`

def sprint(*args):
    """
    A smart print function for JSON-like structures
    
    """
    for arg in args:
        console.print(arg)  # Normal print with `rich`

class State(TypedDict):
    messages: Annotated[
        list, add_messages
    ]  # similar to using MessageState (If I understand well)


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
        sprint("Tools node initialized with tools:", self.tools_by_name)

    def __call__(self, inputs: dict):
        sprint("Tools node called with inputs:", inputs)
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            sprint("Tool call requested:", tool_call)
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        sprint("Tools node outputs:", outputs)
        return {"messages": outputs}


graph_builder = StateGraph(State)


tool = TavilyAnswer(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):

    #sprint("Chatbot called with state:", state)

    response = llm_with_tools.invoke(state["messages"])

    #
    # sprint("Response:", response)

    return {"messages": [response]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
#tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    sprint("Route tools called with state:", state)
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        print("Tools expected -> next step 'tools'")
        return "tools"
    print("No tools -> returning next step END")
    return END


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
# graph_builder.add_conditional_edges(
#    "chatbot",
#    route_tools,
# The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
# It defaults to the identity function, but if you
# want to use a node named something else apart from "tools",
# You can update the value of the dictionary to something else
# e.g., "tools": "my_tools"
#    {"tools": "tools", END: END},
# )

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    #route_tools,
)


memory = MemorySaver()

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
print("Graph builder created")
graph = graph_builder.compile(checkpointer=memory)

print("Graph compiled")

#print(graph.get_graph().draw_ascii())

with open("temp/graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

config = {"configurable": {"thread_id": "1"}}


def stream_graph_updates(user_input: str):
    print("Assistant: ")
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="messages",
    ):
        try:
            text_value = event[0].content[0]["text"]
        except (IndexError, AttributeError, TypeError, KeyError):
            text_value = None  # Default to None if structure is unexpected
            pass
        if text_value == None : pass
        elif text_value == "None": pass
        elif text_value == "": pass
        else: print(text_value, end="")

user_input = "What was the score of the last game of AC Milan soccer team?"
stream_graph_updates(user_input)


#user_input = "when is the next match planned?"
#stream_graph_updates(user_input)


for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
