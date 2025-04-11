"""
Configuration settings for LangChain tracing/observability
WTF is this?
"""

import os
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.tracers import ConsoleTracer

# Disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"


# If you need a custom callback manager without LangSmith
def get_callback_manager():
    """
    Returns a callback manager with tracing disabled.
    """
    callbacks = []

    # Add console tracer if debugging is needed
    # callbacks.append(ConsoleTracer())

    return CallbackManager(handlers=callbacks)


# Use this in places where you need to explicitly disable tracing
def disable_langsmith_tracing():
    """
    Configure LangChain to disable LangSmith tracing.
    Call this function early in your application startup.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_PROJECT"] = ""  # Clear project name

    # Unset API key if environment allows it
    if "LANGCHAIN_API_KEY" in os.environ:
        del os.environ["LANGCHAIN_API_KEY"]
