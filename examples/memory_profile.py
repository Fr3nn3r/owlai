import os
import sys
import psutil
import torch
from memory_profiler import profile
from owlai.nest import AgentManager
from owlai.rag import RAGTool
from owlai.core import OwlAgent
from owlai.config import OWL_AGENTS_CONFIG
import logging
from transformers import PreTrainedModel
import gc
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_torch_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def get_process_memory():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_memory_usage(message: str):
    """Log current memory usage with a message."""
    mem = get_process_memory()
    logging.info(f"{message} - Memory usage: {mem:.2f} MB")


def get_model_size(model_container):
    """Calculate size of a model in MB."""
    try:
        # Try different ways to access the actual model
        model = None
        if hasattr(model_container, "_chat_model_cache"):
            model = model_container._chat_model_cache
        elif hasattr(model_container, "model"):
            model = model_container.model
        elif hasattr(model_container, "_model"):
            model = model_container._model

        if model is None:
            logger.warning(
                f"Could not access model from container: {type(model_container)}"
            )
            return 0

        # Get model size
        total_size = 0
        for param in model.parameters():
            total_size += param.nelement() * param.element_size()
        return total_size / 1024 / 1024  # Convert to MB
    except Exception as e:
        logger.warning(f"Could not calculate model size: {e}")
        return 0


@profile
def analyze_memory(agent):
    """Analyze memory usage of an agent."""
    logging.info(f"Analyzing memory usage for agent: {agent.__class__.__name__}")

    if isinstance(agent, RAGTool):
        if hasattr(agent, "_embeddings") and agent._embeddings is not None:
            embeddings_size = get_model_size(agent._embeddings)
            logging.info(f"Embeddings model size: {embeddings_size:.2f} MB")

        if hasattr(agent, "_reranker") and agent._reranker is not None:
            if isinstance(agent._reranker, CrossEncoder):
                reranker_size = get_model_size(agent._reranker.model)
                logging.info(f"Reranker model size: {reranker_size:.2f} MB")
            else:
                logging.warning("Reranker is not a CrossEncoder instance")
    elif isinstance(agent, OwlAgent):
        # Check for chat model
        if hasattr(agent, "chat_model") and agent.chat_model is not None:
            chat_model_size = get_model_size(agent.chat_model)
            logging.info(f"Chat model size: {chat_model_size:.2f} MB")


@profile
def run_memory_profile():
    """Run a memory profiling session."""
    # Log initial state
    log_memory_usage("Initial state")

    # Initialize agent
    agent_manager = AgentManager(agents_config=OWL_AGENTS_CONFIG, enable_cleanup=True)
    log_memory_usage("After agent initialization")

    # Get focus agent
    focus_agent = agent_manager.get_focus_owl()
    log_memory_usage("After getting focus agent")

    # Analyze model sizes
    if focus_agent:
        analyze_memory(focus_agent)

    # Log final state
    log_memory_usage("Final state")


if __name__ == "__main__":
    run_memory_profile()
