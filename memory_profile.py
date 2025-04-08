import os
import sys
import psutil
import torch
from memory_profiler import profile
from owlai.nest import AgentManager
from owlai.config import OWL_AGENTS_CONFIG
import logging
from transformers import PreTrainedModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_torch_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # Convert to MB


def get_model_size(model_container):
    """Calculate model size in MB if possible"""
    total_size = 0

    # Try different model attributes that might contain the actual model
    if hasattr(model_container, "_model"):
        model = model_container._model
    elif hasattr(model_container, "model"):
        model = model_container.model
    elif isinstance(model_container, PreTrainedModel):
        model = model_container
    else:
        logger.warning(f"Unknown model type: {type(model_container)}")
        return 0

    # Calculate size for PyTorch models
    if isinstance(model, torch.nn.Module):
        total_size = (
            sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        )
        logger.info(f"Model size: {total_size:.2f} MB")
    else:
        logger.warning(f"Cannot calculate size for model type: {type(model)}")

    return total_size


@profile
def analyze_memory():
    print(f"Initial process memory: {get_process_memory():.2f} MB")

    # Initialize AgentManager
    print("\nInitializing AgentManager...")
    initial_mem = get_process_memory()
    manager = AgentManager(agents_config=OWL_AGENTS_CONFIG, enable_cleanup=False)
    after_manager = get_process_memory()
    print(f"AgentManager initialization used: {after_manager - initial_mem:.2f} MB")

    # Get focus agent and analyze its components
    print("\nAnalyzing focus agent components...")
    focus_agent = manager.get_focus_owl()

    # Analyze chat model size
    if hasattr(focus_agent, "chat_model"):
        model_size = get_model_size(focus_agent.chat_model)
        print(f"Chat model size: {model_size:.2f} MB")

    # Analyze embeddings model size if available
    if hasattr(focus_agent, "_embeddings"):
        embeddings_size = get_model_size(focus_agent._embeddings)
        print(f"Embeddings model size: {embeddings_size:.2f} MB")

    # Analyze reranker model size if available
    if hasattr(focus_agent, "_reranker"):
        reranker_size = get_model_size(focus_agent._reranker)
        print(f"Reranker model size: {reranker_size:.2f} MB")

    # Check CUDA memory if available
    cuda_mem = get_torch_memory()
    if cuda_mem > 0:
        print(f"CUDA memory used: {cuda_mem:.2f} MB")

    # Get total process memory at end
    final_mem = get_process_memory()
    print(f"\nTotal process memory: {final_mem:.2f} MB")
    print(f"Memory increase during profiling: {final_mem - initial_mem:.2f} MB")

    return manager  # Return manager to prevent garbage collection


if __name__ == "__main__":
    analyze_memory()
