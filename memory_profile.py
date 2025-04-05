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


def get_model_size(model):
    """Calculate model size in MB if possible"""
    if isinstance(model, PreTrainedModel):
        return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    return 0


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

    # Analyze model size if possible
    if hasattr(focus_agent.chat_model, "model"):
        model_size = get_model_size(focus_agent.chat_model.model)
        print(f"Model parameters size: {model_size:.2f} MB")

    # Check CUDA memory if available
    cuda_mem = get_torch_memory()
    if cuda_mem > 0:
        print(f"CUDA memory used: {cuda_mem:.2f} MB")

    # Get total process memory at end
    final_mem = get_process_memory()
    print(f"\nTotal process memory: {final_mem:.2f} MB")

    return manager  # Return manager to prevent garbage collection


if __name__ == "__main__":
    analyze_memory()
