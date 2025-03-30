from memory_profiler import profile
import psutil
import os
import gc
import logging
from typing import Dict, Any
import sys


def get_process_memory() -> Dict[str, Any]:
    """Get memory usage of the current process"""
    process = psutil.Process(os.getpid())
    return {
        "rss": process.memory_info().rss / 1024 / 1024,  # Resident Set Size in MB
        "vms": process.memory_info().vms / 1024 / 1024,  # Virtual Memory Size in MB
        "percent": process.memory_percent(),
    }


def log_memory_usage(message: str):
    """Log current memory usage"""
    mem = get_process_memory()
    logging.info(f"Memory usage - {message}:")
    logging.info(f"  RSS: {mem['rss']:.2f} MB")
    logging.info(f"  VMS: {mem['vms']:.2f} MB")
    logging.info(f"  Percent: {mem['percent']:.2f}%")


@profile
def profile_memory():
    """Profile memory usage of the application"""
    # Import your main application here
    from main import app, lifespan

    # Log initial memory usage
    log_memory_usage("Initial state")

    # Force garbage collection
    gc.collect()
    log_memory_usage("After garbage collection")

    # Log memory usage after FastAPI app initialization
    log_memory_usage("After FastAPI initialization")

    # You can add more profiling points here based on your needs


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Run memory profiling
    profile_memory()
