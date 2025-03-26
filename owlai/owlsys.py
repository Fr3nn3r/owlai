import platform
import psutil
import json
import GPUtil
import time
import logging
import logging.config
from contextlib import contextmanager
from typing import Optional, Dict, Any
from rich.console import Console
import yaml
import os
import codecs
import sys


def set_cuda_device():
    """Set CUDA device based on availability"""
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            env = os.getenv("OWL_ENV", "development")
            print(f"Using CUDA device in {env} environment")
        else:
            device = "cpu"
            print("CUDA not available, falling back to CPU")
    except ImportError:
        device = "cpu"
        print("PyTorch not available, using CPU")

    return device


class UnicodeStreamHandler(logging.StreamHandler):
    """Custom StreamHandler that handles Unicode properly"""

    def __init__(self, stream=None):
        super().__init__(stream)
        self.stream = stream or sys.stdout

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Use codecs to handle Unicode properly
            stream = codecs.getwriter("utf-8")(stream.buffer)
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Configure logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "owlai.owlsys.UnicodeStreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "standard",
                "filename": "logs/owlai.log",
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": True,
            },
        },
    }

    logging.config.dictConfig(logging_config)


# Create a basic logger first
logger = logging.getLogger("main")

# Set CUDA device immediately
device = set_cuda_device()


@contextmanager
def track_time(event_name: str, execution_log: Optional[Dict[str, Any]] = None):
    start_time = time.time()
    logger.debug(f"Started '{event_name}' with time tracking...")
    try:
        yield  # This is where the actual event execution happens
    finally:
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        human_readable_time = ""
        if hours > 0:
            human_readable_time += f"{int(hours)}h "
        if minutes > 0:
            human_readable_time += f"{int(minutes)}m "
        human_readable_time += f"{seconds:.3f}s"
        logger.debug(f"'{event_name}' - completed in {human_readable_time}.")
        if execution_log:
            execution_log[f"{event_name} - execution time"] = human_readable_time


def sprint(*args):
    """A smart print function for JSON-like structures"""
    console = Console()
    for arg in args:
        console.print(arg)  # Normal print with `rich`


def get_system_info():
    system_info = {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "OS Release": platform.release(),
        "CPU": {
            "Model": platform.processor(),
            "Cores": psutil.cpu_count(logical=False),
            "Threads": psutil.cpu_count(logical=True),
            "Max Frequency (MHz)": (
                psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown"
            ),
        },
        "Memory": {
            "Total (GB)": round(psutil.virtual_memory().total / (1024**3), 2),
            "Available (GB)": round(psutil.virtual_memory().available / (1024**3), 2),
        },
        "Disk": {
            "Total (GB)": round(psutil.disk_usage("/").total / (1024**3), 2),
            "Used (GB)": round(psutil.disk_usage("/").used / (1024**3), 2),
            "Free (GB)": round(psutil.disk_usage("/").free / (1024**3), 2),
        },
        "GPU": [],
    }

    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            system_info["GPU"].append(
                {
                    "Name": gpu.name,
                    "Memory Total (GB)": round(gpu.memoryTotal, 2),
                    "Memory Free (GB)": round(gpu.memoryFree, 2),
                    "Memory Used (GB)": round(gpu.memoryUsed, 2),
                    "Load (%)": gpu.load * 100,
                }
            )
    except Exception as e:
        system_info["GPU"].append({"Error": str(e)})

    return system_info


def encode_text(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("utf-8")
