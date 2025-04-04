print("loading owlsys module")

import platform
import psutil
import json

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
from pythonjsonlogger import jsonlogger  # Import JSON formatter
from dotenv import load_dotenv


# Create a basic logger first
logger = logging.getLogger(__name__)


def set_cuda_device():
    """Set CUDA device based on availability"""
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device")
        else:
            device = "cpu"
            logger.info("CUDA not available, falling back to CPU")

    except ImportError:
        device = "cpu"
        logger.info("PyTorch not available, using CPU")

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
    """Setup logging configuration based on environment"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Map environment to config file
    config_file = f"config/logging.{env}.yaml"

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Logging config file {config_file} not found")

    try:
        with open(config_file, "r") as f:
            logging_config = yaml.safe_load(f)

        # Ensure logs directory exists for file handlers
        for handler in logging_config.get("handlers", {}).values():
            if "filename" in handler:
                os.makedirs(os.path.dirname(handler["filename"]), exist_ok=True)

        logging.config.dictConfig(logging_config)
        logger.info(f"Logging configured from {config_file}")

    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize logging from {config_file}: {str(e)}"
        ) from e


def set_env():
    # Check if OWLAI_ENV environment variable is set
    owlai_env = os.getenv("OWLAI_ENV")
    if owlai_env:
        logging.debug(f"OWLAI_ENV environment variable is set to: {owlai_env}")
    else:
        raise ValueError("OWLAI_ENV environment variable is not set")

    return owlai_env


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
        if device == "cuda":
            import GPUtil

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


load_dotenv()

device = set_cuda_device()

device = "cpu"  # disable GPU for now

env = set_env()

setup_logging()

is_prod = env == "production"
is_dev = env == "development"
is_test = env == "test"

logger.info(f"System initialized for '{env}' environment, CUDA device: '{device}'")
