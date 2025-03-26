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


# Create a basic logger first
logger = logging.getLogger("main")


def set_cuda_device():
    """Set CUDA device based on environment"""
    env = os.getenv("OWL_ENV", "development")
    if env == "production":
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Using CUDA device for production environment")
            else:
                device = "cpu"
                logger.warning("CUDA not available, falling back to CPU in production")
        except ImportError:
            device = "cpu"
            logger.warning("PyTorch not available, using CPU in production")
    else:
        device = "cpu"
        logger.info("Using CPU for development environment")

    return device


def load_logger_config():
    # Only load config if not already configured
    if not logging.getLogger().handlers:
        # Get environment from ENV variable, default to 'development'
        env = os.getenv("OWL_ENV", "development")
        config_path = os.path.join("config", f"logging.{env}.yaml")

        if os.path.exists(config_path):
            with open(config_path, "r") as logger_config:
                config = yaml.safe_load(logger_config)

                # Create log directories if they don't exist
                if "handlers" in config:
                    for handler_name, handler_config in config["handlers"].items():
                        if "filename" in handler_config:
                            log_dir = os.path.dirname(handler_config["filename"])
                            if log_dir and not os.path.exists(log_dir):
                                os.makedirs(log_dir)

                logging.config.dictConfig(config)
                logger.info(f"Loaded logging configuration from {config_path}")
        else:
            # Fallback configuration if yaml file not found
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            )
            logger.warning(
                f"No configuration file found at {config_path}, using default configuration"
            )

    # Set CUDA device based on environment
    return set_cuda_device()


# Load logging config and get device
device = load_logger_config()


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
