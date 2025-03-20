import platform
import psutil
import json
import GPUtil
import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any
from rich.console import Console

logger = logging.getLogger("owlsys")


@contextmanager
def track_time(event_name: str, execution_log: Optional[Dict[str, Any]] = None):
    start_time = time.time()
    logging.debug(f"Started '{event_name}' please wait..")
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
        logging.info(f"'{event_name}' - completed in {human_readable_time}.")
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
