import platform
import psutil
import json
import GPUtil
import time
import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)


@contextmanager
def track_time(event_name: str, execution_log: list = None):
    start_time = time.time()
    logging.debug(f"Started '{event_name}' please wait..")
    try:
        yield  # This is where the actual event execution happens
    finally:
        elapsed_time = time.time() - start_time
        logging.info(f"'{event_name}' completed in {elapsed_time:.4f} [s].")
        if execution_log:
            execution_log.append(
                {f"{event_name}_execution_time": f"{elapsed_time:.4f} [s]."}
            )


# Usage
# with track_time("Data Processing"):
#    time.sleep(2)  # Simulating a long process


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
