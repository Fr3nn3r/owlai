import psutil
import os
import time
import logging
from typing import Dict, Any, Optional
import sys
from datetime import datetime


def get_process_memory() -> Dict[str, Any]:
    """Get memory usage of the current process"""
    process = psutil.Process(os.getpid())
    return {
        "rss": process.memory_info().rss / 1024 / 1024,  # Resident Set Size in MB
        "vms": process.memory_info().vms / 1024 / 1024,  # Virtual Memory Size in MB
        "percent": process.memory_percent(),
        "num_threads": process.num_threads(),
        "num_fds": process.num_fds(),
        "cpu_percent": process.cpu_percent(),
    }


def monitor_memory(interval: float = 1.0, duration: Optional[float] = None):
    """
    Monitor memory usage of the current process

    Args:
        interval: Time between measurements in seconds
        duration: Total monitoring duration in seconds (None for infinite)
    """
    start_time = time.time()
    peak_memory = 0

    logging.info("Starting memory monitoring...")

    try:
        while True:
            if duration and (time.time() - start_time) > duration:
                break

            mem = get_process_memory()
            peak_memory = max(peak_memory, mem["rss"])

            logging.info(f"Memory usage at {datetime.now().strftime('%H:%M:%S')}:")
            logging.info(f"  RSS: {mem['rss']:.2f} MB")
            logging.info(f"  VMS: {mem['vms']:.2f} MB")
            logging.info(f"  Percent: {mem['percent']:.2f}%")
            logging.info(f"  Threads: {mem['num_threads']}")
            logging.info(f"  File Descriptors: {mem['num_fds']}")
            logging.info(f"  CPU Usage: {mem['cpu_percent']}%")
            logging.info(f"  Peak Memory: {peak_memory:.2f} MB")
            logging.info("-" * 50)

            time.sleep(interval)

    except KeyboardInterrupt:
        logging.info("Memory monitoring stopped by user")
    finally:
        logging.info(f"Final peak memory usage: {peak_memory:.2f} MB")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("memory_monitor.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Monitor memory for 60 seconds
    monitor_memory(interval=1.0, duration=60.0)
