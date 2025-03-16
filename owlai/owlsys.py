import platform
import psutil
import json
import GPUtil


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
