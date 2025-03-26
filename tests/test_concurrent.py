import logging
import time
from concurrent.futures import ThreadPoolExecutor
import importlib

# Global variable to track reloads
reload_count = 0


def worker_function():
    global reload_count
    # Simulate RAG-like operation
    time.sleep(0.1)
    # Force reload of a module
    importlib.reload(importlib.import_module("test_concurrent"))
    reload_count += 1
    return reload_count


def main():
    # Test with different import locations
    print("Testing with imports at module level...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: worker_function(), range(10)))
    print(f"Reload count with module-level imports: {results[-1]}")


if __name__ == "__main__":
    # Reset counter
    reload_count = 0

    # Test with imports in main
    print("\nTesting with imports in if __name__ == '__main__'...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: worker_function(), range(10)))
    print(f"Reload count with main-level imports: {results[-1]}")

if __name__ == "__main__":
    main()
