#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the OwlAI application.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the src directory to the path
project_root = Path(__file__).parent.parent.absolute()
src_path = str(project_root / "src")
sys.path.insert(0, src_path)
sys.path.insert(0, str(project_root))

# Set the PYTHONPATH environment variable
current_pythonpath = os.environ.get("PYTHONPATH", "")
paths_to_add = [src_path, str(project_root)]
for path in paths_to_add:
    if path not in current_pythonpath:
        if current_pythonpath:
            current_pythonpath = f"{path}{os.pathsep}{current_pythonpath}"
        else:
            current_pythonpath = path

os.environ["PYTHONPATH"] = current_pythonpath
print(f"PYTHONPATH set to: {os.environ['PYTHONPATH']}")

# Import and run the app
try:
    print("Attempting to import the app module...")
    from src.owlai.app import main

    print("Successfully imported app module. Starting the application...")
    main()
except ImportError as e:
    print(f"Error importing the OwlAI application: {e}")
    print("Detailed traceback:")
    import traceback

    traceback.print_exc()

    print("\nTrying to run directly...")

    # Try to run directly
    try:
        app_path = project_root / "src" / "owlai" / "app.py"
        print(f"Running Python script at: {app_path}")
        env = os.environ.copy()
        result = subprocess.run(
            [sys.executable, str(app_path)],
            check=True,
            env=env,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running OwlAI: {e}")
        if e.stderr:
            print(f"Error output:\n{e.stderr}")
        if e.stdout:
            print(f"Standard output:\n{e.stdout}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
