"""
Example script to run OwlAI system initialization
"""

from owlai.services.system import get_environment, initialize

# Get just the environment variable without running full initialization
env = get_environment()
print(f"Current environment: {env}")

# Uncomment below to run full system initialization if needed
# initialize()
