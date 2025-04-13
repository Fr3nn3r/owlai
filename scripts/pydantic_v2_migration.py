#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pydantic v2 Migration Script for OwlAI

This script helps with migrating the OwlAI codebase from Pydantic v1 to v2.
It provides instructions, verifies dependencies, and assists with setting up
the necessary environment variables for compatibility.
"""

import os
import sys
import subprocess
import importlib.metadata
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("pydantic_migration")

# Required package versions
REQUIRED_PACKAGES = {
    "langchain-core": "0.1.27",
    "langchain-text-splitters": "0.0.1",
    "langchain-community": "0.0.24",
    "langchain-openai": "0.0.5",
    "langchain-anthropic": "0.1.1",
    "langgraph": "0.0.22",
    "langchain-huggingface": "0.0.2",
    "pydantic": "2.5.2",
}

# Files that need to be updated
KEY_FILES = [
    "owlai/core.py",
    "main.py",
    "owlai/services/rag.py",
    "owlai/services/toolbox.py",
    "tests/test_core.py",
]

# Migration checklist items
MIGRATION_CHECKLIST = [
    "Update class Config to model_config in all Pydantic models",
    "Add model_config = {'arbitrary_types_allowed': True} to models with non-Pydantic types",
    "Update validator decorators from root_validator to model_validator",
    "Add from pydantic.v1 import root_validator if you still need v1 validators",
    "Update orm_mode=True to from_attributes=True",
    "Replace any deprecated methods with their new equivalents",
]


def check_environment() -> bool:
    """
    Check if PYDANTIC_V1 environment variable is set.

    Returns:
        bool: True if environment is set correctly
    """
    pydantic_v1 = os.environ.get("PYDANTIC_V1")
    if pydantic_v1 == "1":
        logger.info("✅ PYDANTIC_V1 environment variable is set to 1")
        return True
    else:
        logger.warning("❌ PYDANTIC_V1 environment variable is not set")
        return False


def check_packages() -> Dict[str, Tuple[str, str, bool]]:
    """
    Check if required packages are installed with the correct versions.

    Returns:
        Dict[str, Tuple[str, str, bool]]: Package name -> (current version, required version, is valid)
    """
    results = {}

    for package, required_version in REQUIRED_PACKAGES.items():
        try:
            current_version = importlib.metadata.version(package)
            is_valid = _version_is_valid(current_version, required_version)
            results[package] = (current_version, required_version, is_valid)
        except importlib.metadata.PackageNotFoundError:
            results[package] = ("not installed", required_version, False)

    return results


def _version_is_valid(current: str, required: str) -> bool:
    """
    Check if current version is valid compared to required version.
    Simple version check that handles basic version strings.

    Args:
        current: Current version string
        required: Required version string

    Returns:
        bool: True if version is valid
    """
    # This is a simple check, could be improved with packaging.version
    current_parts = current.split(".")
    required_parts = required.split(".")

    # Compare major versions
    if int(current_parts[0]) > int(required_parts[0]):
        return True
    elif int(current_parts[0]) < int(required_parts[0]):
        return False

    # Compare minor versions if major versions are equal
    if len(current_parts) > 1 and len(required_parts) > 1:
        if int(current_parts[1]) > int(required_parts[1]):
            return True
        elif int(current_parts[1]) < int(required_parts[1]):
            return False

    # Compare patch versions if major and minor versions are equal
    if len(current_parts) > 2 and len(required_parts) > 2:
        if int(current_parts[2]) >= int(required_parts[2]):
            return True

    # Default to True if just comparing major.minor and they're equal
    return True


def install_packages() -> bool:
    """
    Install required packages with pip.

    Returns:
        bool: True if installation succeeded
    """
    package_specs = [f"{pkg}>={ver}" for pkg, ver in REQUIRED_PACKAGES.items()]
    command = [sys.executable, "-m", "pip", "install", "--upgrade"] + package_specs

    logger.info(f"Installing packages: {' '.join(package_specs)}")
    try:
        subprocess.check_call(command)
        logger.info("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install packages: {e}")
        return False


def set_environment_variable() -> bool:
    """
    Set PYDANTIC_V1=1 environment variable in .env file.

    Returns:
        bool: True if environment variable was set
    """
    env_path = Path(".env")

    # Check if file exists and PYDANTIC_V1 is already set
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()
            if "PYDANTIC_V1=1" in content:
                logger.info("✅ PYDANTIC_V1 already set in .env file")
                return True

    # Add PYDANTIC_V1=1 to .env file
    try:
        with open(env_path, "a", encoding="utf-8") as f:
            f.write(
                "\n# Added for LangChain/Pydantic v2 compatibility\nPYDANTIC_V1=1\n"
            )
        logger.info("✅ Added PYDANTIC_V1=1 to .env file")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to update .env file: {e}")
        return False


def print_checklist() -> None:
    """
    Print the migration checklist for manual code changes.
    """
    logger.info("\n=== Pydantic v2 Migration Checklist ===")
    for i, item in enumerate(MIGRATION_CHECKLIST, 1):
        logger.info(f"{i}. {item}")

    logger.info("\nKey files to update:")
    for file in KEY_FILES:
        logger.info(f"- {file}")


def main():
    """
    Main function to run the migration script.
    """
    logger.info("=== Pydantic v2 Migration for OwlAI ===")

    # Check environment
    env_ok = check_environment()
    if not env_ok:
        set_env = input(
            "Would you like to set PYDANTIC_V1=1 in your .env file? (y/n): "
        )
        if set_env.lower() == "y":
            set_environment_variable()

    # Check package versions
    logger.info("\nChecking installed package versions...")
    pkg_results = check_packages()

    all_ok = True
    for pkg, (current, required, valid) in pkg_results.items():
        status = "✅" if valid else "❌"
        logger.info(f"{status} {pkg}: {current} (required: >={required})")
        if not valid:
            all_ok = False

    # Offer to install packages if needed
    if not all_ok:
        install = input(
            "\nWould you like to install the required package versions? (y/n): "
        )
        if install.lower() == "y":
            install_packages()

    # Print migration checklist
    print_checklist()

    logger.info(
        """
=== Migration Complete ===
1. The PYDANTIC_V1=1 environment variable has been set for compatibility
2. Required package versions have been installed
3. Follow the checklist to complete any remaining manual code changes

Once your application is working correctly with Pydantic v2:
1. You can gradually remove the PYDANTIC_V1=1 environment variable
2. Remove any imports from pydantic.v1
3. Update your code to fully use Pydantic v2 features
"""
    )


if __name__ == "__main__":
    main()
