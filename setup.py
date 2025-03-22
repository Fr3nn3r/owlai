#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A minimal setup file that defers to pyproject.toml for configuration.
This is primarily for backward compatibility with tools that don't yet
support pyproject.toml.
"""

from setuptools import setup, find_packages

setup(
    name="owlai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "pydantic>=2.0.0",
    ],
)
