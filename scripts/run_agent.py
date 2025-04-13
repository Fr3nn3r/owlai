#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple script to run the OwlAgent from the command line.
"""

import argparse
import sys
import os
import logging
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

# Add the parent directory to the path so we can import owlai
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from owlai.core import OwlAgent


def setup_logging():
    """Set up logging with rich formatting."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run OwlAgent with a query")
    parser.add_argument(
        "query",
        nargs="?",
        help="The query to send to the agent. If not provided, will enter interactive mode.",
    )
    parser.add_argument(
        "--model-provider",
        "-p",
        default="openai",
        help="The model provider to use (default: openai)",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        default="gpt-4o-mini",
        help="The model name to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.1,
        help="Temperature for generation (default: 0.1)",
    )
    parser.add_argument(
        "--system-prompt",
        "-s",
        default="You are a helpful assistant.",
        help="System prompt (default: 'You are a helpful assistant.')",
    )
    return parser.parse_args()


def main():
    """Main function."""
    # Load environment variables
    load_dotenv()

    # Set up logging
    setup_logging()

    # Parse arguments
    args = parse_args()

    # Create console for rich output
    console = Console()

    # Create agent
    agent = OwlAgent(
        model_provider=args.model_provider,
        model_name=args.model_name,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
    )

    if args.query:
        # Run once with the provided query
        console.print(f"[bold blue]Query:[/bold blue] {args.query}")
        response = agent.run(args.query)
        console.print(f"[bold green]Response:[/bold green] {response}")
    else:
        # Enter interactive mode
        console.print(
            "[bold cyan]OwlAI Interactive Mode[/bold cyan] (type 'exit' to quit)"
        )
        while True:
            try:
                query = input("\nYou: ")
                if query.lower() in ["exit", "quit", "q"]:
                    break

                response = agent.run(query)
                console.print(f"[bold green]OwlAI:[/bold green] {response}")
            except KeyboardInterrupt:
                console.print("\n[bold red]Exiting...[/bold red]")
                break
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()
