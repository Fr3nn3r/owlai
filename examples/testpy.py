import fitz  # PyMuPDF
import re
from langchain_community.document_loaders import PyPDFLoader


def extract_footer(doc):
    footers = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract text and split into lines
        text = page.get_text("text")
        lines = text.split("\n")

        if len(lines) > 1:
            footer = lines[-2:]  # Assume footer is in the last two lines
            footers.append((page_num + 1, " | ".join(footer)))

    return footers


def extract_metadata(pdf_path):
    """
    Extract metadata from a PDF file footer.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        dict: Dictionary containing title, last_modification, and doc_generated_on
    """
    doc = fitz.open(pdf_path)

    # Get footer from the first page
    footers = extract_footer(doc)
    if not footers:
        return {
            "title": "",
            "last_modification": "",
            "doc_generated_on": "",
            "num_pages": 0,
        }

    footer = footers[0][1]

    # Regular Expression to Extract Components
    match = re.match(r"^(.*?)\s*-\s*(.*?)\s*-\s*(.*?)$", footer)
    if match:
        title = match.group(1).strip()
        last_modification = match.group(2).strip()
        doc_generated_on = match.group(3).strip()

        return {
            "title": title,
            "last_modification": last_modification,
            "doc_generated_on": doc_generated_on,
            "num_pages": len(doc),
        }

    return {
        "title": "",
        "last_modification": "",
        "doc_generated_on": "",
        "num_pages": 0,
    }


import os
import asyncio
from typing import List


async def process_file(pdf_path: str) -> List:
    print(f"Processing {pdf_path}")
    metadata = extract_metadata(pdf_path)
    print(metadata)

    loader = PyPDFLoader(
        file_path=pdf_path, extract_images=False, extraction_mode="plain"
    )
    print(f"Starting PDF loading for {pdf_path}...")

    docs = []
    for page_number, doc in enumerate(loader.lazy_load()):
        print(f"{pdf_path} loaded page {page_number + 1}")
        docs.append(doc)

    print(f"PDF loading complete for {pdf_path}")
    return docs


async def process_files_in_batches(files: List[str], batch_size: int = 5):
    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        tasks = []
        for f in batch:
            if f.endswith((".pdf", ".txt")):
                pdf_path = f"data/dataset-0005/{f}"
                tasks.append(asyncio.create_task(process_file(pdf_path)))

        if tasks:
            await asyncio.gather(*tasks)


# Get list of files
# files = os.listdir("data/dataset-0005/")

# Run the async processing
# asyncio.run(process_files_in_batches(files))
# Path to your PDF
# pdf_path = "data/dataset-0004/LEGITEXT000006069576.pdf"

report_path = "C:/Users/fbrun/Documents/GitHub/owlai/data/dataset-0005/20250321-151653-qa_results.json"

import json

with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

from rich.console import Console
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

console = Console()

# Assuming rag_prompt is a long string inside the report dictionary
rag_prompt = report["Test #2"]["metadata"]["rag_prompt"]

# Option 1: Use a Panel for nicer formatting
console.print(Panel(rag_prompt, title="RAG Prompt", expand=True, padding=(1, 2)))
