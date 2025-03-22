from typing import List
from langchain.docstore.document import Document
import fitz  # PyMuPDF
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


def load_fr_law_pdf(pdf_path: str) -> List[Document]:
    """Load and parse a French law PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of Document objects containing the parsed content

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or invalid
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        raise ValueError("PDF file is empty")

    documents = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "page": page_num + 1,
                    },
                )
            )

    doc.close()
    return documents


def analyze_chunk_size_distribution(
    output_dir: str,
    pdf_name: str,
    split_docs: List[Document],
    model_name: str,
) -> str:
    """Analyze and plot the distribution of chunk sizes.

    Args:
        output_dir: Directory to save the plot
        pdf_name: Name of the PDF file
        split_docs: List of split documents
        model_name: Name of the model used for tokenization

    Returns:
        Path to the generated plot file
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Calculate token counts for each chunk
    token_counts = []
    for doc in split_docs:
        tokens = tokenizer.encode(doc.page_content)
        token_counts.append(len(tokens))

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(token_counts, bins=30, edgecolor="black")
    plt.title(f"Chunk Size Distribution - {pdf_name}")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")

    # Add statistics
    mean_tokens = float(np.mean(token_counts))
    std_tokens = float(np.std(token_counts))
    plt.axvline(
        float(mean_tokens), color="r", linestyle="--", label=f"Mean: {mean_tokens:.1f}"
    )
    plt.axvline(
        float(mean_tokens + std_tokens),
        color="g",
        linestyle="--",
        label=f"Std: {std_tokens:.1f}",
    )
    plt.axvline(float(mean_tokens - std_tokens), color="g", linestyle="--")
    plt.legend()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{pdf_name}_chunk_distribution.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def document_curator(content: str, pdf_path: str) -> str:
    """Curate document content by removing footers and cleaning up text.

    Args:
        content: The document content to curate
        pdf_path: Path to the PDF file

    Returns:
        Curated content with footers removed and text cleaned up
    """
    if not content:
        return ""

    # Split content into lines
    lines = content.split("\n")

    # Find and remove footer
    footer_pattern = " - Dernière modification le"
    for i, line in enumerate(lines):
        if footer_pattern in line:
            # Remove footer and any trailing empty lines
            lines = lines[:i]
            break

    # Clean up text
    cleaned_lines = []
    for line in lines:
        # Remove extra whitespace
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)
