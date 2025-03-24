import fitz
import re
from fitz import Document
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from owlai.owlsys import track_time
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

from typing import List
import os
import logging

logger = logging.getLogger("main")


def extract_footer(doc):
    footers = []

    # Check at least the first 10 pages or all pages if less than 10
    max_pages_to_check = min(10, len(doc))
    common_footer = None

    for page_num in range(max_pages_to_check):
        page = doc[page_num]

        # Extract text and split into lines
        text = page.get_text("text")
        lines = text.split("\n")

        if len(lines) > 1:
            current_footer = "".join(
                lines[-2:]
            )  # Assume footer is in the last two lines
            footers.append((page_num + 1, current_footer))

            # Initialize common_footer with the first page's footer
            if page_num == 0:
                common_footer = current_footer
            # Check if footer is consistent across pages
            elif current_footer != common_footer:
                common_footer = None  # Footers don't match

    # If we've checked enough pages and found a consistent footer, return it
    if common_footer and len(footers) >= max_pages_to_check:
        logger.debug(f"Common footer found: '''{common_footer}'''")
        return common_footer

    raise ValueError("No consistent footer found in the document")


def extract_metadata_fr_law(footer: str, doc: Document):
    """
    Extract metadata from a PDF file footer (file expected to follow french law convention).

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        dict: Dictionary containing title, last_modification, and doc_generated_on
    """

    # Regular Expression to Extract Components
    match = re.match(r"^(.*?)\s*-\s*(.*?)\s*-\s*(.*?)$", footer)
    if match:
        title = match.group(1).strip()
        last_modification = match.group(2).strip()
        last_modification = last_modification.replace("Dernière modification le ", "")
        doc_generated_on = match.group(3).strip()
        doc_generated_on = doc_generated_on.replace("Document généré le ", "")

        return {
            "title": title,
            "last_modification_fr": last_modification,
            "doc_generated_on_fr": doc_generated_on,
            "num_pages": len(doc),
        }

    raise ValueError(f"footer '{footer}' not matching french law convention.")


def analyze_chunk_size_distribution(
    input_data_folder,
    filename,
    docs: List[LangchainDocument],
    model_name="thenlper/gte-small",
):
    """
    Analyze and visualize document lengths.

    Args:
        docs: to analyze
        model_name: Name of the embedding model to use
    """
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    # Get max sequence length from SentenceTransformer
    max_seq_len = SentenceTransformer(model_name).max_seq_length
    info_message = (
        f"Model's max sequence size: '{max_seq_len}' Document count: '{len(docs)}'"
    )
    logger.info(info_message)

    # Analyze token lengths (should init tokenizer once... whatever)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lengths = [len(tokenizer.encode(doc.page_content)) for doc in docs]

    # Plot distribution
    import matplotlib.pyplot as plt
    import pandas as pd

    fig = pd.Series(lengths).hist()
    plt.title(f"Distribution of page lengths [tokens] for {filename}")
    file_dir = f"{input_data_folder}/images"
    file_path = f"{file_dir}/chunk_size_distribution-{filename}.png"
    os.makedirs(file_dir, exist_ok=True)
    plt.savefig(file_path)
    plt.close()
    print(f"Distribution of document lengths (in count of tokens) saved to {file_path}")
    return file_path


def load_fr_law_pdf(pdf_path: str) -> list[LangchainDocument]:
    """
    Loads a french law PDF file and returns the content.
    """
    doc: Document = fitz.open(pdf_path)

    file_name = os.path.basename(pdf_path)

    footer = extract_footer(doc)

    metadata = doc.metadata.copy()

    # Merge the two dictionaries - metadata from the document and extracted metadata from the footer
    metadata.update(extract_metadata_fr_law(footer, doc))

    total_pages = metadata.get("num_pages", 0)
    loaded_docs: List[LangchainDocument] = []
    total_page_content = ""

    for page_number, page in tqdm(
        enumerate(doc),
        total=total_pages,
        desc=f"Loading pages from {pdf_path}",
    ):
        # Extract text content from the page
        page_content = page.get_text("text")

        # Remove footer if present to avoid duplication
        page_content = page_content.replace(footer, "")

        # Update page-specific metadata
        # page_metadata = metadata.copy()
        # page_metadata["page_number"] = page_number + 1  # 1-based page numbering
        # page_metadata["source"] = f"{file_name}:{page_number + 1}"

        # Load the whole document to ensure we don't force a split between pages
        total_page_content += page_content

    loaded_docs.append(
        LangchainDocument(
            page_content=total_page_content,
            metadata=metadata,
        )
    )

    return loaded_docs
    # Split documents


def parse_fr_law_docs(docs: List[LangchainDocument]) -> List[LangchainDocument]:

    # All that stuff could be in the parse parameters
    embeddings_chunk_size = 512
    chunk_size = 512
    chunk_overlap = int(embeddings_chunk_size / 5)
    chunk_size = embeddings_chunk_size * 0.9
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    separators = [
        # "\nPartie[^\n]*",
        # "\nLivre[^\n]*",
        # "\nTitre[^\n]*",
        # "\nChapitre[^\n]*",
        # "\nArticle[^\n]*",
        ".{10,}(?=\n)Partie[^\n]*",
        ".{10,}(?=\n)Livre[^\n]*",
        ".{10,}(?=\n)Titre[^\n]*",
        ".{10,}(?=\n)Chapitre[^\n]*",
        ".{10,}(?=\n)Article[^\n]*",
        "\n \n \n",
        "\n\n \n",
        "\n \n\n",
        "\n \n",
        "\n\n",
        "\n",
        ".",
        # " ",
        # "",
    ]

    with track_time("Splitting documents"):
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,  # The number of characters to overlap between chunks
            add_start_index=True,  # If `True`, includes chunk's start index in metadata
            strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
            separators=separators,
            is_separator_regex=True,
        )

        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text))

        docs_processed: List[LangchainDocument] = []

        docs_processed = text_splitter.split_documents(loaded_docs)

        unique_texts = {}
        docs_processed_unique = []
        for idoc in docs_processed:
            if idoc.page_content not in unique_texts:
                unique_texts[idoc.page_content] = True
                metadata_update = idoc.metadata.copy()
                metadata_update["token_count"] = count_tokens(idoc.page_content)
                docs_processed_unique.append(
                    LangchainDocument(
                        page_content=idoc.page_content, metadata=metadata_update
                    )
                )

    return docs_processed_unique


def load_fr_law_pdf_from_folder(folder_path: str) -> list[LangchainDocument]:
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            doc = load_fr_law_pdf(os.path.join(folder_path, file))
            docs.append(doc)
            analyze_chunk_size_distribution(
                folder_path, file, doc, "thenlper/gte-small"
            )
    return docs


doc = load_fr_law_pdf("data/dataset-0006/LEGITEXT000006069568.pdf")

console = Console()
console.print(doc[0])
