import fitz
import re
from fitz import Document
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm


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
        print(f"Common footer found: {common_footer}")
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


def load_fr_law_pdf(pdf_path: str) -> list[LangchainDocument]:
    """
    Loads a french law PDF file and returns the content.
    """
    doc: Document = fitz.open(pdf_path)

    footer = extract_footer(doc)

    metadata = doc.metadata

    # Merge the two dictionaries - metadata from the document and extracted metadata from the footer
    metadata.update(extract_metadata_fr_law(footer, doc))

    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(metadata)

    total_pages = metadata.get("num_pages", 0)
    loaded_docs: List[LangchainDocument] = []

    for page_number, page in tqdm(
        enumerate(doc),
        total=total_pages,
        desc=f"Loading pages from {pdf_path}",
    ):
        # Extract text content from the page
        page_content = page.get_text("text")

        # Remove footer if present to avoid duplication
        # if footer and page_content.endswith(footer):
        page_content = page_content.replace(footer, "")

        # Update page-specific metadata
        page_metadata = metadata.copy()
        page_metadata["page_number"] = page_number + 1  # 1-based page numbering
        page_metadata["source"] = f"{pdf_path}:{page_number + 1}"
        # metadata.update(page.metadata)
        metadata["source"] = f"{pdf_path}:{page_number}"
        # Call document curator if provided
        # if document_curator:
        #     doc_content = document_curator(doc_content, filepath)

        loaded_docs.append(
            LangchainDocument(
                page_content=page_content,
                metadata=metadata,
            )
        )
        # Split documents
        chunk_size = 512
        chunk_overlap = int(chunk_size / 5)
        tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,  # The number of characters to overlap between chunks
            add_start_index=True,  # If `True`, includes chunk's start index in metadata
            strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
            separators=[
                # "Livre",
                # "Titre",
                # "Chapitre",
                "Article",
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )
        
        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text))

        docs_processed = []
        separator = "Article"
        chunk_buffer = ""
        current_chunk = ""
        for idoc in loaded_docs:
            result = text_splitter.split_documents([idoc])
            for i in idoc:
                next_chunk : str = i.page_content
                if separator in next_chunk:
                    current_chunk += next_chunk.split(separator,1)[0] # split the chunk at the first occurrence of the separator
                    next_chunk = next_chunk.split(separator,1)[1]
                    chunk_buffer += current_chunk
                    current_chunk_token_count = count_tokens(chunk_buffer)
                    next_chunk_token_count = count_tokens(next_chunk)
                    if current_chunk_token_count < chunk_size - chunk_overlap :
                        current_chunk = separator
                        continue
                    elif current_chunk_token_count < chunk_size :
                        if current_chunk_token_count + next_chunk_token_count < chunk_size:
                            current_chunk = separator
                            continue
                        elif:
                            docs_processed.append(LangchainDocument(page_content=chunk_buffer, metadata=i.metadata))
                            chunk_buffer = ""
                            current_chunk = separator
                            continue
                        
                        i.metadata["token_count"] = token_count
                        docs_processed.append(LangchainDocument(page_content=chunk_buffer, metadata=i.metadata))
                        chunk_buffer = ""





            docs_processed += result

        unique_texts = {}
        docs_processed_unique = []
        for idoc in docs_processed:
            if idoc.page_content not in unique_texts:
                unique_texts[idoc.page_content] = True
                docs_processed_unique.append(idoc)

        if page_number > 20:
            break

    return docs_processed_unique


docs = load_fr_law_pdf("data/dataset-0005/in_store/LEGITEXT000006069414.pdf")

console = Console()

for i in range(10):
    console.print(docs[i])
