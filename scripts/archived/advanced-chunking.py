import fitz  # PyMuPDF
import re
import os
import pandas as pd

from langchain.schema import Document
from rich.console import Console
from rich.table import Table

console = Console()

# Define regex patterns for different levels of the hierarchy
doc_title_pattern = r"^\s*Code\s+général\s+des\s+impôts,?\s*"


patterns = {
    "code": re.compile(doc_title_pattern, re.MULTILINE | re.IGNORECASE),
    "annexe": re.compile(
        r"^\s*Annexe\s+\w+\s*", re.MULTILINE | re.IGNORECASE  # (?P<heading_id>\w+)
    ),
    # "livre": re.compile(r"^\s*Livre\s+\w+\s*:", re.MULTILINE | re.IGNORECASE),
    # "partie": re.compile(
    #    r"^\s*Première partie\s*:|Deuxième partie\s*:|Troisième partie\s*:",
    #    re.MULTILINE | re.IGNORECASE,
    # ),
    # "titre": re.compile(r"^\s*Titre\s+\w+\s*:", re.MULTILINE | re.IGNORECASE),
    # "chapitre": re.compile(r"^\s*Chapitre\s+\w+\s*:", re.MULTILINE | re.IGNORECASE),
    # "section": re.compile(r"^\s*Section\s+\w+\s*:", re.MULTILINE | re.IGNORECASE),
    # "article": re.compile(r"^\s*Article\s+\w+\s*:", re.MULTILINE | re.IGNORECASE),
    # "paragraphe": re.compile(r"^\s*M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.\s+\w+\s*:", re.MULTILINE | re.IGNORECASE),
}


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF while maintaining order and handling accents correctly."""
    doc = fitz.open(pdf_path)
    text = "\n".join(
        [page.get_text("text", sort=True) for page in doc]
    )  # Sort text for better reading order
    text = text.encode("utf-8", "ignore").decode(
        "utf-8"
    )  # Handle special characters correctly
    # Curate text to remove all the text before the code title
    footer_pattern = r"Code général des impôts, (Annexe )?C|XC|L?X{0,3}(IX|IV|V?I{0,3})CGI(ANIV)?\. - Dernière modification le \d{2} \w+ \d{4} - Document généré le \d{2} \w+ \d{4}"
    text = re.sub(footer_pattern, "", text)
    return text


def split_document(text, pdf_filename):
    """Splits text into hierarchical chunks based on the structure"""
    chunks = []
    hierarchy = {key: None for key in patterns.keys()}

    # Find all structural elements and split accordingly
    matches = []
    for level, pattern in patterns.items():
        matches.extend(
            [
                (
                    match.start(),
                    level,
                    match.group().strip(),
                    match.groupdict().get("heading_id"),
                )
                for match in pattern.finditer(text)
            ]
        )

    # Sort matches by position in text
    matches.sort()

    for i, (start, level, heading, heading_id) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        # Update hierarchy with correct labels
        hierarchy[level] = heading + (f" ({heading_id})" if heading_id else "")

        # Store structured metadata
        chunk = Document(
            page_content=content,
            metadata={
                "source_filename": pdf_filename,
                "level": level,
                "code": hierarchy["code"],
                "annexe": hierarchy["annexe"],
                # "livre": hierarchy["livre"],
                # "partie": hierarchy["partie"],
                # "titre": hierarchy["titre"],
                # "chapitre": hierarchy["chapitre"],
                # "section": hierarchy["section"],
                # "article": hierarchy["article"],
                # "paragraphe": hierarchy["paragraphe"],
                "start_position": start,
                "end_position": end,
                "content_length": len(content),
                "content_preview": content[:100],
                "heading_id": heading_id,
            },
        )

        chunks.append(chunk)

    return chunks


# Example usage
pdf_path = "data/dataset-0004/LEGITEXT000006069576.pdf"
pdf_filename = os.path.basename(pdf_path)

# Extract and split document
text = extract_text_from_pdf(pdf_path)
chunks = split_document(text, pdf_filename)

# Convert to DataFrame for display
df_chunks = pd.DataFrame(chunks)


print(len(chunks))
for chunk in chunks[:10]:  # Show first 5 chunks
    # print(
    #    f"Source: {chunk.metadata['source_filename']} | Level: {chunk.metadata['level']}"
    # )
    console.print(chunk.metadata)
    # console.print(chunk.page_content[:100])
