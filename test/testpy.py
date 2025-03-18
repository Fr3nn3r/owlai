import fitz  # PyMuPDF
import re


def extract_footer(pdf_path):
    doc = fitz.open(pdf_path)
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
    # Get footer from the first page
    footers = extract_footer(pdf_path)
    if not footers:
        return {"title": "", "last_modification": "", "doc_generated_on": ""}

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
        }

    return {"title": "", "last_modification": "", "doc_generated_on": ""}


# Path to your PDF
# pdf_path = "data/dataset-0004/LEGITEXT000006069576.pdf"
pdf_path = "data/dataset-0004/LEGITEXT000006069577 (1).pdf"

# Example usage of extract_metadata function
metadata = extract_metadata(pdf_path)
print("\nExtracted metadata:")
print(metadata)
