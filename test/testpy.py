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


# Path to your PDF
pdf_path = "data/dataset-0004/LEGITEXT000006069576.pdf"

# Run the extraction
footers = extract_footer(pdf_path)

# Display results
for page, footer in footers[:10]:  # Show first 10 pages
    print(f"Page {page}: {footer}")

footer = footers[0][1]

# Regular Expression to Extract Components
match = re.match(r"^(.*?)\s*-\s*(.*?)\s*-\s*(.*?)$", footer)
if match:
    title = match.group(1).strip()
    last_modification = match.group(2).strip()
    doc_generated_on = match.group(3).strip()

print(
    {
        "title": title,
        "last_modification": last_modification,
        "doc_generated_on": doc_generated_on,
    }
)
