from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader

loader = PyMuPDFLoader(
    file_path="data/dataset-0001/Naruto-S01.pdf",
    extract_images=False,
)
docs = loader.load()

print("Naruto-S01", docs[0].page_content, docs[0].metadata)

loader = PyMuPDFLoader(
    file_path="data/dataset-0000/2023_canadian_budget.pdf",
    extract_images=False,
)
docs = loader.load()

print("2023_canadian_budget", docs[0].page_content, docs[0].metadata)

import fitz  # PyMuPDF

doc = fitz.open("data/dataset-0001/Naruto-S01.pdf")
for page in doc:
    text = page.get_text()
    print(text)
