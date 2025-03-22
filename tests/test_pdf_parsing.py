import pytest
from typing import List
from langchain.docstore.document import Document
import tempfile
import os
import fitz  # PyMuPDF
import time


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup fixture that runs after each test"""
    yield
    # Give the system a moment to release file handles
    time.sleep(0.1)
    # Clean up any remaining temporary files
    temp_dir = tempfile.gettempdir()
    for file in os.listdir(temp_dir):
        if file.endswith(".pdf") and file.startswith("tmp"):
            try:
                file_path = os.path.join(temp_dir, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")


@pytest.fixture
def sample_pdf_path():
    """Create a temporary PDF file with French law content"""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        # Create a PDF with French law content
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Code de commerce\nArticle 1\nTest content")
        page.insert_text(
            (50, 700),
            "Code de commerce - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025",
        )
        doc.save(tmp.name)
        doc.close()
        return tmp.name
    finally:
        tmp.close()


@pytest.fixture
def multi_page_pdf_path():
    """Create a temporary multi-page PDF file"""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        doc = fitz.open()
        # Page 1
        page1 = doc.new_page()
        page1.insert_text((50, 50), "Page 1\nContent 1")
        # Page 2
        page2 = doc.new_page()
        page2.insert_text((50, 50), "Page 2\nContent 2")
        # Page 3
        page3 = doc.new_page()
        page3.insert_text((50, 50), "Page 3\nContent 3")
        # Add footer to each page
        for page in [page1, page2, page3]:
            page.insert_text(
                (50, 700),
                "Code de commerce - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025",
            )
        doc.save(tmp.name)
        doc.close()
        return tmp.name
    finally:
        tmp.close()


@pytest.fixture
def french_law_parser():
    """Fixture to provide a French law parser"""
    from owlai.document_parser import FrenchLawParser

    return FrenchLawParser()


def test_parse_pdf(french_law_parser, sample_pdf_path):
    """Test parsing a PDF file"""
    documents = french_law_parser.parse(sample_pdf_path)

    assert len(documents) > 0
    assert all(isinstance(doc, Document) for doc in documents)
    assert "Code de commerce" in documents[0].page_content
    assert "Article 1" in documents[0].page_content


def test_parse_pdf_invalid_file(french_law_parser):
    """Test parsing with invalid file path"""
    with pytest.raises(FileNotFoundError):
        french_law_parser.parse("nonexistent.pdf")


def test_parse_pdf_empty_file():
    """Test parsing an empty PDF file"""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        # Create an empty PDF
        doc = fitz.open()
        doc.new_page()
        doc.save(tmp.name)
        doc.close()

        parser = FrenchLawParser()
        with pytest.raises(ValueError):
            parser.parse(tmp.name)
    finally:
        tmp.close()


def test_parse_pdf_multiple_pages(french_law_parser, multi_page_pdf_path):
    """Test parsing a multi-page PDF file"""
    documents = french_law_parser.parse(multi_page_pdf_path)

    assert len(documents) == 3
    assert all(isinstance(doc, Document) for doc in documents)
    assert "Page 1" in documents[0].page_content
    assert "Page 2" in documents[1].page_content
    assert "Page 3" in documents[2].page_content


def test_parse_pdf_metadata(french_law_parser, sample_pdf_path):
    """Test that PDF metadata is correctly extracted"""
    documents = french_law_parser.parse(sample_pdf_path)

    for doc in documents:
        assert "source" in doc.metadata
        assert "page" in doc.metadata
        assert "title" in doc.metadata
        assert doc.metadata["source"] == sample_pdf_path


def test_parse_pdf_encoding(french_law_parser):
    """Test parsing PDF with different encodings"""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        doc = fitz.open()
        page = doc.new_page()
        # Add text with special characters
        page.insert_text((50, 50), "Légal contenu\nPlus de contenu\nContenu final")
        doc.save(tmp.name)
        doc.close()

        documents = french_law_parser.parse(tmp.name)
        assert len(documents) > 0
        assert "Légal contenu" in documents[0].page_content
    finally:
        tmp.close()


def test_parse_pdf_with_images():
    """Test parsing PDF with images"""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        doc = fitz.open()
        page = doc.new_page()
        # Add text and image
        page.insert_text((50, 50), "Text content")
        # Add a simple rectangle as an image
        page.draw_rect((100, 100, 200, 200), color=(1, 1, 1))
        doc.save(tmp.name)
        doc.close()

        parser = FrenchLawParser()
        documents = parser.parse(tmp.name)
        assert len(documents) > 0
        assert "Text content" in documents[0].page_content
    finally:
        tmp.close()


def test_parse_pdf_with_tables():
    """Test parsing PDF with tables"""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        doc = fitz.open()
        page = doc.new_page()
        # Add text in table-like format
        page.insert_text((50, 50), "Header 1 | Header 2\nValue 1 | Value 2")
        doc.save(tmp.name)
        doc.close()

        parser = FrenchLawParser()
        documents = parser.parse(tmp.name)
        assert len(documents) > 0
        assert "Header 1" in documents[0].page_content
        assert "Value 1" in documents[0].page_content
    finally:
        tmp.close()


def test_parse_pdf_with_links():
    """Test parsing PDF with hyperlinks"""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        doc = fitz.open()
        page = doc.new_page()
        # Add text and link
        page.insert_text((50, 50), "Click here")
        # Add a link annotation
        page.insert_link(
            {"kind": 1, "from": (50, 50, 100, 60), "uri": "https://example.com"}
        )
        doc.save(tmp.name)
        doc.close()

        parser = FrenchLawParser()
        documents = parser.parse(tmp.name)
        assert len(documents) > 0
        assert "Click here" in documents[0].page_content
    finally:
        tmp.close()


def test_parse_pdf_with_rotated_text():
    """Test parsing PDF with rotated text"""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        doc = fitz.open()
        page = doc.new_page()
        # Add rotated text
        page.insert_text((50, 50), "Rotated text", rotate=45)
        doc.save(tmp.name)
        doc.close()

        parser = FrenchLawParser()
        documents = parser.parse(tmp.name)
        assert len(documents) > 0
        assert "Rotated text" in documents[0].page_content
    finally:
        tmp.close()
