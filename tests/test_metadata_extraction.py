import pytest
from typing import List
from langchain.docstore.document import Document
import tempfile
import os
import fitz  # PyMuPDF
import time
from owlai.document_parser import FrenchLawParser


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup temporary files after each test"""
    yield
    # Give the system a moment to release file handles
    time.sleep(0.1)
    # Clean up any remaining temporary PDF files
    for file in os.listdir(tempfile.gettempdir()):
        if file.startswith("tmp") and file.endswith(".pdf"):
            try:
                os.remove(os.path.join(tempfile.gettempdir(), file))
            except PermissionError:
                # Ignore permission errors during cleanup
                pass


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
def french_law_parser():
    """Fixture to provide a French law parser"""
    return FrenchLawParser()


@pytest.fixture
def sample_doc():
    """Create a sample PyMuPDF document for testing"""
    doc = fitz.open()
    doc.new_page()
    return doc


def test_extract_footer(sample_pdf_path, french_law_parser):
    """Test footer extraction from French law document"""
    footer = french_law_parser.extract_footer(sample_pdf_path)
    assert "Code de commerce" in footer
    assert "Dernière modification" in footer
    assert "Document généré" in footer


def test_extract_metadata_fr_law(french_law_parser, sample_doc):
    """Test metadata extraction from French law footer"""
    footer = "Code de commerce - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025"
    metadata = french_law_parser.extract_metadata_fr_law(footer, sample_doc)

    assert metadata["title"] == "Code de commerce"
    assert metadata["last_modification_fr"] == "01 mars 2025"
    assert metadata["doc_generated_on_fr"] == "12 mars 2025"
    assert metadata["num_pages"] == 1


def test_extract_metadata_fr_law_invalid_footer(french_law_parser, sample_doc):
    """Test metadata extraction with invalid footer format"""
    footer = "Invalid footer format"
    with pytest.raises(ValueError):
        french_law_parser.extract_metadata_fr_law(footer, sample_doc)


def test_extract_metadata_fr_law_missing_fields(french_law_parser, sample_doc):
    """Test metadata extraction with missing fields"""
    footer = "Code de commerce - Document généré le 12 mars 2025"
    with pytest.raises(ValueError):
        french_law_parser.extract_metadata_fr_law(footer, sample_doc)


def test_extract_metadata_fr_law_empty_footer(french_law_parser, sample_doc):
    """Test metadata extraction with empty footer"""
    with pytest.raises(ValueError):
        french_law_parser.extract_metadata_fr_law("", sample_doc)


def test_extract_metadata_fr_law_different_date_formats(french_law_parser, sample_doc):
    """Test metadata extraction with different date formats"""
    footer = "Code de commerce - Dernière modification le 1 mars 2025 - Document généré le 12 mars 2025"
    metadata = french_law_parser.extract_metadata_fr_law(footer, sample_doc)

    assert metadata["title"] == "Code de commerce"
    assert metadata["last_modification_fr"] == "1 mars 2025"
    assert metadata["doc_generated_on_fr"] == "12 mars 2025"
    assert metadata["num_pages"] == 1


def test_extract_metadata_fr_law_with_spaces(french_law_parser, sample_doc):
    """Test metadata extraction with extra spaces"""
    footer = "  Code de commerce  -  Dernière modification le 01 mars 2025  -  Document généré le 12 mars 2025  "
    metadata = french_law_parser.extract_metadata_fr_law(footer, sample_doc)

    assert metadata["title"] == "Code de commerce"
    assert metadata["last_modification_fr"] == "01 mars 2025"
    assert metadata["doc_generated_on_fr"] == "12 mars 2025"
    assert metadata["num_pages"] == 1


def test_extract_metadata_fr_law_with_special_chars(french_law_parser, sample_doc):
    """Test metadata extraction with special characters"""
    footer = "Code de commerce (2025) - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025"
    metadata = french_law_parser.extract_metadata_fr_law(footer, sample_doc)

    assert metadata["title"] == "Code de commerce (2025)"
    assert metadata["last_modification_fr"] == "01 mars 2025"
    assert metadata["doc_generated_on_fr"] == "12 mars 2025"
    assert metadata["num_pages"] == 1
