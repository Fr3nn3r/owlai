import pytest
from typing import List
from langchain.docstore.document import Document
import tempfile
import os
import fitz  # PyMuPDF


@pytest.fixture
def sample_pdf_path():
    """Create a temporary PDF file with French law content"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
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


@pytest.fixture
def french_law_parser():
    """Fixture to provide a French law parser"""
    from owlai.document_parser import FrenchLawParser

    return FrenchLawParser()


def test_extract_footer(sample_pdf_path, french_law_parser):
    """Test footer extraction from French law document"""
    footer = french_law_parser.extract_footer(sample_pdf_path)
    assert "Code de commerce" in footer
    assert "Dernière modification" in footer
    assert "Document généré" in footer


def test_extract_metadata_fr_law(french_law_parser):
    """Test metadata extraction from French law footer"""
    footer = "Code de commerce - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025"
    metadata = french_law_parser.extract_metadata_fr_law(footer)

    assert metadata["title"] == "Code de commerce"
    assert metadata["last_modification_fr"] == "01 mars 2025"
    assert metadata["doc_generated_on_fr"] == "12 mars 2025"


def test_extract_metadata_fr_law_invalid_footer(french_law_parser):
    """Test metadata extraction with invalid footer format"""
    footer = "Invalid footer format"
    metadata = french_law_parser.extract_metadata_fr_law(footer)

    assert metadata["title"] == ""
    assert metadata["last_modification_fr"] == ""
    assert metadata["doc_generated_on_fr"] == ""


def test_extract_metadata_fr_law_missing_fields(french_law_parser):
    """Test metadata extraction with missing fields"""
    footer = "Code de commerce - Document généré le 12 mars 2025"
    metadata = french_law_parser.extract_metadata_fr_law(footer)

    assert metadata["title"] == "Code de commerce"
    assert metadata["last_modification_fr"] == ""
    assert metadata["doc_generated_on_fr"] == "12 mars 2025"


def test_extract_metadata_fr_law_empty_footer(french_law_parser):
    """Test metadata extraction with empty footer"""
    metadata = french_law_parser.extract_metadata_fr_law("")

    assert metadata["title"] == ""
    assert metadata["last_modification_fr"] == ""
    assert metadata["doc_generated_on_fr"] == ""


def test_extract_metadata_fr_law_different_date_formats(french_law_parser):
    """Test metadata extraction with different date formats"""
    footer = "Code de commerce - Dernière modification le 1 mars 2025 - Document généré le 12 mars 2025"
    metadata = french_law_parser.extract_metadata_fr_law(footer)

    assert metadata["title"] == "Code de commerce"
    assert metadata["last_modification_fr"] == "1 mars 2025"
    assert metadata["doc_generated_on_fr"] == "12 mars 2025"


def test_extract_metadata_fr_law_with_spaces(french_law_parser):
    """Test metadata extraction with extra spaces"""
    footer = "  Code de commerce  -  Dernière modification le 01 mars 2025  -  Document généré le 12 mars 2025  "
    metadata = french_law_parser.extract_metadata_fr_law(footer)

    assert metadata["title"] == "Code de commerce"
    assert metadata["last_modification_fr"] == "01 mars 2025"
    assert metadata["doc_generated_on_fr"] == "12 mars 2025"


def test_extract_metadata_fr_law_with_special_chars(french_law_parser):
    """Test metadata extraction with special characters"""
    footer = "Code de commerce (2025) - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025"
    metadata = french_law_parser.extract_metadata_fr_law(footer)

    assert metadata["title"] == "Code de commerce (2025)"
    assert metadata["last_modification_fr"] == "01 mars 2025"
    assert metadata["doc_generated_on_fr"] == "12 mars 2025"
