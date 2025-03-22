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


def test_document_curator_footer_removal(french_law_parser, sample_pdf_path):
    """Test removing footer from document content"""
    # Load document
    documents = french_law_parser.parse(sample_pdf_path)

    # Curate document
    curated_content = french_law_parser.document_curator(
        documents[0].page_content, sample_pdf_path
    )

    # Check that footer is removed
    assert "Document généré" not in curated_content
    assert "Dernière modification" not in curated_content
    assert "Code de commerce" in curated_content
    assert "Article 1" in curated_content


def test_document_curator_empty_content(french_law_parser):
    """Test document curator with empty content"""
    curated_content = french_law_parser.document_curator("", "test.pdf")
    assert curated_content == ""


def test_document_curator_single_line(french_law_parser):
    """Test document curator with single line content"""
    curated_content = french_law_parser.document_curator(
        "Single line content", "test.pdf"
    )
    assert curated_content == "Single line content"


def test_document_curator_no_footer(french_law_parser):
    """Test document curator with content without footer"""
    content = "Line 1\nLine 2\nLine 3"
    curated_content = french_law_parser.document_curator(content, "test.pdf")
    assert curated_content == content


def test_document_curator_multiple_pages(french_law_parser):
    """Test document curator with content from multiple pages"""
    content = "Page 1 content\nPage 2 content\nPage 3 content"
    footer = "Code de commerce - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025"
    full_content = f"{content}\n{footer}"

    curated_content = french_law_parser.document_curator(full_content, "test.pdf")
    assert curated_content == content


def test_document_curator_preserve_content(french_law_parser):
    """Test that document curator preserves non-footer content"""
    content = "Important legal text\nMore legal content\nFinal legal text"
    footer = "Code de commerce - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025"
    full_content = f"{content}\n{footer}"

    curated_content = french_law_parser.document_curator(full_content, "test.pdf")
    assert curated_content == content


def test_document_curator_with_spaces(french_law_parser):
    """Test document curator with extra spaces"""
    content = "  Legal content  \n  More content  \n  Final content  "
    footer = "  Code de commerce  -  Dernière modification le 01 mars 2025  -  Document généré le 12 mars 2025  "
    full_content = f"{content}\n{footer}"

    curated_content = french_law_parser.document_curator(full_content, "test.pdf")
    assert curated_content == content


def test_document_curator_with_special_chars(french_law_parser):
    """Test document curator with special characters"""
    content = "Legal content (2025)\nMore content\nFinal content"
    footer = "Code de commerce (2025) - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025"
    full_content = f"{content}\n{footer}"

    curated_content = french_law_parser.document_curator(full_content, "test.pdf")
    assert curated_content == content


def test_document_curator_with_unicode(french_law_parser):
    """Test document curator with Unicode characters"""
    content = "Légal contenu\nPlus de contenu\nContenu final"
    footer = "Code de commerce - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025"
    full_content = f"{content}\n{footer}"

    curated_content = french_law_parser.document_curator(full_content, "test.pdf")
    assert curated_content == content


def test_document_curator_with_line_breaks(french_law_parser):
    """Test document curator with various line break types"""
    content = "Line 1\r\nLine 2\nLine 3\rLine 4"
    footer = "Code de commerce - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025"
    full_content = f"{content}\n{footer}"

    curated_content = french_law_parser.document_curator(full_content, "test.pdf")
    assert curated_content == content
