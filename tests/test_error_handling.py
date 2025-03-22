import pytest
from typing import List
from langchain.docstore.document import Document
import tempfile
import os
import fitz  # PyMuPDF


@pytest.fixture
def french_law_parser():
    """Fixture to provide a French law parser"""
    from owlai.document_parser import FrenchLawParser

    return FrenchLawParser()


def test_parse_nonexistent_file(french_law_parser):
    """Test parsing a nonexistent file"""
    with pytest.raises(FileNotFoundError):
        french_law_parser.parse("nonexistent.pdf")


def test_parse_empty_file():
    """Test parsing an empty file"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        # Create an empty file
        with open(tmp.name, "w") as f:
            f.write("")

        parser = FrenchLawParser()
        with pytest.raises(ValueError):
            parser.parse(tmp.name)


def test_parse_corrupted_pdf():
    """Test parsing a corrupted PDF file"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        # Create a corrupted PDF file
        with open(tmp.name, "w") as f:
            f.write("Not a valid PDF file")

        parser = FrenchLawParser()
        with pytest.raises(ValueError):
            parser.parse(tmp.name)


def test_parse_invalid_file_type():
    """Test parsing a file with invalid extension"""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        parser = FrenchLawParser()
        with pytest.raises(ValueError):
            parser.parse(tmp.name)


def test_parse_file_permission_error():
    """Test parsing a file with permission error"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        # Create a PDF file
        doc = fitz.open()
        doc.new_page()
        doc.save(tmp.name)
        doc.close()

        # Make file read-only
        os.chmod(tmp.name, 0o000)

        parser = FrenchLawParser()
        with pytest.raises(PermissionError):
            parser.parse(tmp.name)


def test_parse_memory_error():
    """Test parsing a file that would cause memory error"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        # Create a very large PDF file
        doc = fitz.open()
        for _ in range(1000):  # Create 1000 pages
            page = doc.new_page()
            # Add a lot of text to each page
            for i in range(100):
                page.insert_text((50, 50 + i * 10), f"Line {i} of text")
        doc.save(tmp.name)
        doc.close()

        parser = FrenchLawParser()
        with pytest.raises(MemoryError):
            parser.parse(tmp.name)


def test_parse_timeout_error():
    """Test parsing a file that would cause timeout"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        # Create a PDF file with complex content
        doc = fitz.open()
        page = doc.new_page()
        # Add a lot of complex content
        for i in range(1000):
            page.insert_text((50, 50 + i * 10), f"Complex content {i}")
            page.draw_rect((100, 100, 200, 200), color=(1, 1, 1))
        doc.save(tmp.name)
        doc.close()

        parser = FrenchLawParser()
        with pytest.raises(TimeoutError):
            parser.parse(tmp.name)


def test_parse_unicode_error():
    """Test parsing a file with invalid Unicode characters"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        doc = fitz.open()
        page = doc.new_page()
        # Add text with invalid Unicode characters
        page.insert_text((50, 50), "Invalid \xff\xfe characters")
        doc.save(tmp.name)
        doc.close()

        parser = FrenchLawParser()
        with pytest.raises(UnicodeDecodeError):
            parser.parse(tmp.name)


def test_parse_io_error():
    """Test parsing a file with IO error"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        # Create a PDF file
        doc = fitz.open()
        doc.new_page()
        doc.save(tmp.name)
        doc.close()

        # Delete the file while it's being processed
        parser = FrenchLawParser()
        with pytest.raises(IOError):
            parser.parse(tmp.name)
            os.remove(tmp.name)


def test_parse_generic_error():
    """Test parsing a file that would cause a generic error"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        # Create a PDF file with invalid content
        with open(tmp.name, "wb") as f:
            f.write(b"\x00" * 1000)  # Write null bytes

        parser = FrenchLawParser()
        with pytest.raises(Exception):
            parser.parse(tmp.name)


def test_parse_error_recovery():
    """Test error recovery after a failed parse"""
    parser = FrenchLawParser()

    # First attempt with invalid file
    with pytest.raises(FileNotFoundError):
        parser.parse("nonexistent.pdf")

    # Second attempt with valid file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Valid content")
        doc.save(tmp.name)
        doc.close()

        documents = parser.parse(tmp.name)
        assert len(documents) > 0
        assert "Valid content" in documents[0].page_content
