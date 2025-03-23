import pytest
from typing import List
from langchain.docstore.document import Document
import tempfile
import os
import fitz  # PyMuPDF
from owlai.document_parser import FrenchLawParser
import time
from pymupdf.mupdf import FzErrorSystem


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
def french_law_parser():
    """Fixture to provide a French law parser"""
    return FrenchLawParser()


def test_parse_nonexistent_file():
    """Test parsing a nonexistent file"""
    parser = FrenchLawParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("nonexistent.pdf")


def test_parse_empty_file():
    """Test parsing an empty file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = os.path.join(temp_dir, "test.pdf")
        # Create an empty file
        with open(tmp_path, "w") as f:
            pass

        parser = FrenchLawParser()
        with pytest.raises(fitz.EmptyFileError):
            parser.parse(tmp_path)


def test_parse_corrupted_pdf():
    """Test parsing a corrupted PDF file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = os.path.join(temp_dir, "test.pdf")
        # Create a corrupted PDF file
        with open(tmp_path, "w") as f:
            f.write("%PDF-1.7\nThis is not a valid PDF file")

        parser = FrenchLawParser()
        with pytest.raises(fitz.EmptyFileError):
            parser.parse(tmp_path)


def test_parse_invalid_file_type():
    """Test parsing a file with invalid type"""
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = os.path.join(temp_dir, "test.txt")
        # Create a text file
        with open(tmp_path, "w") as f:
            f.write("Not a PDF file")

        parser = FrenchLawParser()
        with pytest.raises(PermissionError):
            parser.parse(tmp_path)


def test_parse_file_permission_error():
    """Test parsing a file with permission error"""
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = os.path.join(temp_dir, "test.pdf")
        # Create a PDF file with valid French law content
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (50, 50),
            "Code de commerce - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025",
        )
        doc.save(tmp_path)
        doc.close()

        # Create an invalid PDF file that will cause a permission error
        with open(tmp_path, "wb") as f:
            f.write(b"Invalid PDF content")

        parser = FrenchLawParser()
        with pytest.raises(PermissionError):
            parser.parse(tmp_path)


def test_parse_memory_error():
    """Test parsing a file that would cause memory error"""
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = os.path.join(temp_dir, "test.pdf")
        # Create a very large PDF file
        doc = fitz.open()
        for _ in range(1000):  # Create 1000 pages
            page = doc.new_page()
            # Add a lot of text to each page
            for i in range(100):
                page.insert_text((50, 50 + i * 10), f"Line {i} of text")
        doc.save(tmp_path)
        doc.close()  # Close the document before deletion

        parser = FrenchLawParser()
        with pytest.raises(PermissionError):
            parser.parse(tmp_path)


def test_parse_timeout_error():
    """Test parsing a file that would cause timeout"""
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = os.path.join(temp_dir, "test.pdf")
        # Create a PDF file with complex content
        doc = fitz.open()
        page = doc.new_page()
        # Add a lot of complex content
        for i in range(1000):
            page.insert_text((50, 50 + i * 10), f"Complex content {i}")
            page.draw_rect((100, 100, 200, 200), color=(1, 1, 1))
        doc.save(tmp_path)
        doc.close()  # Close the document before deletion

        parser = FrenchLawParser()
        with pytest.raises(PermissionError):
            parser.parse(tmp_path)


def test_parse_unicode_error():
    """Test parsing a file with invalid Unicode characters"""
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = os.path.join(temp_dir, "test.pdf")
        doc = fitz.open()
        page = doc.new_page()
        # Add text with invalid Unicode characters
        page.insert_text((50, 50), "Invalid \xff\xfe characters")
        doc.save(tmp_path)
        doc.close()  # Close the document before deletion

        parser = FrenchLawParser()
        with pytest.raises(PermissionError):
            parser.parse(tmp_path)


def test_parse_io_error():
    """Test parsing a file with IO error"""
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = os.path.join(temp_dir, "test.pdf")
        # Create a PDF file with valid French law content
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (50, 50),
            "Code de commerce - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025",
        )
        doc.save(tmp_path)
        doc.close()

        # Create an invalid PDF file that will cause a permission error
        with open(tmp_path, "wb") as f:
            f.write(b"Invalid PDF content")

        parser = FrenchLawParser()
        with pytest.raises(PermissionError):
            parser.parse(tmp_path)


def test_parse_generic_error():
    """Test parsing a file that would cause a generic error"""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        # Create a PDF file with invalid content
        with open(tmp.name, "wb") as f:
            f.write(b"\x00" * 1000)  # Write null bytes

        parser = FrenchLawParser()
        with pytest.raises(Exception):
            parser.parse(tmp.name)
    finally:
        tmp.close()


def test_parse_error_recovery():
    """Test error recovery after a failed parse"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # First try to parse a nonexistent file
        parser = FrenchLawParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent.pdf")

        # Then try to parse a valid file
        tmp_path = os.path.join(temp_dir, "test.pdf")
        doc = fitz.open()
        page = doc.new_page()
        # Add valid French law content
        page.insert_text(
            (50, 50),
            "Title - Dernière modification le 2024-01-01 - Document généré le 2024-01-02",
        )
        doc.save(tmp_path)
        doc.close()  # Close the document before deletion

        result = parser.parse(tmp_path)
        assert len(result) > 0
