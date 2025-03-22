import pytest
from typing import List
from langchain.docstore.document import Document
import tempfile
import os
import fitz  # PyMuPDF
import time
import psutil
import sys


@pytest.fixture
def large_pdf_path():
    """Create a temporary large PDF file"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        doc = fitz.open()
        # Create 100 pages with 1000 lines each
        for _ in range(100):
            page = doc.new_page()
            for i in range(1000):
                page.insert_text((50, 50 + i * 10), f"Line {i} of text")
        doc.save(tmp.name)
        doc.close()
        return tmp.name


@pytest.fixture
def french_law_parser():
    """Fixture to provide a French law parser"""
    from owlai.document_parser import FrenchLawParser

    return FrenchLawParser()


def test_parse_performance(french_law_parser, large_pdf_path):
    """Test parsing performance"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss

    documents = french_law_parser.parse(large_pdf_path)

    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss

    # Calculate metrics
    parse_time = end_time - start_time
    memory_used = end_memory - start_memory

    # Assert performance metrics
    assert parse_time < 10.0  # Should parse within 10 seconds
    assert memory_used < 500 * 1024 * 1024  # Should use less than 500MB memory


def test_split_performance(french_law_parser, large_pdf_path):
    """Test document splitting performance"""
    # First parse the document
    documents = french_law_parser.parse(large_pdf_path)

    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss

    # Split documents
    split_docs = french_law_parser.split(documents, chunk_size=100)

    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss

    # Calculate metrics
    split_time = end_time - start_time
    memory_used = end_memory - start_memory

    # Assert performance metrics
    assert split_time < 5.0  # Should split within 5 seconds
    assert memory_used < 200 * 1024 * 1024  # Should use less than 200MB memory


def test_footer_extraction_performance(french_law_parser, large_pdf_path):
    """Test footer extraction performance"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss

    footer = french_law_parser.extract_footer(large_pdf_path)

    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss

    # Calculate metrics
    extract_time = end_time - start_time
    memory_used = end_memory - start_memory

    # Assert performance metrics
    assert extract_time < 2.0  # Should extract within 2 seconds
    assert memory_used < 100 * 1024 * 1024  # Should use less than 100MB memory


def test_metadata_extraction_performance(french_law_parser):
    """Test metadata extraction performance"""
    footer = "Code de commerce - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025"

    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss

    metadata = french_law_parser.extract_metadata_fr_law(footer)

    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss

    # Calculate metrics
    extract_time = end_time - start_time
    memory_used = end_memory - start_memory

    # Assert performance metrics
    assert extract_time < 0.1  # Should extract within 100ms
    assert memory_used < 10 * 1024 * 1024  # Should use less than 10MB memory


def test_document_curator_performance(french_law_parser, large_pdf_path):
    """Test document curator performance"""
    # First parse the document
    documents = french_law_parser.parse(large_pdf_path)

    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss

    # Curate documents
    for doc in documents:
        french_law_parser.document_curator(doc.page_content, large_pdf_path)

    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss

    # Calculate metrics
    curator_time = end_time - start_time
    memory_used = end_memory - start_memory

    # Assert performance metrics
    assert curator_time < 5.0  # Should curate within 5 seconds
    assert memory_used < 200 * 1024 * 1024  # Should use less than 200MB memory


def test_concurrent_processing_performance(french_law_parser, large_pdf_path):
    """Test concurrent processing performance"""
    import concurrent.futures

    # First parse the document
    documents = french_law_parser.parse(large_pdf_path)

    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss

    # Process documents concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for doc in documents:
            future = executor.submit(
                french_law_parser.document_curator, doc.page_content, large_pdf_path
            )
            futures.append(future)

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss

    # Calculate metrics
    process_time = end_time - start_time
    memory_used = end_memory - start_memory

    # Assert performance metrics
    assert process_time < 3.0  # Should process within 3 seconds
    assert memory_used < 300 * 1024 * 1024  # Should use less than 300MB memory


def test_memory_cleanup(french_law_parser, large_pdf_path):
    """Test memory cleanup after processing"""
    # Get initial memory usage
    initial_memory = psutil.Process().memory_info().rss

    # Process documents
    documents = french_law_parser.parse(large_pdf_path)
    split_docs = french_law_parser.split(documents, chunk_size=100)

    # Force garbage collection
    import gc

    gc.collect()

    # Get final memory usage
    final_memory = psutil.Process().memory_info().rss

    # Assert memory cleanup
    assert (
        final_memory - initial_memory < 100 * 1024 * 1024
    )  # Should clean up most memory


def test_cpu_usage(french_law_parser, large_pdf_path):
    """Test CPU usage during processing"""
    # Get initial CPU usage
    initial_cpu = psutil.Process().cpu_percent()

    # Process documents
    documents = french_law_parser.parse(large_pdf_path)
    split_docs = french_law_parser.split(documents, chunk_size=100)

    # Get final CPU usage
    final_cpu = psutil.Process().cpu_percent()

    # Assert CPU usage
    assert final_cpu < 80  # Should not use more than 80% CPU


def test_disk_io_performance(french_law_parser, large_pdf_path):
    """Test disk I/O performance"""
    import psutil

    # Get initial disk I/O counters
    initial_io = psutil.disk_io_counters()

    # Process documents
    documents = french_law_parser.parse(large_pdf_path)
    split_docs = french_law_parser.split(documents, chunk_size=100)

    # Get final disk I/O counters
    final_io = psutil.disk_io_counters()

    # Calculate I/O metrics
    read_bytes = final_io.read_bytes - initial_io.read_bytes
    write_bytes = final_io.write_bytes - initial_io.write_bytes

    # Assert I/O performance
    assert read_bytes < 100 * 1024 * 1024  # Should read less than 100MB
    assert write_bytes < 50 * 1024 * 1024  # Should write less than 50MB
