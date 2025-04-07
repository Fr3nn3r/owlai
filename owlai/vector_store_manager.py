"""
Utility to manage vector store persistence in the database.
Handles saving and loading complete vector store folders (index.faiss and index.pkl) to/from PostgreSQL.
"""

import os
import base64
import logging
from datetime import datetime, timezone
import json
from typing import Optional, Any, cast, Dict, Tuple, List
from uuid import UUID
import faiss
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from owlai.dbmodels import VectorStore
from tqdm import tqdm
import hashlib
import time
import traceback
from contextlib import contextmanager
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)

# Constants for database operations
CHUNK_SIZE = 1024 * 1024  # 1MB chunk size for reading/writing
MAX_RETRIES = 3  # Maximum number of retry attempts for database operations
RETRY_DELAY = (
    2  # Initial delay in seconds for retry (will be used with exponential backoff)
)
CONNECTION_TIMEOUT = 30  # Connection timeout in seconds
KEEPALIVE_IDLE = 15  # Keepalive idle time in seconds
KEEPALIVE_INTERVAL = 5  # Keepalive interval in seconds


def encode_vector_store_files(vector_db_path: str) -> Dict[str, str]:
    """Encode both FAISS index and pickle files to base64 strings.

    Args:
        vector_db_path: Path to the vector_db directory containing index.faiss and index.pkl

    Returns:
        Dictionary containing base64 encoded contents of both files
    """
    logger.debug(f"Encoding vector store files from {vector_db_path}")
    try:
        faiss_path = os.path.join(vector_db_path, "index.faiss")
        pkl_path = os.path.join(vector_db_path, "index.pkl")

        if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing required files in {vector_db_path}")

        # Read and encode both files
        with open(faiss_path, "rb") as f:
            faiss_encoded = base64.b64encode(f.read()).decode("utf-8")

        with open(pkl_path, "rb") as f:
            pkl_encoded = base64.b64encode(f.read()).decode("utf-8")

        encoded_data = {"index.faiss": faiss_encoded, "index.pkl": pkl_encoded}

        logger.debug("Successfully encoded vector store files")
        return encoded_data

    except Exception as e:
        logger.error(f"Failed to encode vector store files: {str(e)}")
        raise


def decode_vector_store_files(encoded_data: str, output_dir: str) -> None:
    """Decode base64 strings back to FAISS index and pickle files.

    Args:
        encoded_data: JSON string containing base64 encoded file contents
        output_dir: Directory where to save the decoded files
    """
    logger.debug(f"Decoding vector store files to {output_dir}")
    try:
        # Parse the JSON string back to dictionary
        files_data = json.loads(encoded_data)

        os.makedirs(output_dir, exist_ok=True)

        # Decode and write FAISS index
        faiss_path = os.path.join(output_dir, "index.faiss")
        with open(faiss_path, "wb") as f:
            f.write(base64.b64decode(files_data["index.faiss"]))

        # Decode and write pickle file
        pkl_path = os.path.join(output_dir, "index.pkl")
        with open(pkl_path, "wb") as f:
            f.write(base64.b64decode(files_data["index.pkl"]))

        logger.debug("Successfully decoded vector store files")

    except Exception as e:
        logger.error(f"Failed to decode vector store files: {str(e)}")
        raise


def chunk_large_data(
    data: Dict[str, str], chunk_size: int = 1024 * 1024
) -> List[Dict[str, str]]:
    """Split large vector store data into manageable chunks.

    Args:
        data: Dictionary containing the encoded files
        chunk_size: Size of each chunk in bytes

    Returns:
        List of dictionaries, each containing a chunk of the data
    """
    chunks = []
    for file_name, content in data.items():
        # Split content into chunks
        content_chunks = [
            content[i : i + chunk_size] for i in range(0, len(content), chunk_size)
        ]

        # Create chunk metadata
        for i, chunk in enumerate(content_chunks):
            chunks.append(
                {
                    "file_name": file_name,
                    "chunk_index": i,
                    "total_chunks": len(content_chunks),
                    "data": chunk,
                }
            )

    return chunks


@contextmanager
def get_fresh_connection(url, operation_name="database operation"):
    """Create a fresh database connection for a single operation."""
    logger.info(f"Creating fresh connection for: {operation_name}")

    # Create engine with optimized settings for large object operations
    engine = create_engine(
        url,
        pool_size=1,
        max_overflow=0,
        pool_timeout=CONNECTION_TIMEOUT,
        pool_recycle=60,  # Recycle connections after 60 seconds
        connect_args={
            "connect_timeout": CONNECTION_TIMEOUT,
            "keepalives": 1,
            "keepalives_idle": KEEPALIVE_IDLE,
            "keepalives_interval": KEEPALIVE_INTERVAL,
            "keepalives_count": 3,
        },
    )

    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        logger.debug(f"Connection established for {operation_name}")
        yield session
        session.commit()
        logger.debug(f"Operation completed successfully: {operation_name}")
    except Exception as e:
        logger.error(f"Error during {operation_name}: {str(e)}")
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()
        logger.debug(f"Connection closed for {operation_name}")


def retry_on_connection_error(max_retries=MAX_RETRIES, base_delay=RETRY_DELAY):
    """Decorator to retry functions on database connection errors with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (OperationalError, SQLAlchemyError) as e:
                    # Check if it's a connection error
                    error_str = str(e).lower()
                    if any(
                        x in error_str
                        for x in ["connection", "ssl", "timeout", "closed"]
                    ):
                        delay = base_delay * (2**attempt)  # Exponential backoff
                        logger.warning(
                            f"Connection error on attempt {attempt+1}/{max_retries}, "
                            f"retrying in {delay} seconds. Error: {str(e)}"
                        )
                        time.sleep(delay)
                        last_exception = e
                    else:
                        # Not a connection error, don't retry
                        raise
                except Exception as e:
                    # Don't retry other types of exceptions
                    raise

            # If we get here, we've exhausted our retries
            if last_exception:
                logger.error(
                    f"Max retries ({max_retries}) reached. Last error: {str(last_exception)}"
                )
                raise last_exception
            raise RuntimeError("Max retries reached for unknown reason")

        return wrapper

    return decorator


@retry_on_connection_error()
def save_vector_store(
    session: Any,
    vector_db_path: str,
    name: str,
    version: str,
    model_name: str,
    progress_bar: Optional[tqdm] = None,
) -> UUID:
    """Save a complete vector store (both index.faiss and index.pkl) to the database using large object storage with chunking."""
    logger.info(f"Saving vector store '{name}' version {version} to database")

    try:
        # Read files in binary mode
        faiss_path = os.path.join(vector_db_path, "index.faiss")
        pkl_path = os.path.join(vector_db_path, "index.pkl")

        if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing required files in {vector_db_path}")

        # Calculate total size for logging
        faiss_size = os.path.getsize(faiss_path)
        pkl_size = os.path.getsize(pkl_path)
        total_size = faiss_size + pkl_size
        logger.info(
            f"Total data size: {total_size/1024/1024:.1f} MB (FAISS: {faiss_size/1024/1024:.1f} MB, PKL: {pkl_size/1024/1024:.1f} MB)"
        )

        if progress_bar:
            progress_bar.set_description(f"Processing {name}")
            progress_bar.update(10)

        # Get raw connection from SQLAlchemy session
        connection = session.connection().connection
        logger.info("Database connection established for saving vector store")

        # Create large objects for each file
        large_objects = {}

        # Process FAISS index with chunking
        logger.info(
            f"Storing FAISS index ({faiss_size/1024/1024:.1f} MB) as large object"
        )
        lobj_faiss = connection.lobject(0, "wb")

        chunks_processed = 0
        total_chunks = faiss_size // CHUNK_SIZE + (1 if faiss_size % CHUNK_SIZE else 0)

        with open(faiss_path, "rb") as f:
            start_time = time.time()
            while chunk := f.read(CHUNK_SIZE):
                try:
                    lobj_faiss.write(chunk)
                    chunks_processed += 1

                    # Log progress periodically
                    if chunks_processed % 10 == 0 or chunks_processed == total_chunks:
                        elapsed = time.time() - start_time
                        logger.info(
                            f"FAISS: Processed {chunks_processed}/{total_chunks} chunks "
                            f"({chunks_processed*CHUNK_SIZE/faiss_size*100:.1f}%) "
                            f"in {elapsed:.1f}s ({chunks_processed*CHUNK_SIZE/1024/1024/elapsed:.1f} MB/s)"
                        )

                    if progress_bar:
                        progress_bar.update(
                            len(chunk) / total_size * 30
                        )  # 30% of progress for FAISS
                except Exception as e:
                    logger.error(
                        f"Error writing FAISS chunk {chunks_processed}: {str(e)}"
                    )
                    raise

        large_objects["index.faiss"] = lobj_faiss.oid
        logger.info(
            f"Successfully stored FAISS index as large object with OID {lobj_faiss.oid}"
        )

        if progress_bar:
            progress_bar.update(5)

        # Process pickle file with chunking
        logger.info(
            f"Storing pickle file ({pkl_size/1024/1024:.1f} MB) as large object"
        )
        lobj_pkl = connection.lobject(0, "wb")
        chunks_processed = 0
        total_chunks = pkl_size // CHUNK_SIZE + (1 if pkl_size % CHUNK_SIZE else 0)

        with open(pkl_path, "rb") as f:
            start_time = time.time()
            while chunk := f.read(CHUNK_SIZE):
                try:
                    lobj_pkl.write(chunk)
                    chunks_processed += 1

                    # Log progress periodically
                    if chunks_processed % 10 == 0 or chunks_processed == total_chunks:
                        elapsed = time.time() - start_time
                        logger.info(
                            f"PKL: Processed {chunks_processed}/{total_chunks} chunks "
                            f"({chunks_processed*CHUNK_SIZE/pkl_size*100:.1f}%) "
                            f"in {elapsed:.1f}s ({chunks_processed*CHUNK_SIZE/1024/1024/elapsed:.1f} MB/s)"
                        )

                    if progress_bar:
                        progress_bar.update(
                            len(chunk) / total_size * 30
                        )  # 30% of progress for PKL
                except Exception as e:
                    logger.error(
                        f"Error writing PKL chunk {chunks_processed}: {str(e)}"
                    )
                    raise

        large_objects["index.pkl"] = lobj_pkl.oid
        logger.info(
            f"Successfully stored pickle file as large object with OID {lobj_pkl.oid}"
        )

        if progress_bar:
            progress_bar.update(5)

        # Create the vector store record with large object references
        logger.info("Creating vector store record in database")
        vector_store = VectorStore(
            name=name,
            version=version,
            model_name=model_name,
            data=json.dumps(
                large_objects
            ),  # Store large object IDs instead of raw data
            created_at=datetime.now(timezone.utc),
        )

        # Add to session and commit
        session.add(vector_store)
        try:
            session.commit()
            logger.info("Database commit successful")
        except SQLAlchemyError as e:
            logger.error(f"Database error during commit: {str(e)}")
            session.rollback()
            raise

        if progress_bar:
            progress_bar.update(100 - progress_bar.n)

        logger.info(f"Successfully saved vector store '{name}' (version {version})")
        return cast(UUID, vector_store.id)

    except Exception as e:
        logger.error(f"Failed to save vector store '{name}': {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        session.rollback()
        raise


@retry_on_connection_error()
def load_vector_store(
    session: Any, name: str, output_dir: str, version: Optional[str] = None
) -> bool:
    """Load a vector store from the database and save it to the specified directory."""
    logger.info(
        f"Loading vector store '{name}'{f' version {version}' if version else ' (latest version)'}"
    )
    try:
        query = select(VectorStore).where(VectorStore.name == name)
        if version:
            query = query.where(VectorStore.version == version)
        else:
            query = query.order_by(VectorStore.created_at.desc())

        vector_store = session.execute(query).scalar_one_or_none()
        if not vector_store:
            logger.warning(f"Vector store '{name}' not found in database")
            return False

        logger.info(f"Found vector store '{name}' version {vector_store.version}")

        # Create vector_db subdirectory in output_dir
        vector_db_dir = os.path.join(output_dir, "vector_db")
        os.makedirs(vector_db_dir, exist_ok=True)

        try:
            # Parse the JSON data to get large object OIDs
            object_ids = json.loads(vector_store.data)
            logger.info(f"Retrieved large object IDs: {object_ids}")

            # Get raw connection
            connection = session.connection().connection

            # Process each large object
            for file_name, oid in object_ids.items():
                logger.info(f"Retrieving large object for {file_name} with OID {oid}")
                file_path = os.path.join(vector_db_dir, file_name)

                try:
                    # Open the large object for reading
                    lobj = connection.lobject(oid, "rb")
                    file_size = lobj.seek(0, 2)  # Seek to end to get size
                    lobj.seek(0)  # Reset to beginning

                    logger.info(
                        f"Large object size for {file_name}: {file_size/1024/1024:.1f} MB"
                    )

                    # Read and write in chunks
                    with open(file_path, "wb") as f:
                        bytes_read = 0
                        start_time = time.time()
                        while chunk := lobj.read(CHUNK_SIZE):
                            f.write(chunk)
                            bytes_read += len(chunk)
                            if bytes_read % (10 * CHUNK_SIZE) == 0:
                                elapsed = time.time() - start_time
                                logger.info(
                                    f"Read {bytes_read/1024/1024:.1f} MB of {file_size/1024/1024:.1f} MB "
                                    f"for {file_name} ({bytes_read/file_size*100:.1f}%) "
                                    f"in {elapsed:.1f}s ({bytes_read/1024/1024/elapsed:.1f} MB/s)"
                                )

                    logger.info(
                        f"Successfully wrote {file_name} ({os.path.getsize(file_path)/1024/1024:.1f} MB)"
                    )

                except Exception as e:
                    logger.error(
                        f"Error reading large object for {file_name}: {str(e)}"
                    )
                    raise

            # Verify FAISS index integrity
            try:
                faiss_path = os.path.join(vector_db_dir, "index.faiss")
                index = faiss.read_index(faiss_path)
                logger.info(f"Successfully verified FAISS index: {index}")
            except Exception as e:
                logger.error(f"Failed to verify FAISS index: {str(e)}")
                raise ValueError(f"FAISS index verification failed: {str(e)}")

            logger.info(f"Successfully loaded vector store '{name}' to {vector_db_dir}")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse vector store data: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    except SQLAlchemyError as e:
        logger.error(f"Database error while loading vector store '{name}': {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading vector store '{name}': {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def import_vector_stores(
    database_url: str, base_path: str, version: str, model_name: str, store_name: str
) -> Dict[str, bool]:
    """Import vector store from a specific directory to the database using fresh connections."""
    results = {}

    # Look for vector_db directory
    vector_db_path = os.path.join(base_path, "vector_db")

    if not os.path.exists(vector_db_path):
        logger.warning(f"Vector store directory not found: {vector_db_path}")
        return results

    # Check if the directory contains the required files
    if os.path.exists(os.path.join(vector_db_path, "index.faiss")) and os.path.exists(
        os.path.join(vector_db_path, "index.pkl")
    ):
        try:
            # Use a fresh connection for storing the vector store
            with get_fresh_connection(database_url, f"Import {store_name}") as session:
                vector_store_id = save_vector_store(
                    session, vector_db_path, store_name, version, model_name
                )
                results[store_name] = True
                logger.info(f"Successfully imported vector store: {store_name}")

                # Verify the import with a separate connection
                with get_fresh_connection(
                    database_url, f"Verify {store_name}"
                ) as verify_session:
                    output_dir = os.path.join("temp", store_name)
                    os.makedirs(output_dir, exist_ok=True)
                    success = load_vector_store(
                        verify_session, store_name, output_dir, version
                    )

                    if not success:
                        logger.warning(
                            f"Verification did not find vector store: {store_name}"
                        )
                        results[store_name] = False

        except Exception as e:
            logger.error(f"Failed to import vector store: {str(e)}")
            logger.error(traceback.format_exc())
            results[store_name] = False
    else:
        logger.warning(f"No valid vector store found in {vector_db_path}")

    return results


def process_vector_store(database_url, store_name, store_path, version, model_name):
    """Process a single vector store with improved connection management and error handling."""
    logger.info(f"Processing vector store '{store_name}' from {store_path}")

    with tqdm(total=100, desc=f"Processing {store_name}", unit="%") as progress_bar:
        try:
            # First check if store already exists using a quick connection
            with get_fresh_connection(
                database_url, f"Check if {store_name} exists"
            ) as check_session:
                existing = check_session.execute(
                    select(VectorStore).where(
                        VectorStore.name == store_name, VectorStore.version == version
                    )
                ).scalar_one_or_none()

            if existing:
                logger.warning(
                    f"Vector store '{store_name}' version {version} already exists, skipping..."
                )
                progress_bar.update(100)
                return False

            progress_bar.update(10)
            logger.info(f"Starting import of '{store_name}' version {version}")

            # Process the import with a dedicated connection for saving
            with get_fresh_connection(
                database_url, f"Save {store_name}"
            ) as save_session:
                try:
                    vector_store_id = save_vector_store(
                        session=save_session,
                        vector_db_path=store_path,
                        name=store_name,
                        version=version,
                        model_name=model_name,
                        progress_bar=progress_bar,
                    )
                    logger.info(
                        f"Successfully saved vector store '{store_name}' with ID {vector_store_id}"
                    )
                except Exception as e:
                    logger.error(f"Error during save: {str(e)}")
                    logger.error(traceback.format_exc())
                    return False

            # Verify with a separate connection
            if vector_store_id:
                with get_fresh_connection(
                    database_url, f"Verify {store_name}"
                ) as verify_session:
                    try:
                        output_dir = os.path.join("temp", store_name)
                        os.makedirs(output_dir, exist_ok=True)
                        success = load_vector_store(
                            session=verify_session,
                            name=store_name,
                            output_dir=output_dir,
                            version=version,
                        )
                        progress_bar.update(20)

                        if success:
                            logger.info(
                                f"Successfully imported and verified {store_name}"
                            )
                            return True
                        else:
                            logger.error(f"Verification failed for {store_name}")
                            return False
                    except Exception as e:
                        logger.error(f"Error during verification: {str(e)}")
                        logger.error(traceback.format_exc())
                        return False
            else:
                logger.error(f"Import failed for {store_name}")
                return False

        except Exception as e:
            logger.error(
                f"Unexpected error in process_vector_store for {store_name}: {str(e)}"
            )
            logger.error(traceback.format_exc())
            return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Database connection string with SSL and timeout settings
    DATABASE_URL = (
        "postgresql://owluser:NyCINUy7Un3JjE28Md3mRjpg5Dd4aKEy@"
        "dpg-cvn7c2ngi27c73bi26hg-a.frankfurt-postgres.render.com/owlai_db"
        "?sslmode=require&connect_timeout=30"
    )

    # Store names in dict keys match the names used in the database
    stores = {
        "rag-fr-general-law": "data/legal-rag-tmp/general",
        "rag-fr-tax-law": "data/legal-rag/fiscal",
        "rag-fr-admin-law": "data/legal-rag/admin",
    }

    try:
        # Process each store in sequence, with validation first
        for store_name, base_path in stores.items():
            vector_db_path = os.path.join(base_path, "vector_db")
            faiss_path = os.path.join(vector_db_path, "index.faiss")
            pkl_path = os.path.join(vector_db_path, "index.pkl")

            logger.info(f"Validating paths for {store_name}:")
            logger.info(f"  Base path: {base_path}")
            logger.info(f"  Vector DB path: {vector_db_path}")
            logger.info(f"  FAISS index: {faiss_path}")
            logger.info(f"  Pickle file: {pkl_path}")

            if not os.path.exists(base_path):
                raise FileNotFoundError(f"Base path does not exist: {base_path}")
            if not os.path.exists(vector_db_path):
                raise FileNotFoundError(
                    f"vector_db directory not found: {vector_db_path}"
                )
            if not os.path.exists(faiss_path):
                raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

        # If validation passes, process each store with a separate connection
        for store_name, store_path in stores.items():
            success = process_vector_store(
                database_url=DATABASE_URL,
                store_name=store_name,
                store_path=os.path.join(store_path, "vector_db"),
                version="0.3.1",
                model_name="thenlper/gte-small",
            )

            if not success:
                logger.error(f"Failed to process {store_name}")
            else:
                logger.info(f"Successfully processed {store_name}")

    except Exception as e:
        logger.error(f"Error during vector store processing: {str(e)}")
        logger.error(traceback.format_exc())
        raise
