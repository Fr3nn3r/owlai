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
from sqlalchemy.exc import SQLAlchemyError
from owlai.dbmodels import VectorStore
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)


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


def save_vector_store(
    session: Any,
    vector_db_path: str,
    name: str,
    version: str,
    model_name: str,
    progress_bar: Optional[tqdm] = None,
) -> UUID:
    """Save a complete vector store (both index.faiss and index.pkl) to the database."""
    logger.info(f"Saving vector store '{name}' version {version} to database")

    try:
        # First encode the files
        encoded_data = encode_vector_store_files(vector_db_path)

        # Calculate total size for progress monitoring
        total_size = sum(len(content) for content in encoded_data.values())
        logger.info(f"Total data size: {total_size/1024/1024:.1f} MB")

        if progress_bar:
            progress_bar.set_description(f"Encoding {name}")
            progress_bar.update(10)

        # Create the vector store record first
        vector_store = VectorStore(
            name=name,
            version=version,
            model_name=model_name,
            data=json.dumps({}),  # Empty data initially
            created_at=datetime.now(timezone.utc),
        )
        session.add(vector_store)
        session.flush()

        if progress_bar:
            progress_bar.set_description(f"Created record for {name}")
            progress_bar.update(10)

        # Split data into smaller chunks (5MB)
        chunks = chunk_large_data(
            encoded_data, chunk_size=5 * 1024 * 1024
        )  # 5MB chunks
        chunk_size = len(chunks)

        logger.info(f"Split data into {chunk_size} chunks of 5MB each")

        # Create a nested progress bar for chunk uploads
        chunk_progress = tqdm(
            total=chunk_size, desc="Uploading chunks", unit="chunk", leave=False
        )

        # Upload chunks with progress monitoring
        accumulated_data = {"index.faiss": "", "index.pkl": ""}
        last_commit = 0  # Track when we last committed

        for i, chunk in enumerate(chunks):
            # Accumulate chunk data
            file_name = chunk["file_name"]
            accumulated_data[file_name] += chunk["data"]
            chunk_progress.update(1)

            # Update accumulated size for this file
            current_size = len(accumulated_data[file_name])

            # If we've accumulated 5MB or this is the last chunk, send the data
            if current_size >= 5 * 1024 * 1024 or i == len(chunks) - 1:
                try:
                    # Update the vector store with accumulated data
                    setattr(vector_store, "data", json.dumps(accumulated_data))
                    session.merge(vector_store)
                    session.flush()

                    # Clear accumulated data for this file
                    accumulated_data[file_name] = ""

                    # Update main progress bar
                    if progress_bar:
                        progress = min(80, 20 + (i + 1) / chunk_size * 60)
                        progress_bar.update(progress - progress_bar.n)

                    # Commit every 20 chunks to avoid long-running transactions
                    if i - last_commit >= 20:
                        session.commit()
                        last_commit = i

                except SQLAlchemyError as e:
                    chunk_progress.close()
                    logger.error(f"Failed uploading chunk {i+1}/{chunk_size}: {str(e)}")
                    raise

        chunk_progress.close()

        # Final commit
        session.commit()

        if progress_bar:
            progress_bar.set_description(f"Completed {name}")
            progress_bar.update(100 - progress_bar.n)

        logger.info(f"Successfully saved vector store '{name}' (version {version})")
        return cast(UUID, vector_store.id)

    except SQLAlchemyError as e:
        logger.error(f"Database error while saving vector store '{name}': {str(e)}")
        session.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error while saving vector store '{name}': {str(e)}")
        session.rollback()
        raise


def load_vector_store(
    session: Any, name: str, output_dir: str, version: Optional[str] = None
) -> bool:
    """Load a vector store from the database and save it to the specified directory.

    Args:
        session: SQLAlchemy session
        name: Name identifier of the vector store
        output_dir: Directory where to save the vector store files
        version: Optional specific version to load (loads latest if not specified)

    Returns:
        True if successful, False if not found
    """
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

        # Decode files into vector_db directory
        decode_vector_store_files(vector_store.data, vector_db_dir)
        logger.info(f"Successfully loaded vector store '{name}' to {vector_db_dir}")
        return True

    except SQLAlchemyError as e:
        logger.error(f"Database error while loading vector store '{name}': {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading vector store '{name}': {str(e)}")
        raise


def import_vector_stores(
    session: Any, base_path: str, version: str, model_name: str, store_name: str
) -> Dict[str, bool]:
    """Import vector store from a specific directory to the database.

    Args:
        session: SQLAlchemy session
        base_path: Base path containing the vector store files
        version: Version string to assign
        model_name: Name of the model used to create embeddings
        store_name: Name to use for the vector store in the database

    Returns:
        Dictionary mapping store names to import success status
    """
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
            save_vector_store(session, vector_db_path, store_name, version, model_name)
            results[store_name] = True
            logger.info(f"Successfully imported vector store: {store_name}")
        except Exception as e:
            logger.error(f"Failed to import vector store: {str(e)}")
            results[store_name] = False
    else:
        logger.warning(f"No valid vector store found in {vector_db_path}")

    return results


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

    # Configure engine with proper settings for large data handling
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,  # Limit concurrent connections
        max_overflow=10,  # Allow up to 10 additional connections
        pool_timeout=30,  # Wait up to 30 seconds for a connection
        pool_recycle=1800,  # Recycle connections after 30 minutes
    )
    Session = sessionmaker(bind=engine)

    # Store names in dict keys match the names used in the database
    stores = {
        "rag-fr-general-law": "data/legal-rag-tmp/general",
        "rag-fr-tax-law": "data/legal-rag/fiscal",
        "rag-fr-admin-law": "data/legal-rag/admin",
    }

    from tqdm import tqdm
    import time
    from contextlib import contextmanager
    from sqlalchemy.exc import OperationalError, SQLAlchemyError

    @contextmanager
    def get_db_session():
        """Context manager for database sessions with automatic cleanup"""
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def retry_on_db_error(func, max_retries=3, delay=5):
        """Decorator to retry database operations with exponential backoff"""

        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except OperationalError as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(
                        f"Database operation failed, retrying in {delay} seconds... Error: {e}"
                    )
                    time.sleep(delay * (2**attempt))
            return None

        return wrapper

    @retry_on_db_error
    def process_vector_store(session, store_name, store_path, version, model_name):
        """Process a single vector store with progress monitoring"""
        logger.info(f"Processing vector store '{store_name}' from {store_path}")

        with tqdm(total=100, desc=f"Processing {store_name}", unit="%") as progress_bar:
            # Check if store already exists
            existing = session.execute(
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

            # Import the store with progress monitoring
            try:
                vector_store_id = save_vector_store(
                    session=session,
                    vector_db_path=store_path,
                    name=store_name,
                    version=version,
                    model_name=model_name,
                    progress_bar=progress_bar,  # Pass the progress bar
                )

                # Verify the import
                if vector_store_id:
                    output_dir = os.path.join("temp", store_name)
                    os.makedirs(output_dir, exist_ok=True)
                    success = load_vector_store(
                        session=session,
                        name=store_name,
                        output_dir=output_dir,
                        version=version,
                    )
                    progress_bar.update(20)

                    if success:
                        logger.info(f"Successfully imported and verified {store_name}")
                        return True
                    else:
                        logger.error(f"Verification failed for {store_name}")
                        return False
                else:
                    logger.error(f"Import failed for {store_name}")
                    return False

            except Exception as e:
                logger.error(f"Error processing {store_name}: {str(e)}")
                return False

    try:
        # Process each store in sequence
        with get_db_session() as session:
            # First validate all paths before starting
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

            # If validation passes, process stores
            for store_name, store_path in stores.items():
                success = process_vector_store(
                    session=session,
                    store_name=store_name,
                    store_path=os.path.join(
                        store_path, "vector_db"
                    ),  # Add vector_db to path
                    version="0.3.1",
                    model_name="thenlper/gte-small",
                )

                if not success:
                    raise RuntimeError(f"Failed to process {store_name}")

                logger.info(f"Successfully processed {store_name}")

    except Exception as e:
        logger.error(f"Error during vector store processing: {str(e)}")
        raise
