"""
Utility to manage vector store persistence in the database.
Handles saving and loading complete vector store folders (index.faiss and index.pkl) to/from PostgreSQL.
"""

import os
import base64
import logging
from datetime import datetime
import json
from typing import Optional, Any, cast, Dict, Tuple
from uuid import UUID
import faiss
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from owlai.dbmodels import VectorStore

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


def save_vector_store(
    session: Any, vector_db_path: str, name: str, version: str, model_name: str
) -> UUID:
    """Save a complete vector store (both index.faiss and index.pkl) to the database.

    Args:
        session: SQLAlchemy session
        vector_db_path: Path to the vector_db directory containing the files
        name: Name identifier for the vector store
        version: Version string
        model_name: Name of the model used to create embeddings

    Returns:
        UUID of the created vector store record
    """
    logger.info(f"Saving vector store '{name}' version {version} to database")
    try:
        encoded_data = encode_vector_store_files(vector_db_path)
        vector_store = VectorStore(
            name=name,
            version=version,
            model_name=model_name,
            data=json.dumps(encoded_data),  # Store as JSON string
            created_at=datetime.utcnow(),
        )
        session.add(vector_store)
        session.commit()
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
        decode_vector_store_files(vector_store.data, output_dir)
        logger.info(f"Successfully loaded vector store '{name}' to {output_dir}")
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

    # Check if the directory contains the required files directly
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

    # Example usage
    DATABASE_URL = "postgresql+psycopg2://owluser:owlsrock@localhost:5432/owlai_db"
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = None

    # Store names in dict keys match the names used in the database
    stores = {
        "rag-fr-general-law-v1": "data/legal-rag-tmp/general",
        "rag-fr-tax-law-v1": "data/legal-rag/fiscal",
        "rag-fr-admin-law-v1": "data/legal-rag/admin",
    }

    try:
        session = Session()

        # Import stores
        for store_name, store_path in stores.items():
            logger.info(f"Processing vector store '{store_name}' from {store_path}")
            try:
                results = import_vector_stores(
                    session=session,
                    base_path=store_path,
                    version="0.3.1",
                    model_name="thenlper/gte-small",
                    store_name=store_name,
                )

                if results:  # If any results were returned
                    if all(results.values()):
                        logger.info(f"Successfully imported vector store: {store_name}")
                    else:
                        logger.warning(f"Failed to import vector store: {store_name}")
                else:
                    logger.warning(f"No vector store found in {store_path}")

            except Exception as e:
                logger.error(f"Error processing {store_name}: {str(e)}")
                continue

        # Load stores to verify
        output_base = "temp"
        os.makedirs(output_base, exist_ok=True)

        for store_name in stores.keys():
            output_dir = os.path.join(output_base, store_name)
            success = load_vector_store(
                session=session, name=store_name, output_dir=output_dir, version="0.3.1"
            )
            if success:
                logger.info(f"Successfully verified {store_name}")
            else:
                logger.warning(f"Failed to verify {store_name}")

    except Exception as e:
        logger.error(f"Global error: {str(e)}")
    finally:
        if session:
            session.close()
