"""
Utility to manage vector store persistence in the database.
Handles saving and loading FAISS indexes to/from PostgreSQL.
"""

import os
import base64
import logging
from datetime import datetime
from typing import Optional, Any, cast, Dict
from uuid import UUID
import faiss
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from owlai.dbmodels import VectorStore

# Configure logging
logger = logging.getLogger(__name__)


def encode_vector_store(index: Any) -> str:
    """Encode a FAISS index to base64 string."""
    logger.debug("Encoding FAISS index to base64")
    try:
        buffer = faiss.serialize_index(index)
        encoded = base64.b64encode(buffer).decode("utf-8")
        logger.debug("Successfully encoded FAISS index")
        return encoded
    except Exception as e:
        logger.error(f"Failed to encode FAISS index: {str(e)}")
        raise


def decode_vector_store(encoded_data: str) -> Any:
    """Decode a base64 string back to FAISS index."""
    logger.debug("Decoding base64 string to FAISS index")
    try:
        buffer = base64.b64decode(encoded_data.encode("utf-8"))
        index = faiss.deserialize_index(buffer)
        logger.debug("Successfully decoded FAISS index")
        return index
    except Exception as e:
        logger.error(f"Failed to decode FAISS index: {str(e)}")
        raise


def save_vector_store(
    session: Any, index: Any, name: str, version: str, model_name: str
) -> UUID:
    """Save a FAISS index to the database.

    Args:
        session: SQLAlchemy session
        index: FAISS index to save
        name: Name identifier for the vector store
        version: Version string
        model_name: Name of the model used to create embeddings

    Returns:
        UUID of the created vector store record
    """
    logger.info(f"Saving vector store '{name}' version {version} to database")
    try:
        encoded_data = encode_vector_store(index)
        vector_store = VectorStore(
            name=name,
            version=version,
            model_name=model_name,
            data=encoded_data,
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
    session: Any, name: str, version: Optional[str] = None
) -> Optional[Any]:
    """Load a FAISS index from the database.

    Args:
        session: SQLAlchemy session
        name: Name identifier of the vector store
        version: Optional specific version to load (loads latest if not specified)

    Returns:
        Loaded FAISS index or None if not found
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
            return None

        logger.info(f"Found vector store '{name}' version {vector_store.version}")
        index = decode_vector_store(vector_store.data)
        logger.info(f"Successfully loaded vector store '{name}'")
        return index

    except SQLAlchemyError as e:
        logger.error(f"Database error while loading vector store '{name}': {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading vector store '{name}': {str(e)}")
        raise


def import_vector_stores(
    session: Any, base_path: str, version: str, model_name: str
) -> Dict[str, bool]:
    """Import vector stores from files into database.

    Args:
        session: SQLAlchemy session
        base_path: Base path containing the vector store files
        version: Version string to assign
        model_name: Name of the model used for embeddings

    Returns:
        Dictionary mapping store names to import success status
    """
    logger.info(f"Starting vector store import process (version {version})")
    stores = {
        "fr-law-admin": "data/legal-rag/admin",
        "fr-law-fiscal": "data/legal-rag/fiscal",
        "fr-law-general": "data/legal-rag-tmp/general",
    }

    results = {}
    total_stores = len(stores)
    imported_count = 0

    for name, path in stores.items():
        # Normalize the path and point to the actual index file
        vector_dir = os.path.normpath(os.path.join(base_path, path, "vector_db"))
        index_file = os.path.join(vector_dir, "index.faiss")
        logger.info(f"Processing vector store '{name}' from {index_file}")

        try:
            if os.path.exists(index_file):
                logger.debug(f"Reading FAISS index from {index_file}")
                index = faiss.read_index(index_file)
                save_vector_store(session, index, name, version, model_name)
                logger.info(f"Successfully imported vector store '{name}'")
                imported_count += 1
                results[name] = True
            else:
                logger.warning(f"Vector store file not found: {index_file}")
                results[name] = False

        except Exception as e:
            logger.error(f"Failed to import vector store '{name}': {str(e)}")
            results[name] = False

        logger.info(f"Progress: {imported_count}/{total_stores} stores imported")

    # Final status report
    success_count = sum(1 for success in results.values() if success)
    logger.info("=" * 50)
    logger.info("Vector Store Import Summary:")
    logger.info(f"Total stores processed: {total_stores}")
    logger.info(f"Successfully imported: {success_count}")
    logger.info(f"Failed imports: {total_stores - success_count}")
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  - {name}: {status}")
    logger.info("=" * 50)

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

    session = Session()
    try:
        results = import_vector_stores(
            session,
            base_path=".",  # Adjust this to your project root
            version="0.3.0",
            model_name="thenlper/gte-small",  # Adjust to your model
        )

        if all(results.values()):
            logger.info("All vector stores were successfully imported!")
        else:
            failed = [name for name, success in results.items() if not success]
            logger.warning(f"Some vector stores failed to import: {', '.join(failed)}")

    except Exception as e:
        logger.error(f"Import process failed: {str(e)}")
    finally:
        session.close()
