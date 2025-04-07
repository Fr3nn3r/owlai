import os
import faiss
import pickle
import logging
import time
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from owlai.dbmodels import VectorStore
from owlai.vector_store_manager import (
    decode_vector_store_files,
    get_fresh_connection,
    retry_on_connection_error,
    CHUNK_SIZE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database connection settings
DATABASE_URL = (
    "postgresql://owluser:NyCINUy7Un3JjE28Md3mRjpg5Dd4aKEy@"
    "dpg-cvn7c2ngi27c73bi26hg-a.frankfurt-postgres.render.com/owlai_db"
    "?sslmode=require&connect_timeout=30"
)


@retry_on_connection_error()
def download_large_object(connection, oid: int, output_path: str) -> bool:
    """Download a large object from the database in chunks."""
    try:
        # Open the large object for reading
        lobj = connection.lobject(oid, "rb")
        file_size = lobj.seek(0, 2)  # Seek to end to get size
        lobj.seek(0)  # Reset to beginning

        logger.info(f"Large object size: {file_size/1024/1024:.1f} MB")

        # Read and write in chunks
        with open(output_path, "wb") as f:
            bytes_read = 0
            start_time = time.time()
            while chunk := lobj.read(CHUNK_SIZE):
                f.write(chunk)
                bytes_read += len(chunk)
                if bytes_read % (10 * CHUNK_SIZE) == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Read {bytes_read/1024/1024:.1f} MB of {file_size/1024/1024:.1f} MB "
                        f"({bytes_read/file_size*100:.1f}%) "
                        f"in {elapsed:.1f}s ({bytes_read/1024/1024/elapsed:.1f} MB/s)"
                    )

        logger.info(
            f"Successfully wrote file ({os.path.getsize(output_path)/1024/1024:.1f} MB)"
        )
        return True

    except Exception as e:
        logger.error(f"Error downloading large object: {str(e)}")
        return False


@retry_on_connection_error()
def download_vector_store(store_name: str, output_dir: str) -> bool:
    """Download vector store files from database and create FAISS index."""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        vector_db_dir = os.path.join(output_dir, "vector_db")
        os.makedirs(vector_db_dir, exist_ok=True)

        # Use get_fresh_connection for better connection handling
        with get_fresh_connection(DATABASE_URL, f"Download {store_name}") as session:
            try:
                # Query the latest version of the vector store
                logger.info(f"Querying vector store '{store_name}'...")
                query = (
                    select(VectorStore)
                    .where(VectorStore.name == store_name)
                    .order_by(VectorStore.created_at.desc())
                )
                vector_store = session.execute(query).scalar_one_or_none()

                if not vector_store:
                    logger.error(f"Vector store '{store_name}' not found")
                    return False

                logger.info(
                    f"Found vector store '{store_name}' version {vector_store.version}"
                )

                # Parse the JSON data to get large object OIDs
                import json

                object_ids = json.loads(str(vector_store.data))
                logger.info(f"Retrieved large object IDs: {object_ids}")

                # Get raw connection
                connection = session.connection().connection

                # Download each file separately
                for file_name, oid in object_ids.items():
                    logger.info(f"Downloading {file_name} (OID: {oid})...")
                    file_path = os.path.join(vector_db_dir, file_name)

                    if not download_large_object(connection, int(oid), file_path):
                        logger.error(f"Failed to download {file_name}")
                        return False

                # Verify FAISS index
                logger.info("Loading FAISS index to verify...")
                faiss_path = os.path.join(vector_db_dir, "index.faiss")
                index = faiss.read_index(faiss_path)

                # Verify pickle file
                logger.info("Loading pickle file to verify...")
                pkl_path = os.path.join(vector_db_dir, "index.pkl")
                with open(pkl_path, "rb") as f:
                    pkl_data = pickle.load(f)

                logger.info(f"Successfully loaded vector store:")
                logger.info(f"- FAISS index dimensions: {index.d}")
                logger.info(f"- FAISS index total vectors: {index.ntotal}")
                logger.info(f"- Pickle data type: {type(pkl_data)}")
                if hasattr(pkl_data, "__len__"):
                    logger.info(f"- Pickle data length: {len(pkl_data)}")

                return True

            except Exception as e:
                logger.error(f"Error during vector store download: {str(e)}")
                raise

    except Exception as e:
        logger.error(f"Error downloading vector store: {str(e)}")
        return False


if __name__ == "__main__":
    # Create temp directory
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Download vector store
    store_name = (
        "rag-fr-general-law"  # You can change this to download a different store
    )
    success = download_vector_store(store_name, temp_dir)

    if success:
        logger.info(f"Vector store downloaded successfully to {temp_dir}")
    else:
        logger.error("Failed to download vector store")
