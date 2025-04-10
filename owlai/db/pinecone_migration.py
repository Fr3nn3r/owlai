import json
import os
import time
import logging
from typing import List, Dict, Any, Tuple, Optional, Sequence, Union, cast
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from owlai.services.system import is_dev
from urllib.parse import urlparse

# Set up logging

logger = logging.getLogger("PineconeMigration")

# Load environment variables
load_dotenv()

# === CONFIGURATION ===
FAISS_INDEX_PATH = "data/cache/rag-fr-law-complete/vector_db"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "eu-west-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "owlai-law")
EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 100
BACKUP_PATH = "faiss_to_pinecone_backup.jsonl"

# Type alias for Pinecone index (to make linter happy)
PineconeIndex = Any


def extract_documents_from_faiss(faiss_store) -> List[Document]:
    """Extract documents from a FAISS vector store using multiple methods"""
    documents = []

    # Try multiple methods to extract documents
    try:
        # Method 1: Direct access to docstore dictionary (most common)
        if hasattr(faiss_store, "docstore") and hasattr(faiss_store.docstore, "_dict"):
            logger.info("Extracting documents using docstore._dict")
            try:
                # The linter doesn't know about the _dict attribute, but it exists at runtime
                # so we'll use a more general approach to access it
                doc_dict = getattr(faiss_store.docstore, "_dict", {})
                documents = list(doc_dict.values())
            except Exception as e:
                logger.warning(f"Error extracting via docstore._dict: {str(e)}")

        # Method 2: Using index to docstore mapping
        if not documents and hasattr(faiss_store, "index_to_docstore_id"):
            logger.info("Extracting documents using index_to_docstore_id")
            try:
                doc_ids = list(faiss_store.index_to_docstore_id.values())
                documents = [faiss_store.docstore.search(doc_id) for doc_id in doc_ids]
            except Exception as e:
                logger.warning(f"Error extracting via index_to_docstore_id: {str(e)}")

        # Method 3: Alternative docstore access
        if (
            not documents
            and hasattr(faiss_store, "docstore")
            and hasattr(faiss_store.docstore, "documents")
        ):
            logger.info("Extracting documents using docstore.documents")
            try:
                # The linter doesn't know about the documents attribute, but it exists at runtime
                doc_dict = getattr(faiss_store.docstore, "documents", {})
                documents = list(doc_dict.values())
            except Exception as e:
                logger.warning(f"Error extracting via docstore.documents: {str(e)}")

    except Exception as e:
        logger.error(f"Error extracting documents: {str(e)}")

    # Convert mixed types to Documents if needed
    result_docs = []
    for doc in documents:
        if isinstance(doc, Document):
            result_docs.append(doc)
        elif isinstance(doc, str):
            # Convert strings to Documents with minimal metadata
            result_docs.append(Document(page_content=doc, metadata={}))
        else:
            logger.warning(f"Unknown document type: {type(doc)}")

    return result_docs


def load_faiss_index():
    """Load the FAISS index and return the store and documents"""
    logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}...")

    try:
        embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        # Add allow_dangerous_deserialization=True since we trust our own files
        faiss_store = FAISS.load_local(
            FAISS_INDEX_PATH, embedding, allow_dangerous_deserialization=True
        )

        # Extract documents from FAISS store
        documents = extract_documents_from_faiss(faiss_store)

        logger.info(f"Loaded {len(documents)} documents from FAISS index")
        return faiss_store, documents
    except Exception as e:
        logger.error(f"Error loading FAISS index: {str(e)}", exc_info=True)
        raise


def initialize_pinecone() -> PineconeIndex:
    """Initialize connection to Pinecone using compatible methods for different versions"""
    logger.info("Connecting to Pinecone...")

    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY environment variable not set")
        raise ValueError("PINECONE_API_KEY environment variable not set")

    try:
        # Delayed import to handle import error gracefully
        try:
            import pinecone
            import requests
            import json
            from urllib.parse import urlparse

            logger.info(
                f"Imported Pinecone module version: {pinecone.__version__ if hasattr(pinecone, '__version__') else 'unknown'}"
            )

            # Get the full host URL
            host_url = os.getenv(
                "PINECONE_HOST",
                "https://owlai-law-sq85boh.svc.apu-57e2-42f6.pinecone.io",
            )

            # Make sure the URL has the https:// prefix
            if not host_url.startswith(("http://", "https://")):
                host_url = f"https://{host_url}"

            logger.info(f"Using Pinecone host: {host_url}")

            # Create a custom index class that uses the direct URL
            class DirectPineconeIndex:
                def __init__(self, host_url, api_key, index_name):
                    self.host_url = host_url.rstrip("/")
                    self.api_key = api_key
                    self.index_name = index_name
                    self.headers = {
                        "Api-Key": self.api_key,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    }

                def describe_index_stats(self):
                    """Get stats about the index"""
                    url = f"{self.host_url}/describe_index_stats"
                    response = requests.get(url, headers=self.headers)
                    response.raise_for_status()
                    return response.json()

                def upsert(self, vectors, namespace=""):
                    """Upsert vectors to the index"""
                    url = f"{self.host_url}/vectors/upsert"

                    # Convert vectors to the format expected by Pinecone API
                    vectors_data = []
                    for vector_id, embedding, metadata in vectors:
                        vectors_data.append(
                            {
                                "id": vector_id,
                                "values": (
                                    embedding.tolist()
                                    if hasattr(embedding, "tolist")
                                    else embedding
                                ),
                                "metadata": metadata,
                            }
                        )

                    # Prepare request data
                    data = {"vectors": vectors_data}
                    if namespace:
                        data["namespace"] = namespace  # type: ignore

                    # Make the request
                    response = requests.post(url, headers=self.headers, json=data)
                    response.raise_for_status()
                    return response.json()

            # Create an index instance using our custom class
            logger.info(f"Creating direct connection to Pinecone index at {host_url}")
            index = DirectPineconeIndex(host_url, PINECONE_API_KEY, PINECONE_INDEX_NAME)

            # Test the connection
            try:
                stats = index.describe_index_stats()
                logger.info(f"Connected to Pinecone index: {stats}")
            except Exception as e:
                logger.warning(f"Could not get index stats: {str(e)}")
                logger.warning(f"Will continue with the migration anyway")

            return index

        except ImportError as e:
            logger.error(f"Failed to import Pinecone: {str(e)}")
            logger.error("Please run: pip install requests")
            raise

    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}", exc_info=True)
        raise


def process_documents(documents, embedding, index):
    """Process documents, compute embeddings, and upload to Pinecone"""
    logger.info(f"Processing {len(documents)} documents in batches of {BATCH_SIZE}")

    total_uploaded = 0
    start_time = time.time()

    with open(BACKUP_PATH, "w", encoding="utf-8") as backup_file:
        for i in tqdm(range(0, len(documents), BATCH_SIZE)):
            batch_docs = documents[i : i + BATCH_SIZE]
            batch_texts = [doc.page_content for doc in batch_docs]
            batch_metadata = [doc.metadata or {} for doc in batch_docs]

            # Add retry logic for embedding API
            retry_attempts = 3
            batch_embeddings = None

            for attempt in range(retry_attempts):
                try:
                    logger.debug(
                        f"Embedding batch {i//BATCH_SIZE+1}/{len(documents)//BATCH_SIZE+1}"
                    )
                    batch_embeddings = embedding.embed_documents(batch_texts)
                    break
                except Exception as e:
                    logger.warning(f"Embedding attempt {attempt+1} failed: {str(e)}")
                    if attempt < retry_attempts - 1:
                        time.sleep(2**attempt)  # Exponential backoff
                    else:
                        logger.error(
                            f"Failed to embed batch after {retry_attempts} attempts"
                        )
                        raise

            if batch_embeddings is None:
                logger.error("Failed to generate embeddings for batch")
                continue

            items = []
            for j, doc in enumerate(batch_docs):
                doc_id = f"doc-{i+j}"
                vector = batch_embeddings[j]
                metadata = batch_metadata[j]
                text = batch_texts[j]

                # Log document details at debug level
                logger.debug(f"Processing document {doc_id}: {text[:100]}...")

                # Save backup entry with JSON serializable values
                vector_list = vector if isinstance(vector, list) else vector.tolist()
                backup_file.write(
                    json.dumps(
                        {
                            "id": doc_id,
                            "vector": vector_list,
                            "metadata": {k: str(v) for k, v in metadata.items()},
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                # Make sure all metadata is string values for Pinecone
                clean_metadata = {
                    k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                    for k, v in metadata.items()
                }

                # Prepare Pinecone upsert
                items.append((doc_id, vector, {"text": text, **clean_metadata}))

            # Upsert to Pinecone with retry logic
            for attempt in range(retry_attempts):
                try:
                    logger.info(
                        f"Upserting batch {i//BATCH_SIZE+1}/{len(documents)//BATCH_SIZE+1} with {len(items)} vectors"
                    )
                    index.upsert(vectors=items)
                    total_uploaded += len(items)
                    break
                except Exception as e:
                    logger.warning(f"Upsert attempt {attempt+1} failed: {str(e)}")
                    if attempt < retry_attempts - 1:
                        time.sleep(2**attempt)  # Exponential backoff
                    else:
                        logger.error(
                            f"Failed to upsert batch after {retry_attempts} attempts"
                        )
                        raise

    elapsed_time = time.time() - start_time
    logger.info(f"Uploaded {total_uploaded} vectors in {elapsed_time:.2f} seconds")
    logger.info(
        f"Average upload rate: {total_uploaded/elapsed_time:.2f} vectors/second"
    )


def main():
    """Main function to execute the migration process"""
    logger.info("Starting FAISS to Pinecone migration")

    try:
        # 1. Load FAISS index
        faiss_store, documents = load_faiss_index()

        if not documents:
            logger.error("No documents found in FAISS index")
            return 1

        # 2. Initialize Pinecone
        index = initialize_pinecone()

        # 3. Recompute embeddings and upload to Pinecone
        embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        process_documents(documents, embedding, index)

        # 4. Verify migration
        try:
            index_stats = index.describe_index_stats()
            logger.info(f"Final index stats: {index_stats}")
        except Exception as e:
            logger.warning(f"Could not get final index stats: {str(e)}")

        logger.info(f"✅ Migration complete! {len(documents)} docs sent to Pinecone.")
        logger.info(f"📁 Backup saved to `{BACKUP_PATH}`")

    except Exception as e:
        logger.error(f"Migration failed: {str(e)}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
