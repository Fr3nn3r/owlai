"""
OwlAI Data Store Module for RAG

Note: We are using Pydantic v1 because it's required by langchain-core and other LangChain components.
This is a temporary solution until LangChain fully supports Pydantic v2.
The deprecation warnings are suppressed in pytest configuration.
"""

print("Loading data module")

from typing import Optional, List, Tuple, Any, Callable, Dict, Literal
import os
import logging
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from pydantic import BaseModel

from tqdm import tqdm
import traceback

from owlai.services.system import track_time, Session
from owlai.services.parser import DefaultParser, create_instance

# Get logger using the module name

logger = logging.getLogger(__name__)


class RAGDataStore(BaseModel):
    """
    RAGDataStore is a class that manages the data store for the RAG system.
    It is responsible for loading and saving the vector store, as well as for parsing the documents.
    """

    name: str
    version: str
    input_data_folder: Optional[str] = None
    cache_data_folder: str = "data/cache"
    parser: DefaultParser

    _vector_store: Optional[FAISS] = None
    _images_folder: str = "images"
    _in_store_documents_folder: str = "in_store"
    _vector_store_folder: str = "vector_db"
    _db_session: Optional[Any] = None

    def __init__(self, **kwargs):
        """
        Initialize the RAGDataStore with the provided configuration.

        This constructor handles the initialization of the RAGDataStore object,
        setting up all necessary parameters for document processing and vector storage.
        """
        super().__init__(**kwargs)
        # Normalize paths immediately upon initialization
        if self.input_data_folder:
            self.input_data_folder = os.path.normpath(self.input_data_folder)
            if not os.path.exists(self.input_data_folder):
                logger.warning(
                    f"Input data folder does not exist: {self.input_data_folder}"
                )

        self.cache_data_folder = os.path.normpath(self.cache_data_folder)

        # Ensure cache_data_folder exists
        os.makedirs(self.cache_data_folder, exist_ok=True)

        self._db_session = Session()

        if self.parser.implementation == "DefaultParser":
            logger.debug(f"Using DefaultParser")
        else:
            logger.debug(f"Using custom parser: {self.parser.implementation}")
            new_parser = create_instance(self.parser.implementation, **kwargs["parser"])
            if not isinstance(new_parser, DefaultParser):
                raise Exception(f"{self.parser.implementation} is not a valid parser")
            self.parser = new_parser

    def _normalize_path(self, *paths: str) -> str:
        """
        Normalize and join paths in a platform-independent way.
        """
        return os.path.normpath(os.path.join(*paths))

    def load_vector_store(
        self, embedding_model: HuggingFaceEmbeddings
    ) -> Optional[FAISS]:
        """
        Load vector store with the following priority:
        1. Try loading from cache folder (cache_data_folder/name/vector_db)
        2. Try loading from database and save to cache if successful
        3. Try loading from input folder and save to cache if successful
        4. Return None if no vector store is found

        Args:
            embedding_model: HuggingFaceEmbeddings instance to use for the vector store

        Returns:
            Optional FAISS vector store
        """
        # 1. Try loading from cache first
        cache_path = self._normalize_path(
            self.cache_data_folder, self.name, self._vector_store_folder
        )
        logger.debug(f"Looking for cached vector database at: {cache_path}")

        if os.path.exists(cache_path):
            logger.info(f"Loading vector database from cache: {cache_path}")
            try:
                return FAISS.load_local(
                    cache_path,
                    embedding_model,
                    distance_strategy=DistanceStrategy.COSINE,
                    allow_dangerous_deserialization=True,
                )
            except Exception as e:
                logger.error(f"Failed to load from cache: {str(e)}")

        # 2. Try loading from database
        if self._db_session is not None:
            logger.debug(f"Attempting to load vector store '{self.name}' from database")
            try:
                from sqlalchemy import select
                from owlai.dbmodels import VectorStore
                from owlai.vector_store_manager import decode_vector_store_files

                query = select(VectorStore).where(VectorStore.name == self.name)
                if self.version:
                    query = query.where(VectorStore.version == self.version)
                else:
                    query = query.order_by(VectorStore.created_at.desc())

                vector_store_record = self._db_session.execute(
                    query
                ).scalar_one_or_none()
                if vector_store_record:
                    logger.info(f"Found vector store '{self.name}' in database")

                    # Create temporary directory for database files
                    temp_db_dir = self._normalize_path(cache_path)
                    os.makedirs(os.path.dirname(temp_db_dir), exist_ok=True)

                    # Decode files from database
                    decode_vector_store_files(
                        vector_store_record.data, temp_db_dir, self._db_session
                    )

                    # Load the vector store
                    vector_store = FAISS.load_local(
                        temp_db_dir,
                        embedding_model,
                        distance_strategy=DistanceStrategy.COSINE,
                        allow_dangerous_deserialization=True,
                    )

                    # Save to cache for future use
                    logger.info(f"Saving database vector store to cache: {cache_path}")
                    vector_store.save_local(cache_path)

                    return vector_store
                else:
                    logger.error(
                        f"No vector store found in database for {self.name} {self.version}"
                    )
            except Exception as e:
                logger.error(f"Failed to load from database: {str(e)}")

        else:
            logger.error(f"Database session in no initialized.")

        # 3. Try loading from input folder
        if not self.input_data_folder:
            logger.warning("No input data folder defined, skipping input folder load")
            return None

        input_path = self._normalize_path(
            self.input_data_folder, self._vector_store_folder
        )
        logger.debug(f"Looking for vector database at: {input_path}")

        if os.path.exists(input_path):
            logger.info(f"Loading vector database from input folder: {input_path}")
            try:
                vector_store = FAISS.load_local(
                    input_path,
                    embedding_model,
                    distance_strategy=DistanceStrategy.COSINE,
                    allow_dangerous_deserialization=True,
                )

                # Save to cache for next time
                logger.info(f"Saving vector database to cache: {cache_path}")
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                vector_store.save_local(cache_path)

                return vector_store
            except Exception as e:
                logger.error(f"Failed to load from input folder: {str(e)}")

        logger.warning("Vector database not found in cache, database or input folder")
        return None

    def load_dataset(
        self,
        embedding_model: HuggingFaceEmbeddings,
        metadata_extractor: Optional[Callable] = None,
        document_curator: Optional[Callable] = None,
    ) -> Optional[FAISS]:
        """
        Loads an existing vector store if exists in the input_data_folder.
        Processes documents and adds them to the store.
        Processed documents are moved to the 'in_store' folder.
        New vector stores are saved both to input folder and cache.

        Args:
            embedding_model: The embedding model to use
            metadata_extractor: Optional callback function for extracting metadata
            document_curator: Optional callback function for curating documents

        Returns:
            FAISS vector store or None if no documents were processed
        """
        if not self.input_data_folder:
            logger.warning("No input data folder defined")
            return None

        # First try loading existing vector store (will check cache first)
        vector_store = self.load_vector_store(embedding_model)

        # Get list of PDF and text files
        try:
            files = [
                f
                for f in os.listdir(self.input_data_folder)
                if f.lower().endswith((".pdf", ".txt"))
            ]
        except Exception as e:
            logger.error(f"Error listing directory {self.input_data_folder}: {str(e)}")
            return vector_store

        logger.debug(
            f"Found {len(files)} new documents to process in {self.input_data_folder}"
        )

        if len(files) > 0:
            # Create in_store_folder before processing files
            in_store_folder = self._normalize_path(
                self.input_data_folder, self._in_store_documents_folder
            )
            os.makedirs(in_store_folder, exist_ok=True)

            # Process each file individually
            with track_time(f"Loading {len(files)} document(s) into vector database"):
                for filename in files:
                    # Normalize paths for source and destination
                    source_path = self._normalize_path(self.input_data_folder, filename)
                    dest_path = self._normalize_path(in_store_folder, filename)

                    logger.info(
                        f"Processing file: {filename} size: {os.path.getsize(source_path)}"
                    )

                    try:
                        split_docs = self.parser.load_and_split_document(
                            source_path,
                            filename,
                            embedding_model.model_name,
                            metadata_extractor,
                            document_curator,
                        )

                        # Create or update vector store
                        if vector_store is None:
                            logger.debug("Creating new vector store")
                            vector_store = FAISS.from_documents(
                                split_docs,
                                embedding_model,
                                distance_strategy=DistanceStrategy.COSINE,
                            )
                        else:
                            logger.debug("Loading new documents into vector store")
                            batch_store = FAISS.from_documents(
                                split_docs,
                                embedding_model,
                                distance_strategy=DistanceStrategy.COSINE,
                            )
                            logger.debug("Merging new documents into vector store")
                            vector_store.merge_from(batch_store)

                        # Move processed file to 'in_store' folder
                        os.rename(source_path, dest_path)

                    except Exception as e:
                        logger.error(f"Error processing {filename}: {str(e)}")
                        logger.error(f"Error details: {traceback.format_exc()}")
                        continue

            # Save to both input folder and cache if we have a vector store and processed files
            if vector_store is not None and len(files) > 0:
                # Save to input folder
                input_vector_path = self._normalize_path(
                    self.input_data_folder, self._vector_store_folder
                )
                os.makedirs(input_vector_path, exist_ok=True)
                vector_store.save_local(input_vector_path)
                logger.info(
                    f"Vector database saved to input folder: {input_vector_path}"
                )

                # Save to cache
                cache_vector_path = self._normalize_path(
                    self.cache_data_folder, self.name, self._vector_store_folder
                )
                os.makedirs(os.path.dirname(cache_vector_path), exist_ok=True)
                vector_store.save_local(cache_vector_path)
                logger.info(f"Vector database saved to cache: {cache_vector_path}")

        return vector_store
