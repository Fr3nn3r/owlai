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

from owlai.owlsys import track_time
from owlai.parser import DefaultParser, create_instance

# Get logger using the module name

logger = logging.getLogger(__name__)


class RAGDataStore(BaseModel):

    input_data_folder: str

    parser: DefaultParser

    _vector_store: Optional[FAISS] = None
    _images_folder: str = "images"
    _in_store_documents_folder: str = "in_store"
    _vector_store_folder: str = "vector_db"

    def __init__(self, **kwargs):
        """
        Initialize the RAGDataStore with the provided configuration.

        This constructor handles the initialization of the RAGDataStore object,
        setting up all necessary parameters for document processing and vector storage.
        """
        super().__init__(**kwargs)
        # Normalize input path immediately upon initialization
        self.input_data_folder = os.path.normpath(self.input_data_folder)

        # Ensure input_data_folder exists
        if not os.path.exists(self.input_data_folder):
            logger.warning(
                f"Input data folder does not exist: {self.input_data_folder}"
            )

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
        Load vector store from disk using the provided embedding model.

        Args:
            embedding_model: HuggingFaceEmbeddings instance to use for the vector store

        Returns:
            Optional FAISS vector store
        """
        file_path = self._normalize_path(
            self.input_data_folder, self._vector_store_folder
        )
        logger.debug(f"Looking for vector database at: {file_path}")
        FAISS_vector_store = None

        if os.path.exists(file_path):
            logger.debug(f"Loading the vector database from disk: {file_path}")
            FAISS_vector_store = FAISS.load_local(
                file_path,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
                allow_dangerous_deserialization=True,
            )
        else:
            logger.error(f"Vector database not found in {file_path}")

        return FAISS_vector_store

    def load_dataset(
        self,
        embedding_model: HuggingFaceEmbeddings,
        metadata_extractor: Optional[Callable] = None,
        document_curator: Optional[Callable] = None,
    ) -> Optional[FAISS]:
        """
        Loads an existing vector store if exists in the input_data_folder.
        Processes documents and adds them to the store.
        Processed documents are moved to the 'in_store' folder

        Args:
            input_data_folder: Path to the folder containing documents
            embedding_model: The embedding model to use
            chunk_size: Size of text chunks for splitting documents
            metadata_extractor: Optional callback function for extracting metadata

        Returns:
            FAISS vector store or None if no documents were processed
        """
        # Normalize all paths at the start
        vector_db_file_path = self._normalize_path(
            self.input_data_folder, self._vector_store_folder
        )
        in_store_folder = self._normalize_path(
            self.input_data_folder, self._in_store_documents_folder
        )
        vector_store = None

        if os.path.exists(vector_db_file_path):
            logger.debug(
                f"Loading existing vector database from: {vector_db_file_path}"
            )
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
                        try:
                            os.rename(source_path, dest_path)
                            logger.debug(f"Moved {source_path} to {dest_path}")
                        except Exception as move_error:
                            logger.error(
                                f"Error moving file {source_path} to {dest_path}: {str(move_error)}"
                            )
                            # If rename fails, try to copy and then delete
                            try:
                                import shutil

                                shutil.copy2(source_path, dest_path)
                                os.remove(source_path)
                                logger.debug(
                                    f"Successfully copied and deleted {source_path} to {dest_path}"
                                )
                            except Exception as fallback_error:
                                logger.error(
                                    f"Fallback copy-delete also failed for {source_path}: {str(fallback_error)}"
                                )

                    except Exception as e:
                        logger.error(f"Error processing {filename}: {str(e)}")
                        logger.error(f"Error details: {traceback.format_exc()}")
                        continue

            # Save to disk
            if vector_store is not None and len(files) > 0:
                try:
                    vector_store.save_local(vector_db_file_path)
                    logger.info(f"Vector database saved to {vector_db_file_path}")
                except Exception as save_error:
                    logger.error(
                        f"Error saving vector store to {vector_db_file_path}: {str(save_error)}"
                    )

        return vector_store
