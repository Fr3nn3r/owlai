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

    def load_vector_store(
        self, input_data_folder: str, embedding_model: HuggingFaceEmbeddings
    ) -> Optional[FAISS]:
        file_path = os.path.normpath(
            os.path.join(input_data_folder, self._vector_store_folder)
        )
        logger.debug(f"Looking for vector database at: {file_path}")
        FAISS_vector_store = None

        if os.path.exists(file_path):
            with track_time(f"Loading the vector database from disk: {file_path}"):
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
        # Use os.path.join for platform-independent path handling
        # Normalize paths to ensure OS-compatible separators
        input_data_folder = os.path.normpath(self.input_data_folder)
        vector_db_file_path = os.path.join(input_data_folder, self._vector_store_folder)
        in_store_folder = os.path.join(
            input_data_folder, self._in_store_documents_folder
        )
        vector_store = None

        if os.path.exists(vector_db_file_path):
            logger.debug(
                f"Loading existing vector database from: {vector_db_file_path}"
            )
            vector_store = self.load_vector_store(
                self.input_data_folder, embedding_model
            )

        # Get list of PDF and text files
        files = [
            f
            for f in os.listdir(self.input_data_folder)
            if f.endswith((".pdf", ".txt"))
        ]
        logger.debug(
            f"Found {len(files)} new documents to process in {self.input_data_folder}"
        )

        if len(files) > 0:
            # Process each file individually
            with track_time(f"Loading {len(files)} document(s) into vector database"):
                for filename in files:
                    filepath = os.path.join(self.input_data_folder, filename)
                    logger.info(
                        f"Processing file: {filename} size: {os.path.getsize(filepath)}"
                    )

                    try:
                        split_docs = self.parser.load_and_split_document(
                            filepath,
                            filename,
                            embedding_model.model_name,
                            metadata_extractor,
                            document_curator,
                        )

                        # Create or update vector store
                        if vector_store is None:
                            vector_store = FAISS.from_documents(
                                split_docs,
                                embedding_model,
                                distance_strategy=DistanceStrategy.COSINE,
                            )
                        else:
                            batch_store = FAISS.from_documents(
                                split_docs,
                                embedding_model,
                                distance_strategy=DistanceStrategy.COSINE,
                            )
                            vector_store.merge_from(batch_store)

                        # Move processed file to 'in_store' folder
                        os.makedirs(in_store_folder, exist_ok=True)
                        os.rename(filepath, os.path.join(in_store_folder, filename))

                    except Exception as e:
                        logger.error(f"Error processing {filename}: {str(e)}")
                        logger.error(f"Error details: {traceback.format_exc()}")
                        continue

            # Save to disk
            if vector_store is not None and len(files) > 0:
                vector_store.save_local(vector_db_file_path)
                logger.info(f"Vector database saved to {vector_db_file_path}")

        return vector_store
