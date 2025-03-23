from typing import List, Dict, Any, Optional, Callable
from langchain_community.docstore.document import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import logging
import os
import time
from owlai.owlsys import track_time
from owlai.core.logging_setup import get_logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.document_loaders import TextLoader
from tqdm import tqdm
import traceback
import fitz

logger = get_logger("vector_store")


class VectorStore:
    """Vector store for document embeddings with FAISS backend."""

    input_data_folder: str

    def __init__(
        self,
        embedding_model: HuggingFaceEmbeddings,
        input_data_folder: str,
    ):
        """
        Initialize the vector store.

        Args:
            embedding_model: The embedding model to use
            input_data_folder: List of input data folders to load vector stores from
        """
        self.embedding_model = embedding_model
        self.input_data_folder = input_data_folder

        if os.path.exists(input_data_folder):
            store_path = f"{input_data_folder}/vector_db"
            disk_vector_store = self.load_vector_store_from_disk(
                store_path, self.embedding_model
            )
            files_vector_store = self.load_dataset(
                input_data_folder, self.embedding_model, chunk_size=512
            )

            if disk_vector_store is not None and files_vector_store is not None:
                self.docstore = disk_vector_store
                self.docstore.merge_from(files_vector_store)
            elif disk_vector_store is not None:
                self.docstore = disk_vector_store
            elif files_vector_store is not None:
                self.docstore = files_vector_store
            else:
                raise ValueError("No vector store found in {input_data_folder}")

        self.docstore = (
            None  # Initialize as None, will be created when first document is added
        )
        self.documents = []  # List of documents

    def load_vector_store_from_disk(
        self, store_path: str, embedding_model: HuggingFaceEmbeddings
    ) -> Optional[FAISS]:
        """
        Load a vector store from disk.

        Args:
            store_path: Path to the vector store
            embedding_model: The embedding model to use
        """
        KNOWLEDGE_VECTOR_DATABASE = None

        if os.path.exists(store_path):
            logger.info(f"Loading the vector database from disk: {store_path}")

            with track_time(f"Loading the vector database from disk: {store_path}"):
                KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
                    store_path,
                    embedding_model,
                    distance_strategy=DistanceStrategy.COSINE,
                    allow_dangerous_deserialization=True,
                )

        else:
            raise FileNotFoundError(f"Vector database not found in {store_path}")

        return KNOWLEDGE_VECTOR_DATABASE

    def analyze_chunk_size_distribution(
        self,
        input_data_folder,
        filename,
        docs: List[LangchainDocument],
        model_name="thenlper/gte-small",
    ):
        """
        Analyze and visualize document lengths.

        Args:
            docs: to analyze
            model_name: Name of the embedding model to use
        """
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer

        # Get max sequence length from SentenceTransformer
        max_seq_len = SentenceTransformer(model_name).max_seq_length
        info_message = (
            f"Model's max sequence size: '{max_seq_len}' Document count: '{len(docs)}'"
        )
        logger.debug(info_message)

        # Analyze token lengths (should init tokenizer once... whatever)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        lengths = [len(tokenizer.encode(doc.page_content)) for doc in docs]

        # Plot distribution
        import matplotlib.pyplot as plt
        import pandas as pd

        fig = pd.Series(lengths).hist()
        plt.title(f"Distribution of page lengths [tokens] for {filename}")
        file_dir = f"{input_data_folder}/images"
        file_path = f"{file_dir}/chunk_size_distribution-{filename}.png"
        os.makedirs(file_dir, exist_ok=True)
        plt.savefig(file_path)
        plt.close()
        logger.debug(
            f"Distribution of document lengths (in count of tokens) saved to {file_path}"
        )
        return file_path

    def split_documents(
        self,
        chunk_size: int,  # The maximum number of tokens in a chunk
        input_docs: List[LangchainDocument],
        tokenizer_name: str,
    ) -> List[LangchainDocument]:
        """
        Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
            chunk_size=chunk_size,
            chunk_overlap=int(
                chunk_size / 10
            ),  # The number of characters to overlap between chunks
            add_start_index=True,  # If `True`, includes chunk's start index in metadata
            strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
            separators=[
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )
        logger.debug(f"Splitting {len(input_docs)} documents")

        docs_processed = []
        for doc in tqdm(input_docs, desc="Splitting documents"):
            result = text_splitter.split_documents([doc])
            docs_processed += result

        logger.debug(
            f"Splitted {len(input_docs)} documents into {len(docs_processed)} chunks"
        )
        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in tqdm(docs_processed, desc="Removing duplicates"):
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        logger.debug(
            f"Removed {len(docs_processed) - len(docs_processed_unique)} duplicates from {len(docs_processed)} chunks"
        )

        return docs_processed_unique

    def load_and_split_document(
        self,
        filepath: str,
        input_data_folder: str,
        filename: str,
        chunk_size: int,
        model_name: str,
        metadata_extractor: Optional[Callable] = None,
        document_curator: Optional[Callable] = None,
    ) -> List[LangchainDocument]:
        """
        Loads a document file and splits it into chunks.

        Args:
            filepath: Path to the document file
            input_data_folder: Folder containing the documents
            filename: Name of the document file
            chunk_size: Size of text chunks for splitting
            model_name: Name of the embedding model for tokenization
            metadata_extractor: Optional callback function that takes a document and returns
                                additional metadata as a dictionary to be added to the document

        Returns:
            List of split LangchainDocument objects
        """
        # Call metadata extractor if provided
        metadata = {}
        if metadata_extractor:
            try:
                additional_metadata = metadata_extractor(filepath)
                if additional_metadata and isinstance(additional_metadata, dict):
                    metadata.update(additional_metadata)
            except Exception as e:
                logger.error(f"Error in metadata extractor for {filename}: {str(e)}")
                logger.error(f"Error details: {traceback.format_exc()}")

        # Load document
        if filename.endswith(".pdf"):
            doc = fitz.open(filepath)
            total_pages = len(doc)
            loaded_docs: List[LangchainDocument] = []

            with track_time(f"Loading document: '{filename}'"):
                for page_number in tqdm(
                    range(total_pages), desc=f"Loading pages from {filename}"
                ):
                    page = doc[page_number]
                    page_content = page.get_text("text")

                    # Call document curator if provided
                    if document_curator:
                        page_content = document_curator(page_content, filepath)

                    # Update metadata
                    page_metadata = metadata.copy()
                    page_metadata.update(
                        {
                            "source": f"{filename}:{page_number}",
                            "page_number": page_number + 1,
                            "num_pages": total_pages,
                        }
                    )

                    loaded_docs.append(
                        LangchainDocument(
                            page_content=page_content,
                            metadata=page_metadata,
                        )
                    )
            doc.close()
        else:  # .txt files
            loader = TextLoader(filepath)
            docs = loader.lazy_load()
            loaded_docs = []
            for doc in docs:
                doc_content = doc.page_content
                if document_curator:
                    doc_content = document_curator(doc_content, filepath)
                loaded_docs.append(
                    LangchainDocument(
                        page_content=doc_content,
                        metadata=metadata,
                    )
                )

        # Split documents
        split_docs = self.split_documents(
            chunk_size,
            loaded_docs,
            tokenizer_name=model_name,
        )

        # Analyze document chunks before splitting
        pre_split_file = self.analyze_chunk_size_distribution(
            input_data_folder,
            "pre-split-" + filename,
            loaded_docs,
            model_name,
        )
        metadata["pre_split_file"] = pre_split_file

        # Analyze post-split chunks and add to metadata
        post_split_file = self.analyze_chunk_size_distribution(
            input_data_folder,
            "post-split-" + filename,
            split_docs,
            model_name,
        )
        metadata["post_split_file"] = post_split_file

        for i in range(min(5, len(split_docs))):
            logger.debug(f"Split doc {i}: {split_docs[i].metadata}")

        return split_docs

    def load_dataset(
        self,
        input_data_folder: str,
        embedding_model: HuggingFaceEmbeddings,
        chunk_size: int = 512,
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
        vector_db_file_path = f"{input_data_folder}/vector_db"
        in_store_folder = f"{input_data_folder}/in_store"
        vector_store = None

        if os.path.exists(vector_db_file_path):
            logger.info(f"Loading existing vector database from: {vector_db_file_path}")
            vector_store = self.load_vector_store_from_disk(
                vector_db_file_path, embedding_model
            )

        # Get list of PDF and text files
        files = [
            f for f in os.listdir(input_data_folder) if f.endswith((".pdf", ".txt"))
        ]
        logger.info(
            f"Found {len(files)} new documents to process in {input_data_folder}"
        )

        # Process each file individually
        with track_time(f"{len(files)} document(s) loading into vector database"):
            for filename in files:
                filepath = os.path.join(input_data_folder, filename)
                logger.info(
                    f"Processing file: {filename} size: {os.path.getsize(filepath)}"
                )

                try:
                    split_docs = self.load_and_split_document(
                        filepath,
                        input_data_folder,
                        filename,
                        chunk_size,
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
                    in_store_folder = os.path.join(input_data_folder, "in_store")
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

    def add_documents(self, documents: List[LangchainDocument]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
        """
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        if self.docstore is None:
            # Create new vector store with first batch of documents
            self.docstore = FAISS.from_texts(
                texts,
                self.embedding_model,
                metadatas=metadatas,
                distance_strategy=DistanceStrategy.COSINE,
            )
        else:
            # Add to existing vector store
            self.docstore.add_texts(texts, metadatas=metadatas)

        logger.info(f"Added {len(documents)} documents to vector store")
        self.documents.extend(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[LangchainDocument]:
        """
        Search for similar documents.

        Args:
            query: The search query
            k: Number of results to return
            filter: Optional filter criteria

        Returns:
            List of similar documents

        Raises:
            ValueError: If query is empty or k is invalid
        """
        if not query:
            raise ValueError("Query cannot be empty")

        if k <= 0:
            raise ValueError("k must be positive")

        return self.docstore.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )

    def save_local(self, path: str) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Path to save the vector store

        Raises:
            ValueError: If no documents have been added to the vector store
        """
        if self.docstore is None:
            raise ValueError(
                "Cannot save empty vector store. Please add documents first."
            )

        self.docstore.save_local(path)
        logger.info(f"Saved vector store to {path}")

    def load_local(self, path: str) -> Optional[FAISS]:
        """
        Load the vector store from disk.

        Args:
            path: Path to load the vector store from

        Raises:
            FileNotFoundError: If the vector store doesn't exist
        """
        return self.load_vector_store_from_disk(path, self.embedding_model)

    def merge(self, other_store: "VectorStore") -> None:
        """
        Merge another vector store into this one.

        Args:
            other_store: The vector store to merge
        """
        if self.docstore is None or other_store.docstore is None:
            raise ValueError("Both vector stores must be initialized before merging")
        self.docstore.merge_from(other_store.docstore)

    def get_document_count(self) -> int:
        """
        Get the number of documents in the vector store.

        Returns:
            Number of documents
        """
        return self.docstore.index.ntotal

    def get_document_by_id(self, doc_id: str) -> Optional[LangchainDocument]:
        """
        Get a document by its ID.

        Args:
            doc_id: The document ID

        Returns:
            The document if found, None otherwise
        """
        for doc in self.documents:
            if doc.metadata.get("id") == doc_id:
                return doc
        return None

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by its ID.

        Args:
            doc_id: The document ID

        Returns:
            True if document was deleted, False otherwise
        """
        for i, doc in enumerate(self.documents):
            if doc.metadata.get("id") == doc_id:
                del self.documents[i]
                return True
        return False

    def get_documents(self) -> List[LangchainDocument]:
        """
        Get all documents from the vector store.

        Returns:
            List of all documents
        """
        return self.documents
