from typing import List, Dict, Any, Optional
from langchain_community.docstore.document import Document as LangchainDocument
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import logging
import os
import time

logger = logging.getLogger("main")


class VectorStore:
    """Vector store for document embeddings with FAISS backend."""

    def __init__(self, embedding_model: HuggingFaceEmbeddings):
        """
        Initialize the vector store.

        Args:
            embedding_model: The embedding model to use
        """
        self.embedding_model = embedding_model
        self.docstore = FAISS.from_documents(
            [],
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )

    def add_documents(self, documents: List[LangchainDocument]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
        """
        if not documents:
            return

        batch_store = FAISS.from_documents(
            documents,
            self.embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )

        if len(self.docstore.docstore.docs) == 0:
            self.docstore = batch_store
        else:
            self.docstore.merge_from(batch_store)

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
        """
        self.docstore.save_local(path)
        logger.info(f"Vector store saved to {path}")

    def load_local(self, path: str) -> None:
        """
        Load the vector store from disk.

        Args:
            path: Path to load the vector store from

        Raises:
            FileNotFoundError: If the vector store doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store not found at {path}")

        start_time = time.time()
        self.docstore = FAISS.load_local(
            path,
            self.embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True,
        )
        end_time = time.time()

        logger.info(
            f"Vector store loaded from disk in {end_time - start_time:.2f} seconds"
        )

    def merge(self, other_store: "VectorStore") -> None:
        """
        Merge another vector store into this one.

        Args:
            other_store: The vector store to merge
        """
        self.docstore.merge_from(other_store.docstore)

    def get_document_count(self) -> int:
        """
        Get the number of documents in the store.

        Returns:
            Number of documents
        """
        return len(self.docstore.docstore.docs)

    def get_document_by_id(self, doc_id: str) -> Optional[LangchainDocument]:
        """
        Get a document by its ID.

        Args:
            doc_id: The document ID

        Returns:
            The document if found, None otherwise
        """
        return self.docstore.docstore.docs.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by its ID.

        Args:
            doc_id: The document ID

        Returns:
            True if document was deleted, False otherwise
        """
        if doc_id in self.docstore.docstore.docs:
            del self.docstore.docstore.docs[doc_id]
            return True
        return False
