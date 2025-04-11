"""
OwlAI RAG Module

Note: We are using Pydantic v1 because it's required by langchain-core and other LangChain components.
This is a temporary solution until LangChain fully supports Pydantic v2.
The deprecation warnings are suppressed in pytest configuration.
"""

print("Loading rag module")

from typing import Optional, List, Tuple, Any, Callable, Dict, Literal, Union, Type
import os
import logging
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import warnings
import traceback
from owlai.services.datastore import FAISS_DataStore
from langchain_core.tools import BaseTool, ArgsSchema
from owlai.services.embeddings import EmbeddingManager
from owlai.services.reranker import RerankerManager
from owlai.services.system import sprint
import requests
import json
from urllib.parse import urlparse
from loguru import logger
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.docstore.document import Document
from langchain_core.documents import Document as LCDocument
from pydantic import BaseModel, Field, root_validator
from pinecone import Pinecone, FetchResponse, Index


logger = logging.getLogger(__name__)

warnings.simplefilter("ignore", category=FutureWarning)

from pinecone import Pinecone, FetchResponse

from owlai.services.data_provider import DataProvider


class DefaultToolInput(BaseModel):
    """Input for tool."""

    query: str = Field(description="A query to the tool")


class FAISS_RAG_Retriever(BaseModel):
    """Class responsible for retrieving relevant chunks from the knowledge base"""

    num_retrieved_docs: int = 30  # Commonly called k
    num_docs_final: int = 5
    embeddings_model_name: str = "thenlper/gte-small"
    reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    model_kwargs: Dict[str, Any] = {}
    encode_kwargs: Dict[str, Any] = {}
    multi_process: bool = True
    datastore: Optional[FAISS_DataStore] = None

    def load_dataset(self, embeddings: HuggingFaceEmbeddings) -> Optional[FAISS]:
        """
        Load or create vector store using the provided embeddings.

        Args:
            embeddings: HuggingFaceEmbeddings instance to use

        Returns:
            Optional FAISS vector store
        """
        if not isinstance(embeddings, HuggingFaceEmbeddings):
            raise ValueError("embeddings must be an instance of HuggingFaceEmbeddings")
        return self.datastore.load_vector_store(embeddings)

    def retrieve_relevant_chunks(
        self,
        query: str,
        knowledge_base: Optional[FAISS],
    ) -> List[LangchainDocument]:
        """
        Retrieve the k most relevant document chunks for a given query.

        Args:
            query: The user query to find relevant documents for
            knowledge_base: The vector database containing indexed documents

        Returns:
            Tuple containing a list of retrieved and reranked LangchainDocument objects with scores and metadata
        """
        if knowledge_base is None:
            raise Exception("Invalid FAISS vector store")

        logger.debug(f"Documents search")
        retrieved_docs = knowledge_base.similarity_search(
            query=query, k=self.num_retrieved_docs
        )

        logger.debug(f"{len(retrieved_docs)} documents retrieved")

        return retrieved_docs


class FAISS_RAG_Tool(DataProvider):
    """
    RAG Tool implementation with local inference.
    """

    name: str = "sad_unamed_rag_tool"
    description: str
    args_schema: Optional[ArgsSchema] = DefaultToolInput

    retriever: FAISS_RAG_Retriever
    _vector_store: Optional[FAISS] = None
    _embeddings: Optional[HuggingFaceEmbeddings] = None
    _reranker: Optional[Any] = None
    _prompt: Optional[PromptTemplate] = None
    _db_session = None

    def __init__(self, *args, db_session: Optional[Any] = None, **kwargs):
        """Initialize RAGTool with optional database session for vector store caching.

        Args:
            db_session: Optional SQLAlchemy session for vector store DB operations
            *args, **kwargs: Additional arguments passed to parent
        """
        try:
            logger.debug(f"Starting RAGTool initialization")

            # Initialize the base class (BaseTool)
            logger.debug("Initializing base class")
            super().__init__(**kwargs)
            logger.debug(f"Base class initialization completed {self.name}")

            # Store DB session
            self._db_session = db_session

            # Initialize embeddings using EmbeddingManager
            logger.debug(
                f"Getting embeddings with model: '{self.retriever.embeddings_model_name}' multi_process: '{self.retriever.multi_process}'"
            )
            embeddings = EmbeddingManager.get_embedding(
                model_name=self.retriever.embeddings_model_name,
                multi_process=self.retriever.multi_process,
                model_kwargs=self.retriever.model_kwargs,
                encode_kwargs=self.retriever.encode_kwargs,
            )
            if not isinstance(embeddings, HuggingFaceEmbeddings):
                raise ValueError("Failed to initialize embeddings")
            self._embeddings = embeddings
            logger.debug("Embeddings initialization completed")

            # Try to initialize reranker using RerankerManager
            logger.debug(
                f"Attempting to initialize reranker: {self.retriever.reranker_name}"
            )
            try:
                reranker_name = self.retriever.reranker_name
                self._reranker = RerankerManager.get_reranker(
                    model_name=reranker_name,
                    model_kwargs=self.retriever.model_kwargs,
                )
                logger.debug(
                    f"Successfully initialized cross-encoder reranker: {reranker_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize reranker model: {str(e)}")
                logger.warning("Falling back to basic retrieval without reranking")
                self._reranker = None

            # Pass db_session to datastore
            if self._db_session and self.retriever.datastore:
                self.retriever.datastore._db_session = self._db_session

            if self.retriever.datastore:
                # Load vector store
                logger.debug(f"Loading vector store {self.retriever.datastore.name}")
                self._vector_store = self.retriever.load_dataset(self._embeddings)
                if self._vector_store is None:
                    logger.warning(
                        "No vector stores found: you must set the vector store manually."
                    )
                else:
                    logger.debug(f"Data store loaded: {self.retriever.datastore.name}")

            logger.debug(f"RAGTool initialization completed successfully {self.name}")

        except Exception as e:
            logger.error(f"Error during RAGTool initialization: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            raise

    def rerank_documents(
        self, query: str, documents: List[LangchainDocument], k: int = 5
    ) -> List[LangchainDocument]:
        """
        Rerank documents using sentence-transformers cross-encoder.

        Args:
            query: The search query
            documents: List of documents to rerank
            k: Number of documents to return

        Returns:
            Reranked list of documents
        """
        if not self._reranker:
            return documents[:k]

        # Prepare sentence pairs for cross-encoder
        sentence_pairs = [(query, doc.page_content) for doc in documents]

        try:
            # Get cross-encoder scores
            scores = self._reranker.predict(sentence_pairs)

            # Create list of (score, doc) tuples and sort by score
            scored_docs = list(zip(scores, documents))
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            # Update metadata with scores
            for i, (score, doc) in enumerate(scored_docs[:k]):
                doc.metadata["rerank_score"] = float(score)
                doc.metadata["rerank_position"] = i + 1

            logger.debug(
                f"Reranking completed from {len(scored_docs)} documents to {k}"
            )

            return [doc for _, doc in scored_docs[:k]]

        except Exception as e:
            logger.warning(f"Error during reranking: {str(e)}")
            logger.warning("Falling back to original document order")
            return documents[:k]

    def get_rag_resources(self, question: str) -> Dict[str, Any]:
        """
        Runs the RAG query against the vector store and returns an answer to the question.
        Args:
            question: a string containing the question to answer.

        Returns:
            A dictionary containing the question, answer, and metadata.
        """
        logger.info(f"Tool: '{self.name}' - Semantics query: '{question}'")

        answer: Dict[str, Any] = {"question": question}

        metadata = {
            "query": question,
            "k": self.retriever.num_retrieved_docs,
            "num_docs_final": self.retriever.num_docs_final,
            "reranking_enabled": self._reranker is not None,
        }

        # Get initial documents
        retrieved_docs = self.retriever.retrieve_relevant_chunks(
            query=question,
            knowledge_base=self._vector_store,
        )

        # Jeep is there it is useful
        # from owlai.services.system import sprint
        # sprint(retrieved_docs[0].metadata)

        metadata["num_docs_retrieved"] = len(retrieved_docs)
        metadata["retrieved_docs"] = [
            {
                "id": doc.id,
                "title": doc.metadata.get("title", "Unknown Title"),
                "score": doc.metadata.get("", "Unknown"),
                "source": doc.metadata.get("source", "Unknown Source"),
            }
            for doc in retrieved_docs
        ]

        # Rerank documents if reranker is available
        reranked_docs = self.rerank_documents(
            question, retrieved_docs, k=self.retriever.num_docs_final
        )

        for idoc in reranked_docs:
            logger.debug(
                f"Document '{idoc.metadata.get('title', 'Unknown Title')}' - Rerank Score: {idoc.metadata.get('rerank_score', 'Unknown')}"
            )

        docs_content = f"Query: '{question}'\nDocuments:\n\n".join(
            [
                f"{idx+1}. [Source: {doc.metadata.get('title', 'Unknown Title')} - Score: {doc.metadata.get('rerank_score', 'Unknown')}] \"{doc.page_content}\""
                for idx, doc in enumerate(reranked_docs)
            ]
        )

        metadata["reranked_docs"] = [
            {
                "id": doc.id,
                "title": doc.metadata.get("title", "Unknown Title"),
                "score": doc.metadata.get("rerank_score", "Unknown"),
                "source": doc.metadata.get("source", "Unknown Source"),
            }
            for doc in reranked_docs
        ]

        answer["answer"] = docs_content
        answer["metadata"] = metadata

        return answer

    def get_document_content_by_id(self, id: str) -> str:
        """
        Get the content of a document by its ID.
        """
        return self._vector_store.get_by_ids([id])[0].page_content


class Pinecone_RAG_Tool(DataProvider):
    """
    RAG Tool implementation using Pinecone API for retrieval instead of local FAISS.
    This allows for more scalable and distributed vector search without running inference locally.
    """

    name: str = "pinecone_rag"
    description: str = "Searches through documents using Pinecone vector database"
    args_schema: Optional[ArgsSchema] = DefaultToolInput

    num_retrieved_docs: int = 30  # Commonly called k
    num_docs_final: int = 5

    pinecone_api_key: str = Field(
        default_factory=lambda: os.getenv("PINECONE_API_KEY", "")
    )
    pinecone_host: str = Field(
        default_factory=lambda: os.getenv(
            "PINECONE_HOST", "https://owlai-law-sq85boh.svc.apu-57e2-42f6.pinecone.io"
        )
    )
    pinecone_namespace: str = Field(
        default_factory=lambda: os.getenv("PINECONE_NAMESPACE", "")
    )
    embeddings_model_name: str = "text-embedding-3-large"
    embedding: Optional[OpenAIEmbeddings] = None

    pc: Optional[Pinecone] = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))

    def __init__(self, *args, **kwargs):
        """Initialize PineconeRAGTool with Pinecone connection settings."""
        super().__init__(*args, **kwargs)

        # Ensure Pinecone host has https:// prefix
        if self.pinecone_host and not self.pinecone_host.startswith(
            ("http://", "https://")
        ):
            self.pinecone_host = f"https://{self.pinecone_host}"

        if not self.pinecone_api_key:
            logger.warning("No Pinecone API key provided. RAG queries will fail.")

        self.embedding = OpenAIEmbeddings(model="text-embedding-3-large")

        logger.debug(f"Initialized PineconeRAGTool with host: {self.pinecone_host}")

    def get_rag_resources(self, question: str) -> Dict[str, Any]:
        """
        Runs the RAG query against Pinecone vector store and returns an answer to the question.

        Args:
            question: a string containing the question to answer.

        Returns:
            A dictionary containing the question, answer, and metadata.
        """

        answer: Dict[str, Any] = {"question": question}

        metadata = {
            "query": question,
            "k": self.num_retrieved_docs,
            "num_docs_final": self.num_docs_final,
            "reranking_enabled": False,
            "vector_store": "pinecone",
            "host": self.pinecone_host,
            "namespace": self.pinecone_namespace or "default",
        }

        try:
            # Generate embeddings for the query using the embedding model
            retry_attempts = 3
            query_embedding = None

            for attempt in range(retry_attempts):
                try:
                    logger.debug(
                        f"Generating embedding for query (attempt {attempt+1})"
                    )
                    query_embedding = self.embedding.embed_query(question)
                    break
                except Exception as e:
                    logger.warning(f"Embedding attempt {attempt+1} failed: {str(e)}")
                    if attempt < retry_attempts - 1:
                        import time

                        time.sleep(2**attempt)  # Exponential backoff
                    else:
                        logger.error(
                            f"Failed to embed query after {retry_attempts} attempts"
                        )
                        raise

            if query_embedding is None:
                raise ValueError("Failed to generate embedding for query")

            # Structure the API request to Pinecone
            headers = {
                "Api-Key": self.pinecone_api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            request_data = {
                "vector": query_embedding,
                "topK": self.num_docs_final,
                "includeMetadata": True,
                "includeValues": False,  # We don't need the vector values
            }

            # Add namespace if specified
            if self.pinecone_namespace:
                request_data["namespace"] = self.pinecone_namespace

            # Make the query request to Pinecone
            query_url = f"{self.pinecone_host}/query"

            response = None
            for attempt in range(retry_attempts):
                try:
                    logger.debug(f"Querying Pinecone (attempt {attempt+1})")
                    response = requests.post(
                        query_url, headers=headers, json=request_data
                    )
                    response.raise_for_status()
                    break
                except Exception as e:
                    logger.warning(
                        f"Pinecone query attempt {attempt+1} failed: {str(e)}"
                    )
                    if attempt < retry_attempts - 1:
                        import time

                        time.sleep(2**attempt)  # Exponential backoff
                    else:
                        logger.error(
                            f"Failed to query Pinecone after {retry_attempts} attempts"
                        )
                        raise

            if response is None:
                raise ValueError("Failed to get response from Pinecone")

            # Process the response from Pinecone
            results = response.json()

            logger.debug(f"Pinecone returned {len(results.get('matches', []))} matches")

            # Convert Pinecone matches to LangchainDocument format
            retrieved_docs: List[LangchainDocument] = []
            for match in results.get("matches", []):
                doc_metadata = match.get("metadata", {})
                # Create a LangchainDocument with the content and metadata
                doc = LangchainDocument(
                    page_content=doc_metadata.get("text", ""),
                    metadata={
                        "id": match.get("id", ""),
                        "title": doc_metadata.get("title", "Unknown Title"),
                        "source": doc_metadata.get("source", "Unknown Source"),
                        "pc_score": match.get("score", 0),
                        **{k: v for k, v in doc_metadata.items() if k != "text"},
                    },
                )
                retrieved_docs.append(doc)

            logger.info(f"Retrieved {len(retrieved_docs)} documents from Pinecone")

            metadata["num_docs_retrieved"] = len(retrieved_docs)
            metadata["retrieved_docs"] = [
                {
                    "id": doc.metadata.get("id", ""),
                    "title": doc.metadata.get("title", "Unknown Title"),
                    "pc_score": doc.metadata.get("pc_score", 0),
                    "source": doc.metadata.get("source", "Unknown Source"),
                }
                for doc in retrieved_docs
            ]

            docs_content = f"Query: '{question}'\nDocuments:\n\n" + "\n\n".join(
                [
                    f"{idx+1}. [Source: {doc.metadata.get('title', 'Unknown Title')} - "
                    f"Score: {doc.metadata.get('pc_score', doc.metadata.get('score', 'Unknown'))}] "
                    f'"{doc.page_content}"'
                    for idx, doc in enumerate(retrieved_docs)
                ]
            )

            answer["answer"] = docs_content
            answer["metadata"] = metadata

        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            logger.error(traceback.format_exc())
            answer["answer"] = f"Error retrieving information: {str(e)}"
            answer["metadata"] = metadata

        return answer

    def describe_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.
        Matches the interface used in the pinecone_migration.py DirectPineconeIndex class.

        Returns:
            Dictionary with index statistics
        """
        url = f"{self.pinecone_host}/describe_index_stats"
        headers = {
            "Api-Key": self.pinecone_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {"error": str(e)}

    def get_document_content_by_id(self, id: str) -> str:
        """
        Get the content of a document by its ID.
        """

        index = self.pc.Index(host=os.getenv("PINECONE_HOST", ""))
        response: FetchResponse = index.fetch(ids=[id])

        # View result
        item = response.vectors.get(id)

        # Ensure the 'text' field is treated as a string
        return str(item.metadata.get("text", ""))


def main():
    logger.info("Starting main function")


if __name__ == "__main__":
    main()
