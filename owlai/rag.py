"""
OwlAI RAG Module

Note: We are using Pydantic v1 because it's required by langchain-core and other LangChain components.
This is a temporary solution until LangChain fully supports Pydantic v2.
The deprecation warnings are suppressed in pytest configuration.
"""

print("Loading rag module")

from typing import Optional, List, Tuple, Any, Callable, Dict, Literal
import os
import logging
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from pydantic import BaseModel
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import warnings
from tqdm import tqdm
import fitz
import re
import traceback
from owlai.owlsys import encode_text, track_time, setup_logging, sprint
from owlai.core import OwlAgent
from owlai.parser import DefaultParser, create_instance
from owlai.data import RAGDataStore
from langchain_core.tools import BaseTool, ArgsSchema
from pydantic import BaseModel, Field
from owlai.embeddings import EmbeddingManager
from owlai.models import RAGPretrainedModel

logger = logging.getLogger(__name__)

warnings.simplefilter("ignore", category=FutureWarning)


class DefaultToolInput(BaseModel):
    """Input for tool."""

    query: str = Field(description="A query to the tool")


class RAGRetriever(BaseModel):
    """Class responsible for retrieving relevant chunks from the knowledge base"""

    num_retrieved_docs: int = 30  # Commonly called k
    num_docs_final: int = 5
    embeddings_model_name: str = "thenlper/gte-small"
    reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    model_kwargs: Dict[str, Any] = {}
    encode_kwargs: Dict[str, Any] = {}
    multi_process: bool = True
    datastore: RAGDataStore

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
        reranker: Optional[Any] = None,
    ) -> Tuple[List[LangchainDocument], dict]:
        """
        Retrieve the k most relevant document chunks for a given query.

        Args:
            query: The user query to find relevant documents for
            knowledge_base: The vector database containing indexed documents
            reranker: Optional reranker model to rerank results

        Returns:
            Tuple containing a list of retrieved and reranked LangchainDocument objects with scores and metadata
        """
        if knowledge_base is None:
            raise Exception("Invalid FAISS vector store")

        metadata = {
            "query": query,
            "k": self.num_retrieved_docs,
            "num_docs_final": self.num_docs_final,
            "reranking_enabled": reranker is not None,
        }

        with track_time(f"Documents search", metadata):
            retrieved_docs = knowledge_base.similarity_search(
                query=query, k=self.num_retrieved_docs
            )
            metadata["num_docs_retrieved"] = len(retrieved_docs)
            metadata["retrieved_docs"] = {
                i: {
                    "title": doc.metadata.get("title", "No title"),
                    "source": doc.metadata.get("source", "Unknown source"),
                }
                for i, doc in enumerate(retrieved_docs)
            }
            logger.debug(f"{len(retrieved_docs)} documents retrieved")

        # If no reranker, just return top k docs
        if not reranker:
            logger.info("No reranker available, using basic retrieval")
            return retrieved_docs[: self.num_docs_final], metadata

        return retrieved_docs, metadata


class RAGTool(BaseTool):
    """
    RAG Agent implementation that extends OwlAgent with RAG capabilities
    """

    name: str = "sad_unamed_rag_tool"
    description: str
    args_schema: Optional[ArgsSchema] = DefaultToolInput

    retriever: RAGRetriever
    _vector_store: Optional[FAISS] = None
    _embeddings: Optional[HuggingFaceEmbeddings] = None
    _reranker: Optional[Any] = None
    _prompt: Optional[PromptTemplate] = None

    def __init__(self, *args, **kwargs):
        try:
            logger.debug(f"Starting RAGTool initialization")

            # Initialize the base class (BaseTool)
            logger.debug("Initializing base class")
            super().__init__(**kwargs)
            logger.debug(f"Base class initialization completed {self.name}")

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

            # Try to initialize reranker
            logger.debug(
                f"Attempting to initialize reranker: {self.retriever.reranker_name}"
            )
            try:
                from sentence_transformers import CrossEncoder

                reranker_name = self.retriever.reranker_name
                self._reranker = CrossEncoder(reranker_name)
                logger.debug(
                    f"Successfully initialized cross-encoder reranker: {reranker_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize reranker model: {str(e)}")
                logger.warning("Falling back to basic retrieval without reranking")
                self._reranker = None

            # Load vector store
            logger.debug("Loading vector store")
            self._vector_store = self.retriever.load_dataset(self._embeddings)
            if self._vector_store is None:
                logger.warning(
                    "No vector stores found: you must set the vector store manually."
                )
            else:
                logger.debug(
                    f"Data store loaded: {self.retriever.datastore.input_data_folder}"
                )

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

            logger.debug(f"Reranking completed for {len(scored_docs)} documents")

            return [doc for _, doc in scored_docs[:k]]

        except Exception as e:
            logger.warning(f"Error during reranking: {str(e)}")
            logger.warning("Falling back to original document order")
            return documents[:k]

    def get_rag_sources(self, question: str) -> Dict[str, Any]:
        """
        Runs the RAG query against the vector store and returns an answer to the question.
        Args:
            question: a string containing the question to answer.

        Returns:
            A dictionary containing the question, answer, and metadata.
        """
        logger.debug(f"Running RAG query: '{question}'")

        answer: Dict[str, Any] = {"question": question}

        # Get initial documents
        retrieved_docs, metadata = self.retriever.retrieve_relevant_chunks(
            query=question,
            knowledge_base=self._vector_store,
            reranker=None,  # We'll do reranking separately
        )

        # Rerank documents if reranker is available
        reranked_docs = self.rerank_documents(
            question, retrieved_docs, k=self.retriever.num_docs_final
        )

        with track_time("Model invocation with RAG context", metadata):
            docs_content = "\n\n".join(
                [
                    f"{idx+1}. [Source : {doc.metadata.get('title', 'Unknown Title')} - {doc.metadata.get('source', '')}] \"{doc.page_content}\""
                    for idx, doc in enumerate(reranked_docs)
                ]
            )

        answer["answer"] = docs_content
        answer["metadata"] = metadata

        return answer

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Override BaseTool._run to ensure we use our implementation"""
        logger.debug(f"[RqAGTool._run] Called with query: {query}")
        return self.message_invoke(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Override BaseTool._arun to ensure we use our implementation"""
        return self.message_invoke(query)

    def message_invoke(self, message: str) -> str:
        """Override OwlAgent.message_invoke with RAG specific implementation"""
        logger.debug(
            f"[RAGTool.message_invoke] Called from {self.name} with message: {message}"
        )
        logger.debug(f"RAG engine not keeping context for now")
        answer = self.get_rag_sources(message)
        if "answer" not in answer or answer["answer"] == "":
            raise Exception("No answer found")
        return answer.get("answer", "?????")


class RAGAgent(OwlAgent):

    # JSON defined properties
    retriever: RAGRetriever

    # Runtime updated properties
    _init_completed = False
    _prompt = None
    _vector_store = None
    _embeddings: Optional[HuggingFaceEmbeddings] = None
    _reranker = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embeddings = EmbeddingManager.get_embedding(
            model_name=self.retriever.embeddings_model_name,
            multi_process=self.retriever.multi_process,
            model_kwargs=self.retriever.model_kwargs,
            encode_kwargs=self.retriever.encode_kwargs,
        )
        reranker_name = self.retriever.reranker_name
        self._reranker = RAGPretrainedModel.from_pretrained(reranker_name)
        self._prompt = PromptTemplate.from_template(self.system_prompt)

        if self._embeddings is None:
            raise ValueError("Failed to initialize embeddings")

        self._vector_store = self.retriever.load_dataset(self._embeddings)

        if self._vector_store is None:
            logger.warning(
                "No vector stores found: you must set the vector store manually."
            )
        else:
            logger.info(
                f"Data store loaded: {self.retriever.datastore.input_data_folder}"
            )

    def rerank_documents(
        self, query: str, documents: List[LangchainDocument], k: int = 5
    ) -> List[LangchainDocument]:
        """
        Rerank documents using the reranker model.

        Args:
            query: The search query
            documents: List of documents to rerank
            k: Number of documents to return

        Returns:
            Reranked list of documents
        """
        if not self._reranker:
            return documents[:k]

        try:
            # Prepare sentence pairs for reranking
            sentence_pairs = [(query, doc.page_content) for doc in documents]

            # Get scores from reranker
            scores = self._reranker.predict(sentence_pairs)

            # Create list of (score, doc) tuples and sort by score
            scored_docs = list(zip(scores, documents))
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            # Update metadata with scores
            for i, (score, doc) in enumerate(scored_docs[:k]):
                doc.metadata["rerank_score"] = float(score)
                doc.metadata["rerank_position"] = i + 1

            logger.debug(f"Reranking completed for {len(scored_docs)} documents")

            return [doc for _, doc in scored_docs[:k]]

        except Exception as e:
            logger.warning(f"Error during reranking: {str(e)}")
            logger.warning("Falling back to original document order")
            return documents[:k]

    def get_rag_sources(self, question: str) -> Dict[str, Any]:
        """
        Get relevant documents for a question using RAG.

        Args:
            question: The question to find sources for

        Returns:
            Dictionary containing question, answer (document content), and metadata
        """
        logger.debug(f"Getting RAG sources for: '{question}'")

        answer: Dict[str, Any] = {"question": question}

        # Get initial documents without reranking
        retrieved_docs, metadata = self.retriever.retrieve_relevant_chunks(
            query=question,
            knowledge_base=self._vector_store,
            reranker=None,  # We'll do reranking separately
        )

        # Rerank documents
        reranked_docs = self.rerank_documents(
            question, retrieved_docs, k=self.retriever.num_docs_final
        )

        # Format document content
        docs_content = "\n\n".join(
            [
                f"{idx+1}. [Source: {doc.metadata.get('title', 'Unknown Title')} - {doc.metadata.get('source', '')}] \"{doc.page_content}\""
                for idx, doc in enumerate(reranked_docs)
            ]
        )

        answer["answer"] = docs_content
        answer["metadata"] = metadata

        return answer

    def invoke_rag(self, question: str) -> Dict[str, Any]:
        """
        Runs the RAG query against the vector store and returns an answer to the question.
        Args:
            question: a string containing the question to answer.

        Returns:
            A dictionary containing the question, answer, and metadata.
        """
        logger.debug(f"Running RAG query: '{question}'")

        answer: Dict[str, Any] = {"question": question}

        docs_content = self.get_rag_sources(question)["answer"]

        with track_time("Model invocation with RAG context"):

            if self._prompt is None:
                raise Exception("Prompt is not set")

            rag_prompt = self._prompt.format(question=question, context=docs_content)
            message = SystemMessage(rag_prompt)
            messages = self.chat_model.invoke([message])

        answer["answer"] = str(messages.content) if messages.content is not None else ""

        return answer

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Override BaseTool._run to ensure we use our implementation"""
        logger.debug(f"[RAGOwlAgent._run] Called with query: {query}")
        return self.message_invoke(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Override BaseTool._arun to ensure we use our implementation"""
        return self.message_invoke(query)

    def message_invoke(self, message: str) -> str:
        """Override OwlAgent.message_invoke with RAG specific implementation"""
        logger.debug(
            f"[RAGOwlAgent.message_invoke] Called from {self.name} with message: {message}"
        )
        logger.warning(f"RAG engine not keeping context for now")
        answer = self.invoke_rag(message)
        if "answer" not in answer or answer["answer"] == "":
            raise Exception("No answer found")
        return answer.get("answer", "?????")

    async def stream_message(self, message: str):
        """Stream a response from the RAG agent"""
        logger.debug(
            f"[RAGOwlAgent.stream_message] Called from {self.name} with message: {message}"
        )

        # Get initial documents without reranking
        retrieved_docs, metadata = self.retriever.retrieve_relevant_chunks(
            query=message,
            knowledge_base=self._vector_store,
            reranker=None,  # Skip reranking in the retriever
        )

        # Perform reranking separately with error handling
        reranked_docs = self.rerank_documents(
            message, retrieved_docs, k=self.retriever.num_docs_final
        )

        # Format documents content
        docs_content = self.get_rag_sources(message)["answer"]

        if self._prompt is None:
            raise Exception("Prompt is not set")

        # Create the RAG prompt
        rag_prompt = self._prompt.format(question=message, context=docs_content)

        # Stream the response
        async for chunk in self.chat_model.astream([SystemMessage(rag_prompt)]):
            if chunk.content:
                yield chunk.content


def main():
    logger.info("Starting main function")


if __name__ == "__main__":
    main()
