print("Loading rag module")
from typing import Optional, List, Tuple, Any, Callable
import os
import time
import logging
import warnings
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from ragatouille import RAGPretrainedModel
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, ArgsSchema
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_community.document_loaders import TextLoader
import traceback
from langchain_core.tools.base import ArgsSchema
import fitz
from fitz import Page
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import (
    HuggingFaceEmbeddings as LegacyHuggingFaceEmbeddings,
)

from owlai.owlsys import encode_text
import sys
from pathlib import Path
from owlai.vector_store import VectorStore

# sys.path.append(str(Path(__file__).parent.parent))

from owlai.core import OwlAgent
from owlai.core.logging_setup import get_logger
from owlai.db import TOOLS_CONFIG, RAG_AGENTS_CONFIG
from owlai.owlsys import track_time
from tqdm import tqdm

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="langchain")

import fitz

logger = get_logger("main")

from sentence_transformers import CrossEncoder
from langchain.prompts import PromptTemplate
from owlai.core.rag_config import RAGConfig


class OwlMemoryInput(BaseModel):
    """Input schema for OwlMemoryTool."""

    query: str = Field(
        description="a natural language question to answer from the knowledge base"
    )


""" Class config for HuggingFace embeddings, and FAISS vector store """


class RAGConfig(BaseModel):
    num_retrieved_docs: int
    num_docs_final: int
    embeddings_model_name: str
    reranker_name: str
    input_data_folders: List[str]
    model_kwargs: Dict[str, Any]
    encode_kwargs: Dict[str, Any]
    multi_process: bool = True
    default_queries: Optional[List[str]] = None


""" Class based on HuggingFace embeddings, and FAISS vector store """


class RAGOwlAgent(OwlAgent):

    # JSON defined properties
    retriever: RAGConfig
    default_queries: Optional[List[str]] = None

    # Runtime updated properties
    _init_completed = False
    _prompt = None
    _vector_stores = None
    _embeddings = None
    _reranker = None

    def __init__(self, config, embedding_model=None, *args, **kwargs):
        # Initialize base class first
        super().__init__(config, *args, **kwargs)

        self.retriever = config.retriever

        # Set default queries from config
        self.default_queries = config.default_queries

        # Set up embeddings
        self._embeddings = (
            embedding_model
            if embedding_model
            else HuggingFaceEmbeddings(
                model_name=config.retriever.embeddings_model_name,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        )

        # Set up reranker
        self._reranker = RAGPretrainedModel.from_pretrained(
            config.retriever.reranker_name
        )

        # Load datasets from input data folders

        self.vector_store = VectorStore(
            self._embeddings, self.retriever.input_data_folders
        )

        self._init_completed = True

    def retrieve_relevant_chunks(
        self,
        query: str,
        knowledge_base: Optional[FAISS],
        reranker: Optional[RAGPretrainedModel] = None,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
    ) -> Tuple[List[LangchainDocument], dict]:
        """
        Retrieve the k most relevant document chunks for a given query.

        Args:
            query: The user query to find relevant documents for
            knowledge_base: The vector database containing indexed documents
            reranker: Optional reranker model to rerank results
            num_retrieved_docs: Number of initial documents to retrieve
            num_docs_final: Number of documents to return after reranking

        Returns:
            Tuple containing a list of retrieved and reranked LangchainDocument objects with scores and metadata
        """
        logger.info(
            f"Starting retrieval for query: '{query}' with k={num_retrieved_docs}"
        )
        metadata = {
            "query": query,
            "k": num_retrieved_docs,
            "num_docs_final": num_docs_final,
        }

        if knowledge_base is None:
            logger.warning("Knowledge base is None, returning empty results")
            return [], metadata

        with track_time(f"Documents search", metadata):
            try:
                retrieved_docs = knowledge_base.similarity_search(
                    query=query, k=min(num_retrieved_docs, knowledge_base.index.ntotal)
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
            except Exception as e:
                logger.error(f"Error during similarity search: {str(e)}")
                return [], metadata

        # If no reranker or no docs retrieved, just return top k docs
        if not reranker or not retrieved_docs:
            return retrieved_docs[:num_docs_final], metadata

        # Rerank results
        logger.debug(
            f"Reranking {len(retrieved_docs)} documents chunks to {num_docs_final} please wait..."
        )

        with track_time("Documents chunks reranking", metadata):
            try:
                # Create mapping of content to original doc for later matching
                content_to_doc = {doc.page_content: doc for doc in retrieved_docs}

                # Get reranked results
                reranked_results = reranker.rerank(
                    query,
                    [doc.page_content for doc in retrieved_docs],
                    k=num_docs_final,
                )

                if not reranked_results:
                    logger.warning("Reranker returned no results, using original order")
                    return retrieved_docs[:num_docs_final], metadata

                # Match reranked results back to original docs and add scores to doc metadata
                reranked_docs = []
                for rank, result in enumerate(reranked_results):
                    doc = content_to_doc[result["content"]]
                    doc.metadata["rerank_score"] = result["score"]
                    doc.metadata["rerank_position"] = result["rank"]
                    reranked_docs.append(doc)

                # Add reranked docs metadata
                metadata["selected_docs"] = {
                    i: {
                        "title": doc.metadata.get("title", "No title"),
                        "source": doc.metadata.get("source", "Unknown source"),
                        "rerank_score": doc.metadata.get("rerank_score", 0.0),
                        "rerank_position": doc.metadata.get("rerank_position", -1),
                    }
                    for i, doc in enumerate(reranked_docs)
                }

                for i in range(min(5, len(reranked_docs))):
                    if reranked_docs[i].metadata.get("rerank_score", 0.0) < 15:
                        logger.warning(
                            f"Reranked doc {i} has a score of {reranked_docs[i].metadata.get('rerank_score', 0.0)}"
                        )

                return reranked_docs, metadata

            except Exception as e:
                logger.error(f"Error during reranking: {str(e)}")
                # Fall back to original order if reranking fails
                return retrieved_docs[:num_docs_final], metadata

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
        answer = self.rag_question(message)
        if "answer" not in answer or answer["answer"] == "":
            raise Exception("No answer found")
        return answer.get("answer", "?????")

    def load_dataset_from_split_docs(
        self,
        split_docs: List[LangchainDocument],
        input_data_folder: str,
        input_store: Optional[FAISS] = None,
    ) -> Optional[FAISS]:
        """
        Loads a dataset from pre-split documents into a FAISS vector store.

        Args:
            split_docs: List of pre-split LangchainDocument objects
            input_data_folder: Path to the folder containing documents
            input_store: Optional existing FAISS vector store to merge into

        Returns:
            FAISS vector store or None if no documents were processed

        Raises:
            ValueError: If input_store is provided but is not a FAISS instance
        """
        vector_db_file_path = f"{input_data_folder}/vector_db"
        vector_store = input_store

        if self._embeddings is None:
            raise Exception("No embedding model provided")

        if input_store is not None and not isinstance(input_store, FAISS):
            raise ValueError("input_store must be a FAISS instance")

        logger.debug(f"Vector store: {vector_store}")

        if os.path.exists(vector_db_file_path) and vector_store is None:
            logger.info(f"Loading existing vector database from: {vector_db_file_path}")
            vector_store = self.load_vector_store(input_data_folder, self._embeddings)

        if vector_store is None:
            logger.info(
                f"Creating new vector database from {len(split_docs)} documents"
            )
            vector_store = FAISS.from_documents(
                split_docs,
                self._embeddings,
                distance_strategy=DistanceStrategy.COSINE,
            )
        else:
            logger.info(
                f"Merging {len(split_docs)} documents into existing vector database"
            )
            batch_store = FAISS.from_documents(
                split_docs,
                self._embeddings,
                distance_strategy=DistanceStrategy.COSINE,
            )
            vector_store.merge_from(batch_store)

        # Save to disk
        vector_store.save_local(vector_db_file_path)
        logger.info(f"Vector database saved to {vector_db_file_path}")

        return vector_store

    def rag_question(self, question: str) -> Dict[str, Any]:
        """
        Runs the RAG query against the vector store and returns an answer to the question.
        Args:
            question: a string containing the question to answer.

        Returns:
            A dictionary containing the question, answer, and metadata.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        logger.debug(f"Running RAG query: '{question}'")

        answer: Dict[str, Any] = {"question": question}

        # TODO: think about passing parameters in a structure
        k = self.retriever.num_retrieved_docs
        k_final = self.retriever.num_docs_final

        reranked_docs, metadata = self.retrieve_relevant_chunks(
            query=question,
            knowledge_base=self._vector_stores,
            reranker=self._reranker,
            num_retrieved_docs=k,
            num_docs_final=k_final,
        )

        if not reranked_docs:
            answer["answer"] = "I don't know based on the provided sources."
            answer["metadata"] = metadata
            return answer

        with track_time("Model invocation with RAG context", metadata):
            docs_content = "\n\n".join(
                [
                    f"{idx+1}. [Source : {doc.metadata.get('title', 'Unknown Title')} - {doc.metadata.get('source', 'Unknown Source')}] \"{doc.page_content}\""
                    for idx, doc in enumerate(reranked_docs)
                ]
            )

            if self._prompt is None:
                raise Exception("Prompt is not set")

            rag_prompt = self._prompt.format(question=question, context=docs_content)
            rag_prompt = encode_text(rag_prompt)
            # Add the RAG prompt to the metadata for debugging and analysis purposes
            metadata["rag_prompt"] = rag_prompt

            # logger.debug(f"Final prompt: {rag_prompt}")
            message = SystemMessage(rag_prompt)
            messages = self.chat_model.invoke([message])
        # logger.debug(f"Raw RAG answer: {messages.content}")

        answer["answer"] = (
            encode_text(str(messages.content))
            if messages.content is not None
            else "I don't know based on the provided sources."
        )
        answer["metadata"] = metadata

        return answer


def main():
    # Import fitz here to prevent reloading in concurrent processes
    import fitz

    config = RAG_AGENTS_CONFIG[0]
    load_logger_config()

    rag_tool = RAGOwlAgent(**config)

    if hasattr(rag_tool, "default_queries") and rag_tool.default_queries:
        for iq in rag_tool.default_queries:
            logger.info(iq)
            logger.info(rag_tool.rag_question(iq).get("answer"))
            logger.info("-" * 100)


if __name__ == "__main__":
    main()
