print("Loading rag module")
from typing import Optional, List, Tuple, Any
import os
import time
import logging
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from ragatouille import RAGPretrainedModel
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from .core import OwlAgent
from .db import TOOLS_CONFIG
import warnings

logging.getLogger("tqdm").setLevel(logging.WARNING)
warnings.simplefilter("ignore", category=FutureWarning)
import sentence_transformers

sentence_transformers.util.tqdm = lambda x, *args, **kwargs: x

logger = logging.getLogger("ragtool")


class OwlMemoryInput(BaseModel):

    query: str = Field(
        description="a natural language question to answer from the knowledge base"
    )


class LocalRAGTool(OwlAgent):

    _prompt = None
    _vector_stores = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embeddings_model_name = TOOLS_CONFIG["owl_memory_tool"]["embeddings_model_name"]
        self._embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model_name,
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
        reranker_name = TOOLS_CONFIG["owl_memory_tool"]["reranker_name"]
        self._reranker = RAGPretrainedModel.from_pretrained(reranker_name)
        self._prompt = PromptTemplate.from_template(self.system_prompt)

        input_data_folders = TOOLS_CONFIG["owl_memory_tool"]["input_data_folders"]

        self._vector_stores = None
        for ifolder in input_data_folders:
            current_store = self.load_vector_store(ifolder, self._embeddings)
            if self._vector_stores is None:
                self._vector_stores = current_store
            else:
                self._vector_stores.merge_from(current_store)

        if self._vector_stores is None:
            raise ValueError("No vector stores found")

        logger.info(f"Loaded dataset stores: {input_data_folders}")

    def rag_question(self, question: str) -> str:
        """
        Runs the RAG query against the vector store and returns an answer to the question.
        Args:
            question: a string containing the question to answer.
        """

        k = TOOLS_CONFIG["owl_memory_tool"]["num_retrieved_docs"]
        k_final = TOOLS_CONFIG["owl_memory_tool"]["num_docs_final"]

        retrieved_docs, reranked_docs = self.retrieve_relevant_chunks(
            query=question,
            knowledge_base=self._vector_stores,
            reranker=self._reranker,
            num_retrieved_docs=k,
            num_docs_final=k_final,
        )

        def _encode_text(text: str) -> str:
            return text.encode("ascii", errors="replace").decode("utf-8")

        if reranked_docs is not None:
            for doc in reranked_docs:
                logger.debug(
                    f"Reranked document: {doc['rank']} {doc['score']} {_encode_text(doc['content'][:100])}"
                )
            docs_content = "\n\n".join(
                _encode_text(doc["content"]) for doc in reranked_docs
            )
        else:
            docs_content = "\n\n".join(
                _encode_text(doc.page_content) for doc in retrieved_docs
            )

        message_with_question_and_context = self._prompt.format(
            question=question, context=docs_content
        )
        currated_message_with_question_and_context = (
            message_with_question_and_context.encode("ascii", errors="replace").decode(
                "utf-8"
            )
        )
        messages = [SystemMessage(currated_message_with_question_and_context)]
        messages = self.chat_model.invoke(messages)

        logger.debug(f"Raw RAG answer: {messages.content}")
        return messages.content

    def load_vector_store(
        self, input_data_folder: str, embedding_model: HuggingFaceEmbeddings
    ):
        file_path = f"{input_data_folder}/vector_db"

        if os.path.exists(file_path):
            logger.info(f"Loading the vector database from disk: {file_path}")
            start_time = time.time()
            KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
                file_path,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
                allow_dangerous_deserialization=True,
            )
            end_time = time.time()
            logger.info(
                f"Vector database loaded from disk in {end_time - start_time:.2f} seconds"
            )
        else:
            logger.error(f"Vector database not found in {file_path}")

        return KNOWLEDGE_VECTOR_DATABASE

    def retrieve_relevant_chunks(
        self,
        query: str,
        knowledge_base: FAISS,
        reranker: Optional[RAGPretrainedModel] = None,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
    ) -> Tuple[List[LangchainDocument], List[dict[str, Any]]]:
        """
        Retrieve the k most relevant document chunks for a given query.

        Args:
            query: The user query to find relevant documents for
            knowledge_base: The vector database containing indexed documents
            k: Number of documents to retrieve (default 5)

        Returns:
            List of retrieved LangchainDocument objects
        """
        logger.info(
            f"Starting retrieval for query: {query} with k={num_retrieved_docs}"
        )
        start_time = time.time()
        retrieved_docs = knowledge_base.similarity_search(
            query=query, k=num_retrieved_docs
        )
        end_time = time.time()

        logger.info(
            f"{len(retrieved_docs)} documents retrieved in {end_time - start_time:.2f} seconds"
        )
        # logger.debug(f"Top documents: {retrieved_docs[0].page_content}")
        logger.debug(f"Top document metadata: {retrieved_docs[0].metadata}")

        # Optionally rerank results
        reranked_docs = None
        if reranker:
            logger.info("Reranking documents chunks please wait...")
            start_time = time.time()
            relevant_docs_content_only = [
                doc.page_content for doc in retrieved_docs
            ]  # Keep only the text
            reranked_docs = reranker.rerank(
                query, relevant_docs_content_only, k=num_docs_final
            )
            # reranked_docs = [doc["content"] for doc in reranked_docs]
            end_time = time.time()
            logger.info(f"Documents reranked in {end_time - start_time:.2f} seconds")
            # Careful the docs encoding my be crap here

        retrieved_docs = retrieved_docs[:num_docs_final]

        return retrieved_docs, reranked_docs


class OwlMemoryTool(BaseTool, LocalRAGTool):
    """Tool that retrieves information from the owl memory base"""

    name: str = "owl_memory_tool"
    description: str = "Gets answers from the knowledge base"
    args_schema: Type[BaseModel] = OwlMemoryInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Union[List[Dict[str, str]], str], Dict]:
        """Use the tool."""
        return self.rag_question(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Union[List[Dict[str, str]], str], Dict]:
        """Use the tool asynchronously."""
        return self.rag_question(query)
