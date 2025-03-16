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
    _embeddings = None
    _reranker = None

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
            current_store = self.load_or_create_vector_store(ifolder, self._embeddings)
            if self._vector_stores is None:
                self._vector_stores = current_store
            else:
                self._vector_stores.merge_from(current_store)

        if self._vector_stores is None:
            raise ValueError("No vector stores found")

        logger.info(f"Loaded dataset stores: {input_data_folders}")

    def visualize_embeddings(
        self,
        knowledge_base: FAISS,
        docs_processed: List[LangchainDocument],
        user_query: str,
        query_vector: List[float],
    ):
        """
        Visualize document embeddings and query vector in 2D space using PaCMAP.

        Args:
            knowledge_base: The FAISS vector store
            docs_processed: List of processed documents
            user_query: The query string
            query_vector: The query's embedding vector
        """
        import pacmap
        import numpy as np
        import pandas as pd
        import plotly.express as px

        embedding_projector = pacmap.PaCMAP(
            n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1
        )

        embeddings_2d = [
            list(knowledge_base.index.reconstruct_n(idx, 1)[0])
            for idx in range(len(docs_processed))
        ] + [query_vector]

        # Fit the data
        documents_projected = embedding_projector.fit_transform(
            np.array(embeddings_2d), init="pca"
        )

        df = pd.DataFrame.from_dict(
            [
                {
                    "x": documents_projected[i, 0],
                    "y": documents_projected[i, 1],
                    "source": docs_processed[i].metadata["source"].split("/")[1],
                    "extract": docs_processed[i].page_content[:100] + "...",
                    "symbol": "circle",
                    "size_col": 4,
                }
                for i in range(len(docs_processed))
            ]
            + [
                {
                    "x": documents_projected[-1, 0],
                    "y": documents_projected[-1, 1],
                    "source": "User query",
                    "extract": user_query,
                    "size_col": 100,
                    "symbol": "star",
                }
            ]
        )

        # Visualize the embedding
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="source",
            hover_data="extract",
            size="size_col",
            symbol="symbol",
            color_discrete_map={"User query": "black"},
            width=1000,
            height=700,
        )
        fig.update_traces(
            marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )
        fig.update_layout(
            legend_title_text="<b>Chunk source</b>",
            title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>",
        )
        return fig

    def analyze_chunk_size_distribution(self, docs, model_name="thenlper/gte-small"):
        """
        Analyze and visualize document lengths before and after processing.

        Args:
            raw_docs: Original documents before splitting
            processed_docs: Documents after splitting
            model_name: Name of the embedding model to use
        """
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer

        # Get max sequence length from SentenceTransformer
        max_seq_len = SentenceTransformer(model_name).max_seq_length
        info_message = (
            f"Model's max sequence size: '{max_seq_len}' Document count: '{len(docs)}'"
        )
        logger.info(info_message)

        # Analyze token lengths
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        lengths = [len(tokenizer.encode(doc.page_content)) for doc in docs]

        # Plot distribution
        import matplotlib.pyplot as plt
        import pandas as pd

        fig = pd.Series(lengths).hist()
        plt.title("Distribution of document lengths (in count of tokens)")
        plt.savefig(f"chunk_size_distribution-{id(docs)}.png")
        plt.close()
        logger.info(
            f"Distribution of document lengths (in count of tokens) saved to chunk_size_distribution-{id(docs)}.png"
        )

    def load_or_create_vector_store(
        self,
        input_data_folder: str,
        embedding_model: HuggingFaceEmbeddings,
        chunk_size: int = 512,
    ) -> FAISS:
        """
        Loads an existing vector store or creates a new one if it doesn't exist.

        Args:
            input_data_folder: Path to the folder containing documents
            embedding_model: The embedding model to use
            chunk_size: Size of text chunks for splitting documents

        Returns:
            FAISS vector store
        """
        file_path = f"{input_data_folder}/vector_db"

        if os.path.exists(file_path):
            logger.info(f"Loading existing vector database from: {file_path}")
            return self.load_vector_store(input_data_folder, embedding_model)
        else:
            logger.info("Creating new vector database...")
            # Load raw documents
            start_time = time.time()
            raw_documents = self.load_documents(input_data_folder)
            logger.info(
                f"{len(raw_documents)} documents loaded in {time.time() - start_time:.2f} seconds"
            )

            # Analyze and split documents
            self.analyze_chunk_size_distribution(
                raw_documents, embedding_model.model_name
            )
            split_docs = self.split_documents(
                chunk_size, raw_documents, tokenizer_name=embedding_model.model_name
            )

            # Create and save vector store
            start_time = time.time()
            vector_store = FAISS.from_documents(
                split_docs,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
            )
            logger.info(
                f"Vector database created in {time.time() - start_time:.2f} seconds"
            )

            # Save to disk
            vector_store.save_local(file_path)

            return vector_store

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

        # logger.debug(f"Final prompt: {currated_message_with_question_and_context}")
        messages = [SystemMessage(currated_message_with_question_and_context)]
        messages = self.chat_model.invoke(messages)

        # logger.debug(f"Raw RAG answer: {messages.content}")
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
