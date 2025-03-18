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
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from .core import OwlAgent
from .db import TOOLS_CONFIG
import warnings
from tqdm import tqdm

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
            logger.warning(
                "No vector stores found: you must set the vector store manually."
            )
        else:
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

    def analyze_chunk_size_distribution(
        self, input_data_folder, filename, docs, model_name="thenlper/gte-small"
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
        logger.info(info_message)

        # Analyze token lengths
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        lengths = [len(tokenizer.encode(doc.page_content)) for doc in docs]

        # Plot distribution
        import matplotlib.pyplot as plt
        import pandas as pd

        fig = pd.Series(lengths).hist()
        plt.title("Distribution of document lengths (in count of tokens)")
        file_path = f"{input_data_folder}/chunk_size_distribution-{filename}.png"
        plt.savefig(file_path)
        plt.close()
        logger.info(
            f"Distribution of document lengths (in count of tokens) saved to chunk_size_distribution-{filename}.png"
        )
        return file_path

    def split_documents(
        self,
        chunk_size: int,  # The maximum number of tokens in a chunk
        knowledge_base: List[LangchainDocument],
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
        logger.debug(f"Splitting {len(knowledge_base)} documents")

        docs_processed = []
        for doc in tqdm(knowledge_base, desc="Splitting documents"):
            result = text_splitter.split_documents([doc])
            docs_processed += result

        logger.info(
            f"Splitted {len(knowledge_base)} documents into {len(docs_processed)} chunks"
        )
        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in tqdm(docs_processed, desc="Removing duplicates"):
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        logger.info(
            f"Removed {len(docs_processed) - len(docs_processed_unique)} duplicates from {len(docs_processed)} chunks"
        )

        return docs_processed_unique

    def load_or_create_vector_store(
        self,
        input_data_folder: str,
        embedding_model: HuggingFaceEmbeddings,
        chunk_size: int = 512,
        metadata_extractor: Optional[callable] = None,
    ) -> FAISS:
        """
        Loads an existing vector store or creates a new one if it doesn't exist.
        Processes documents one by one to manage memory usage.

        Args:
            input_data_folder: Path to the folder containing documents
            embedding_model: The embedding model to use
            chunk_size: Size of text chunks for splitting documents
            metadata_extractor: Optional callback function for extracting metadata

        Returns:
            FAISS vector store
        """
        file_path = f"{input_data_folder}/vector_db"

        if os.path.exists(file_path):
            logger.info(f"Loading existing vector database from: {file_path}")
            return self.load_vector_store(input_data_folder, embedding_model)

        logger.info("Creating new vector database...")
        vector_store = None

        # Get list of PDF and text files
        files = [
            f for f in os.listdir(input_data_folder) if f.endswith((".pdf", ".txt"))
        ]
        logger.info(f"Found {len(files)} documents to process in {input_data_folder}")

        start_time = time.time()

        # Process each file individually
        for filename in tqdm(files, desc="Processing documents"):
            filepath = os.path.join(input_data_folder, filename)
            logger.info(
                f"Processing file: {filename} size: {os.path.getsize(filepath)}"
            )

            try:
                split_docs = self.load_and_split_documents(
                    filepath,
                    input_data_folder,
                    filename,
                    chunk_size,
                    embedding_model.model_name,
                    metadata_extractor,
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

                logger.info(
                    f"Processed {filename} in {time.time() - start_time:.2f} seconds"
                )

            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                logger.error(f"Error details: {traceback.format_exc()}")
                continue

        total_time = time.time() - start_time
        logger.info(f"Vector database created in {total_time:.2f} seconds")

        # Save to disk
        vector_store.save_local(file_path)
        logger.info(f"Vector database saved to {file_path}")

        return vector_store

    def load_and_split_documents(
        self,
        filepath: str,
        input_data_folder: str,
        filename: str,
        chunk_size: int,
        model_name: str,
        metadata_extractor: Optional[callable] = None,
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
        # Load document
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(
                file_path=filepath,
                extract_images=False,
                extraction_mode="plain",
            )
            docs = loader.load()
        else:  # .txt files
            loader = TextLoader(filepath)
            docs = loader.load()

        # Convert to LangchainDocuments
        current_docs: List[LangchainDocument] = []
        for doc in docs:
            metadata = {
                "source": doc.metadata["source"],
            }

            # Call metadata extractor if provided
            if metadata_extractor:
                try:
                    additional_metadata = metadata_extractor(filepath)
                    if additional_metadata and isinstance(additional_metadata, dict):
                        metadata.update(additional_metadata)
                except Exception as e:
                    logger.error(
                        f"Error in metadata extractor for {filename}: {str(e)}"
                    )
                    logger.error(f"Error details: {traceback.format_exc()}")

            current_docs.append(
                LangchainDocument(
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )

        # Analyze document chunks before splitting
        pre_split_file = self.analyze_chunk_size_distribution(
            input_data_folder,
            "pre-split-" + filename,
            current_docs,
            model_name,
        )

        # Add pre-split distribution file path to metadata
        for doc in current_docs:
            doc.metadata["pre_split_distribution"] = pre_split_file

        # Split documents
        split_docs = self.split_documents(
            chunk_size,
            current_docs,
            tokenizer_name=model_name,
        )

        # Analyze post-split chunks and add to metadata
        post_split_file = self.analyze_chunk_size_distribution(
            input_data_folder,
            "post-split-" + filename,
            split_docs,
            model_name,
        )

        for i in range(min(5, len(split_docs))):
            logger.debug(f"Split doc {i}: {split_docs[i].metadata}")

        return split_docs

    def rag_question(self, question: str, no_context: bool = False) -> str:
        """
        Runs the RAG query against the vector store and returns an answer to the question.
        Args:
            question: a string containing the question to answer.
        """

        k = TOOLS_CONFIG["owl_memory_tool"]["num_retrieved_docs"]
        k_final = TOOLS_CONFIG["owl_memory_tool"]["num_docs_final"]

        # If no_context is True, the question is answered without context (for testing purposes)
        if no_context:
            message_with_question_no_context = self._prompt.format(
                question=question, context=""
            )
            messages = [SystemMessage(message_with_question_no_context)]
            messages = self.chat_model.invoke(messages)

            return messages.content

        else:
            reranked_docs = self.retrieve_relevant_chunks(
                query=question,
                knowledge_base=self._vector_stores,
                reranker=self._reranker,
                num_retrieved_docs=k,
                num_docs_final=k_final,
            )

        def _encode_text(text: str) -> str:
            return text.encode("ascii", errors="replace").decode("utf-8")

        docs_content = "\n\n".join(
            _encode_text(doc.page_content) for doc in reranked_docs
        )

        message_with_question_and_context = self._prompt.format(
            question=question, context=docs_content
        )
        currated_message_with_question_and_context = (
            message_with_question_and_context.encode("ascii", errors="replace").decode(
                "utf-8"
            )
        )

        logger.debug(f"Final prompt: {currated_message_with_question_and_context}")
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
    ) -> List[LangchainDocument]:
        """
        Retrieve the k most relevant document chunks for a given query.

        Args:
            query: The user query to find relevant documents for
            knowledge_base: The vector database containing indexed documents
            reranker: Optional reranker model to rerank results
            num_retrieved_docs: Number of initial documents to retrieve
            num_docs_final: Number of documents to return after reranking

        Returns:
            List of retrieved and reranked LangchainDocument objects with scores
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

        # If no reranker, just return top k docs
        if not reranker:
            return retrieved_docs[:num_docs_final]

        # Rerank results
        logger.info("Reranking documents chunks please wait...")
        start_time = time.time()

        # Create mapping of content to original doc for later matching
        content_to_doc = {doc.page_content: doc for doc in retrieved_docs}

        # Get reranked results
        reranked_results = reranker.rerank(
            query, [doc.page_content for doc in retrieved_docs], k=num_docs_final
        )
        end_time = time.time()
        logger.info(f"Documents reranked in {end_time - start_time:.2f} seconds")

        # Match reranked results back to original docs and add scores
        reranked_docs = []
        for rank, result in enumerate(reranked_results):
            doc = content_to_doc[result["content"]]
            doc.metadata["rerank_score"] = result["score"]
            doc.metadata["rerank_position"] = result["rank"]
            reranked_docs.append(doc)

        logger.debug(f"Top document metadata: {reranked_docs[0].metadata}")

        return reranked_docs


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
