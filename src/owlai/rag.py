print("Loading rag module")
from typing import Optional, List, Tuple, Any, Callable
import os
import time
import logging
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from .core import OwlAgent
from .db import TOOLS_CONFIG
import warnings
from tqdm import tqdm
from owlai.owlsys import track_time

warnings.simplefilter("ignore", category=FutureWarning)

logger = logging.getLogger("ragtool")


class OwlMemoryInput(BaseModel):
    """Input schema for OwlMemoryTool."""

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
            print(f"Loading dataset from {ifolder}")
            current_store = self.load_dataset(ifolder, self._embeddings)
            if current_store is not None:
                if self._vector_stores is None:
                    self._vector_stores = current_store
                else:
                    print(f"Merging dataset from {ifolder}")
                    self._vector_stores.merge_from(current_store)

        if self._vector_stores is None:
            logger.warning(
                "No vector stores found: you must set the vector store manually."
            )
        else:
            logger.info(f"Loaded dataset stores: {input_data_folders}")

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

        logger.debug(
            f"Splitted {len(knowledge_base)} documents into {len(docs_processed)} chunks"
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

    def load_dataset(
        self,
        input_data_folder: str,
        embedding_model: HuggingFaceEmbeddings,
        chunk_size: int = 512,
        metadata_extractor: Optional[Callable] = None,
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
            vector_store = self.load_vector_store(input_data_folder, embedding_model)

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

    def load_and_split_document(
        self,
        filepath: str,
        input_data_folder: str,
        filename: str,
        chunk_size: int,
        model_name: str,
        metadata_extractor: Optional[Callable] = None,
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
            loader = PyPDFLoader(
                file_path=filepath,
                extract_images=False,
                extraction_mode="plain",
            )
            docs = loader.lazy_load()
        else:  # .txt files
            loader = TextLoader(filepath)
            docs = loader.lazy_load()

        # Convert to LangchainDocuments
        total_pages = metadata.get("num_pages", 0)
        loaded_docs: List[LangchainDocument] = []
        with track_time(f"Loading document: '{filename}'"):
            for page_number, doc in tqdm(
                enumerate(docs),
                total=total_pages,
                desc=f"Loading pages from {filename}",
            ):
                # logger.debug(f"Loading page {page_number} of {total_pages}")
                metadata.update(doc.metadata)
                metadata["source"] = f"{filename}:{page_number}"

                loaded_docs.append(
                    LangchainDocument(
                        page_content=doc.page_content,
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

    def rag_question(self, question: str) -> Dict[str, Any]:
        """
        Runs the RAG query against the vector store and returns an answer to the question.
        Args:
            question: a string containing the question to answer.

        Returns:
            A dictionary containing the question, answer, and metadata.
        """

        answer: Dict[str, Any] = {"question": question}

        # TODO: think about passing parameters in a structure
        k = TOOLS_CONFIG["owl_memory_tool"]["num_retrieved_docs"]
        k_final = TOOLS_CONFIG["owl_memory_tool"]["num_docs_final"]

        reranked_docs, metadata = self.retrieve_relevant_chunks(
            query=question,
            knowledge_base=self._vector_stores,
            reranker=self._reranker,
            num_retrieved_docs=k,
            num_docs_final=k_final,
        )

        with track_time("Model invocation with RAG context", metadata):

            def _encode_text(text: str) -> str:
                return text.encode("ascii", errors="replace").decode("utf-8")

            docs_content = "\n\n".join(
                _encode_text(doc.page_content) for doc in reranked_docs
            )

            docs_content = "\n\n".join(
                [
                    f"{idx+1}. [Source : {doc.metadata.get('title', 'Unknown Title')} - {doc.metadata.get('source', 'Unknown Source')}] \"{doc.page_content}\""
                    for idx, doc in enumerate(reranked_docs)
                ]
            )

            if self._prompt is None:
                raise Exception("Prompt is not set")

            rag_prompt = self._prompt.format(question=question, context=docs_content)
            rag_prompt = rag_prompt.encode("ascii", errors="replace").decode("utf-8")

            logger.debug(f"Final prompt: {rag_prompt}")
            messages = [SystemMessage(rag_prompt)]
            messages = self.chat_model.invoke(messages)

        logger.debug(f"Raw RAG answer: {messages.content}")

        answer["answer"] = str(messages.content) if messages.content is not None else ""
        answer["metadata"] = metadata

        return answer

    def load_vector_store(
        self, input_data_folder: str, embedding_model: HuggingFaceEmbeddings
    ) -> Optional[FAISS]:
        file_path = f"{input_data_folder}/vector_db"
        KNOWLEDGE_VECTOR_DATABASE = None

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
        knowledge_base: Optional[FAISS],
        reranker: Optional[RAGPretrainedModel] = None,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
    ) -> Tuple[List[LangchainDocument], dict]:
        """
        Retrieve the k most relevant document chunks for a given query.

        Args:0
            query: The user query to find relevant documents for
            knowledge_base: The vector database containing indexed documents
            reranker: Optional reranker model to rerank results
            num_retrieved_docs: Number of initial documents to retrieve
            num_docs_final: Number of documents to return after reranking

        Returns:
            Tuple containing a list of retrieved and reranked LangchainDocument objects with scores and metadata
        """
        logger.debug(
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
            retrieved_docs = knowledge_base.similarity_search(
                query=query, k=num_retrieved_docs
            )
            print([doc.metadata for doc in retrieved_docs])
            metadata["num_docs_retrieved"] = len(retrieved_docs)
            metadata["retrieved_docs"] = {
                i: {
                    "title": doc.metadata.get("title", "No title"),
                    "source": doc.metadata.get("source", "Unknown source"),
                }
                for i, doc in enumerate(retrieved_docs)
            }
            logger.debug(f"{len(retrieved_docs)} documents retrieved")
        # print([doc.metadata for doc in retrieved_docs])

        # If no reranker, just return top k docs
        if not reranker:
            return retrieved_docs[:num_docs_final], metadata

        # Rerank results
        logger.debug("Reranking documents chunks please wait...")

        with track_time("Documents chunks reranking", metadata):

            # Create mapping of content to original doc for later matching
            content_to_doc = {doc.page_content: doc for doc in retrieved_docs}

            # Get reranked results
            reranked_results = reranker.rerank(
                query, [doc.page_content for doc in retrieved_docs], k=num_docs_final
            )

            # Match reranked results back to original docs and add scores to doc metadata
            reranked_docs = []
            for rank, result in enumerate(reranked_results):
                doc = content_to_doc[result["content"]]
                doc.metadata["rerank_score"] = result["score"]
                doc.metadata["rerank_position"] = result["rank"]
                reranked_docs.append(doc)

            # Add reranked docs metadata to metadata (you can't have too much metadata)
            metadata["reranked_docs"] = {
                i: {
                    "title": doc.metadata.get("title", "No title"),
                    "source": doc.metadata.get("source", "Unknown source"),
                    "rerank_score": doc.metadata.get("rerank_score", 0.0),
                    "rerank_position": doc.metadata.get("rerank_position", -1),
                }
                for i, doc in enumerate(reranked_docs)
            }

        for i in range(min(5, len(reranked_docs))):
            logger.debug(f"Reranked doc {i}: {reranked_docs[i].metadata}")

        return reranked_docs, metadata


class OwlMemoryTool(BaseTool):
    """Tool that retrieves information from the owl memory base"""

    name: str = "owl_memory_tool"
    description: str = "Gets answers from the knowledge base"
    args_schema: Type[BaseModel] = OwlMemoryInput
    _rag_tool = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create the RAG tool component instead of inheriting from it
        logger.info("Initializing RAG tool")
        self._rag_tool = LocalRAGTool(**kwargs)

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        """Process tool inputs according to BaseTool pattern."""
        logger.debug(f"Tool invoked with input: {input}")
        if isinstance(input, dict) and "query" in input:
            return self._run(input["query"])
        elif isinstance(input, str):
            return self._run(input)
        else:
            return self._run(str(input))

    # For OwlAgent compatibility
    def message_invoke(self, message: str) -> str:
        """Handle invoke calls from OwlAgent pattern."""
        return self._run(message)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        logger.debug(f"Running tool with query: {query}")
        if self._rag_tool is None:
            return "Error: RAG tool is not initialized"
        answer = self._rag_tool.rag_question(query)
        return answer["answer"]

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        if self._rag_tool is None:
            return "Error: RAG tool is not initialized"
        answer = self._rag_tool.rag_question(query)
        return answer["answer"]
