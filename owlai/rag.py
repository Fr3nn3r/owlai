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

    def generate_test_report(
        self, json_file_path: str, output_html_path: str = None
    ) -> str:
        """
        Generate a professional HTML test report from JSON test results.

        Args:
            json_file_path: Path to the JSON file containing test results
            output_html_path: Optional path to save the HTML report. If None, returns the HTML string.

        Returns:
            HTML string if output_html_path is None, otherwise None
        """
        import json
        from datetime import datetime
        import os

        # Read JSON data
        with open(json_file_path, "r", encoding="utf-8") as f:
            test_results = json.load(f)

        # Generate HTML report
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RAG System Test Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                    padding-bottom: 20px;
                    border-bottom: 2px solid #eee;
                }}
                .header h1 {{
                    color: #2c3e50;
                    margin-bottom: 10px;
                }}
                .metadata {{
                    color: #666;
                    font-size: 0.9em;
                }}
                .test-case {{
                    margin-bottom: 30px;
                    padding: 20px;
                    border: 1px solid #e0e0e0;
                    border-radius: 5px;
                    background-color: #fff;
                }}
                .test-case:hover {{
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    transition: box-shadow 0.3s ease;
                }}
                .question {{
                    color: #2c3e50;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .answer {{
                    color: #34495e;
                    white-space: pre-wrap;
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-top: 10px;
                }}
                .stats {{
                    display: flex;
                    justify-content: space-around;
                    margin: 20px 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                .stat-item {{
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .stat-label {{
                    color: #666;
                    font-size: 0.9em;
                }}
                @media (max-width: 768px) {{
                    .container {{
                        padding: 15px;
                    }}
                    .stats {{
                        flex-direction: column;
                        gap: 15px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>RAG System Test Report</h1>
                    <div class="metadata">
                        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                        Test File: {os.path.basename(json_file_path)}
                    </div>
                </div>

                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-value">{len(test_results)}</div>
                        <div class="stat-label">Total Test Cases</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{sum(1 for result in test_results if result.get('answer', '').strip())}</div>
                        <div class="stat-label">Answered Questions</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{sum(1 for result in test_results if not result.get('answer', '').strip())}</div>
                        <div class="stat-label">Unanswered Questions</div>
                    </div>
                </div>

                <div class="test-cases">
        """

        for i, result in enumerate(test_results, 1):
            html += f"""
                    <div class="test-case">
                        <div class="question">Q{i}: {result['question']}</div>
                        <div class="answer">{result.get('answer', 'No answer provided')}</div>
                    </div>
            """

        html += """
                </div>
            </div>
        </body>
        </html>
        """

        if output_html_path:
            with open(output_html_path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"Test report generated and saved to: {output_html_path}")
            return None
        return html

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
        plt.savefig(f"{input_data_folder}/chunk_size_distribution-{filename}.png")
        plt.close()
        logger.info(
            f"Distribution of document lengths (in count of tokens) saved to chunk_size_distribution-{filename}.png"
        )

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
    ) -> FAISS:
        """
        Loads an existing vector store or creates a new one if it doesn't exist.
        Processes documents one by one to manage memory usage.

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
                current_docs = [
                    LangchainDocument(
                        page_content=doc.page_content,
                        metadata={
                            "source": doc.metadata["source"],
                        },
                    )
                    for doc in docs
                ]

                # Analyze document chunks before splitting
                self.analyze_chunk_size_distribution(
                    input_data_folder,
                    "pre-split-" + filename,
                    current_docs,
                    embedding_model.model_name,
                )

                # Split documents
                split_docs = self.split_documents(
                    chunk_size,
                    current_docs,
                    tokenizer_name=embedding_model.model_name,
                )

                self.analyze_chunk_size_distribution(
                    input_data_folder,
                    "post-split-" + filename,
                    split_docs,
                    embedding_model.model_name,
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
                continue

        total_time = time.time() - start_time
        logger.info(f"Vector database created in {total_time:.2f} seconds")

        # Save to disk
        vector_store.save_local(file_path)
        logger.info(f"Vector database saved to {file_path}")

        return vector_store

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
