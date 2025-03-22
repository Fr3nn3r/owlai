from typing import List, Dict, Any, Optional
from langchain_community.docstore.document import Document as LangchainDocument
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun
import logging
import os

from .document_parser import FrenchLawParser
from .vector_store import VectorStore

logger = logging.getLogger("main")


class RAGAgent:
    """RAG agent for French law question answering."""

    def __init__(
        self,
        embedding_model: HuggingFaceEmbeddings,
        llm: HuggingFaceHub,
        vector_store_path: Optional[str] = None,
    ):
        """
        Initialize the RAG agent.

        Args:
            embedding_model: The embedding model to use
            llm: The language model to use
            vector_store_path: Optional path to load existing vector store
        """
        self.parser = FrenchLawParser()
        self.vector_store = VectorStore(embedding_model)
        self.llm = llm

        if vector_store_path:
            self.vector_store.load_local(vector_store_path)

        # Create the QA chain
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        If the question is not in French, translate it to French first.
        Always answer in French.

        Context:
        {context}

        Question: {question}
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.docstore.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
        )

    def process_documents(self, pdf_path: str, chunk_size: int = 1000) -> None:
        """
        Process documents and add them to the vector store.

        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of document chunks
        """
        # Parse the PDF
        documents = self.parser.parse(pdf_path)

        # Split into chunks
        chunks = self.parser.split(documents, chunk_size=chunk_size)

        # Add to vector store
        self.vector_store.add_documents(chunks)

        logger.info(f"Processed {len(chunks)} chunks from {pdf_path}")

    def save_vector_store(self, path: str) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Path to save the vector store
        """
        self.vector_store.save_local(path)

    def load_vector_store(self, path: str) -> None:
        """
        Load the vector store from disk.

        Args:
            path: Path to load the vector store from
        """
        self.vector_store.load_local(path)

    def query(
        self,
        question: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG agent with a question.

        Args:
            question: The question to answer
            k: Number of documents to retrieve
            filter: Optional filter criteria

        Returns:
            Dictionary containing answer and source documents
        """
        # Run the chain
        result = self.chain(
            {"query": question},
            callbacks=CallbackManagerForChainRun.get_noop_manager(),
        )

        return {
            "answer": result["result"],
            "source_documents": result["source_documents"],
        }

    def get_document_count(self) -> int:
        """
        Get the number of documents in the vector store.

        Returns:
            Number of documents
        """
        return self.vector_store.get_document_count()

    def analyze_chunk_distribution(self, output_dir: str) -> None:
        """
        Analyze and visualize chunk size distribution.

        Args:
            output_dir: Directory to save analysis results
        """
        self.parser.analyze_chunk_size_distribution(
            self.vector_store.docstore.docstore.docs.values(),
            output_dir,
        )
