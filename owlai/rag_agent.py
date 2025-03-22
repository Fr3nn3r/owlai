from typing import List, Dict, Any, Optional
from langchain_community.docstore.document import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
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
        llm: HuggingFaceEndpoint,
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
        self.chain = None  # Initialize chain as None, will be created when needed

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

        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

    def _ensure_chain(self):
        """Ensure the QA chain is created and ready to use."""
        if self.chain is None:
            if self.vector_store.docstore is None:
                raise ValueError(
                    "No documents available in the vector store. Please add documents first."
                )

            # Configure retriever with search parameters
            retriever = self.vector_store.docstore.as_retriever(
                search_kwargs={
                    "k": 4,  # Number of documents to retrieve
                    "fetch_k": 20,  # Number of documents to fetch before filtering
                    "lambda_mult": 0.5,  # Diversity of results
                }
            )

            self.chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": self.prompt},
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

    def query(self, query: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.

        Args:
            query: The question to answer

        Returns:
            Dict containing the answer and metadata

        Raises:
            ValueError: If no documents are available in the vector store
        """
        logger.info(f"Starting RAG query: '{query}'")

        # Ensure chain is created
        self._ensure_chain()

        # Get relevant documents
        logger.debug("Performing similarity search...")
        relevant_docs = self.vector_store.similarity_search(
            query,
            k=4,  # Default number of documents to retrieve
        )
        logger.info(f"Retrieved {len(relevant_docs)} relevant documents")

        # Log details of retrieved documents
        for i, doc in enumerate(relevant_docs):
            logger.debug(
                f"Document {i+1} Source: {doc.metadata.get('source', 'Unknown')}"
            )

        # Prepare context from retrieved documents
        context = "\n\n".join(
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(relevant_docs)
        )

        # Use the chain to get the answer with our retrieved context
        logger.debug("Using chain to generate answer...")
        result = self.chain.invoke(
            {"query": query, "context": context, "source_documents": relevant_docs}
        )
        logger.info("Received response from chain")

        return {
            "answer": result["result"],
            "source_documents": relevant_docs,  # Include source documents in result
            "metadata": {
                "num_docs_retrieved": len(relevant_docs),
                "sources": [
                    doc.metadata.get("source", "Unknown") for doc in relevant_docs
                ],
            },
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
            output_dir, "chunk_distribution", self.vector_store.documents
        )

    def process_dataset(
        self, dataset_path: str, analysis_dir: Optional[str] = None
    ) -> None:
        """
        Process all PDF files in a dataset directory and save the vector store.

        Args:
            dataset_path: Path to the dataset directory containing PDF files
            analysis_dir: Optional directory to save analysis results. If None, will use dataset_path/analysis

        Raises:
            ValueError: If no PDF files are found in the dataset directory
        """
        logger.info(f"Processing dataset from directory: {dataset_path}")

        # Check if directory exists and contains PDF files
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset directory not found: {dataset_path}")

        vector_db_path = os.path.join(dataset_path, "vector_db")
        pdf_files = [f for f in os.listdir(dataset_path) if f.endswith(".pdf")]

        # Try to load existing vector store if it exists
        if os.path.exists(vector_db_path):
            logger.info(f"Loading existing vector database from: {vector_db_path}")
            self.load_vector_store(vector_db_path)
        else:
            logger.info("No existing vector database found, creating new one")

        if pdf_files:
            # Process each PDF file in the dataset
            for filename in pdf_files:
                file_path = os.path.join(dataset_path, filename)
                logger.info(f"Processing file: {filename}")
                self.process_documents(file_path)

            # Save updated vector store
            self.save_vector_store(vector_db_path)
            logger.info(f"Vector store saved to: {vector_db_path}")

            # Analyze chunk distribution
            if analysis_dir is None:
                analysis_dir = os.path.join(dataset_path, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            self.analyze_chunk_distribution(analysis_dir)
            logger.info(f"Analysis saved to: {analysis_dir}")
        else:
            logger.warning("No PDF files found in dataset directory")
