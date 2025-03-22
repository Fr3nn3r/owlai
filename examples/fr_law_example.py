import os
import logging
from dotenv import load_dotenv
from typing import Optional, List, Tuple, Any, Callable
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.llms import HuggingFaceHub

from owlai.rag import RAGOwlAgent
from owlai.db import RAG_AGENTS_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load environment variables (for HuggingFace API token)
    load_dotenv()

    # Initialize the RAG agent with default config
    rag_agent = RAGOwlAgent(**RAG_AGENTS_CONFIG[0])

    # Example PDF path - replace with your French law PDF
    pdf_path = "data/dataset-0005/in_store/LEGITEXT000006069577.pdf"

    # Process the PDF if it exists
    if os.path.exists(pdf_path) and rag_agent._embeddings is not None:
        logger.info(f"Processing PDF: {pdf_path}")
        rag_agent.load_dataset(os.path.dirname(pdf_path), rag_agent._embeddings)

    # Example questions to test the system
    questions = [
        "Quelles sont les conditions de validité d'un contrat en droit français ?",
        "Expliquez le principe de la responsabilité civile en droit français.",
        "Quelles sont les principales obligations de l'employeur en matière de santé et sécurité au travail ?",
    ]

    # Query the system
    for question in questions:
        logger.info(f"\nQuestion: {question}")
        result = rag_agent.rag_question(question)

        print("\nRéponse:")
        print(result["answer"])

        print("\nSources utilisées:")
        for i, doc in enumerate(result["metadata"].get("retrieved_docs", {}).values()):
            print(f"- {doc['title']} ({doc['source']})")

        print("-" * 80)

    # Analyze chunk distribution
    if rag_agent._vector_stores is not None:
        # Get documents using similarity search with a dummy query
        docs = rag_agent._vector_stores.similarity_search(
            "", k=rag_agent.retriever.num_docs_final
        )
        rag_agent.analyze_chunk_size_distribution("data/dataset-0005/", "example", docs)


if __name__ == "__main__":
    main()
