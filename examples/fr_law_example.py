import os
import logging
from dotenv import load_dotenv
from typing import Optional, List, Tuple, Any, Callable
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.llms import HuggingFaceHub
from pymupdf.mupdf import pdf_page

from owlai.rag import RAGOwlAgent
from owlai.db import RAG_AGENTS_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load environment variables (for HuggingFace API token)
    load_dotenv()

    config = RAG_AGENTS_CONFIG[0]

    # Initialize the RAG agent with default config
    rag_agent = RAGOwlAgent(**config)

    # Example questions to test the system
    questions = rag_agent.default_queries

    # Query the system
    for question in questions:  # Limit to first 5 questions for testing
        logger.info(f"Processing question: {question}")
        logger.info(f"\nQuestion: {question}")
        result = rag_agent.rag_question(question)

        logger.info("\nRéponse:")
        logger.info(result["answer"])

        logger.info("\nSources utilisées (top documents réordonnés):")
        for i, doc in enumerate(result["metadata"].get("reranked_docs", {}).values()):
            logger.info(f"- {doc['title']} ({doc['source']})")

        logger.info("-" * 80)


if __name__ == "__main__":
    main()
