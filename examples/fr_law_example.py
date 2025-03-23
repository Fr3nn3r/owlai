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
from owlai.db import RAG_AGENTS_CONFIG_V2
from owlai.core.config import AgentConfig
from owlai.core.logging_setup import get_logger, setup_logging

# Load environment variables and setup logging
load_dotenv()
setup_logging()

# Get logger for this example
logger = get_logger("examples.fr_law_example")


def main():
    # Load environment variables (for HuggingFace API token)
    load_dotenv()

    # Convert dictionary to AgentConfig model
    config_dict = RAG_AGENTS_CONFIG_V2[0]
    config = AgentConfig(**config_dict)

    # Initialize the RAG agent with default config
    rag_agent = RAGOwlAgent(config)

    # Example questions to test the system
    questions = rag_agent.default_queries if rag_agent.default_queries else []

    # Query the system
    for question in questions:  # Limit to first 5 questions for testing
        logger.info(f"Question: {question}")
        result = rag_agent.rag_question(question)

        logger.info("\nRéponse:")
        logger.info(result["answer"])

        logger.info("Sources utilisées (top documents réordonnés):")
        for i, doc in enumerate(result["metadata"].get("reranked_docs", {}).values()):
            logger.info(f"- {doc['title']} ({doc['source']})")


if __name__ == "__main__":
    main()
