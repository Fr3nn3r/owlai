import os
import sys
import logging
import warnings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from owlai.rag_agent import RAGAgent
from owlai.owlsys import load_logger_config
from owlai.rag import RAGOwlAgent
from owlai.db import RAG_AGENTS_CONFIG

# Configure console for UTF-8 output
if sys.platform == "win32":
    import locale

    try:
        # Try to set UTF-8 locale on Windows
        locale.setlocale(locale.LC_ALL, "")
        # Force UTF-8 encoding for stdout (Python 3.7+)
        if sys.version_info >= (3, 7) and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except (locale.Error, AttributeError):
        pass

# Suppress HuggingFace deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Load environment variables and configure logging
load_dotenv()

# Debug prints for logging configuration
load_logger_config()

# Configure logging before getting logger
logger = logging.getLogger("naruto")


def process_dataset(dataset_path: str, rag_agent: RAGAgent) -> None:
    """
    Process a dataset directory, handling file management and orchestration.

    Args:
        dataset_path: Path to the dataset directory
        rag_agent: Initialized RAG agent
    """
    vector_store_path = os.path.join(dataset_path, "vector_store")

    # Check for existing vector store
    if os.path.exists(vector_store_path):
        logger.info(f"Loading existing vector store from {vector_store_path}")
        rag_agent.load_vector_store(vector_store_path)
        logger.info(
            f"Loaded {rag_agent.get_document_count()} documents from vector store"
        )
    else:
        logger.info("No existing vector store found, will process files from scratch")

    # Find new files to process
    files_to_process = [
        f
        for f in os.listdir(dataset_path)
        if f.endswith(".pdf") and not f.startswith(".")
    ]

    if files_to_process:
        logger.info(f"Found {len(files_to_process)} new files to process")
        for filename in files_to_process:
            pdf_path = os.path.join(dataset_path, filename)
            logger.info(f"Processing {filename}")
            rag_agent.process_documents(pdf_path, chunk_size=512)

        # Save updated vector store
        rag_agent.save_vector_store(vector_store_path)
        logger.info(f"Updated vector store saved to {vector_store_path}")
    else:
        logger.info("No new files to process")

    # Create analysis directory and run analysis
    analysis_dir = os.path.join(dataset_path, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    rag_agent.analyze_chunk_distribution(analysis_dir)
    logger.info(f"Chunk distribution analysis saved to {analysis_dir}")


def main():
    # Set up logging
    load_logger_config()
    logger = logging.getLogger("naruto")

    # Initialize the RAG agent
    embedding_model = HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    llm = HuggingFaceEndpoint(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
    )

    rag_agent = RAGAgent(
        embedding_model=embedding_model,
        llm=llm,
    )

    # Process the dataset
    dataset_path = "data/dataset-0001"
    rag_agent.process_dataset(dataset_path)

    # Test the RAG system
    test_questions = [
        "Tell me about Orochimaru",
        "Who is itachi?",
        "Who is madara?",
    ]

    for question in test_questions:
        logger.info(f"Question: {question}")
        result = rag_agent.query(question)
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Sources: {result['metadata']['sources']}")


if __name__ == "__main__":
    main()
