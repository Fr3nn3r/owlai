"""
Singleton manager for embedding models to ensure sharing across RAG tools.
"""

import logging
from typing import Dict, Optional
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Singleton manager for embedding models to ensure sharing across RAG tools.
    Implements a cache to avoid loading the same model multiple times.
    """

    _instances: Dict[str, HuggingFaceEmbeddings] = {}

    @classmethod
    def get_embedding(
        cls,
        model_name: str,
        multi_process: bool = True,
        model_kwargs: Optional[dict] = None,
        encode_kwargs: Optional[dict] = None,
    ) -> HuggingFaceEmbeddings:
        """
        Get or create an embedding model instance.

        Args:
            model_name: Name of the HuggingFace model to use
            multi_process: Whether to use multiple processes for encoding
            model_kwargs: Additional kwargs for model initialization
            encode_kwargs: Additional kwargs for encoding

        Returns:
            HuggingFaceEmbeddings instance
        """
        # Create a unique key based on all parameters
        key = f"{model_name}_{multi_process}_{hash(str(model_kwargs))}_{hash(str(encode_kwargs))}"

        if key not in cls._instances:
            logger.debug(f"Creating new embedding model instance for {model_name}")
            cls._instances[key] = HuggingFaceEmbeddings(
                model_name=model_name,
                multi_process=multi_process,
                model_kwargs=model_kwargs or {},
                encode_kwargs=encode_kwargs or {},
            )
            logger.debug(
                f"Successfully created embedding model instance for {model_name}"
            )
        else:
            logger.debug(f"Reusing existing embedding model instance for {model_name}")

        return cls._instances[key]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached embedding models."""
        cls._instances.clear()
        logger.debug("Cleared embedding model cache")
