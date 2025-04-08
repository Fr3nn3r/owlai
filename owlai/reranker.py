"""
Singleton manager for reranker models to ensure sharing across RAG tools.
"""

import logging
from typing import Dict, Optional
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RerankerManager:
    """
    Singleton manager for reranker models to ensure sharing across RAG tools.
    Implements a cache to avoid loading the same model multiple times.
    """

    _instances: Dict[str, CrossEncoder] = {}

    @classmethod
    def get_reranker(
        cls,
        model_name: str,
        model_kwargs: Optional[dict] = None,
    ) -> CrossEncoder:
        """
        Get or create a reranker model instance.

        Args:
            model_name: Name of the cross-encoder model to use
            model_kwargs: Additional kwargs for model initialization

        Returns:
            CrossEncoder instance
        """
        # Create a unique key based on all parameters
        key = f"{model_name}_{hash(str(model_kwargs))}"

        if key not in cls._instances:
            logger.debug(f"Creating new reranker model instance for {model_name}")
            cls._instances[key] = CrossEncoder(model_name, **(model_kwargs or {}))
            logger.debug(
                f"Successfully created reranker model instance for {model_name}"
            )
        else:
            logger.debug(f"Reusing existing reranker model instance for {model_name}")

        return cls._instances[key]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached reranker models."""
        cls._instances.clear()
        logger.debug("Cleared reranker model cache")
