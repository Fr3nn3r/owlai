"""
OwlAI Models Module

Contains model abstractions and implementations for RAG and other AI functionalities.
"""

import logging
from typing import List, Union, Optional
import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RAGPretrainedModel:
    """
    Wrapper class for pretrained models used in RAG.
    Currently supports sentence-transformers CrossEncoder models.
    """

    def __init__(self, model: CrossEncoder):
        """
        Initialize with a pretrained model.

        Args:
            model: The underlying model (currently CrossEncoder)
        """
        self.model = model

    @classmethod
    def from_pretrained(cls, model_name: str) -> "RAGPretrainedModel":
        """
        Create a RAGPretrainedModel from a model name.

        Args:
            model_name: Name of the pretrained model to load

        Returns:
            RAGPretrainedModel instance
        """
        try:
            model = CrossEncoder(model_name)
            return cls(model)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise

    def predict(self, sentence_pairs: List[tuple]) -> np.ndarray:
        """
        Get predictions for sentence pairs.

        Args:
            sentence_pairs: List of (sentence1, sentence2) tuples to compare

        Returns:
            Array of similarity scores
        """
        try:
            return self.model.predict(sentence_pairs)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
