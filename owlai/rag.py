import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
import os
import time
import datasets
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    Pipeline,
)
import transformers
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import pacmap
import numpy as np
import plotly.express as px
from ragatouille import RAGPretrainedModel
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from transformers import PreTrainedTokenizer

import logging
import logging.config
import yaml
import json

from tqdm import tqdm

transformers.logging.set_verbosity_error()

logger = logging.getLogger("ragtool")


def load_vector_store(input_data_folder: str, embedding_model: HuggingFaceEmbeddings):
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
    query: str, knowledge_base, k: int = 5
) -> List[LangchainDocument]:
    """
    Retrieve the k most relevant document chunks for a given query.

    Args:
        query: The user query to find relevant documents for
        knowledge_base: The vector database containing indexed documents
        k: Number of documents to retrieve (default 5)

    Returns:
        List of retrieved LangchainDocument objects
    """
    logger.info(f"Starting retrieval for query: {query} with k={k}")
    start_time = time.time()
    retrieved_docs = knowledge_base.similarity_search(query=query, k=k)
    end_time = time.time()

    logger.info(
        f"{len(retrieved_docs)} documents retrieved in {end_time - start_time:.2f} seconds"
    )
    # logger.debug(f"Top documents: {retrieved_docs[0].page_content}")
    logger.debug(f"Top document metadata: {retrieved_docs[0].metadata}")

    return retrieved_docs
