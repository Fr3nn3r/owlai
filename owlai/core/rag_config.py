from typing import Dict, List, Any
from pydantic import BaseModel


class RAGConfig(BaseModel):
    """Configuration for RAG components"""

    num_retrieved_docs: int
    num_docs_final: int
    embeddings_model_name: str
    reranker_name: str
    input_data_folders: List[str]
    model_kwargs: Dict[str, Any]
    encode_kwargs: Dict[str, Any]
    multi_process: bool = True
