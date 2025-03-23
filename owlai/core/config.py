from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from owlai.core.rag_config import RAGConfig
from enum import Enum


class Provider(str, Enum):
    """Supported LLM providers"""

    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENAI = "openai"
    MISTRALAI = "mistralai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    AZURE = "azure"
    FIREWORKS = "fireworks"
    TOGETHER = "together"
    VERTEXAI = "vertexai"
    BEDROCK = "bedrock"
    COHERE = "cohere"
    DEEPINFRA = "deepinfra"
    GROQ = "groq"
    PERPLEXITY = "perplexity"
    ZHIPUAI = "zhipuai"
    YI = "yi"
    CUSTOM = "custom"


class ModelConfig(BaseModel):
    """Configuration for language models"""

    provider: Provider = Field(..., description="Model provider (e.g., 'openai')")
    model_name: str = Field(..., description="Name of the model to use")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Model temperature"
    )
    max_tokens: int = Field(
        default=2048, gt=0, description="Maximum tokens per response"
    )
    context_size: int = Field(default=4096, gt=0, description="Maximum context size")


class AgentConfig(BaseModel):
    """Configuration for AI agents"""

    name: str = Field(..., min_length=1, description="Unique name for the agent")
    description: str = Field(..., min_length=1, description="Description of the agent")
    system_prompt: str = Field(
        ..., min_length=1, description="System prompt for the agent"
    )
    llm_config: ModelConfig = Field(..., description="Model configuration")
    default_queries: Optional[List[str]] = Field(
        default=None, description="Default queries to run"
    )
    tools_names: List[str] = Field(
        default_factory=list, description="Names of tools to use"
    )
    retriever: Optional[RAGConfig] = Field(
        default=None, description="RAG configuration for retrieval"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()

    @field_validator("description")
    @classmethod
    def validate_description(cls, v):
        if not v.strip():
            raise ValueError("Description cannot be empty")
        return v.strip()

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v):
        if not v.strip():
            raise ValueError("System prompt cannot be empty")
        return v.strip()
