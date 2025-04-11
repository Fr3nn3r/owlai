from typing import Any, Dict, Union, Optional
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class DataProvider(BaseTool):
    """Data provider that uses a store implementation."""

    def query(self, query: str) -> Dict[str, Any]:
        """Query the data store through the provider."""

        logger.info(
            f"Processing Data Query - provider: '{self.name}' - query: '{query}'"
        )

        # Get RAG resources
        rag_resources = self.get_rag_resources(query)

        logger.info(f"{self.name} execution completed")

        return rag_resources.get("answer", None)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[Dict[str, Any], str]:
        """Override BaseTool._run to use the store's query method."""
        return self.query(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[Dict[str, Any], str]:
        """Override BaseTool._arun to use the store's query method asynchronously."""
        return self.query(query)

    @abstractmethod
    def get_rag_resources(self, query: str) -> Dict[str, Any]:
        """Abstract method to get RAG resources. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_document_content_by_id(self, id: str) -> str:
        """Abstract method to get document content by ID. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")
