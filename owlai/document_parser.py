import fitz  # PyMuPDF
import re
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("main")


class FrenchLawParser:
    """Parser for French law documents with specialized handling for French legal format."""

    def __init__(self):
        """Initialize the French law parser."""
        self.tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Define separators for text splitting
        self.separators = [
            ".{10,}(?=\n)Partie[^\n]*",
            ".{10,}(?=\n)Livre[^\n]*",
            ".{10,}(?=\n)Titre[^\n]*",
            ".{10,}(?=\n)Chapitre[^\n]*",
            ".{10,}(?=\n)Article[^\n]*",
            "\n \n \n",
            "\n\n \n",
            "\n \n\n",
            "\n \n",
            "\n\n",
            "\n",
            ".",
        ]

    def parse(self, pdf_path: str) -> List[Document]:
        """
        Parse a French law PDF file and return a list of documents.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of Document objects containing the parsed content

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the PDF is empty or invalid
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            raise ValueError("PDF file is empty")

        file_name = os.path.basename(pdf_path)
        footer = self.extract_footer(doc)
        metadata = doc.metadata
        metadata.update(self.extract_metadata_fr_law(footer, doc))

        total_pages = metadata.get("num_pages", 0)
        loaded_docs: List[Document] = []
        total_page_content = ""

        for page_number, page in enumerate(doc):
            page_content = page.get_text("text")
            page_content = page_content.replace(footer, "")

            page_metadata = metadata.copy()
            page_metadata["page_number"] = page_number + 1
            page_metadata["source"] = f"{file_name}:{page_number + 1}"

            total_page_content += page_content

        loaded_docs.append(
            Document(
                page_content=total_page_content,
                metadata=page_metadata,
            )
        )

        return loaded_docs

    def split(
        self,
        documents: List[Document],
        chunk_size: int = 512,
        chunk_overlap: Optional[int] = None,
    ) -> List[Document]:
        """
        Split documents into chunks of specified size.

        Args:
            documents: List of documents to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks

        Returns:
            List of split Document objects

        Raises:
            ValueError: If chunk_size is invalid
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if chunk_overlap is None:
            chunk_overlap = int(chunk_size / 5)

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            separators=self.separators,
            is_separator_regex=True,
        )

        docs_processed = text_splitter.split_documents(documents)

        # Remove duplicates and add token count
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                metadata_update = doc.metadata.copy()
                metadata_update["token_count"] = len(
                    self.tokenizer.encode(doc.page_content)
                )
                docs_processed_unique.append(
                    Document(page_content=doc.page_content, metadata=metadata_update)
                )

        return docs_processed_unique

    def extract_footer(self, doc: fitz.Document) -> str:
        """
        Extract the common footer from a document.

        Args:
            doc: PyMuPDF document object

        Returns:
            The common footer text

        Raises:
            ValueError: If no consistent footer is found
        """
        footers = []
        max_pages_to_check = min(10, len(doc))
        common_footer = None

        for page_num in range(max_pages_to_check):
            page = doc[page_num]
            text = page.get_text("text")
            lines = text.split("\n")

            if len(lines) > 1:
                current_footer = "".join(lines[-2:])
                footers.append((page_num + 1, current_footer))

                if page_num == 0:
                    common_footer = current_footer
                elif current_footer != common_footer:
                    common_footer = None

        if common_footer and len(footers) >= max_pages_to_check:
            logger.debug(f"Common footer found: '''{common_footer}'''")
            return common_footer

        raise ValueError("No consistent footer found in the document")

    def extract_metadata_fr_law(
        self, footer: str, doc: fitz.Document
    ) -> Dict[str, Any]:
        """
        Extract metadata from a French law document footer.

        Args:
            footer: The footer text
            doc: PyMuPDF document object

        Returns:
            Dictionary containing extracted metadata

        Raises:
            ValueError: If footer doesn't match French law convention
        """
        match = re.match(r"^(.*?)\s*-\s*(.*?)\s*-\s*(.*?)$", footer)
        if match:
            title = match.group(1).strip()
            last_modification = match.group(2).strip()
            last_modification = last_modification.replace(
                "Dernière modification le ", ""
            )
            doc_generated_on = match.group(3).strip()
            doc_generated_on = doc_generated_on.replace("Document généré le ", "")

            return {
                "title": title,
                "last_modification_fr": last_modification,
                "doc_generated_on_fr": doc_generated_on,
                "num_pages": len(doc),
            }

        raise ValueError(f"footer '{footer}' not matching french law convention.")

    def document_curator(self, doc_content: str, file_path: str) -> str:
        """
        Curate document content by removing footers.

        Args:
            doc_content: The document content to curate
            file_path: Path to the document file

        Returns:
            Curated document content with footer removed
        """
        lines = doc_content.split("\n")

        if len(lines) > 2:
            curated_content = "\n".join(lines[:-2])
            logger.info(f"Removed footer from document: {file_path}")
        else:
            curated_content = doc_content
            logger.info(f"No footer found in document: {file_path}")

        return curated_content

    def analyze_chunk_size_distribution(
        self,
        input_data_folder: str,
        filename: str,
        docs: List[Document],
        model_name: str = "thenlper/gte-small",
    ) -> str:
        """
        Analyze and visualize document lengths.

        Args:
            input_data_folder: Folder to save analysis results
            filename: Name of the document file
            docs: Documents to analyze
            model_name: Name of the embedding model

        Returns:
            Path to the generated distribution plot
        """
        max_seq_len = SentenceTransformer(model_name).max_seq_length
        info_message = (
            f"Model's max sequence size: '{max_seq_len}' Document count: '{len(docs)}'"
        )
        logger.info(info_message)

        lengths = [len(self.tokenizer.encode(doc.page_content)) for doc in docs]

        fig = pd.Series(lengths).hist()
        plt.title(f"Distribution of page lengths [tokens] for {filename}")
        file_dir = f"{input_data_folder}/images"
        file_path = f"{file_dir}/chunk_size_distribution-{filename}.png"
        os.makedirs(file_dir, exist_ok=True)
        plt.savefig(file_path)
        plt.close()

        logger.debug(
            f"Distribution of document lengths (in count of tokens) saved to {file_path}"
        )
        return file_path
