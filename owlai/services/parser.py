"""
OwlAI Document Parser Module

Note: We are using Pydantic v1 because it's required by langchain-core and other LangChain components.
This is a temporary solution until LangChain fully supports Pydantic v2.
The deprecation warnings are suppressed in pytest configuration.
"""

print("Loading parser module")
from typing import Optional, List, Tuple, Any, Callable
import os
import logging
from langchain.docstore.document import Document as LangchainDocument

from typing import Dict, List, Literal, Optional, Tuple, Type, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer

import warnings
from tqdm import tqdm

import fitz
import re
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


from tqdm import tqdm

from typing import List
import os
import logging
import traceback
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", category=FutureWarning)

# Get logger using the module name
logger = logging.getLogger(__name__)


# Fix fitz import to resolve type checking issues
try:
    from fitz import Document as PyMuPDFDocument, Page as PyMuPDFPage, open as fitz_open

    # Type aliases for type checking
    Document = PyMuPDFDocument  # type: ignore[assignment]
    Page = PyMuPDFPage  # type: ignore[assignment]
except ImportError:
    # For type hinting only
    class Document:
        def __len__(self) -> int:
            return 0

        def __getitem__(self, index: int) -> "Page":
            raise NotImplementedError

        @property
        def metadata(self) -> dict:
            return {}

    class Page:
        def get_text(self, text_type: str) -> str:
            return ""


# Implementation starts here
class DefaultParser(BaseModel):
    """
    Default parser for RAGDataStore.
    """

    implementation: str = "DefaultParser"
    output_data_folder: str = os.path.normpath("data/dataset-0001")
    chunk_size: float = 512
    chunk_overlap: float = 50
    add_start_index: bool = True
    strip_whitespace: bool = True
    separators: List[str] = ["\n\n", "\n", " ", ""]
    extract_images: bool = False
    extraction_mode: Literal["plain", "layout"] = "plain"

    _image_folder: str = "images"

    def _split_documents(
        self,
        input_docs: List[LangchainDocument],
        tokenizer_name: str,
    ) -> List[LangchainDocument]:
        """
        Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
        """
        try:
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            # Ensure chunk_size and chunk_overlap are integers
            chunk_size = int(self.chunk_size)
            chunk_overlap = int(self.chunk_overlap)

            # Configure text splitter with more conservative settings
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
                strip_whitespace=True,
                separators=[
                    "\n\n",
                    "\n",
                    ".",
                    "!",
                    "?",
                    ",",
                    " ",
                    "",
                ],  # Simplified separators
                is_separator_regex=False,  # Disable regex for more reliable splitting
            )
            logger.debug(
                f"Splitting {len(input_docs)} documents with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
            )

            docs_processed = []
            for doc in tqdm(input_docs, desc="Splitting documents"):
                try:
                    # Clean and normalize the text content
                    content = doc.page_content
                    if not content or not isinstance(content, str):
                        logger.warning(
                            f"Skipping document with invalid content type: {type(content)}"
                        )
                        continue

                    # Normalize text: remove problematic characters and normalize whitespace
                    content = (
                        content.replace("\u2015", "-")
                        .replace("\u300c", '"')
                        .replace("\u300d", '"')
                        .replace("\u2018", "'")
                        .replace("\u2019", "'")
                        .replace("\u201c", '"')
                        .replace("\u201d", '"')
                    )
                    # Normalize whitespace and remove multiple spaces
                    content = " ".join(content.split())

                    if not content:
                        logger.warning(
                            "Skipping document with empty content after cleaning"
                        )
                        continue

                    # Create a new document with cleaned content
                    cleaned_doc = LangchainDocument(
                        page_content=content, metadata=doc.metadata.copy()
                    )

                    # Initialize tokens variable
                    tokens = []
                    try:
                        # First try to split by tokens
                        tokens = tokenizer.encode(content)
                        if len(tokens) <= chunk_size:
                            # If content is small enough, keep it as is
                            docs_processed.append(cleaned_doc)
                        else:
                            # Otherwise use the text splitter
                            result = text_splitter.split_documents([cleaned_doc])
                            docs_processed.extend(result)
                    except Exception as split_error:
                        logger.error(f"Error during text splitting: {str(split_error)}")
                        logger.error(
                            f"Content length: {len(content)}, Token count: {len(tokens)}"
                        )
                        # If splitting fails, add the whole document as a single chunk
                        docs_processed.append(cleaned_doc)

                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}")
                    logger.error(f"Document metadata: {doc.metadata}")
                    # Add the original document as a fallback
                    docs_processed.append(doc)
                    continue

            logger.debug(
                f"Splitted {len(input_docs)} documents into {len(docs_processed)} chunks"
            )

            # Remove duplicates
            unique_texts = {}
            docs_processed_unique = []
            for doc in tqdm(docs_processed, desc="Removing duplicates"):
                if doc.page_content not in unique_texts:
                    unique_texts[doc.page_content] = True
                    docs_processed_unique.append(doc)

            logger.debug(
                f"Removed {len(docs_processed) - len(docs_processed_unique)} duplicates from {len(docs_processed)} chunks"
            )

            return docs_processed_unique

        except Exception as e:
            logger.error(f"Error in _split_documents: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            # Return original documents as fallback
            return input_docs

    def load_and_split_document(
        self,
        filepath: str,
        filename: str,
        model_name: str,
        metadata_extractor: Optional[Callable] = None,
        document_curator: Optional[Callable] = None,
    ) -> List[LangchainDocument]:
        """
        Loads a document file and splits it into chunks.

        Args:
            filepath: Path to the document file
            input_data_folder: Folder containing the documents
            filename: Name of the document file
            chunk_size: Size of text chunks for splitting
            model_name: Name of the embedding model for tokenization
            metadata_extractor: Optional callback function that takes a document and returns
                                additional metadata as a dictionary to be added to the document

        Returns:
            List of split LangchainDocument objects
        """
        # Call metadata extractor if provided
        metadata = {}
        if metadata_extractor:
            try:
                additional_metadata = metadata_extractor(filepath)
                if additional_metadata and isinstance(additional_metadata, dict):
                    metadata.update(additional_metadata)
            except Exception as e:
                logger.error(f"Error in metadata extractor for {filename}: {str(e)}")
                logger.error(f"Error details: {traceback.format_exc()}")

        # Load document
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(
                file_path=filepath,
                extract_images=self.extract_images,
                extraction_mode=self.extraction_mode,
            )
            docs = loader.lazy_load()
        else:  # .txt files
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.lazy_load()

        # Convert to LangchainDocuments
        total_pages = metadata.get("num_pages", 0)
        loaded_docs: List[LangchainDocument] = []
        logger.debug(f"Loading document: '{filename}'")

        for page_number, doc in tqdm(
            enumerate(docs),
            total=total_pages,
            desc=f"Loading pages from {filename}",
        ):
            try:
                # Ensure the text is properly encoded
                doc_content = doc.page_content.encode("utf-8", errors="replace").decode(
                    "utf-8"
                )
                metadata.update(doc.metadata)
                metadata["source"] = f"{filename}:{page_number}"

                # Call document curator if provided
                if document_curator:
                    doc_content = document_curator(doc_content, filepath)

                loaded_docs.append(
                    LangchainDocument(
                        page_content=doc_content,
                        metadata=metadata,
                    )
                )
            except Exception as e:
                logger.error(f"Error processing page {page_number}: {str(e)}")
                logger.error(f"Error details: {traceback.format_exc()}")
                continue

        # Split documents
        split_docs = self._split_documents(
            loaded_docs,
            tokenizer_name=model_name,
        )

        # Analyze document chunks before splitting
        pre_split_file = self.analyze_chunk_size_distribution(
            "pre-split-" + filename,
            loaded_docs,
            model_name,
        )
        metadata["pre_split_file"] = pre_split_file

        # Analyze post-split chunks and add to metadata
        post_split_file = self.analyze_chunk_size_distribution(
            "post-split-" + filename,
            split_docs,
            model_name,
        )
        metadata["post_split_file"] = post_split_file

        for i in range(min(5, len(split_docs))):
            logger.debug(f"Split doc {i}: {split_docs[i].metadata}")

        return split_docs

    def analyze_chunk_size_distribution(
        self,
        source_doc_filename: str,
        docs: List[LangchainDocument],
        model_name="thenlper/gte-small",
    ) -> str:
        """
        Counts number of tokens in each document and saves to a file to visualize.

        Args:
            docs: to analyze
            model_name: Name of the embedding model to use
        """

        # Get max sequence length from tokenizer instead of SentenceTransformer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        max_seq_len = tokenizer.model_max_length
        info_message = (
            f"Model's max sequence size: '{max_seq_len}' Document count: '{len(docs)}'"
        )
        logger.debug(info_message)

        # Analyze token lengths
        lengths = [len(tokenizer.encode(doc.page_content)) for doc in docs]
        total_tokens = sum(lengths)
        logger.debug(f"Total token count in documents: {total_tokens}")

        fig = pd.Series(lengths).hist()
        plt.title(
            f"Chunk lengths [tokens] for {source_doc_filename} total tokens: {total_tokens}"
        )
        # Create a path for saving visualization images
        file_dir = os.path.normpath(
            os.path.join(self.output_data_folder, self._image_folder)
        )
        file_path = os.path.normpath(
            os.path.join(file_dir, f"chunk_size_distribution-{source_doc_filename}.png")
        )
        os.makedirs(file_dir, exist_ok=True)
        plt.savefig(file_path)
        plt.close()
        logger.debug(f"Document lengths [tokens] saved to {file_path}")
        return file_path


class FrenchLawParser(DefaultParser):

    implementation: str = "FrenchLawParser"

    def extract_footer(self, doc: Document) -> str:
        footers = []

        # Check at least the first 10 pages or all pages if less than 10
        max_pages_to_check = min(10, len(doc))
        common_footer = None

        for page_num in range(max_pages_to_check):
            page = doc[page_num]  # type: Page

            # Extract text and split into lines
            text = page.get_text("text")  # type: ignore
            lines = text.split("\n")

            if len(lines) > 1:
                current_footer = "".join(
                    lines[-2:]
                )  # Assume footer is in the last two lines
                footers.append((page_num + 1, current_footer))

                # Initialize common_footer with the first page's footer
                if page_num == 0:
                    common_footer = current_footer
                # Check if footer is consistent across pages
                elif current_footer != common_footer:
                    common_footer = None  # Footers don't match

        # If we've checked enough pages and found a consistent footer, return it
        if common_footer and len(footers) >= max_pages_to_check:
            logger.debug(f"Common footer found: '''{common_footer}'''")
            return common_footer

        raise ValueError("No consistent footer found in the document")

    def extract_metadata_fr_law(self, footer: str, doc: Document) -> dict:
        """
        Extract metadata from a PDF file footer (file expected to follow french law convention).

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            dict: Dictionary containing title, last_modification, and doc_generated_on
        """

        # Regular Expression to Extract Components
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

    def load_fr_law_pdf(self, pdf_path: str) -> list[LangchainDocument]:
        """
        Loads a french law PDF file and returns the content.
        """
        # Normalize file path to ensure consistent separators
        pdf_path = os.path.normpath(pdf_path)

        # Verify the file exists
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

        # Check file extension
        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError(f"File must be a PDF: {pdf_path}")

        logger.debug(f"Loading French law PDF from: {pdf_path}")

        # Open document with proper type handling
        doc: Document = fitz_open(pdf_path)  # type: ignore[assignment]

        if not doc or doc is None:
            raise ValueError(f"Failed to load document from {pdf_path}")

        file_name = os.path.basename(pdf_path)

        footer = self.extract_footer(doc)

        metadata = doc.metadata.copy()

        # Merge the two dictionaries - metadata from the document and extracted metadata from the footer
        metadata.update(self.extract_metadata_fr_law(footer, doc))
        metadata["source"] = file_name

        total_pages = int(metadata.get("num_pages", 0))
        loaded_docs: List[LangchainDocument] = []
        total_page_content = ""

        # Use a simpler loop approach with tqdm
        for page_number in tqdm(
            range(total_pages), desc=f"Loading pages from {pdf_path}"
        ):
            # Explicitly type page to avoid linter errors with PyMuPDF
            page = doc[page_number]  # type: ignore
            # Extract text content from the page
            page_content = page.get_text("text")  # type: ignore

            # Remove footer if present to avoid duplication
            page_content = page_content.replace(footer, "")

            # Load the whole document to ensure we don't force a split between pages
            total_page_content += page_content

        loaded_docs.append(
            LangchainDocument(
                page_content=total_page_content,
                metadata=metadata,
            )
        )

        return loaded_docs
        # Split documents

    def parse_fr_law_docs(
        self, loaded_docs: List[LangchainDocument]
    ) -> List[LangchainDocument]:

        # All that stuff could be in the parser parameters
        embeddings_chunk_size = 512
        chunk_size = 512
        chunk_overlap = int(embeddings_chunk_size / 5)
        chunk_size = embeddings_chunk_size * 0.9
        tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        separators = [
            # "\nPartie[^\n]*",
            # "\nLivre[^\n]*",
            # "\nTitre[^\n]*",
            # "\nChapitre[^\n]*",
            # "\nArticle[^\n]*",
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
            # " ",
            # "",
        ]
        # Analyze document chunks before splitting

        logger.debug("Splitting documents")
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,  # The number of characters to overlap between chunks
            add_start_index=True,  # If `True`, includes chunk's start index in metadata
            strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
            separators=separators,
            is_separator_regex=True,
        )

        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text))

        docs_processed: List[LangchainDocument] = []

        docs_processed = text_splitter.split_documents(loaded_docs)

        logger.debug("Document split, creating langchain documents")

        unique_texts = {}
        docs_processed_unique = []
        for idoc in docs_processed:
            if idoc.page_content not in unique_texts:
                unique_texts[idoc.page_content] = True
                metadata_update = idoc.metadata.copy()
                metadata_update["token_count"] = count_tokens(idoc.page_content)
                docs_processed_unique.append(
                    LangchainDocument(
                        page_content=idoc.page_content, metadata=metadata_update
                    )
                )

        logger.debug("Langchain documents created, parsing completed")

        return docs_processed_unique

    def load_and_split_document(
        self,
        filepath: str,
        filename: str,
        model_name: str,
        metadata_extractor: Optional[Callable] = None,
        document_curator: Optional[Callable] = None,
    ) -> List[LangchainDocument]:

        loaded_docs = self.load_fr_law_pdf(filepath)

        logger.debug("Document loaded, analyzing chunk size distribution")
        self.analyze_chunk_size_distribution(
            "pre-split-" + filename,
            loaded_docs,
            model_name,
        )

        parsed_docs = self.parse_fr_law_docs(loaded_docs)

        logger.debug("Documents parsed, analyzing chunk size distribution")
        self.analyze_chunk_size_distribution(
            "post-split-" + filename,
            parsed_docs,
            model_name,
        )

        logger.debug(
            f"Document contains {len(parsed_docs)} chunks to add to vector store"
        )

        return parsed_docs


# Instantiate by class name string
def create_instance(class_name: str, **kwargs):
    """Create an instance of a class by its name string"""
    try:
        logger.debug(f"Creating instance of class: {class_name}")
        cls = globals()[class_name]
        return cls(**kwargs)
    except KeyError:
        raise Exception(f"Class {class_name} not found in globals")
    except Exception as e:
        raise Exception(f"Error creating instance of {class_name}: {str(e)}")
