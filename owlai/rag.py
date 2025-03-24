print("Loading rag module")
from typing import Optional, List, Tuple, Any, Callable
import os
import logging
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from ragatouille import RAGPretrainedModel
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple, Literal
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers.util import fullname
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# Plot distribution
import matplotlib.pyplot as plt
import pandas as pd
from owlai.owlsys import encode_text
import traceback

from owlai.core import OwlAgent
from owlai.db import TOOLS_CONFIG, RAG_AGENTS_CONFIG
from owlai.owlsys import track_time, load_logger_config, sprint
import warnings
from tqdm import tqdm

warnings.simplefilter("ignore", category=FutureWarning)

logger = logging.getLogger("main")


class DefaultParser(BaseModel):
    """
    Default parser for RAGDataStore.
    """

    implementation: str = "DefaultParser"
    output_data_folder: str
    chunk_size: float
    chunk_overlap: float
    add_start_index: bool
    strip_whitespace: bool
    separators: List[str]
    extract_images: bool
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

                    # Split the document with error handling
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
        with track_time(f"Loading document: '{filename}'"):
            for page_number, doc in tqdm(
                enumerate(docs),
                total=total_pages,
                desc=f"Loading pages from {filename}",
            ):
                try:
                    # Ensure the text is properly encoded
                    doc_content = doc.page_content.encode(
                        "utf-8", errors="replace"
                    ).decode("utf-8")
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

        # Get max sequence length from SentenceTransformer
        max_seq_len = SentenceTransformer(model_name).max_seq_length
        info_message = (
            f"Model's max sequence size: '{max_seq_len}' Document count: '{len(docs)}'"
        )
        logger.debug(info_message)

        # Analyze token lengths (should init tokenizer once... whatever)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        lengths = [len(tokenizer.encode(doc.page_content)) for doc in docs]

        fig = pd.Series(lengths).hist()
        plt.title(f"Distribution of page lengths [tokens] for {source_doc_filename}")
        # Create a path for saving visualization images
        file_dir = os.path.join(self.output_data_folder, self._image_folder)
        file_path = os.path.join(
            file_dir, f"chunk_size_distribution-{source_doc_filename}.png"
        )
        os.makedirs(file_dir, exist_ok=True)
        plt.savefig(file_path)
        plt.close()
        logger.debug(
            f"Distribution of document lengths (in count of tokens) saved to {file_path}"
        )
        return file_path


import fitz
import re
from fitz import Document, Page


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

        doc: Document = fitz.open(pdf_path)

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

        with track_time("Splitting documents"):
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
        return self.parse_fr_law_docs(loaded_docs)


class RAGDataStore(BaseModel):

    input_data_folder: str

    parser: DefaultParser

    _vector_store: Optional[FAISS] = None
    _images_folder: str = "images"
    _in_store_documents_folder: str = "in_store"
    _vector_store_folder: str = "vector_db"

    def __init__(self, **kwargs):
        """
        Initialize the RAGDataStore with the provided configuration.

        This constructor handles the initialization of the RAGDataStore object,
        setting up all necessary parameters for document processing and vector storage.
        """
        super().__init__(**kwargs)
        # Ensure input_data_folder exists
        if not os.path.exists(self.input_data_folder):
            logger.warning(
                f"Input data folder does not exist: {self.input_data_folder}"
            )

        if self.parser.implementation == "DefaultParser":
            logger.info(f"Using DefaultParser")
        else:
            logger.warning(f"Using custom parser: {self.parser.implementation}")
            self.parser = create_instance(self.parser.implementation)
            if self.parser is None:
                raise Exception(f"{self.parser.implementation} is not a valid parser")

    def load_vector_store(
        self, input_data_folder: str, embedding_model: HuggingFaceEmbeddings
    ) -> Optional[FAISS]:
        file_path = os.path.normpath(
            os.path.join(input_data_folder, self._vector_store_folder)
        )
        logger.debug(f"Looking for vector database at: {file_path}")
        FAISS_vector_store = None

        if os.path.exists(file_path):
            with track_time(f"Loading the vector database from disk: {file_path}"):
                FAISS_vector_store = FAISS.load_local(
                    file_path,
                    embedding_model,
                    distance_strategy=DistanceStrategy.COSINE,
                    allow_dangerous_deserialization=True,
                )
        else:
            logger.error(f"Vector database not found in {file_path}")

        return FAISS_vector_store

    def load_dataset(
        self,
        embedding_model: HuggingFaceEmbeddings,
        metadata_extractor: Optional[Callable] = None,
        document_curator: Optional[Callable] = None,
    ) -> Optional[FAISS]:
        """
        Loads an existing vector store if exists in the input_data_folder.
        Processes documents and adds them to the store.
        Processed documents are moved to the 'in_store' folder

        Args:
            input_data_folder: Path to the folder containing documents
            embedding_model: The embedding model to use
            chunk_size: Size of text chunks for splitting documents
            metadata_extractor: Optional callback function for extracting metadata

        Returns:
            FAISS vector store or None if no documents were processed
        """
        # Use os.path.join for platform-independent path handling
        # Normalize paths to ensure OS-compatible separators
        input_data_folder = os.path.normpath(self.input_data_folder)
        vector_db_file_path = os.path.join(input_data_folder, self._vector_store_folder)
        in_store_folder = os.path.join(
            input_data_folder, self._in_store_documents_folder
        )
        vector_store = None

        if os.path.exists(vector_db_file_path):
            logger.info(f"Loading existing vector database from: {vector_db_file_path}")
            vector_store = self.load_vector_store(
                self.input_data_folder, embedding_model
            )

        # Get list of PDF and text files
        files = [
            f
            for f in os.listdir(self.input_data_folder)
            if f.endswith((".pdf", ".txt"))
        ]
        logger.info(
            f"Found {len(files)} new documents to process in {self.input_data_folder}"
        )

        if len(files) > 0:
            # Process each file individually
            with track_time(f"Loading {len(files)} document(s) into vector database"):
                for filename in files:
                    filepath = os.path.join(self.input_data_folder, filename)
                    logger.info(
                        f"Processing file: {filename} size: {os.path.getsize(filepath)}"
                    )

                    try:
                        split_docs = self.parser.load_and_split_document(
                            filepath,
                            filename,
                            embedding_model.model_name,
                            metadata_extractor,
                            document_curator,
                        )

                        # Create or update vector store
                        if vector_store is None:
                            vector_store = FAISS.from_documents(
                                split_docs,
                                embedding_model,
                                distance_strategy=DistanceStrategy.COSINE,
                            )
                        else:
                            batch_store = FAISS.from_documents(
                                split_docs,
                                embedding_model,
                                distance_strategy=DistanceStrategy.COSINE,
                            )
                            vector_store.merge_from(batch_store)

                        # Move processed file to 'in_store' folder
                        os.makedirs(in_store_folder, exist_ok=True)
                        os.rename(filepath, os.path.join(in_store_folder, filename))

                    except Exception as e:
                        logger.error(f"Error processing {filename}: {str(e)}")
                        logger.error(f"Error details: {traceback.format_exc()}")
                        continue

            # Save to disk
            if vector_store is not None and len(files) > 0:
                vector_store.save_local(vector_db_file_path)
                logger.info(f"Vector database saved to {vector_db_file_path}")

        return vector_store


class RAGRetriever(BaseModel):
    """Classe responsible for retrieving relevant chunks from the knowledge base"""

    num_retrieved_docs: int = 30  # Commonly called k
    num_docs_final: int = 5
    embeddings_model_name: str = "thenlper/gte-small"
    reranker_name: str = "colbert-ir/colbertv2.0"
    model_kwargs: Dict[str, Any] = {}
    encode_kwargs: Dict[str, Any] = {}
    multi_process: bool = True
    datastore: RAGDataStore

    def retrieve_relevant_chunks(
        self,
        query: str,
        knowledge_base: Optional[FAISS],
        reranker: Optional[RAGPretrainedModel] = None,
    ) -> Tuple[List[LangchainDocument], dict]:
        """
        Retrieve the k most relevant document chunks for a given query.

        Args:0
            query: The user query to find relevant documents for
            knowledge_base: The vector database containing indexed documents
            reranker: Optional reranker model to rerank results
            num_retrieved_docs: Number of initial documents to retrieve
            num_docs_final: Number of documents to return after reranking

        Returns:
            Tuple containing a list of retrieved and reranked LangchainDocument objects with scores and metadata
        """

        if knowledge_base is None:
            raise Exception("Invalid FAISS vector store")

        metadata = {
            "query": query,
            "k": self.num_retrieved_docs,
            "num_docs_final": self.num_docs_final,
        }

        with track_time(f"Documents search", metadata):
            retrieved_docs = knowledge_base.similarity_search(
                query=query, k=self.num_retrieved_docs
            )
            metadata["num_docs_retrieved"] = len(retrieved_docs)
            metadata["retrieved_docs"] = {
                i: {
                    "title": doc.metadata.get("title", "No title"),
                    "source": doc.metadata.get("source", "Unknown source"),
                }
                for i, doc in enumerate(retrieved_docs)
            }
            logger.debug(f"{len(retrieved_docs)} documents retrieved")
            sprint(retrieved_docs[0])

        # If no reranker, just return top k docs
        if not reranker:
            return retrieved_docs[: self.num_docs_final], metadata

        # Rerank results
        logger.debug(
            f"Reranking {len(retrieved_docs)} documents chunks to {self.num_docs_final} please wait..."
        )

        with track_time("Documents chunks reranking", metadata):

            # Create mapping of content to original doc for later matching
            content_to_doc = {doc.page_content: doc for doc in retrieved_docs}

            # Get reranked results
            reranked_results = reranker.rerank(
                query,
                [doc.page_content for doc in retrieved_docs],
                k=self.num_docs_final,
            )

            # Match reranked results back to original docs and add scores to doc metadata
            reranked_docs = []
            for rank, result in enumerate(reranked_results):
                doc = content_to_doc[result["content"]]
                doc.metadata["rerank_score"] = result["score"]
                doc.metadata["rerank_position"] = result["rank"]
                reranked_docs.append(doc)

            # Add reranked docs metadata to metadata (you can't have too much metadata)
            metadata["selected_docs"] = {
                i: {
                    "title": doc.metadata.get("title", "No title"),
                    "source": doc.metadata.get("source", "Unknown source"),
                    "rerank_score": doc.metadata.get("rerank_score", 0.0),
                    "rerank_position": doc.metadata.get("rerank_position", -1),
                }
                for i, doc in enumerate(reranked_docs)
            }

        for i in range(min(5, len(reranked_docs))):
            # logger.debug(
            #    f"Reranked doc {i}: {reranked_docs[i].metadata} {reranked_docs[i].page_content[:100]}"
            # )
            if reranked_docs[i].metadata.get("rerank_score", 0.0) < 15:
                logger.warning(
                    f"Reranked doc {i} has a score of {reranked_docs[i].metadata.get('rerank_score', 0.0)}"
                )

        return reranked_docs, metadata

    def load_dataset(self, embeddings: HuggingFaceEmbeddings) -> Optional[FAISS]:
        return self.datastore.load_dataset(embeddings)


class RAGAgent(OwlAgent):

    # JSON defined properties
    retriever: RAGRetriever

    # Runtime updated properties
    _init_completed = False
    _prompt = None
    _vector_store = None
    _embeddings = None
    _reranker = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.retriever.embeddings_model_name,
            multi_process=self.retriever.multi_process,
            model_kwargs=self.retriever.model_kwargs,
            encode_kwargs=self.retriever.encode_kwargs,
        )
        reranker_name = self.retriever.reranker_name
        self._reranker = RAGPretrainedModel.from_pretrained(reranker_name)
        self._prompt = PromptTemplate.from_template(self.system_prompt)

        # input_data_folders = self.retriever.datastore.input_data_folder

        self._vector_store = self.retriever.load_dataset(self._embeddings)

        if self._vector_store is None:
            logger.warning(
                "No vector stores found: you must set the vector store manually."
            )
        else:
            logger.info(
                f"Data store loaded: {self.retriever.datastore.input_data_folder}"
            )

    def invoke_rag(self, question: str) -> Dict[str, Any]:
        """
        Runs the RAG query against the vector store and returns an answer to the question.
        Args:
            question: a string containing the question to answer.

        Returns:
            A dictionary containing the question, answer, and metadata.
        """
        logger.debug(f"Running RAG query: '{question}'")

        answer: Dict[str, Any] = {"question": question}

        reranked_docs, metadata = self.retriever.retrieve_relevant_chunks(
            query=question,
            knowledge_base=self._vector_store,
            reranker=self._reranker,
        )

        with track_time("Model invocation with RAG context", metadata):

            logger.debug(f"TOP RERANKED DOCS: {reranked_docs[0].page_content}")
            docs_content = "\n\n".join(
                [
                    f"{idx+1}. [Source : {doc.metadata.get('title', 'Unknown Title')} - {doc.metadata.get('source', '')}] \"{doc.page_content}\""
                    for idx, doc in enumerate(reranked_docs)
                ]
            )
            logger.debug(f"DOCS CONTENT: {docs_content}")
            if self._prompt is None:
                raise Exception("Prompt is not set")

            rag_prompt = self._prompt.format(question=question, context=docs_content)

            # Add the RAG prompt to the metadata for debugging and analysis purposes
            metadata["rag_prompt"] = rag_prompt
            logger.debug(f"Final prompt: {rag_prompt}")
            message = SystemMessage(rag_prompt)
            messages = self.chat_model.invoke([message])
            logger.debug(f"RAG answer: {messages.content}")

        answer["answer"] = str(messages.content) if messages.content is not None else ""
        answer["metadata"] = metadata

        return answer

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Override BaseTool._run to ensure we use our implementation"""
        logger.debug(f"[RAGOwlAgent._run] Called with query: {query}")
        return self.message_invoke(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Override BaseTool._arun to ensure we use our implementation"""
        return self.message_invoke(query)

    def message_invoke(self, message: str) -> str:
        """Override OwlAgent.message_invoke with RAG specific implementation"""
        logger.debug(
            f"[RAGOwlAgent.message_invoke] Called from {self.name} with message: {message}"
        )
        logger.warning(f"RAG engine not keeping context for now")
        answer = self.invoke_rag(message)
        if "answer" not in answer or answer["answer"] == "":
            raise Exception("No answer found")
        return answer.get("answer", "?????")


# Instantiate by class name string
def create_instance(class_name):
    cls = globals()[class_name]
    return cls()


def main():

    # sprint(globals())

    # print("FrenchLawParser" in globals())

    config = RAG_AGENTS_CONFIG[0]

    load_logger_config()

    # sprint(config)

    rag_tool = RAGAgent(**config)

    if hasattr(rag_tool, "default_queries") and rag_tool.default_queries:
        for iq in rag_tool.default_queries:
            print(iq)
            print(rag_tool.invoke_rag(iq).get("answer"))
            print("-" * 100)
            break


if __name__ == "__main__":
    main()
