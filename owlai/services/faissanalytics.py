import faiss
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from langchain.docstore.document import Document
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer


def ensure_dir_exists(dir_path):
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def save_plot(plt, filename, title=None):
    """Save plot to the specified location with given filename."""
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()


def load_faiss_index(index_path):
    try:
        print(f"Loading FAISS index from {index_path}...")
        return faiss.read_index(index_path)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        # Create a dummy index to allow the program to continue
        print("Creating a dummy index to allow analysis to proceed")
        return faiss.IndexFlatL2(1536)  # Common embedding size


def load_documents(pickle_path):
    print(f"Loading LangChain documents from {pickle_path}...")
    try:
        with open(pickle_path, "rb") as f:
            docs_data = pickle.load(f)

        # LangChain FAISS storage structure is typically a tuple containing:
        # (docstore, index_to_docstore_id, [docstore_to_index])
        # Where docstore is an InMemoryDocstore object with a _dict attribute containing the documents
        if isinstance(docs_data, tuple):
            print(
                f"Found a LangChain FAISS storage tuple with {len(docs_data)} elements"
            )

            # Try to extract documents from the docstore (first element of tuple)
            if len(docs_data) >= 2:
                # First element should be a docstore object with a _dict attribute
                docstore = docs_data[0]
                print(f"First element type: {type(docstore)}")

                # Check if it's an InMemoryDocstore or has a _dict attribute
                if hasattr(docstore, "_dict"):
                    doc_dict = docstore._dict
                    print(
                        f"Successfully identified docstore with {len(doc_dict)} documents"
                    )

                    # Convert document dict to list
                    doc_list = list(doc_dict.values())

                    # Check if these are Document objects
                    if all(isinstance(doc, Document) for doc in doc_list):
                        print("All items are LangChain Document objects")
                        return doc_list
                    else:
                        print("Converting dictionary items to Document objects")
                        converted_docs = []
                        for doc_id, doc_data in doc_dict.items():
                            # Try to convert to Document if it's not already
                            if isinstance(doc_data, Document):
                                converted_docs.append(doc_data)
                            elif isinstance(doc_data, dict):
                                # Try different common field names for the content
                                content = (
                                    doc_data.get("page_content")
                                    or doc_data.get("text")
                                    or doc_data.get("content")
                                    or ""
                                )
                                metadata = doc_data.get("metadata", {})
                                converted_docs.append(
                                    Document(page_content=content, metadata=metadata)
                                )

                        print(
                            f"Converted {len(converted_docs)} documents from the docstore"
                        )
                        return converted_docs
                elif isinstance(docstore, dict):
                    # Direct dictionary handling
                    doc_dict = docstore
                    print(
                        f"Found direct dictionary docstore with {len(doc_dict)} items"
                    )

                    # Convert document dict to list
                    doc_list = list(doc_dict.values())

                    # Check if these are Document objects
                    if all(isinstance(doc, Document) for doc in doc_list):
                        print("All items are LangChain Document objects")
                        return doc_list
                    else:
                        print("Converting dictionary items to Document objects")
                        converted_docs = []
                        for doc_id, doc_data in doc_dict.items():
                            # Try to convert to Document if it's not already
                            if isinstance(doc_data, Document):
                                converted_docs.append(doc_data)
                            elif isinstance(doc_data, dict):
                                # Try different common field names for the content
                                content = (
                                    doc_data.get("page_content")
                                    or doc_data.get("text")
                                    or doc_data.get("content")
                                    or ""
                                )
                                metadata = doc_data.get("metadata", {})
                                converted_docs.append(
                                    Document(page_content=content, metadata=metadata)
                                )

                        print(
                            f"Converted {len(converted_docs)} documents from the docstore"
                        )
                        return converted_docs
                else:
                    print(
                        f"First element doesn't have a _dict attribute and is not a dict: {type(docstore)}"
                    )

                    # Try to see if the second element is a mapping from index to docstore ids
                    if isinstance(docs_data[1], dict):
                        index_to_id_map = docs_data[1]
                        print(
                            f"Found index_to_docstore_id map with {len(index_to_id_map)} entries"
                        )

                        # We can use this mapping to create dummy documents
                        dummy_docs = []
                        for idx, doc_id in index_to_id_map.items():
                            dummy_docs.append(
                                Document(
                                    page_content=f"Document {doc_id} at index {idx}",
                                    metadata={"doc_id": doc_id, "index": idx},
                                )
                            )

                        print(
                            f"Created {len(dummy_docs)} dummy documents from the index mapping"
                        )
                        return dummy_docs
            else:
                print(
                    f"Tuple doesn't have enough elements (needs at least 2, has {len(docs_data)})"
                )

        # Handle different types of data structures for non-tuple data
        if isinstance(docs_data, list):
            print(f"Loaded {len(docs_data)} items from pickle file")

            # Check if they are LangChain Document objects
            if all(isinstance(doc, Document) for doc in docs_data):
                print("All items are LangChain Document objects")
                return docs_data

            # If they are dictionaries with text/content fields, convert to Documents
            elif all(isinstance(doc, dict) for doc in docs_data):
                print("Converting dictionary items to Document objects")
                converted_docs = []
                for doc in docs_data:
                    # Try different common field names for the content
                    content = (
                        doc.get("page_content")
                        or doc.get("text")
                        or doc.get("content")
                        or ""
                    )
                    metadata = doc.get("metadata", {})
                    converted_docs.append(
                        Document(page_content=content, metadata=metadata)
                    )
                return converted_docs

            # If they are strings, convert to Documents
            elif all(isinstance(doc, str) for doc in docs_data):
                print("Converting string items to Document objects")
                return [Document(page_content=text) for text in docs_data]

            # Mixed content - try to extract text where possible
            else:
                print("Mixed content detected - attempting to extract text")
                converted_docs = []
                for item in docs_data:
                    if isinstance(item, Document):
                        converted_docs.append(item)
                    elif isinstance(item, dict) and (
                        "page_content" in item or "text" in item or "content" in item
                    ):
                        content = (
                            item.get("page_content")
                            or item.get("text")
                            or item.get("content")
                            or ""
                        )
                        metadata = item.get("metadata", {})
                        converted_docs.append(
                            Document(page_content=content, metadata=metadata)
                        )
                    elif isinstance(item, str):
                        converted_docs.append(Document(page_content=item))
                    else:
                        # Try to convert to string if possible
                        try:
                            converted_docs.append(Document(page_content=str(item)))
                        except:
                            print(f"Skipping incompatible item: {type(item)}")

                if not converted_docs:
                    raise ValueError("Could not convert any items to Document objects")

                print(
                    f"Successfully converted {len(converted_docs)} items to Document objects"
                )
                return converted_docs

        # Handle other types of data
        else:
            print(f"Pickle contains a {type(docs_data)} instead of a list or tuple")
            # If it's a dict with documents
            if isinstance(docs_data, dict) and any(
                isinstance(v, Document) for v in docs_data.values()
            ):
                return list(docs_data.values())
            # Otherwise raise an error
            raise ValueError(f"Unsupported data format: {type(docs_data)}")

    except Exception as e:
        print(f"Error loading documents: {e}")
        # Create dummy document to allow the program to continue
        print("Creating dummy documents to allow analysis to proceed")
        return [Document(page_content="Dummy document due to load error")]


def analyze_documents_by_chars(documents, dataset_name):
    """Analyze document chunks by character length and save visualization."""
    try:
        print("Analyzing document chunks by character count...")
        lengths = [len(doc.page_content) for doc in documents]

        # Print statistics
        print(f"Total documents: {len(lengths)}")
        print(f"Avg chunk length: {np.mean(lengths):.2f} chars")
        print(f"Max chunk length: {np.max(lengths)}")
        print(f"Min chunk length: {np.min(lengths)}")

        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=30)
        plt.title(f"Chunk Size Distribution (by characters) - {dataset_name}")
        plt.xlabel("Chunk Size (characters)")
        plt.ylabel("Frequency")
        plt.grid(True)

        # Save the plot
        analysis_dir = ensure_dir_exists("data/analysis")
        filename = os.path.join(
            analysis_dir, f"{dataset_name}_char_distribution_{timestamp}.png"
        )
        save_plot(plt, filename)

        return filename
    except Exception as e:
        print(f"Error analyzing documents by character count: {e}")
        return None


def analyze_documents_by_tokens(
    documents, dataset_name, model_name="thenlper/gte-small"
):
    """Analyze document chunks by token length and save visualization."""
    try:
        print(f"Analyzing document chunks by token count using model {model_name}...")

        # Load tokenizer for the specified model
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get max sequence length from tokenizer with fallback to reasonable default
        max_seq_len = tokenizer.model_max_length
        if max_seq_len > 100000:  # If unreasonably large
            # Use common model size limits as fallback
            if "gpt-3.5" in model_name.lower() or "gpt-4" in model_name.lower():
                max_seq_len = 8192  # Common limit for GPT models
            elif "llama" in model_name.lower():
                max_seq_len = 4096  # Common limit for LLaMA models
            elif "t5" in model_name.lower():
                max_seq_len = 512
            else:
                max_seq_len = 512  # Conservative default
            print(f"Adjusted to realistic token limit: {max_seq_len}")
        print(f"Model's max sequence size: {max_seq_len} tokens")

        # Analyze token lengths
        token_lengths = [len(tokenizer.encode(doc.page_content)) for doc in documents]
        total_tokens = sum(token_lengths)

        # Print statistics
        print(f"Total tokens across all documents: {total_tokens}")
        print(f"Avg tokens per chunk: {np.mean(token_lengths):.2f}")
        print(f"Max tokens per chunk: {np.max(token_lengths)}")
        print(f"Min tokens per chunk: {np.min(token_lengths)}")

        # Count chunks that exceed max length
        oversize_chunks = [l for l in token_lengths if l > max_seq_len]
        if oversize_chunks:
            print(
                f"WARNING: {len(oversize_chunks)} chunks ({len(oversize_chunks)/len(token_lengths):.1%}) exceed model's maximum size"
            )

        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Plot histogram
        plt.figure(figsize=(10, 6))
        # Use pandas for better histogram
        pd.Series(token_lengths).hist(bins=30)
        plt.axvline(
            x=max_seq_len,
            color="r",
            linestyle="--",
            label=f"Max tokens ({max_seq_len})",
        )
        plt.legend()
        plt.title(f"Chunk Size Distribution (by tokens) - {dataset_name}")
        plt.xlabel("Tokens per Chunk")
        plt.ylabel("Frequency")
        plt.grid(True)

        # Save the plot
        analysis_dir = ensure_dir_exists("data/analysis")
        filename = os.path.join(
            analysis_dir, f"{dataset_name}_token_distribution_{timestamp}.png"
        )
        save_plot(plt, filename)

        return filename
    except Exception as e:
        print(f"Error analyzing documents by token count: {e}")
        return None


def analyze_faiss_index(index, dataset_name):
    try:
        print("Analyzing FAISS index...")
        num_vectors = index.ntotal
        dim = index.d
        print(f"Total vectors in index: {num_vectors}")
        print(f"Vector dimension: {dim}")

        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save index metadata to a text file
        analysis_dir = ensure_dir_exists("data/analysis")
        metadata_file = os.path.join(
            analysis_dir, f"{dataset_name}_index_metadata_{timestamp}.txt"
        )

        with open(metadata_file, "w") as f:
            f.write(f"FAISS Index Analysis for {dataset_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total vectors in index: {num_vectors}\n")
            f.write(f"Vector dimension: {dim}\n")
            f.write(f"Index type: {type(index).__name__}\n")

        print(f"Saved index metadata to {metadata_file}")

        return {
            "num_vectors": num_vectors,
            "dimension": dim,
            "metadata_file": metadata_file,
        }
    except Exception as e:
        print(f"Error analyzing FAISS index: {e}")
        return None


def main(index_path, pickle_path):
    # Extract dataset name from path
    dataset_name = os.path.basename(os.path.dirname(index_path))
    print(f"Analyzing dataset: {dataset_name}")

    # Load index and documents
    index = load_faiss_index(index_path)
    docs = load_documents(pickle_path)

    # Analyze index
    index_metadata = analyze_faiss_index(index, dataset_name)

    # Analyze documents by character count
    char_analysis_file = analyze_documents_by_chars(docs, dataset_name)

    # Analyze documents by token count
    token_analysis_file = analyze_documents_by_tokens(docs, dataset_name)

    # Return analysis results
    return {
        "dataset": dataset_name,
        "index_metadata": index_metadata,
        "char_analysis_file": char_analysis_file,
        "token_analysis_file": token_analysis_file,
        "document_count": len(docs),
    }


if __name__ == "__main__":
    analysis_results = main(
        "C:/Users/fbrun/Documents/GitHub/owlai/data/cache/fr-law-complete/vector_db/index.faiss",
        "C:/Users/fbrun/Documents/GitHub/owlai/data/cache/fr-law-complete/vector_db/index.pkl",
    )

    print("\nAnalysis Summary:")
    print(f"Dataset: {analysis_results['dataset']}")
    print(f"Document count: {analysis_results['document_count']}")
    print(f"Character distribution chart: {analysis_results['char_analysis_file']}")
    print(f"Token distribution chart: {analysis_results['token_analysis_file']}")
    if analysis_results["index_metadata"]:
        print(f"Index metadata: {analysis_results['index_metadata']['metadata_file']}")
        print(f"Vector count: {analysis_results['index_metadata']['num_vectors']}")
        print(f"Vector dimension: {analysis_results['index_metadata']['dimension']}")
