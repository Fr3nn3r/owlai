import faiss
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from langchain.docstore.document import Document


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


def analyze_documents(documents):
    try:
        print("Analyzing document chunks...")
        lengths = [len(doc.page_content) for doc in documents]

        print(f"Total documents: {len(lengths)}")
        print(f"Avg chunk length: {np.mean(lengths):.2f} chars")
        print(f"Max chunk length: {np.max(lengths)}")
        print(f"Min chunk length: {np.min(lengths)}")

        plt.hist(lengths, bins=30)
        plt.title("Chunk Size Distribution (by characters)")
        plt.xlabel("Chunk Size (characters)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Error analyzing documents: {e}")


def analyze_faiss_index(index):
    print("Analyzing FAISS index...")
    num_vectors = index.ntotal
    dim = index.d
    print(f"Total vectors in index: {num_vectors}")
    print(f"Vector dimension: {dim}")


def main(index_path, pickle_path):
    index = load_faiss_index(index_path)
    docs = load_documents(pickle_path)

    analyze_faiss_index(index)
    analyze_documents(docs)


if __name__ == "__main__":
    main(
        "C:/Users/fbrun/Documents/GitHub/owlai/data/cache/naruto-complete/vector_db/index.faiss",
        "C:/Users/fbrun/Documents/GitHub/owlai/data/cache/naruto-complete/vector_db/index.pkl",
    )
