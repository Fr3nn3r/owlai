from tqdm import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
import os
import time
import datasets
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, Pipeline
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import pacmap
import numpy as np
import plotly.express as px
from ragatouille import RAGPretrainedModel
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def main():

    #pd.set_option("display.max_colwidth", None)  # This will be helpful when visualizing retriever outputs

    #ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")

    loader = TextLoader("data/input/paul_graham_essay.txt")
    paul_graham_docs = loader.load()
    print(f"Number of paul_graham_docs: {len(paul_graham_docs)}")

    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=doc.page_content, metadata={"source": doc.metadata["source"]}) for doc in tqdm(paul_graham_docs)
    ]

    #RAW_KNOWLEDGE_BASE = [
    #    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
    #]
    #Your chunk size is allowed to vary from one snippet to the other.
    #Since there will always be some noise in your retrieval, increasing the top_k increases the chance to get relevant elements in your retrieved snippets. 
    #the summed length of your retrieved documents should not be too high

    #The goal is to prepare a collection of semantically relevant snippets. 
    # So their size should be adapted to precise ideas: too small will truncate ideas, and too large will dilute them.

    # We use a hierarchical list of separators specifically tailored for splitting Markdown documents
    # This list is taken from LangChain's MarkdownTextSplitter class
    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # The maximum number of characters in a chunk: we selected this value arbitrarily
        chunk_overlap=100,  # The number of characters to overlap between chunks
        add_start_index=True,  # If `True`, includes chunk's start index in metadata
        strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    #for doc in RAW_KNOWLEDGE_BASE:
        #docs_processed += text_splitter.split_documents([doc])

    #We also have to keep in mind that when embedding documents, we will use an embedding model that accepts a certain maximum sequence length max_seq_length.
    #So we should make sure that our chunk sizes are below this limit because any longer chunk will be truncated before processing, thus losing relevancy

    #from sentence_transformers import SentenceTransformer

    # To get the value of the max sequence_length, we will query the underlying `SentenceTransformer` object used in the RecursiveCharacterTextSplitter
    #print(f"Model's maximum sequence length: {SentenceTransformer('thenlper/gte-small').max_seq_length}")
    #print(f"{len(RAW_KNOWLEDGE_BASE)} inupt docs ----- {len(docs_processed)} documents after splitting")

    #from transformers import AutoTokenizer

    #tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
    #lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]

    # Plot the distribution of document lengths, counted as the number of tokens
    #fig = pd.Series(lengths).hist()
    #plt.title("Distribution of document lengths in the knowledge base (in count of tokens) chunk_size=1000 chunk_overlap=100 MARKDOWN_SEPARATORS")
    #plt.show()

    EMBEDDING_MODEL_NAME = "thenlper/gte-small"


    def split_documents(
        chunk_size: int,
        knowledge_base: List[LangchainDocument],
        tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
    ) -> List[LangchainDocument]:
        """
        Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )

        docs_processed = []
        for doc in knowledge_base:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique


    docs_processed = split_documents(
        512,  # We choose a chunk size adapted to our model
        RAW_KNOWLEDGE_BASE,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )

    print(f"{len(docs_processed)} documents after splitting")

    # Let's visualize the chunk sizes we would have in tokens from a common model
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
    fig = pd.Series(lengths).hist()
    plt.title("Distribution of document lengths in the knowledge base (in count of tokens) chunk_size=512")
    #plt.show()


    #################### Building the vector database

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    file_path = "data/paul_graham_vector_db"

    if os.path.exists(file_path):
        print("Loading the vector database from disk")
        start_time = time.time()
        KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
            file_path,
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True
        )
        end_time = time.time()
        print(f"Vector database loaded from disk in {end_time - start_time:.2f} seconds")
    else:
    
        print("Encoding documents please wait...")
  

        start_time = time.time()
        KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
            docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )
        end_time = time.time()
        print(f"Time taken to build vector database: {end_time - start_time:.2f} seconds")

        print("Encoding completed")

        # Save the vector database to disk
        KNOWLEDGE_VECTOR_DATABASE.save_local(file_path)

    # Embed a user query in the same space

    user_query = "Explain what happened in the summer of 2016?"

    query_vector = embedding_model.embed_query(user_query)

    #embedding_projector = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1)

    #embeddings_2d = [
    #    list(KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n(idx, 1)[0]) for idx in range(len(docs_processed))
    #] + [query_vector]

    # Fit the data (the index of transformed data corresponds to the index of the original data)
    #documents_projected = embedding_projector.fit_transform(np.array(embeddings_2d), init="pca")

    #df = pd.DataFrame.from_dict(
    #    [
    #        {
    #            "x": documents_projected[i, 0],
    #            "y": documents_projected[i, 1],
    #            "source": docs_processed[i].metadata["source"].split("/")[1],
    #            "extract": docs_processed[i].page_content[:100] + "...",
    #            "symbol": "circle",
    #            "size_col": 4,
    #        }
    #        for i in range(len(docs_processed))
        #]
        #+ [
    #            {
    #                "x": documents_projected[-1, 0],
    #                "y": documents_projected[-1, 1],
    #            "source": "User query",
    #            "extract": user_query,
    #            "size_col": 100,
    #            "symbol": "star",
    #        }
    #    ]
    #)

    # Visualize the embedding
    #fig = px.scatter(
    #    df,
    #    x="x",
    #    y="y",
    #    color="source",
    #    hover_data="extract",
    #    size="size_col",
    #    symbol="symbol",
    #    color_discrete_map={"User query": "black"},
    #    width=1000,
    #    height=700,
    #)
    #fig.update_traces(
    #    marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")),
    #    selector=dict(mode="markers"),
    #)
    #fig.update_layout(
    #    legend_title_text="<b>Chunk source</b>",
    #    title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>",
    #)
    #fig.show()

    print(f"\nStarting retrieval for {user_query=}...")
    start_time = time.time()
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    end_time = time.time()
    print(f"Retrieved documents in {end_time - start_time:.2f} seconds")
    print("\n==================================Top document==================================")
    print(retrieved_docs[0].page_content)
    print("==================================Metadata==================================")
    print(retrieved_docs[0].metadata)

    READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )

    READER_LLM("What is 4+4? Answer:")

    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
    give a comprehensive answer to the question.
    Respond only to the question asked, response should be concise and relevant to the question.
    Provide the number of the source document when relevant.
    If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
            "role": "user",
            "content": """Context:
    {context}
    ---
    Now here is the question you need to answer.

    Question: {question}""",
        },
    ]
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )
    print(RAG_PROMPT_TEMPLATE)

    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]  # We only need the text of the documents
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=user_query, context=context)

    # Redact an answer
    answer = READER_LLM(final_prompt)[0]["generated_text"]
    print(answer)

    RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    def answer_with_rag(
        question: str,
        llm: Pipeline,
        knowledge_index: FAISS,
        reranker: Optional[RAGPretrainedModel] = None,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
    ) -> Tuple[str, List[LangchainDocument]]:
        # Gather documents with retriever
        print("=> Retrieving documents...")
        start_time = time.time()
        relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
        end_time = time.time()
        print(f"Retrieved documents in {end_time - start_time:.2f} seconds")
        relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

        # Optionally rerank results
        if reranker:
            print("=> Reranking documents...")
            relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
            relevant_docs = [doc["content"] for doc in relevant_docs]

        relevant_docs = relevant_docs[:num_docs_final]

        # Build the final prompt
        context = "\nExtracted documents:\n"
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

        final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

        # Redact an answer
        print("=> Generating answer...")
        answer = llm(final_prompt)[0]["generated_text"]

        return answer, relevant_docs

    questions = [
        "Who is Rich Draves?",
        "What happened to Paul Graham in the fall of 1992?",
        "What can boost accuracy while maintaining GPU training and inference efficiency?",
        "What did Paul Graham do growing up?",
        "What did Paul Graham do during his school days?",
        "What languages did Paul Graham use?",
        "Who was Rich Draves?",
    ]
    for question in questions:
        answer, relevant_docs = answer_with_rag(question, READER_LLM, KNOWLEDGE_VECTOR_DATABASE, reranker=RERANKER)
        print("==================================Question==================================")
        print(f"{question}")
        print("==================================Answer==================================")
        print(f"{answer}")
        #print("==================================Source docs==================================")
        #for i, doc in enumerate(relevant_docs):
        #    print(f"Document {i}------------------------------------------------------------")
        #    print(doc[:100])

if __name__ == "__main__":
    print("Application started")
    from multiprocessing import freeze_support
    freeze_support()
    main()
