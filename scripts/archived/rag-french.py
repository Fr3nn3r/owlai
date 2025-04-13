# from tqdm import tqdm
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
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import pacmap
import numpy as np
import plotly.express as px
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from transformers import PreTrainedTokenizer

import logging
import logging.config
import yaml
import json

from tqdm import tqdm

from sentence_transformers import SentenceTransformer


def load_logger_config():
    with open("logging.yaml", "r") as logger_config:
        config = yaml.safe_load(logger_config)
        logging.config.dictConfig(config)


transformers.logging.set_verbosity_error()

load_logger_config()

logger = logging.getLogger("ragtool")


def main():

    def load_documents(input_folder: str) -> List[LangchainDocument]:
        """
        Load all documents from the input folder.
        Supports PDF and text files.
        Returns a list of LangchainDocument objects.
        """
        documents = []

        for filename in os.listdir(input_folder):
            filepath = os.path.join(input_folder, filename)

            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path=filepath, extract_images=False)
                docs = loader.load()
            elif filename.endswith(".txt"):
                loader = TextLoader(filepath)
                docs = loader.load()
            else:
                continue

            logger.info(f"Loading file: {filename} please wait...")

            documents.extend(
                [
                    LangchainDocument(
                        page_content=doc.page_content,
                        metadata={"source": doc.metadata["source"]},
                    )
                    for doc in docs
                ]
            )

        logger.info(f"Loaded {len(documents)} documents from {input_folder}")
        return documents

    # Your chunk size is allowed to vary from one snippet to the other.
    # Since there will always be some noise in your retrieval, increasing the top_k increases the chance to get relevant elements in your retrieved snippets.
    # the summed length of your retrieved documents should not be too high
    # The goal is to prepare a collection of semantically relevant chunks.
    # So their size should be adapted to precise ideas: too small will truncate ideas, and too large will dilute them.
    # We also have to keep in mind that when embedding documents, we will use an embedding model that accepts a certain maximum sequence length max_seq_length.
    # So we should make sure that our chunk sizes are below this limit because any longer chunk will be truncated before processing, thus losing relevancy
    def analyze_chunk_size_distribution(docs, model_name="thenlper/gte-small"):
        """
        Analyze and visualize document lengths before and after processing.

        Args:
            raw_docs: Original documents before splitting
            processed_docs: Documents after splitting
            model_name: Name of the embedding model to use
        """
        # Get max sequence length from SentenceTransformer
        max_seq_len = SentenceTransformer(model_name).max_seq_length
        info_message = (
            f"Model's max sequence size: '{max_seq_len}' Document count: '{len(docs)}'"
        )
        logger.info(info_message)

        # Analyze token lengths
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        lengths = [len(tokenizer.encode(doc.page_content)) for doc in docs]

        # Plot distribution
        fig = pd.Series(lengths).hist()
        plt.title("Distribution of document lengths (in count of tokens)")
        plt.savefig(f"{input_data_folder}/chunk_size_distribution-{id(docs)}.png")
        plt.close()
        logger.info(
            f"Distribution of document lengths (in count of tokens) saved to {input_data_folder}/chunk_size_distribution-{id(docs)}.png"
        )

    def split_documents(
        chunk_size: int,  # The maximum number of tokens in a chunk
        knowledge_base: List[LangchainDocument],
        tokenizer_name: str,
    ) -> List[LangchainDocument]:
        """
        Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
            chunk_size=chunk_size,
            chunk_overlap=int(
                chunk_size / 10
            ),  # The number of characters to overlap between chunks
            add_start_index=True,  # If `True`, includes chunk's start index in metadata
            strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
            separators=[
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )
        logger.debug(f"Splitting {len(knowledge_base)} documents")

        docs_processed = []
        for doc in tqdm(knowledge_base, desc="Splitting documents"):
            result = text_splitter.split_documents([doc])
            docs_processed += result

        logger.info(
            f"Splitted {len(knowledge_base)} documents into {len(docs_processed)} chunks"
        )
        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in tqdm(docs_processed, desc="Removing duplicates"):
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        logger.info(
            f"Removed {len(docs_processed) - len(docs_processed_unique)} duplicates from {len(docs_processed)} chunks"
        )

        return docs_processed_unique

    def load_and_index_documents(
        input_data_folder: str, embedding_model: HuggingFaceEmbeddings
    ):

        file_path = f"{input_data_folder}/vector_db"

        SPLIT_DOCUMENTS = None

        if os.path.exists(file_path):
            logger.info("Loading the vector database from disk")
            start_time = time.time()
            KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
                file_path,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
                allow_dangerous_deserialization=True,
            )
            end_time = time.time()
            print(
                f"Vector database loaded from disk in {end_time - start_time:.2f} seconds"
            )
        else:
            logger.info("Loading documents please wait...")
            start_time = time.time()
            RAW_KNOWLEDGE_BASE = load_documents(input_data_folder)
            end_time = time.time()
            logger.info(
                f"{len(RAW_KNOWLEDGE_BASE)} Documents loaded in {end_time - start_time:.2f} seconds"
            )
            analyze_chunk_size_distribution(
                RAW_KNOWLEDGE_BASE, embedding_model.model_name
            )

            logger.info("Splitting documents please wait...")
            start_time = time.time()
            SPLIT_DOCUMENTS = split_documents(
                512,  # We choose a chunk size adapted to our model
                RAW_KNOWLEDGE_BASE,
                tokenizer_name=embedding_model.model_name,
            )
            end_time = time.time()
            logger.info(f"Documents split in {end_time - start_time:.2f} seconds")
            # Analyze the documents after splitting
            analyze_chunk_size_distribution(SPLIT_DOCUMENTS, embedding_model.model_name)

            # Building the vector database

            start_time = time.time()
            KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
                SPLIT_DOCUMENTS,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
            )
            end_time = time.time()
            logger.info(f"Vector database built in {end_time - start_time:.2f} seconds")

            if not os.path.exists(f"{input_data_folder}/embeddings_visualization.html"):
                logger.info("Generating embedding visualization...")
                user_query = "Explain what happened in the summer of 2016?"
                query_vector = embedding_model.embed_query(user_query)
                fig = visualize_embeddings(
                    KNOWLEDGE_VECTOR_DATABASE, SPLIT_DOCUMENTS, user_query, query_vector
                )
                fig.write_html(f"{input_data_folder}/embeddings_visualization.html")

            # Save the vector database to disk
            KNOWLEDGE_VECTOR_DATABASE.save_local(file_path)

        return KNOWLEDGE_VECTOR_DATABASE

    def visualize_embeddings(
        KNOWLEDGE_VECTOR_DATABASE, docs_processed, user_query, query_vector
    ):
        embedding_projector = pacmap.PaCMAP(
            n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1
        )

        embeddings_2d = [
            list(KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n(idx, 1)[0])
            for idx in range(len(docs_processed))
        ] + [query_vector]

        # Fit the data (the index of transformed data corresponds to the index of the original data)
        documents_projected = embedding_projector.fit_transform(
            np.array(embeddings_2d), init="pca"
        )

        df = pd.DataFrame.from_dict(
            [
                {
                    "x": documents_projected[i, 0],
                    "y": documents_projected[i, 1],
                    "source": docs_processed[i].metadata["source"].split("/")[1],
                    "extract": docs_processed[i].page_content[:100] + "...",
                    "symbol": "circle",
                    "size_col": 4,
                }
                for i in range(len(docs_processed))
            ]
            + [
                {
                    "x": documents_projected[-1, 0],
                    "y": documents_projected[-1, 1],
                    "source": "User query",
                    "extract": user_query,
                    "size_col": 100,
                    "symbol": "star",
                }
            ]
        )

        # Visualize the embedding
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="source",
            hover_data="extract",
            size="size_col",
            symbol="symbol",
            color_discrete_map={"User query": "black"},
            width=1000,
            height=700,
        )
        fig.update_traces(
            marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )
        fig.update_layout(
            legend_title_text="<b>Chunk source</b>",
            title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>",
        )
        return fig

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
        logger.info(f"\nStarting retrieval for query: {query} with k={k}")
        start_time = time.time()
        retrieved_docs = knowledge_base.similarity_search(query=query, k=k)
        end_time = time.time()

        logger.info(
            f"{len(retrieved_docs)} documents retrieved in {end_time - start_time:.2f} seconds"
        )
        # logger.debug(f"Top documents: {retrieved_docs[0].page_content}")
        logger.debug(f"Top document metadata: {retrieved_docs[0].metadata}")

        return retrieved_docs

    def init_reader_llm() -> Tuple[Pipeline, PreTrainedTokenizer]:
        READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            READER_MODEL_NAME, quantization_config=bnb_config, low_cpu_mem_usage=True
        )

        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            READER_MODEL_NAME
        )

        reader_llm = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )

        # Test the model
        reader_llm("What is 4+4? Answer:")

        return reader_llm, tokenizer

    def get_rag_prompt_template(tokenizer: PreTrainedTokenizer):
        prompt_in_chat_format = [
            {
                "role": "system",
                "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
If the answer cannot be deduced from the context, do not give an answer.
If the answer cannot be deduced from the context, be brief.
Avoid using the word "context" or "extract" in the answer.
Avoid startements like "Based on the context provided" or "in the context provided" in the answer.""",
                # Provide the number of the source document when relevant.
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

        prompt_template = tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )
        return prompt_template

    def answer_with_rag(
        question: str,
        llm: Pipeline,
        knowledge_index: FAISS,
        reranker: Optional[RAGPretrainedModel] = None,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
        prompt_template: Optional[str] = None,
    ) -> Tuple[str, List[LangchainDocument]]:
        # Gather documents with retriever
        relevant_docs = retrieve_relevant_chunks(
            question, knowledge_index, k=num_retrieved_docs
        )

        relevant_docs = [
            doc.page_content for doc in relevant_docs
        ]  # Keep only the text

        # Optionally rerank results
        if reranker:
            logger.info("Reranking documents chunks please wait...")
            relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
            relevant_docs = [doc["content"] for doc in relevant_docs]

        relevant_docs = relevant_docs[:num_docs_final]

        # Build the final prompt
        context = "\nExtracted documents:\n"
        context += "".join(
            [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]
        )

        final_prompt = prompt_template.format(question=question, context=context)

        # Redact an answer
        start_time = time.time()
        answer = llm(final_prompt)[0]["generated_text"]
        end_time = time.time()
        logger.info(f"Answer generated in {end_time - start_time:.2f} seconds")

        return answer, relevant_docs

    # Start of the main function ######################################################
    EMBEDDING_MODEL_NAME = "camembert-base"

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={
            "normalize_embeddings": True
        },  # Set `True` for cosine similarity
    )

    datasets = [
        {
            "input_data_folder": "data/dataset-0000",
            "questions": [
                "Who is Rich Draves?",
                "What happened to Paul Graham in the fall of 1992?",
                "What was the last result of the AC Milan soccer team?",
                "What did Paul Graham do growing up?",
                "What did Paul Graham do during his school days?",
                "What languages did Paul Graham use?",
                "How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?",
                "How much was allocated to a implement a means-tested dental care program in the 2023 Canadian federal budget?",
                "What is the color of henry the fourth white horse?",
            ],
        },
        {
            "input_data_folder": "data/dataset-0001",
            "questions": [
                "Who is Naruto?",
                "Provide details about Orochimaru",
                "Who is the strongest ninja in the world?",
                "How is sasuke's personality",
                "Who is the sensei of naruto?",
                "What is a sharingan?",
                "What is the akatsuki?",
                "Who is the first Hokage?",
                "What was the last result of the AC Milan soccer team?",
                "What is the color of henry the fourth white horse?",
            ],
        },
        {
            "input_data_folder": "data/dataset-0003",
            "questions": [
                "What is Arakis?",
                "Who is Duncan Idaho?",
                "Who is the traitor?",
                "What are the powers of the Bene Gesserit?",
                "How does Paul defeat the Emperor?",
            ],
        },
        {
            "input_data_folder": "data/dataset-0002",
            "questions": [
                "Quelles sont les principales sources du droit en France ?",
                "Quelle est la différence entre le droit civil et la common law, et quel système utilise la France ?",
                "Quelles sont les principales branches du droit français ?",
                "Quel est le rôle de la Constitution française dans le système juridique ?",
                "Comment fonctionne le système judiciaire en France et quels sont les principaux types de tribunaux ?",
                "Quels sont les principes clés du droit des contrats en France ?",
                "Quels sont les droits des employés en vertu du droit du travail français ?",
                "Comment fonctionne le droit pénal en France et quels sont les principaux types d'infractions ?",
                "Quelles sont les règles essentielles régissant la propriété en France ?",
                "Comment le système juridique français protège-t-il les droits de l'homme et les libertés fondamentales ?",
            ],
        },
    ]

    dataset = datasets[3]

    input_data_folder = dataset["input_data_folder"]

    KNOWLEDGE_VECTOR_DATABASE = load_and_index_documents(
        input_data_folder, embedding_model
    )

    logger.info("Initializing reader LLM...")
    start_time = time.time()
    READER_LLM, tokenizer = init_reader_llm()
    end_time = time.time()
    logger.info(f"Reader LLM initialized in {end_time - start_time:.2f} seconds")

    logger.info("Initializing reranker...")
    start_time = time.time()
    RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    end_time = time.time()
    logger.info(f"Reranker initialized in {end_time - start_time:.2f} seconds")
    prompt_template = get_rag_prompt_template(
        tokenizer
    )  # PASS EITHER template or tokenizer to answer_with_rag...

    questions = dataset["questions"]

    qa_results = []
    for question in questions:
        answer, relevant_docs = answer_with_rag(
            question,
            READER_LLM,
            KNOWLEDGE_VECTOR_DATABASE,
            num_retrieved_docs=5,
            reranker=RERANKER,
            prompt_template=prompt_template,
        )
        logger.info(f"USER QUERY : {question}")
        logger.info(f"ANSWER : {answer}")

        qa_results.append({"question": question, "answer": answer})

    # Save results to JSON file
    output_file = os.path.join(input_data_folder, "qa_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_file}")
    # logger.info(f"SOURCE DOCS : {relevant_docs}")


if __name__ == "__main__":
    print("Application started")
    from multiprocessing import freeze_support

    freeze_support()
    main()
