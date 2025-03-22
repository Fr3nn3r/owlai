if __name__ == "__main__":
    import time  # this prevents reloading when multi processing is used
    from datetime import datetime

    main_start_time = datetime.now()

    import os
    import sys
    import logging
    import logging.config
    import yaml
    import json
    import warnings
    import transformers  # Add missing import

    from tqdm import tqdm

    import fitz  # PyMuPDF
    import re

    from langchain.docstore.document import Document as LangchainDocument
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import SystemMessage
    from langchain_core.runnables import RunnableConfig
    from ragatouille import RAGPretrainedModel
    from pydantic import BaseModel, Field
    from langchain_core.tools import BaseTool, ArgsSchema
    from langchain_core.callbacks import (
        AsyncCallbackManagerForToolRun,
        CallbackManagerForToolRun,
    )
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    import traceback
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer

    from owlai.rag import RAGOwlAgent
    from owlai.core import OwlAgent
    from owlai.db import TOOLS_CONFIG, PROMPT_CONFIG
    from langchain.chat_models import init_chat_model

    from owlai.owlsys import get_system_info, track_time, load_logger_config, sprint
    from owlai.owlsys import encode_text

    warnings.simplefilter("ignore", category=FutureWarning)

    # Import functions from fr-law-load-docs.py
    from fr_law_load_docs import (
        load_fr_law_pdf,
        analyze_chunk_size_distribution,
        extract_footer,
        extract_metadata_fr_law,
    )

    transformers.logging.set_verbosity_error()

    # Use centralized logging configuration
    load_logger_config()
    logger = logging.getLogger("main")

    def document_curator(doc_content: str, file_path: str) -> str:
        """
        Curates documents to be used by the RAG engine by removing footers.

        Args:
            doc_content (str): The document content to curate
            file_path (str): Path to the document file

        Returns:
            str: Curated document content with footer removed
        """
        # Split content into lines
        lines = doc_content.split("\n")

        # Remove the last two lines (footer) if there are at least 2 lines
        if len(lines) > 2:
            curated_content = "\n".join(lines[:-2])
            logger.info(f"Removed footer from document: {file_path}")
        else:
            curated_content = doc_content
            logger.info(f"No footer found in document: {file_path}")

        return curated_content

    def index_and_save_to_disk(
        folder_path: str, vector_store: FAISS, embedding_model: SentenceTransformer
    ):
        docs = load_fr_law_pdf_from_folder(folder_path)
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
        in_store_folder = os.path.join(input_data_folder, "in_store")
        os.makedirs(in_store_folder, exist_ok=True)
        os.rename(filepath, os.path.join(in_store_folder, filename))

    def main():

        execution_log = {}

        print("QQQQQQQQQQQQQQQQQQQStarting test")

        with track_time("Loading resources", execution_log):
            # Define question sets
            question_sets = {
                "general_law_questions": [
                    "Quelles sont les différences essentielles entre la responsabilité contractuelle et délictuelle ?",
                    "Expliquez les conditions de validité d'un contrat et les conséquences juridiques de leur non-respect.",
                    "Comment s'articule le mécanisme de la prescription en matière civile ? Citez des exemples de délais spécifiques.",
                    "Quelles sont les principales différences entre une SARL et une SAS en termes de gouvernance et de responsabilité ?",
                    "Expliquez le régime juridique des actes de concurrence déloyale et leurs sanctions.",
                    "Comment s'opère la cession d'un fonds de commerce et quelles sont les formalités obligatoires ?",
                    "Quelles sont les modalités légales de rupture d'un CDI et leurs spécificités procédurales ?",
                    "Expliquez les obligations de l'employeur en matière de santé et sécurité au travail.",
                    "Comment caractériser juridiquement le harcèlement moral et quelles sont les voies de recours pour un salarié ?",
                    "Quelles sont les conditions de validité d'un bail commercial et ses clauses essentielles ?",
                    "Expliquez les différentes servitudes et leurs modes d'établissement.",
                    "Comment s'opère la vente en l'état futur d'achèvement (VEFA) et quelles garanties offre-t-elle ?",
                    "Un client vous consulte pour un litige avec son fournisseur. Quelles informations collecteriez-vous et quelle serait votre méthodologie d'analyse ?",
                    "Comment rédigeriez-vous une clause de non-concurrence qui soit à la fois protectrice pour l'entreprise et juridiquement valable ?",
                    "Face à un conflit d'intérêts potentiel, quelle démarche adopteriez-vous ?",
                    "Explique la gestion en france de la confusion des peines",
                ],
                "fiscal_law_questions": [
                    "Quelles sont les principales obligations fiscales d'une entreprise soumise à l'impôt sur les sociétés (IS) en France ?",
                    "Quels sont les critères permettant de déterminer si une opération est soumise à la TVA en France ?",
                    "Quelles sont les principales conventions fiscales internationales et comment permettent-elles d'éviter la double imposition ?",
                    "Quels sont les principaux droits et obligations d'une entreprise lors d'un contrôle fiscal ?",
                    "Quels sont les recours possibles pour une entreprise contestant un redressement fiscal ?",
                    "Quels sont les principaux impôts applicables aux transmissions de patrimoine en France ?",
                    "Quelles sont les obligations déclaratives en matière de prix de transfert pour les entreprises multinationales ?",
                    "Comment la notion d'abus de droit fiscal est-elle définie en droit français et quelles en sont les conséquences ?",
                    "Quels sont les principaux enjeux de la conformité fiscale pour une entreprise et comment un juriste fiscaliste peut-il y contribuer ?",
                    "Pouvez-vous nous parler d'une récente réforme fiscale qui a eu un impact significatif sur les entreprises en France ?",
                ],
            }

            # Define datasets with references to question sets
            datasets = {
                "first_try": {
                    "input_data_folder": "data/dataset-0002",
                    "questions": question_sets["general_law_questions"],
                },
                "fiscal_law_only": {
                    "input_data_folder": "data/dataset-0004",  # dataset 4 droit fiscal
                    "questions": question_sets["fiscal_law_questions"],
                },
                "large_v1": {
                    "input_data_folder": "data/dataset-0005",
                    "questions": question_sets["general_law_questions"]
                    + question_sets["fiscal_law_questions"],
                },
                "load_only": {
                    "input_data_folder": "data/dataset-0005",  # for loading only
                    "questions": [
                        "Explique la gestion en france de la confusion des peines"
                    ],
                },
                "large_v2": {
                    "input_data_folder": "data/dataset-0006",  # for loading only
                    "questions": [
                        "Explique la gestion en france de la confusion des peines"
                    ],
                },
            }

            dataset = datasets["large_v2"]

            RAG_AGENTS_CONFIG = [
                {
                    "name": "rag-fr-law-v1",
                    "description": "Agent expecting a french law question",
                    "args_schema": {
                        "query": {
                            "type": "string",
                            "description": "Any question about french law expressed in french",
                        }
                    },
                    "model_provider": "mistralai",
                    "model_name": "mistral-large-latest",
                    "max_tokens": 4096,
                    "temperature": 0.1,
                    "context_size": 4096,
                    "tools_names": [],
                    "system_prompt": PROMPT_CONFIG["rag-fr-v2"],
                    "default_queries": [],
                    "retriever": {
                        "num_retrieved_docs": 30,
                        "num_docs_final": 5,
                        "embeddings_model_name": "thenlper/gte-small",
                        "reranker_name": "colbert-ir/colbertv2.0",
                        "input_data_folders": [],
                        "model_kwargs": {"device": "cuda"},
                        "encode_kwargs": {"normalize_embeddings": True},
                        "multi_process": True,
                    },
                }
            ]
            input_data_folder = dataset["input_data_folder"]

            CONTROL_LLM_CONFIG = [
                {
                    "name": "control-llm",
                    "description": "Agent without RAG to compare answers with the RAG engine",
                    "model_provider": "openai",
                    "model_name": "gpt-4o",
                    "max_tokens": 4096,
                    "temperature": 0.1,
                    "context_size": 4096,
                    "tools_names": [],
                    "system_prompt": PROMPT_CONFIG["rag-fr-control-llm-v1"],
                    "default_queries": [],
                }
            ]

            rag_tool = RAGOwlAgent(**RAG_AGENTS_CONFIG[0])
            control_llm = OwlAgent(**CONTROL_LLM_CONFIG[0])

            input_data_folder = dataset["input_data_folder"]

            docs = []
            KNOWLEDGE_VECTOR_DATABASE = None
            total_chunks = 0
            for file in os.listdir(input_data_folder):
                if file.endswith(".pdf"):
                    logger.info(f"Processing file: {file}")
                    doc = load_fr_law_pdf(os.path.join(input_data_folder, file))
                    total_chunks += len(doc)
                    docs.append(doc)
                    analyze_chunk_size_distribution(
                        input_data_folder, file, doc, "thenlper/gte-small"
                    )
                    logger.info(f"Loading vector database from split docs")
                    KNOWLEDGE_VECTOR_DATABASE = rag_tool.load_dataset_from_split_docs(
                        docs,
                        input_data_folder,
                        rag_tool._embeddings,
                        KNOWLEDGE_VECTOR_DATABASE,
                    )

            rag_tool._vector_stores = KNOWLEDGE_VECTOR_DATABASE

            print(f"Total chunks in store: {total_chunks}")

            qa_results = {}
            # qa_results["system_info"] = get_system_info()

            # Merge the two dictionaries to store test parameters
            qa_results["test_parameters"] = {
                "rag_config": RAG_AGENTS_CONFIG[0],
                "control_llm_config": CONTROL_LLM_CONFIG[0],
            }

            qa_results["dataset"] = dataset

            questions = dataset["questions"]
            # Generate a filename for the test report using the current date and time
            start_date_time = time.strftime("%Y%m%d-%H%M%S")
            test_report_filename = f"{start_date_time}-qa_results.json"

        for i, question in enumerate(questions[0:5], 1):

            with track_time(f"RAG Question {i}", execution_log):
                answer = rag_tool.rag_question(question)

                logger.info(f"USER QUERY : {question}")
                logger.info(f"ANSWER : {answer['answer']}")

            with track_time(f"Control LLM Question {i}", execution_log):
                # Build the prompt from the template and the question
                control_prompt = PROMPT_CONFIG["rag-fr-control-llm-v1"].format(
                    question=question
                )
                # Invoke the model with the formatted prompt
                answer_control_llm = control_llm.invoke(control_prompt)
                logger.info(f"CONTROL LLM ANSWER : {answer_control_llm}")

            qa_results[f"Test #{i}"] = {
                "question": question,
                "answer": answer["answer"],
                "answer_control_llm": answer_control_llm,
                "metadata": answer["metadata"],
            }

        qa_results["execution_log"] = execution_log

        # Save results to JSON file
        output_file = os.path.join(input_data_folder, test_report_filename)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(qa_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
        # logger.info(f"SOURCE DOCS : {relevant_docs}")

    print("Application started")
    from multiprocessing import freeze_support
    from dotenv import load_dotenv

    load_dotenv()
    freeze_support()
    main()
