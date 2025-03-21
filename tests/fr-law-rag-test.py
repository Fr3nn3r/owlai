if __name__ == "__main__":
    import time  # this prevents reloading when multi processing is used
    from datetime import datetime

    main_start_time = datetime.now()

    import os
    import transformers

    import logging
    import logging.config
    import yaml
    import json

    from owlai.rag import RAGOwlAgent
    from owlai.core import OwlAgent
    from owlai.db import TOOLS_CONFIG, PROMPT_CONFIG
    from langchain.chat_models import init_chat_model

    from tqdm import tqdm

    from owlai.owlsys import get_system_info, track_time

    import fitz  # PyMuPDF
    import re

    def load_logger_config():
        with open("logging.yaml", "r") as logger_config:
            config = yaml.safe_load(logger_config)
            logging.config.dictConfig(config)

    transformers.logging.set_verbosity_error()

    load_logger_config()

    logger = logging.getLogger("main")

    def extract_footer(doc):
        footers = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text and split into lines
            text = page.get_text("text")
            lines = text.split("\n")

            if len(lines) > 1:
                footer = lines[-2:]  # Assume footer is in the last two lines
                footers.append((page_num + 1, " | ".join(footer)))

        return footers

    def extract_metadata_fr_law(pdf_path):
        """
        Extract metadata from a PDF file footer (file expected to follow french law convention).

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            dict: Dictionary containing title, last_modification, and doc_generated_on
        """
        doc = fitz.open(pdf_path)

        # Get footer from the first page
        footers = extract_footer(doc)
        if not footers:
            raise ValueError(f"No footer found in the document '{pdf_path}'")

        footer = footers[0][1]

        # Regular Expression to Extract Components
        match = re.match(r"^(.*?)\s*-\s*(.*?)\s*-\s*(.*?)$", footer)
        if match:
            title = match.group(1).strip()
            last_modification = match.group(2).strip()
            doc_generated_on = match.group(3).strip()

            return {
                "title": title,
                "last_modification": last_modification,
                "doc_generated_on": doc_generated_on,
                "num_pages": len(doc),
            }

        raise ValueError(f"footer '{footer}' not matching french law convention.")

    def document_curator(doc_content: str, file_path: str) -> str:
        """
        Curates documents to be used by the RAG engine.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            dict: Dictionary containing title, last_modification, and doc_generated_on
        """
        doc = fitz.open(pdf_path)

        # Get footer from the first page
        footers = extract_footer(doc)
        if not footers:
            raise ValueError(f"No footer found in the document '{pdf_path}'")

        for footer in footers:
            doc_content = doc_content.replace(footer, "")
            logger.info(f"Removed footer: {footer}")

        return doc_content

    def load_fr_law_pdf(pdf_path: str) -> str:
        """
        Loads a french law PDF file and returns the content.
        """
        doc = fitz.open(pdf_path)
        
        for i in range(len(doc)):
        page = doc[i]               # Page is loaded here
        text = page.get_text()      # Text is extracted now
        print(f"Page {i + 1}:", text[:200])
        
        return doc.get_text("text")




    def main():

        execution_log = {}

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
            }

            dataset = datasets["large_v1"]

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

            # Using the metadata extractor function with the vector store creation
            # this is allowing the caller to specify how the metadata is extracted from the documents
            # and stored in the vector store
            with track_time("Vector store loading", execution_log):
                KNOWLEDGE_VECTOR_DATABASE = rag_tool.load_dataset(
                    input_data_folder,
                    embedding_model,
                    metadata_extractor=extract_metadata_fr_law,
                    document_curator=document_curator,
                )

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
