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
                        "input_data_folders": [dataset["input_data_folder"]],
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

            qa_results = {}
            qa_results["system_info"] = get_system_info()

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

        for i, question in enumerate(questions, 1):

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
